// Qwen3.5/Qwen3.8 text-layer adapter for llama.cpp b10837 GGML.
// The graph mirrors src/models/qwen35.cpp. GGUF conversion already applies
// zero-centered norm offsets, -exp(A_log), and the DeltaNet value-head reorder.
// No custom quantization/attention CUDA kernels and no GGML RPC protocol here.
#include "adapter.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef DIAL_GGML_CUDA
#include "ggml-cuda.h"
#include <cuda_runtime_api.h>
#endif
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using T = ggml_tensor;
void require(bool ok, const std::string & message) {
    if (!ok) throw std::runtime_error(message);
}
void error_text(char * out, size_t size, const char * message) {
    if (out && size) std::snprintf(out, size, "%s", message);
}
struct Context {
    ggml_context * ctx = nullptr;
    Context(size_t bytes = 2 * 1024 * 1024) {
        ctx = ggml_init({bytes, nullptr, true});
        require(ctx != nullptr, "GGML metadata allocation failed");
    }
    ~Context() { ggml_free(ctx); }
    Context(const Context &) = delete;
    Context & operator=(const Context &) = delete;
};
struct Weights {
    Context context;
    ggml_backend_buffer_t buffer = nullptr;
    std::map<std::string, T *> tensors;
    uint64_t bytes = 0;
    ~Weights() { if (buffer) ggml_backend_buffer_free(buffer); }
    T * at(const std::string & name) { return tensors.at(name); }
};
struct Graph {
    Context context;
    ggml_cgraph * graph;
    ggml_gallocr_t alloc;
    T * input = nullptr;
    T * output = nullptr;
    T * positions = nullptr;
    T * rows = nullptr;
    T * mask = nullptr;
    int64_t tokens;
    int64_t attention_length;
    Graph(ggml_backend_t backend, int64_t n, size_t layers = 1, int64_t kv_length = 0) :
        context(2 * 1024 * 1024 + layers * 128 * 1024),
        graph(ggml_new_graph_custom(context.ctx, 512 * layers, false)),
        alloc(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend))),
        tokens(n), attention_length(kv_length) {
        require(alloc != nullptr, "GGML graph allocator creation failed");
    }
    ~Graph() { ggml_gallocr_free(alloc); }
    ggml_context * ctx() { return context.ctx; }
    void finish(ggml_backend_t backend) {
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            T * node = ggml_graph_node(graph, i);
            require(ggml_backend_supports_op(backend, node),
                    std::string("GGML backend does not support ") + ggml_op_name(node->op));
        }
        require(ggml_gallocr_alloc_graph(alloc, graph), "GGML graph allocation failed (out of memory)");
    }
};
struct Layer {
    int index;
    bool linear;
    std::unique_ptr<Weights> weights;
    std::string prefix() const { return "blk." + std::to_string(index) + "."; }
    T * w(const std::string & suffix) { return weights->at(prefix() + suffix); }
};
struct Engine {
    dial_ggml_config c;
    gguf_context * gguf = nullptr;
    ggml_context * metadata = nullptr;
    ggml_backend_t backend = nullptr;
    std::ifstream file;
    int device;
    int64_t capacity;
    uint64_t graph_calls = 0;
    uint64_t range_calls = 0;
    bool cuda_graphs_available = false;
    std::map<int, std::unique_ptr<Layer>> layers;
    std::unique_ptr<Weights> head_weights;
    std::unique_ptr<Graph> head_graph;
    std::unique_ptr<Graph> embed_graph;
    Engine(const char * path, const dial_ggml_config & cfg, int dev) :
        c(cfg), file(path, std::ios::binary), device(dev), capacity((cfg.max_seq + 255) / 256 * 256) {}
    ~Engine() {
        embed_graph.reset(); head_graph.reset(); head_weights.reset(); layers.clear();
        if (backend) ggml_backend_free(backend);
        if (gguf) gguf_free(gguf);
        if (metadata) ggml_free(metadata);
    }
    void init(const char * path, int threads) {
        require(c.hidden > 0 && c.intermediate > 0 && c.vocab > 0 && c.layers > 0 &&
                c.max_seq > 0 && c.max_seq <= INT32_MAX - 255 && c.q_heads > 0 && c.kv_heads > 0 &&
                c.q_heads % c.kv_heads == 0 && c.head_dim > 0 && c.rotary_dim > 0 &&
                c.rotary_dim <= c.head_dim && c.rotary_dim % 2 == 0 && c.key_dim == c.value_dim &&
                c.key_heads > 0 && c.value_heads % c.key_heads == 0 && c.conv_kernel > 1,
                "Invalid or unsupported Qwen3.8 GGML dimensions");
        require(file.is_open(), "Cannot open GGUF weights");
        gguf = gguf_init_from_file(path, {true, &metadata});
        require(gguf && metadata, "Cannot read GGUF metadata");
        auto arch_key = gguf_find_key(gguf, "general.architecture");
        require(arch_key >= 0 && gguf_get_kv_type(gguf, arch_key) == GGUF_TYPE_STRING &&
                std::string(gguf_get_val_str(gguf, arch_key)) == "qwen35",
                "Expected a llama.cpp b10837 qwen35 GGUF (Qwen3.8 text)");
        // Fail before loading if HF config and GGUF describe different models.
        for (auto pair : std::vector<std::pair<std::string, int64_t>>{
                {"qwen35.embedding_length", c.hidden}, {"qwen35.block_count", c.layers},
                {"qwen35.feed_forward_length", c.intermediate},
                {"qwen35.attention.head_count", c.q_heads},
                {"qwen35.attention.head_count_kv", c.kv_heads},
                {"qwen35.rope.dimension_count", c.rotary_dim},
                {"qwen35.ssm.conv_kernel", c.conv_kernel},
                {"qwen35.ssm.inner_size", c.value_heads*c.value_dim},
                {"qwen35.ssm.state_size", c.key_dim},
                {"qwen35.ssm.group_count", c.key_heads},
                {"qwen35.ssm.time_step_rank", c.value_heads}}) {
            int64_t key = gguf_find_key(gguf, pair.first.c_str());
            require(key >= 0 && gguf_get_kv_type(gguf, key) == GGUF_TYPE_UINT32 &&
                    gguf_get_val_u32(gguf, key) == pair.second,
                    "HF config/GGUF mismatch or missing metadata: " + pair.first);
        }
        for (auto pair : std::vector<std::pair<std::string, float>>{
                {"qwen35.rope.freq_base", c.rope_theta},
                {"qwen35.attention.layer_norm_rms_epsilon", c.norm_eps}}) {
            int64_t key = gguf_find_key(gguf, pair.first.c_str());
            require(key >= 0 && gguf_get_kv_type(gguf, key) == GGUF_TYPE_FLOAT32 &&
                    std::fabs(gguf_get_val_f32(gguf, key)-pair.second) <= std::max(1e-9f, std::fabs(pair.second)*1e-5f),
                    "HF config/GGUF mismatch or missing metadata: " + pair.first);
        }
        if (device >= 0) {
#ifdef DIAL_GGML_CUDA
            require(cudaSetDevice(device) == cudaSuccess, "Cannot select GGML CUDA device");
            backend = ggml_backend_cuda_init(device);
#else
            throw std::runtime_error("Adapter was built without GGML_CUDA; rebuild on this node");
#endif
        } else {
            backend = ggml_backend_cpu_init();
            if (backend) ggml_backend_cpu_set_n_threads(backend, std::max(1, threads));
        }
        require(backend != nullptr, "Cannot initialize GGML backend");
        if (device >= 0) {
            auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
            auto features = reinterpret_cast<ggml_backend_get_features_t>(
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features"));
            if (features) {
                for (auto * f = features(reg); f && f->name; ++f) {
                    if (std::strcmp(f->name, "USE_GRAPHS") == 0 && f->value && std::strcmp(f->value, "1") == 0)
                        cuda_graphs_available = true;
                }
            }
            if (std::getenv("GGML_CUDA_DISABLE_GRAPHS")) cuda_graphs_available = false;
        }
    }
    bool has(const std::string & name) { return gguf_find_tensor(gguf, name.c_str()) >= 0; }
    struct Spec { std::string name; int64_t n0, n1; bool f32 = false; };
    std::unique_ptr<Weights> load(const std::vector<Spec> & specs) {
        auto w = std::make_unique<Weights>();
        for (const auto & spec : specs) {
            T * source = ggml_get_tensor(metadata, spec.name.c_str());
            require(source && source->ne[0] == spec.n0 && source->ne[1] == spec.n1 &&
                    source->ne[2] == 1 && source->ne[3] == 1,
                    "GGUF tensor missing or incorrect shape: " + spec.name);
            if (spec.f32) require(source->type == GGML_TYPE_F32 || source->type == GGML_TYPE_F16 ||
                                  source->type == GGML_TYPE_BF16, "Unsupported small GGUF tensor type: " + spec.name);
            T * tensor = ggml_new_tensor_2d(w->context.ctx, spec.f32 ? GGML_TYPE_F32 : source->type,
                                            spec.n0, spec.n1);
            ggml_set_name(tensor, spec.name.c_str());
            w->tensors[spec.name] = tensor;
            w->bytes += ggml_nbytes(tensor);
        }
        w->buffer = ggml_backend_alloc_ctx_tensors(w->context.ctx, backend);
        require(w->buffer != nullptr, "GGML weight allocation failed (out of memory)");
        ggml_backend_buffer_set_usage(w->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        // Only assigned tensors are read, in bounded chunks. The GGUF is not
        // dequantized/mmap'ed into a full CPU/GPU model on either node.
        std::vector<char> chunk(8 * 1024 * 1024);
        for (const auto & spec : specs) {
            auto id = gguf_find_tensor(gguf, spec.name.c_str());
            T * source = ggml_get_tensor(metadata, spec.name.c_str());
            T * tensor = w->at(spec.name);
            file.clear();
            file.seekg(gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, id));
            if (tensor->type != source->type) {
                std::vector<char> raw(ggml_nbytes(source));
                require(bool(file.read(raw.data(), raw.size())), "Truncated GGUF: " + spec.name);
                std::vector<float> floats(ggml_nelements(source));
                if (source->type == GGML_TYPE_F16) {
                    ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(raw.data()), floats.data(), floats.size());
                } else {
                    ggml_bf16_to_fp32_row(reinterpret_cast<const ggml_bf16_t *>(raw.data()), floats.data(), floats.size());
                }
                ggml_backend_tensor_set(tensor, floats.data(), 0, ggml_nbytes(tensor));
            } else {
                size_t bytes = ggml_nbytes(tensor);
                for (size_t offset = 0; offset < bytes;) {
                    size_t count = std::min(chunk.size(), bytes - offset);
                    require(bool(file.read(chunk.data(), count)), "Truncated GGUF: " + spec.name);
                    ggml_backend_tensor_set(tensor, chunk.data(), offset, count);
                    offset += count;
                }
            }
        }
        return w;
    }
    void prepare_layer(int index, bool linear) {
        require(index >= 0 && index < c.layers, "GGML layer index out of range");
        if (layers.count(index)) {
            require(layers.at(index)->linear == linear, "Inconsistent GGML layer type");
            return;
        }
        auto l = std::make_unique<Layer>(); l->index = index; l->linear = linear;
        std::string p = l->prefix();
        std::vector<Spec> specs{
            {p+"attn_norm.weight", c.hidden, 1, true},
            {p+"post_attention_norm.weight", c.hidden, 1, true},
            {p+"ffn_gate.weight", c.hidden, c.intermediate},
            {p+"ffn_up.weight", c.hidden, c.intermediate},
            {p+"ffn_down.weight", c.intermediate, c.hidden}};
        if (linear) {
            int64_t key = c.key_dim*c.key_heads, value = c.value_dim*c.value_heads;
            std::vector<Spec> extra{
                {p+"attn_qkv.weight", c.hidden, 2*key+value},
                {p+"attn_gate.weight", c.hidden, value},
                {p+"ssm_beta.weight", c.hidden, c.value_heads},
                {p+"ssm_alpha.weight", c.hidden, c.value_heads},
                {p+"ssm_conv1d.weight", c.conv_kernel, 2*key+value, true},
                {p+"ssm_dt.bias", c.value_heads, 1, true},
                {p+"ssm_a", c.value_heads, 1, true},
                {p+"ssm_norm.weight", c.value_dim, 1, true},
                {p+"ssm_out.weight", value, c.hidden}};
            specs.insert(specs.end(), extra.begin(), extra.end());
        } else {
            std::vector<Spec> extra{
                {p+"attn_q.weight", c.hidden, 2*c.head_dim*c.q_heads},
                {p+"attn_k.weight", c.hidden, c.head_dim*c.kv_heads},
                {p+"attn_v.weight", c.hidden, c.head_dim*c.kv_heads},
                {p+"attn_output.weight", c.head_dim*c.q_heads, c.hidden},
                {p+"attn_q_norm.weight", c.head_dim, 1, true},
                {p+"attn_k_norm.weight", c.head_dim, 1, true}};
            specs.insert(specs.end(), extra.begin(), extra.end());
        }
        l->weights = load(specs); layers[index] = std::move(l);
    }
    void prepare_head() {
        if (head_weights) return;
        std::string output = has("output.weight") ? "output.weight" : "token_embd.weight";
        std::vector<Spec> specs{{"token_embd.weight", c.hidden, c.vocab},
                               {"output_norm.weight", c.hidden, 1, true}};
        if (output != "token_embd.weight") specs.push_back({output, c.hidden, c.vocab});
        head_weights = load(specs);
        head_graph = std::make_unique<Graph>(backend, 1);
        auto ctx = head_graph->ctx();
        head_graph->input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.hidden, 1);
        ggml_set_input(head_graph->input);
        T * norm = ggml_mul(ctx, ggml_rms_norm(ctx, head_graph->input, c.norm_eps),
                           head_weights->at("output_norm.weight"));
        head_graph->output = ggml_mul_mat(ctx, head_weights->at(output), norm);
        head_graph->finish(backend);
    }
    void input(T * tensor, const void * data, bool device_io) {
        require(device_io == (device >= 0), "GGML input/backend device mismatch (no host fallback)");
        if (!device_io) ggml_backend_tensor_set(tensor, data, 0, ggml_nbytes(tensor));
        else {
#ifdef DIAL_GGML_CUDA
            require(cudaSetDevice(device) == cudaSuccess &&
                    cudaMemcpy(tensor->data, data, ggml_nbytes(tensor), cudaMemcpyDeviceToDevice) == cudaSuccess,
                    "GGML CUDA input copy failed");
#endif
        }
    }
    void output(T * tensor, void * data, bool device_io) {
        if (!device_io) ggml_backend_tensor_get(tensor, data, 0, ggml_nbytes(tensor));
        else {
#ifdef DIAL_GGML_CUDA
            require(cudaMemcpy(data, tensor->data, ggml_nbytes(tensor), cudaMemcpyDeviceToDevice) == cudaSuccess,
                    "GGML CUDA output copy failed");
#endif
        }
    }
    void compute(Graph & graph) {
        require(ggml_backend_graph_compute(backend, graph.graph) == GGML_STATUS_SUCCESS,
                "GGML graph execution failed");
        ggml_backend_synchronize(backend);
        ++graph_calls;
    }
};

T * norm(ggml_context * ctx, T * x, T * w, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), w);
}
T * l2_norm(ggml_context * ctx, T * x, float eps) {
    float n = x->ne[0];
    return ggml_scale(ctx, ggml_rms_norm(ctx, x, eps/n), 1.0f/std::sqrt(n));
}
struct State {
    Engine & e;
    Layer & l;
    Context persistent;
    ggml_backend_buffer_t buffer = nullptr;
    T * conv = nullptr;
    T * recurrent = nullptr;
    T * keys = nullptr;
    T * values = nullptr;
    int64_t next_position = 0;
    // Bounded cache: one prefill graph and one decode graph. Attention grows
    // in 256-position buckets; unused 4K KV capacity is not scanned each token.
    std::unique_ptr<Graph> prefill;
    std::unique_ptr<Graph> decode;
    State(Engine & engine, Layer & layer) : e(engine), l(layer) {
        const auto & c = e.c;
        if (l.linear) {
            int64_t channels = 2*c.key_dim*c.key_heads + c.value_dim*c.value_heads;
            conv = ggml_new_tensor_3d(persistent.ctx, GGML_TYPE_F32, c.conv_kernel-1, channels, 1);
            recurrent = ggml_new_tensor_4d(persistent.ctx, GGML_TYPE_F32,
                                          c.value_dim, c.value_dim, c.value_heads, 1);
        } else {
            keys = ggml_new_tensor_2d(persistent.ctx, GGML_TYPE_F16, c.head_dim*c.kv_heads, e.capacity);
            values = ggml_dup_tensor(persistent.ctx, keys);
        }
        buffer = ggml_backend_alloc_ctx_tensors(persistent.ctx, e.backend);
        require(buffer != nullptr, "GGML state allocation failed (out of memory)");
        ggml_backend_buffer_clear(buffer, 0);
    }
    ~State() {
        decode.reset(); prefill.reset();
        if (buffer) ggml_backend_buffer_free(buffer);
    }
    T * full(Graph & g, T * x) {
        auto ctx = g.ctx(); const auto & c = e.c; int64_t n = g.tokens, h = c.head_dim;
        T * qg = ggml_mul_mat(ctx, l.w("attn_q.weight"), x);
        T * q = ggml_view_3d(ctx, qg, h, c.q_heads, n, 2*h*4, 2*h*c.q_heads*4, 0);
        T * gate = ggml_view_3d(ctx, qg, h, c.q_heads, n, 2*h*4, 2*h*c.q_heads*4, h*4);
        gate = ggml_cont_2d(ctx, gate, h*c.q_heads, n);
        T * k = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, l.w("attn_k.weight"), x), h, c.kv_heads, n);
        T * v = ggml_mul_mat(ctx, l.w("attn_v.weight"), x);
        q = norm(ctx, q, l.w("attn_q_norm.weight"), c.norm_eps);
        k = norm(ctx, k, l.w("attn_k_norm.weight"), c.norm_eps);
        if (!g.positions) {
            g.positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n); ggml_set_input(g.positions);
            g.rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n); ggml_set_input(g.rows);
            int64_t padded_tokens = (n+31)/32*32;
            g.mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, g.attention_length, padded_tokens);
            ggml_set_input(g.mask);
        }
        // Text-only MRoPE: all three axes have identical positions, so this
        // is exactly the partial NEOX rotation. Images/videos are rejected.
        q = ggml_rope_ext(ctx, q, g.positions, nullptr, c.rotary_dim, GGML_ROPE_TYPE_NEOX,
                          c.max_seq, c.rope_theta, 1, 0, 1, 0, 0);
        k = ggml_rope_ext(ctx, k, g.positions, nullptr, c.rotary_dim, GGML_ROPE_TYPE_NEOX,
                          c.max_seq, c.rope_theta, 1, 0, 1, 0, 0);
        k = ggml_set_rows(ctx, keys, ggml_cont_2d(ctx, k, h*c.kv_heads, n), g.rows);
        v = ggml_set_rows(ctx, values, v, g.rows);
        k = ggml_view_2d(ctx, k, h*c.kv_heads, g.attention_length, k->nb[1], 0);
        v = ggml_view_2d(ctx, v, h*c.kv_heads, g.attention_length, v->nb[1], 0);
        k = ggml_permute(ctx, ggml_reshape_3d(ctx, k, h, c.kv_heads, g.attention_length), 0, 2, 1, 3);
        v = ggml_permute(ctx, ggml_reshape_3d(ctx, v, h, c.kv_heads, g.attention_length), 0, 2, 1, 3);
        q = ggml_permute(ctx, q, 0, 2, 1, 3);
        T * attn = ggml_flash_attn_ext(ctx, q, k, v, g.mask, 1.0f/std::sqrt(float(h)), 0, 0);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
        attn = ggml_reshape_2d(ctx, attn, h*c.q_heads, n);
        return ggml_mul_mat(ctx, l.w("attn_output.weight"), ggml_mul(ctx, attn, ggml_sigmoid(ctx, gate)));
    }
    T * linear(Graph & g, T * x, T * & next_conv, T * & next_state) {
        auto ctx = g.ctx(); const auto & c = e.c; int64_t n = g.tokens;
        int64_t kd = c.key_dim, kh = c.key_heads, vd = c.value_dim, vh = c.value_heads;
        int64_t channels = 2*kd*kh+vd*vh;
        T * qkv = ggml_mul_mat(ctx, l.w("attn_qkv.weight"), x);
        T * z = ggml_mul_mat(ctx, l.w("attn_gate.weight"), x);
        T * beta = ggml_sigmoid(ctx, ggml_reshape_4d(ctx,
                      ggml_mul_mat(ctx, l.w("ssm_beta.weight"), x), 1, vh, n, 1));
        T * alpha = ggml_mul_mat(ctx, l.w("ssm_alpha.weight"), x);
        alpha = ggml_softplus(ctx, ggml_add(ctx, alpha, l.w("ssm_dt.bias")));
        T * gate = ggml_reshape_4d(ctx, ggml_mul(ctx, alpha, l.w("ssm_a")), 1, vh, n, 1);
        qkv = ggml_cont(ctx, ggml_transpose(ctx, qkv)); // [time, channels]
        qkv = ggml_reshape_3d(ctx, qkv, n, channels, 1);
        T * all = ggml_concat(ctx, conv, qkv, 0);
        next_conv = ggml_view_3d(ctx, all, c.conv_kernel-1, channels, 1,
                                 (c.conv_kernel-1+n)*4, (c.conv_kernel-1+n)*channels*4, n*4);
        T * mixed = ggml_silu(ctx, ggml_ssm_conv(ctx, all, l.w("ssm_conv1d.weight")));
        T * q = ggml_view_4d(ctx, mixed, kd, kh, n, 1, kd*4, channels*4, channels*n*4, 0);
        T * k = ggml_view_4d(ctx, mixed, kd, kh, n, 1, kd*4, channels*4, channels*n*4, kd*kh*4);
        T * v = ggml_view_4d(ctx, mixed, vd, vh, n, 1, vd*4, channels*4, channels*n*4, 2*kd*kh*4);
        q = l2_norm(ctx, q, c.norm_eps); k = l2_norm(ctx, k, c.norm_eps);
        // GGML's value-head tiling matches the upstream GGUF converter. Do not
        // undo its reorder and do not interleave_repeat HF key/value heads.
        T * packed = ggml_gated_delta_net(ctx, q, k, v, gate, beta, recurrent, 1);
        T * attn = ggml_view_4d(ctx, packed, vd, vh, n, 1, vd*4, vd*vh*4, vd*vh*n*4, 0);
        next_state = ggml_view_4d(ctx, packed, vd, vd, vh, 1, vd*4, vd*vd*4, vd*vd*vh*4, vd*vh*n*4);
        attn = norm(ctx, attn, l.w("ssm_norm.weight"), c.norm_eps);
        z = ggml_reshape_4d(ctx, z, vd, vh, n, 1);
        attn = ggml_reshape_2d(ctx, ggml_mul(ctx, attn, ggml_silu(ctx, z)), vd*vh, n);
        return ggml_mul_mat(ctx, l.w("ssm_out.weight"), attn);
    }
    T * append(Graph & g, T * input) {
        auto ctx = g.ctx();
        const auto & c = e.c;
        T * x = norm(ctx, input, l.w("attn_norm.weight"), c.norm_eps);
        T * next_conv = nullptr, * next_state = nullptr;
        T * mixed = l.linear ? linear(g, x, next_conv, next_state) : full(g, x);
        x = ggml_add(ctx, input, mixed);
        T * normalized = norm(ctx, x, l.w("post_attention_norm.weight"), c.norm_eps);
        T * up = ggml_mul_mat(ctx, l.w("ffn_up.weight"), normalized);
        T * gate = ggml_silu(ctx, ggml_mul_mat(ctx, l.w("ffn_gate.weight"), normalized));
        T * mlp = ggml_mul_mat(ctx, l.w("ffn_down.weight"), ggml_mul(ctx, up, gate));
        T * output = ggml_add(ctx, x, mlp);
        // Complete all consumers of the old recurrent state BEFORE mutating
        // it. cpy nodes are expanded after the layer output, and their source
        // views keep both packed GDN results and conv inputs alive.
        ggml_build_forward_expand(g.graph, output);
        if (l.linear) {
            ggml_build_forward_expand(g.graph, ggml_cpy(ctx, next_conv, conv));
            ggml_build_forward_expand(g.graph, ggml_cpy(ctx, next_state, recurrent));
        }
        return output;
    }
    std::unique_ptr<Graph> build(int64_t tokens, int64_t attention_length) {
        auto g = std::make_unique<Graph>(e.backend, tokens, 1, attention_length);
        g->input = ggml_new_tensor_2d(g->ctx(), GGML_TYPE_F32, e.c.hidden, tokens);
        ggml_set_input(g->input);
        g->output = append(*g, g->input);
        g->finish(e.backend);
        return g;
    }
    void check_position(int64_t tokens, int64_t position) const {
        require(tokens > 0 && position >= 0 && position <= e.c.max_seq - tokens,
                "GGML request exceeds configured context length");
        require(position == 0 || position == next_position,
                "GGML cache position mismatch; start a new request at position 0");
    }
    void reset_if_new(int64_t position) {
        if (position == 0) { ggml_backend_buffer_clear(buffer, 0); next_position = 0; }
    }
    int64_t active_length(int64_t tokens, int64_t position) const {
        return std::min(e.capacity, (position + tokens + 255) / 256 * 256);
    }
    static void set_attention_inputs(Graph & g, int64_t position) {
        if (!g.positions) return;
        int64_t tokens = g.tokens;
        std::vector<int32_t> pos(tokens); std::vector<int64_t> rows(tokens);
        for (int64_t i = 0; i < tokens; ++i) { pos[i] = position+i; rows[i] = position+i; }
        ggml_backend_tensor_set(g.positions, pos.data(), 0, tokens*sizeof(int32_t));
        ggml_backend_tensor_set(g.rows, rows.data(), 0, tokens*sizeof(int64_t));
        std::vector<ggml_fp16_t> mask(ggml_nelements(g.mask), ggml_fp32_to_fp16(-INFINITY));
        for (int64_t i = 0; i < tokens; ++i) {
            std::fill(mask.begin()+i*g.attention_length,
                      mask.begin()+i*g.attention_length+position+i+1, ggml_fp32_to_fp16(0.0f));
        }
        ggml_backend_tensor_set(g.mask, mask.data(), 0, ggml_nbytes(g.mask));
    }
    void forward(const float * input, int64_t tokens, int64_t position, float * output, bool device_io) {
        check_position(tokens, position);
        reset_if_new(position);
        int64_t attention_length = active_length(tokens, position);
        auto & slot = tokens == 1 ? decode : prefill;
        if (!slot || slot->tokens != tokens || slot->attention_length != attention_length)
            slot = build(tokens, attention_length);
        auto & g = *slot;
        e.input(g.input, input, device_io);
        set_attention_inputs(g, position);
        e.compute(g); e.output(g.output, output, device_io); next_position = position+tokens;
    }
};
struct Range {
    Engine & e;
    std::vector<State *> states;
    std::unique_ptr<Graph> prefill;
    std::unique_ptr<Graph> decode;
    Range(Engine & engine, void * const * handles, int64_t count) : e(engine) {
        require(handles && count > 0 && count <= e.c.layers, "Invalid GGML shard size");
        for (int64_t i = 0; i < count; ++i) {
            auto * s = static_cast<State *>(handles[i]);
            require(s && &s->e == &e, "GGML shard state belongs to a different engine");
            require(i == 0 || s->l.index == states.back()->l.index+1,
                    "GGML shard layers must be consecutive");
            states.push_back(s);
        }
    }
    std::unique_ptr<Graph> build(int64_t tokens, int64_t attention_length) {
        auto g = std::make_unique<Graph>(e.backend, tokens, states.size(), attention_length);
        g->input = ggml_new_tensor_2d(g->ctx(), GGML_TYPE_F32, e.c.hidden, tokens);
        ggml_set_input(g->input);
        T * x = g->input;
        for (auto * s : states) x = s->append(*g, x);
        g->output = x; g->finish(e.backend);
        return g;
    }
    void forward(const float * input, int64_t tokens, int64_t position, float * output, bool device_io) {
        // Validate every state before resetting or mutating any of them.
        for (auto * s : states) s->check_position(tokens, position);
        for (auto * s : states) s->reset_if_new(position);
        int64_t attention_length = states.front()->active_length(tokens, position);
        auto & slot = tokens == 1 ? decode : prefill;
        if (!slot || slot->tokens != tokens || slot->attention_length != attention_length)
            slot = build(tokens, attention_length);
        auto & g = *slot;
        e.input(g.input, input, device_io);
        State::set_attention_inputs(g, position);
        e.compute(g); e.output(g.output, output, device_io);
        for (auto * s : states) s->next_position = position+tokens;
        ++e.range_calls;
    }
};
template<typename F> int guarded(char * error, size_t size, F fn) {
    try { fn(); return 0; }
    catch (const std::exception & e) { error_text(error, size, e.what()); return -1; }
    catch (...) { error_text(error, size, "Unknown GGML adapter failure"); return -1; }
}
} // namespace

extern "C" {
uint32_t dial_ggml_abi_version() { return 2; }
void * dial_ggml_open(const char * path, const dial_ggml_config * c, int device, int threads,
                      char * error, size_t size) {
    try {
        auto e = std::make_unique<Engine>(path, *c, device);
        e->init(path, threads); return e.release();
    } catch (const std::exception & e) { error_text(error, size, e.what()); return nullptr; }
}
void dial_ggml_close(void * engine) { delete static_cast<Engine *>(engine); }
int dial_ggml_prepare_layer(void * engine, int layer, int linear, char * error, size_t size) {
    return guarded(error, size, [&]{ static_cast<Engine *>(engine)->prepare_layer(layer, linear != 0); });
}
int dial_ggml_prepare_head(void * engine, char * error, size_t size) {
    return guarded(error, size, [&]{ static_cast<Engine *>(engine)->prepare_head(); });
}
void * dial_ggml_create_state(void * engine, int layer, char * error, size_t size) {
    try {
        auto & e = *static_cast<Engine *>(engine);
        require(e.layers.count(layer) != 0, "GGML layer weights have not been prepared");
        return new State(e, *e.layers.at(layer));
    } catch (const std::exception & e) { error_text(error, size, e.what()); return nullptr; }
}
void dial_ggml_destroy_state(void * state) { delete static_cast<State *>(state); }
void * dial_ggml_create_range(void * engine, void * const * states, int64_t count,
                             char * error, size_t size) {
    try { return new Range(*static_cast<Engine *>(engine), states, count); }
    catch (const std::exception & e) { error_text(error, size, e.what()); return nullptr; }
}
void dial_ggml_destroy_range(void * range) { delete static_cast<Range *>(range); }
int dial_ggml_forward_range(void * engine, void * range, const float * input, int64_t tokens,
                            int64_t position, float * output, int device_io, char * error, size_t size) {
    return guarded(error, size, [&]{
        auto & r = *static_cast<Range *>(range);
        require(&r.e == engine, "GGML shard belongs to a different engine");
        r.forward(input, tokens, position, output, device_io != 0);
    });
}
int dial_ggml_forward(void * engine, void * state, const float * input, int64_t tokens,
                      int64_t position, float * output, int device_io, char * error, size_t size) {
    return guarded(error, size, [&]{
        auto & s = *static_cast<State *>(state);
        require(&s.e == engine, "GGML state belongs to a different engine");
        s.forward(input, tokens, position, output, device_io != 0);
    });
}
int dial_ggml_embed(void * engine, const int32_t * tokens, int64_t count,
                    float * output, int device_io, char * error, size_t size) {
    return guarded(error, size, [&]{
        auto & e = *static_cast<Engine *>(engine); e.prepare_head();
        require(count > 0 && count <= e.c.max_seq, "Invalid GGML embedding token count");
        require((device_io != 0) == (e.device >= 0), "GGML embedding output device mismatch");
        for (int64_t i = 0; i < count; ++i) require(tokens[i] >= 0 && tokens[i] < e.c.vocab,
                                                   "GGML embedding token id out of range");
        if (!e.embed_graph || e.embed_graph->tokens != count) {
            auto g = std::make_unique<Graph>(e.backend, count);
            g->input = ggml_new_tensor_1d(g->ctx(), GGML_TYPE_I32, count); ggml_set_input(g->input);
            g->output = ggml_get_rows(g->ctx(), e.head_weights->at("token_embd.weight"), g->input);
            g->finish(e.backend); e.embed_graph = std::move(g);
        }
        auto & g = *e.embed_graph;
        ggml_backend_tensor_set(g.input, tokens, 0, count*sizeof(int32_t));
        e.compute(g); e.output(g.output, output, device_io != 0);
    });
}
int dial_ggml_head(void * engine, const float * input, float * output, int device_io,
                   char * error, size_t size) {
    return guarded(error, size, [&]{
        auto & e = *static_cast<Engine *>(engine); e.prepare_head();
        auto & g = *e.head_graph;
        e.input(g.input, input, device_io != 0); e.compute(g); e.output(g.output, output, device_io != 0);
    });
}
uint64_t dial_ggml_weight_bytes(void * engine) {
    auto & e = *static_cast<Engine *>(engine); uint64_t bytes = 0;
    for (const auto & l : e.layers) bytes += l.second->weights->bytes;
    if (e.head_weights) bytes += e.head_weights->bytes;
    return bytes;
}
uint64_t dial_ggml_weight_tensors(void * engine) {
    auto & e = *static_cast<Engine *>(engine); uint64_t count = 0;
    for (const auto & l : e.layers) count += l.second->weights->tensors.size();
    if (e.head_weights) count += e.head_weights->tensors.size();
    return count;
}
uint64_t dial_ggml_graph_calls(void * engine) { return static_cast<Engine *>(engine)->graph_calls; }
uint64_t dial_ggml_range_calls(void * engine) { return static_cast<Engine *>(engine)->range_calls; }
uint64_t dial_ggml_cuda_graphs_available(void * engine) {
    return static_cast<Engine *>(engine)->cuda_graphs_available;
}
}
