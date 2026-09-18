#pragma once
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif
// All pointers returned by this ABI are owned; destroy states before the engine.
// Every call on an engine must be serialized by its caller, including destruction.
typedef struct {
    int64_t hidden, intermediate, vocab, layers, q_heads, kv_heads, head_dim;
    int64_t rotary_dim, max_seq, conv_kernel, key_dim, key_heads, value_dim, value_heads;
    float norm_eps, rope_theta;
} dial_ggml_config;
uint32_t dial_ggml_abi_version(void);
void * dial_ggml_open(const char * path, const dial_ggml_config * config,
                     int device, int threads, char * error, size_t error_size);
void dial_ggml_close(void * engine);
int dial_ggml_prepare_layer(void * engine, int layer, int linear,
                           char * error, size_t error_size);
int dial_ggml_prepare_head(void * engine, char * error, size_t error_size);
void * dial_ggml_create_state(void * engine, int layer, char * error, size_t error_size);
void dial_ggml_destroy_state(void * state);
// A range borrows consecutive per-layer states. Destroy it before those states.
void * dial_ggml_create_range(void * engine, void * const * states, int64_t count,
                             char * error, size_t error_size);
void dial_ggml_destroy_range(void * range);
// Float32 contiguous [tokens, hidden] boundaries. device_io=1 means CUDA pointers,
// with synchronous device-to-device copies; it NEVER falls back to host execution.
int dial_ggml_forward(void * engine, void * state, const float * input,
                      int64_t tokens, int64_t position, float * output, int device_io,
                      char * error, size_t error_size);
// One graph/interop boundary for a whole local shard, not one per layer.
int dial_ggml_forward_range(void * engine, void * range, const float * input,
                            int64_t tokens, int64_t position, float * output, int device_io,
                            char * error, size_t error_size);
int dial_ggml_embed(void * engine, const int32_t * tokens, int64_t count,
                    float * output, int device_io, char * error, size_t error_size);
int dial_ggml_head(void * engine, const float * input, float * output, int device_io,
                   char * error, size_t error_size);
uint64_t dial_ggml_weight_bytes(void * engine);
uint64_t dial_ggml_weight_tensors(void * engine);
uint64_t dial_ggml_graph_calls(void * engine);
uint64_t dial_ggml_range_calls(void * engine);
// Build/environment availability, not a promise that every graph is captured.
uint64_t dial_ggml_cuda_graphs_available(void * engine);
#ifdef __cplusplus
}
#endif
