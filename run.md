# 当前推荐方式：文本按 NPU+CPU 混合执行

## 1. 最小启动：文本全走本机 CPU 软件路径

```bash
cd /home/seaway/sdb/ljl/Dial_llama
RUST_LOG=info ./target/release/dial-cli \
  --mode master \
  --api 0.0.0.0:8082 \
  --model /home/seaway/sdb/ljl/model/Qwen3-VL-8B-Instruct \
  --cpu \
  --text-decode-mode cpu-only
```

## 2. 如果在 RK3588 上，视觉可以继续走 RKNN，文本仍走 CPU

```bash
cd /home/seaway/sdb/ljl/Dial_llama
RUST_LOG=info ./target/release/dial-cli \
  --mode master \
  --api 0.0.0.0:8082 \
  --model /home/seaway/sdb/ljl/model/Qwen3-VL-8B-Instruct \
  --cpu \
  --text-decode-mode cpu-only \
  --vision-rknn /home/seaway/sdb/ljl/Dial_llama/transmodel/qwen3_vl_8b_vision_448_deepstack_fp16.rknn \
  --vision-fixed-side 448
```

## 3. RK3588 推荐：文本 prefill/decode 走 NPU 前缀，剩余层走 CPU

```bash
cd /home/seaway/sdb/ljl/Dial_llama
export DIAL_TEXT_DECODE_PAST_BUCKETS=128,256,512,1024
RUST_LOG=info ./target/release/dial-cli \
  --mode master \
  --api 0.0.0.0:8082 \
  --model /home/seaway/sdb/ljl/model/Qwen3-VL-8B-Instruct \
  --cpu \
  --text-rknn-dir /home/seaway/sdb/ljl/Dial_llama/transmodel/chunks_rknn_full \
  --text-rknn-prefill \
  --text-decode-mode npu-cpu \
  --vision-rknn /home/seaway/sdb/ljl/Dial_llama/transmodel/qwen3_vl_8b_vision_448_deepstack_fp16.rknn \
  --vision-fixed-side 448
```

## 4. 更稳的文本子图拆分：QKV 走 RKNN，Attention/KV 走 CPU

先导出单层 QKV ONNX：

```bash
python /home/seaway/sdb/ljl/Dial_llama/tools/export_qwen3_vl_text_qkv_onnx.py \
  --model-dir /home/seaway/sdb/ljl/model/Qwen3-VL-8B-Instruct \
  --layers 0-35 \
  --out-dir /home/seaway/sdb/ljl/Dial_llama/transmodel/qkv_onnx
```

再把这些 ONNX 转成 RKNN：

```bash
python /home/seaway/sdb/ljl/Dial_llama/tools/convert_onnx_to_rknn.py \
  --onnx-dir /home/seaway/sdb/ljl/Dial_llama/transmodel/qkv_onnx \
  --out-dir /home/seaway/sdb/ljl/Dial_llama/transmodel/qkv_rknn \
  --pattern '*.onnx' \
  --target-platform rk3588
```

启动时只让 decode 的 `q_proj/k_proj/v_proj` 走 RKNN；RoPE、KV cache、mask、attention softmax、
`o_proj` 和后续 MLP 仍走 CPU 软件路径：

```bash
cd /home/seaway/sdb/ljl/Dial_llama
RUST_LOG=info ./target/release/dial-cli \
  --mode master \
  --api 0.0.0.0:8082 \
  --model /home/seaway/sdb/ljl/model/Qwen3-VL-8B-Instruct \
  --cpu \
  --text-qkv-rknn-dir /home/seaway/sdb/ljl/Dial_llama/transmodel/qkv_rknn \
  --text-decode-mode cpu-only
```

这条路径当前只在 `seq_len=1` 的 decode 步启用 QKV-RKNN，prefill 继续走 CPU。

## 5. 实验：文本 0-1 层走 RKLLM，剩余层继续本机/CPU

先在 RK3588 上验证 `.rkllm` 能返回 layer 1 后、final norm 前的 hidden。这个脚本会用真实
embedding 输入，并和 PyTorch 参考边界 hidden 做 `max_abs/rms` 对比：

```bash
cd /home/seaway/sdb/ljl/Dial_llama/build/rkllm_hidden_smoke_aarch64
./run_prefix_l00_l01_no_norm_verify.sh \
  /home/seaway/sdb/ljl/model/qwen3vl_text_prefix_l00_l01_causal_lm_no_norm_w8a8_calib2.rkllm
```

如果最后打印 `RKLLM hidden smoke OK`，且 `hidden diff vs reference` 的 RMS 在可接受范围内，再启动主服务：

```bash
cd /home/seaway/sdb/ljl/Dial_llama
export LD_LIBRARY_PATH=/home/seaway/sdb/ljl/rknn-llm-src/rkllm-runtime/Linux/librkllm_api/aarch64:$LD_LIBRARY_PATH
export RKLLM_LOG_LEVEL=1
export QWEN3VL_PROFILE_SUMMARY=1
RUST_LOG=info ./target/release/dial-cli \
  --mode master \
  --api 0.0.0.0:8082 \
  --model /home/seaway/sdb/ljl/model/Qwen3-VL-8B-Instruct \
  --cpu \
  --text-decode-mode cpu-only \
  --text-rkllm-model /home/seaway/sdb/ljl/model/qwen3vl_text_prefix_l00_l01_causal_lm_no_norm_w8a8_calib2.rkllm \
  --text-rkllm-lib /home/seaway/sdb/ljl/rknn-llm-src/rkllm-runtime/Linux/librkllm_api/aarch64/librkllmrt.so \
  --text-rkllm-prefix-layers 2
```

这个 RKLLM 文件用官方 toolkit 导出，真实计算层是 text layers 0-1；`lm_head` 只是为了满足
RKLLM 解析器的 `OUTPUT` 要求，runtime 里用 `RKLLM_INPUT_EMBED` 和
`RKLLM_INFER_GET_LAST_HIDDEN_LAYER`，不会把它作为最终 logits 使用。

这条路径当前只用于纯文本。带图片时不会走 0-1 RKLLM，因为 deepstack 注入需要进入前几层内部，RKLLM runtime 没有这个注入接口。

# 一、qt系统启动
```
///编译
export LD_LIBRARY_PATH=/home/lijilin/miniconda3/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/home/lijilin/miniconda3/lib:$LIBRARY_PATH
cmake --build /home/seaway/sdb/ljl/iSure-master/iSure-master/build-ninja -j4


/// 启动
cd /home/seaway/sdb/ljl/iSure-master/iSure-master
export LD_LIBRARY_PATH=/home/lijilin/miniconda3/lib:$LD_LIBRARY_PATH
./build-ninja/iSure

```
