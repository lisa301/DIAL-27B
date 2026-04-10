
# 一、服务器端

## 1、执行指令：

```raw
//挂载硬盘
sudo mount /dev/sda1 /media/nvidia/Elements/

cargo clean #清除缓存
cargo build --release --features cuda  #有GPU＋cuda，没有去掉

./target/release/dial-cli  --mode worker  --address 0.0.0.0:10128

```
### 2、model地址
```raw
/// qwen3-vl-8B模型   ,transformers一共36层，需修改topology.yml
/media/nvidia/Elements/Qwen3-vl/Qwen3-VL-8B-Instruct


/// qwen3-vl-2B模型  ,transformers一共28层，需修改topology.yml
/media/nvidia/Elements/Qwen3-vl/Qwen3-VL-2B-Instruct
```

### 3、yml地址
```raw
/home/nvidia/Dial_llama/topology_qwen3vl.yml
```

# 二、客户端执行指令：

### 1、终端1：（与服务器连通）
```raw
cargo clean #清除缓存
cargo build --release  #有GPU＋cuda，没有去掉


./target/release/dial-cli --api 0.0.0.0:8082

```
```raw
/// vision-max-side限制图片所占的token数，如果太大的话，首token会很慢
./target/release/dial-cli \
  --mode master \
  --api 0.0.0.0:8082 \
  --vision-max-side 512 \
  --vision-no-upscale 
```

### 2、终端2：（大模型问答推理）
```raw
./target/release/dial-cli  --api-client http://172.16.30.39:8082 --ask "1+1等于多少"


SPM_TRACE_TRANSFER=1 RUST_LOG=info \
./target/release/dial-cli  --api-client http://127.0.0.1:8082 --image test3.png --ask "图片中是否有人摔倒？"
```
### 3、model地址
```raw
/// qwen3-vl-8B模型
/home/firefly/Documents/Qwen3-VL-8B-Instruct

/// qwen3-vl-2B模型
/userdata/Qwen3-VL-2B-Instruct
```
### 4、yml地址
```raw
/home/firefly/Documents/Dial_llama/topology_qwen3vl.yml
```

# 三、客户端使用NPU执行指令：
### 1、qwen3-vl-2B 单张图片(服务器端执行)
```raw
./target/release/dial-cli \
  --mode master \
  --api 0.0.0.0:8082 \
  --vision-rknn /home/firefly/Documents/Dial_llama/vision_448.rknn \
  --vision-fixed-side 448
```

### 2、qwen3-vl-8B 单张图片(服务器端执行)
```raw
./target/release/dial-cli \
  --mode master \
  --api 0.0.0.0:8082 \
  --vision-rknn /home/firefly/Documents/Dial_llama/qwen3_vl_8b_vision_448.rknn \
  --vision-fixed-side 448

./target/release/dial-cli \
  --mode master \
  --api 0.0.0.0:8082 \
  --vision-rknn /home/firefly/Documents/Dial_llama/qwen3_vl_8b_vision_448_deepstack_fp16.rknn \
  --vision-fixed-side 448



```

### 3、qwen3-vl-8B 视频流(客户端执行)
```raw
  python3 /home/firefly/Documents/Dial_llama/tool/stream_video_client.py \
  --source "rtsp://172.16.30.113:8554/live/a" \
  --interval-sec 20 \
  --spm-cli /home/firefly/Documents/Dial_llama/target/release/dial-cli \
  --api-client http://127.0.0.1:8082 \
  --keep-frames \
  --prompt "请回答图片中是否有人摔倒"
```



# 四、查看板子资源状态常用命令
## 1、RK3588的NPU状态
```raw
watch -n 1 cat /sys/kernel/debug/rknpu/load
```
## 2、查看CPU和MEM状态
```raw
htop
```
## 3、查看ORIN的CPU、MEM和GPU状态
```raw
jtop
```

# 五、三个RK3588运行，两台worker一台mode
### 1、服务器端
```raw
./target/release/dial-cli  --mode worker  --name worker1  --address 0.0.0.0:10128
./target/release/dial-cli  --mode worker  --name worker2  --address 0.0.0.0:10128
```
### 2、客户端
```raw
./target/release/dial-cli \
  --mode master \
  --api 0.0.0.0:8082 \
  --vision-rknn /home/firefly/Documents/Dial_llama/qwen3_vl_8b_vision_448_deepstack_fp16.rknn \
  --vision-fixed-side 448
```

### 3、各个性能参数含义

| 参数 | 含义|
|------|------|
| ttft_s | 首 token 延迟，从请求开始到第一个非空输出。包含图片编码、prefill、首 token 解码|
| total_s |本次请求总耗时 |
| tps | 平均生成速率。	generated_tokens / total_s|
|decode_tps | 解码阶段速率（去掉首 token 阶段） |
|dist_overhead_s|分布式“非计算”开销（网络+序列化+协议往返等）|
|remote_compute_s|远端 worker 纯计算时间累计|
|remote_requests|本次请求中发往远端 worker 的请求次数（收到 Tensor 响应就 +1）|


### 4、使用Linux TC流量控制常用命令
#### 1、安装tc
```
sudo apt update && sudo apt install iproute2 -y
```
#### 2、查看tc
```
# 查看 tc 版本（确认工具存在）
tc -V
# 查看系统网络接口（找到你要配置的网卡名，比如 eth0、ens33、wlan0 等）
ip addr
```
#### 3、查看当前网络带宽
```
ethtool eth1 
```


### 5、把conda环境打包并在另一个设备上使用
#### 1、解压
```raw
mkdir -p ~/envs/rknn9
tar -xzf rknn9.tar.gz -C ~/envs/rknn9
```
#### 2、激活
```raw
conda env create -f environment.yml
```
#### 3、第一次激活后，立刻修复路径
```raw
conda-unpack
```
#### 4、再检查
```raw
python --version
which python
```
#### 5、开机自启为conda环境
```raw
source /home/firefly/envs/rknn9/bin/activate
source ~/.bashrc
```
### 6.使用不同网段固定IP

