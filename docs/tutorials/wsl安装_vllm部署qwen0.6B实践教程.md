# Windows 11 WSL 部署 vLLM + Qwen-0.6B 实践教程

本教程记录如何在 Windows 11 的 WSL2 环境下，部署 vLLM 并运行 Qwen-0.6B 大模型，并通过 API 实现推理调用。适合初学者参考，也可作为 B 站视频教程脚本。

---

## 1. 环境准备

### 1.1 安装 WSL2 和 Ubuntu

- 在 Windows 11 中启用 WSL2，推荐安装 Ubuntu 24。

- 参考[微软官方文档](https://docs.microsoft.com/zh-cn/windows/wsl/install)：

- 如果不是第一次安装wsl，可能会出现报错，我的处理如下：
```
winget search wsl #查找WSL的ID，输出内容让ai解释就行
winget uninstall Microsoft.WSL #卸载WSL，曾经安装过

#下载Linux内核更新包

wsl --set-default-version 2 # 设置WSL2为默认版本


```

下载[Linux内核更新包](https://github.com/microsoft/WSL/releases)

### 1.2 创建普通用户（可选）

- 避免长期用 root，提升安全性。
  ```bash
  sudo adduser vllmuser
  sudo usermod -aG sudo vllmuser
  ```

---


### 1.3 安装显卡驱动和 CUDA

- Windows 需安装最新版 NVIDIA 驱动（支持 WSL）。
- 在 WSL 内安装 [CUDA Toolkit_12.4](https://developer.nvidia.com/cuda-12-4-0-download-archive)和 cuDNN，确保 `nvidia-smi` 能正常显示显卡信息。

```
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run

sudo sh cuda_12.4.0_550.54.14_linux.run
#显示缺少gcc
sudo apt update
sudo apt install gcc

```

[cuDNN](https://developer.nvidia.com/rdp/cudnn-archive)安装
可以复制链接迅雷下载
**包名发给ai**

你的文件在 Windows **D盘**：
`D:\cudnn-local-repo-ubuntu2004-8.9.7.29_1.0-1_amd64.deb`

在 WSL 里对应的路径是：
```
/mnt/d/cudnn-local-repo-ubuntu2004-8.9.7.29_1.0-1_amd64.deb
```

---

### A.直接运行这 4 条命令

#### 1. 进入 D 盘
```bash
cd /mnt/d
```

#### 2. 安装 deb 包
```bash
sudo dpkg -i cudnn-local-repo-ubuntu2004-8.9.7.29_1.0-1_amd64.deb
```

#### 3. 复制密钥（必须）
```bash
sudo cp /var/cudnn-local-repo-ubuntu2004-8.9.7.29/cudnn-*-keyring.gpg /usr/share/keyrings/
```

#### 4. 安装 cuDNN
```bash
sudo apt update
sudo apt install -y libcudnn8 libcudnn8-dev
```

---

### B.验证是否成功
```bash
cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

出现这个就是 ✅ 成功：
```
#define CUDNN_MAJOR 8
#define CUDNN_MINOR 9
#define CUDNN_PATCHLEVEL 7
```

---




## 2. Python 环境与依赖

### 2.1 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
source $HOME/.local/bin/env
uv --version
```

### 2.2 创建项目目录与虚拟环境

```bash
mkdir -p ~/projects
mkdir -p ~/models
cd ~/projects
uv venv vllm --python 3.12 --seed
source vllm/bin/activate
uv init
```

---

## 3. 安装 vLLM 及相关依赖
[欢迎来到 vLLM — vLLM 文档](https://docs.vllm.com.cn/en/latest/getting_started/installation/gpu.html)
### 3.1 安装 vLLM

- 推荐安装最新版 vLLM。会自动安装相关依赖。
```bash
uv pip install vllm -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3.2 （可选）安装 flash-attn

```bash
uv pip install flash-attn --no-build-isolation
```

### 3.3 安装 modelscope 并下载模型

```bash
uv pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
modelscope download --model Qwen/Qwen3-0.6B --local_dir /home/vllmuser/models/Qwen3-0.6B
```
![[modelscope模型下载.png]]
---

## 4. 启动 vLLM 服务

```bash
vllm serve /home/vllmuser/models/Qwen3-0.6B --dtype auto --port 6006 --max_model_len 8192 --gpu_memory_utilization 0.8
```

- 启动成功后，终端会显示 `Starting vLLM API server on http://0.0.0.0:6006`
- 用 `ss -tlnp | grep 6006` 确认端口监听
![[vllm启动模型成功.png]]
---

## 5. 测试 API 推理

### 5.1 用 curl 测试

```bash
WSL分配的实际IP
hostname -I

# 输出如：192.168.42.240

curl http://192.168.42.240:6006/v1/models
```

### 5.2 用 Python 脚本调用

```python
import requests

API_URL = "http://192.168.42.240:6006/v1/completions"
payload = {
    "model": "Qwen3-0.6B",
    "prompt": "你好，请用一句话介绍一下你自己。",
    "max_tokens": 128,
    "temperature": 0.7,
    "top_p": 0.95
}
response = requests.post(API_URL, json=payload, timeout=60)
print(response.json())
```

### 5.3 用 Chat API 调用

```python
import requests
import json

url = 'http://192.168.42.240:6006/v1/chat/completions'
data = {
    "model": "/home/vllmuser/models/Qwen3-0.6B",# 模型路径
    "messages": [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": "Tell me something about large language models."}
    ],
    "temperature": 0.7, "top_p": 0.8, "repetition_penalty": 1.05, "max_tokens": 1024
}
headers = {'Content-Type': 'application/json'}
resp = requests.post(url, data=json.dumps(data), headers=headers)
print(resp.json())
```
![[调用模型api成功1.png]]
![[调用模型api成功2.png]]
---

## 6. 常见问题与排查

- 端口无法访问：用 `hostname -I` 查看 WSL IP，Windows 用该 IP 访问。
- API 404：确认接口路径为 `/v1/completions` 或 `/v1/chat/completions`。
- 无返回内容：检查模型是否加载成功，日志有无报错。
- 显存不足：可尝试更小模型或调整 `--max_model_len`。

---

## 7. 视频录制建议

- 推荐录制步骤：环境准备 → 安装依赖 → 下载模型 → 启动服务 → API 调用演示 → 常见问题排查
- 可用 OBS Studio 录屏，配合命令行和浏览器演示

---

如有问题欢迎评论区留言交流！
