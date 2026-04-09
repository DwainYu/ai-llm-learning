import requests

# vLLM API 地址（改为WSL本地IP）
API_URL = "http://192.168.42.240:6006/v1/completions"

# 注意：vLLM 0.8.5 的接口路由为 /v1/completions
# 如果你访问 /completions 报错，请确认 API_URL 是否为 http://localhost:6006/v1/completions

# 构造请求数据
payload = {
    "model": "Qwen3-0.6B",
    "prompt": "你好，请用一句话介绍一下你自己。",
    "max_tokens": 128,
    "temperature": 0.7,
    "top_p": 0.95
}

# 如果你想用 bash 命令调用接口，可以这样写：
# curl -X POST http://localhost:6006/v1/completions \
#   -H "Content-Type: application/json" \
#   -d '{"model": "Qwen3-0.6B", "prompt": "你好，请用一句话介绍一下你自己。", "max_tokens": 128, "temperature": 0.7, "top_p": 0.95}'

# 运行本脚本的命令如下（在终端或WSL中执行）：
# python call_vllm_api_example.py

# 常见报错说明：
# requests.exceptions.ConnectionError: ... Failed to establish a new connection: [Errno 111] Connection refused
# 说明 vLLM 服务没有启动或端口号不对。
# 请确保你已在WSL终端运行了如下命令，并看到 "Starting vLLM API server on http://0.0.0.0:6006"：
# vllm serve /home/vllmuser/models/Qwen3-0.6B --dtype auto --port 6006 --max_model_len 8192 --gpu_memory_utilization 0.8
# 然后再运行本脚本。

# 发送 POST 请求
# 注意：接口路径应为 /v1/completions，不是 /completions
# 如果你收到 404 Not Found，说明 API_URL 配置有误或服务端版本不符
response = requests.post(API_URL, json=payload, timeout=60)
result = response.json()

# 输出结果
print("生成结果：")
# 打印完整响应内容，便于排查无返回内容的原因
print(result)
print(result.get("choices", [{}])[0].get("text", "无返回内容"))
