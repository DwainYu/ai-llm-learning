import requests
import json
import base64
import time
 
 
start = time.time()
# 1.url
url = 'http://192.168.42.240:6006/v1/chat/completions'
 
 
# 2.data
## 2.1如果server.py启动，用这个data
data = {"model": "/home/vllmuser/models/Qwen3-0.6B",  # 使用完整模型路径作为 model 字段
        "messages": [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},     # 系统命令，一般不要改
                     {"role": "user",
                      "content": "Tell me something about large language models."}],    # 用户命令，一般改这里
        "temperature": 0.7,"top_p": 0.8,"repetition_penalty": 1.05,"max_tokens": 1024}

# 3.将字典转换为 JSON 字符串
json_payload = json.dumps(data)
 
 
# 4.发送 POST 请求
headers = {'Content-Type': 'application/json'}
response = requests.post(url, data=json_payload, headers=headers)
 
 
# 5.打印响应内容
resp_json = response.json()
choices = resp_json.get("choices", [])
if choices and "message" in choices[0] and "content" in choices[0]["message"]:
    print(choices[0]["message"]["content"])
else:
    print("响应内容异常：", resp_json)
# print(response.json())        # 调试用
 
 
print("\n总时间：", time.time()-start, "秒")

