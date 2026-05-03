import os
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import OpenAI
import uvicorn

app = FastAPI()
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


# 1. 写一个 Python 生成器函数
def stream_generator(query: str):
    messages = [{"role": "user", "content": query}]
    stream = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=True
    )

    # 2. 严格遵循 SSE 协议格式：`data: 你的数据\n\n`
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            # 为了安全防止乱码，通常包一层 JSON
            payload = json.dumps({"text": content}, ensure_ascii=False)   # text这个字段，前端约定好接收这个字段就行
            yield f"data: {payload}\n\n"   # SSE 协议要求每条消息以 `data: ` 开头，结尾以两个换行符 `\n\n` 结束

    # 发送结束标记
    yield "data: [DONE]\n\n"

#
# 路由装饰器——注册接口、绑定路径 + 请求方式
# 类型 = HTTP POST 方法装饰器
# 作用 = 创建一个接收 POST 请求的接口
# 3. 暴露一个 POST 接口
@app.post("/chat")  #路由装饰器
async def chat_endpoint(request_data: dict):   # 错误写成了request_data: str，导致无法解析 JSON 请求体
    query = request_data.get("query", "你好")

    # 4. 返回 StreamingResponse，大模型吐一个字，FastAPI 往前端推一个字
    return StreamingResponse(
        stream_generator(query),
        media_type="text/event-stream"   # SSE 的 MIME 类型，告诉浏览器这是一个事件流
    )


if __name__ == "__main__":
    print(
        '''启动服务器，在终端运行：
        Invoke-RestMethod -Uri http://127.0.0.1:8000/chat -Method Post -Body '{"query":"什么是AI?"}' -ContentType "application/json"
        '''
    )
    uvicorn.run(app, host="127.0.0.1", port=8000)