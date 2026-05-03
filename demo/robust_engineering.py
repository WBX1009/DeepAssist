import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI, APIConnectionError, RateLimitError

# 1. 配置标准企业级日志格式
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger("LLM_Service")

# Mock Client (替换为你的真实配置)
client = OpenAI(api_key="sk-xxxx", base_url="https://api.deepseek.com")

# 2. 引入神仙库 Tenacity 进行防御编程
# 规则：最多重试 3 次；每次等待时间按指数增长 (1s, 2s, 4s)；只针对网络断开或限流进行重试，若是权限报错(401)则不重试直接死。
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((APIConnectionError, RateLimitError)),
    reraise=True # 如果 3 次都失败了，把异常抛出给上层
)
def safe_chat_call(prompt: str) -> str:
    logger.info(f"🚀 准备请求大模型 (Prompt: {prompt[:5]}...)")
    
    # 这里为了演示重试，你可以故意把 base_url 写错，看它会不会重试 3 次
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        timeout=5.0 # 强制 5 秒超时，防止死等
    )
    logger.info("✅ 请求成功！")
    return response.choices[0].message.content

if __name__ == "__main__":
    try:
        ans = safe_chat_call("你好")
        print(ans)
    except Exception as e:
        logger.error(f"❌ 经历了最大重试后依然失败: {e}")