import os
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
from typing import List, Iterator
from openai import OpenAI

# =======================================================
# 🏛️ Domain Layer (领域层) - 系统的灵魂与法则
# =======================================================

# --- 1. 数据契约 (Data Contract / Entity) ---
class Message(BaseModel):
    """定义系统中标准的消息结构，任何人不准乱传 dict"""
    role: str = Field(..., description="角色: user, assistant, system")
    content: str = Field(..., description="消息正文内容")   # 类体不空，不需要pass

# --- 2. 行为契约 (Behavior Contract / Interface) ---
class BaseLLM(ABC):
    """定义大模型能力的动作标准。输入输出必须是我们在 Domain 中定义的数据契约！"""
    
    @abstractmethod
    def chat_stream(self, messages: List[Message]) -> Iterator[str]:
        pass   # 方法体为空，必须写pass


# =======================================================
# 🔌 Infrastructure Layer (基础设施层) - 遵守契约的打工人
# =======================================================

class DeepSeekClient(BaseLLM):
    """DeepSeek 的具体实现，它必须向 Domain 的契约低头"""
    def __init__(self):
        # 实际应从 settings 读取
        self.client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY", "sk-xxx"), 
            base_url="https://api.deepseek.com"
        )

    def chat_stream(self, messages: List[Message]) -> Iterator[str]:
        # ⚠️ 基础设施层的核心脏活：把业务的数据契约，转换为第三方 SDK 需要的格式
        # message.model_dump() 会安全地将 Pydantic 实体转为 openai 要求的 dict 格式
        sdk_messages =[msg.model_dump() for msg in messages]
        
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=sdk_messages,  # 喂给第三方 SDK
            stream=True
        )
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                yield content


# =======================================================
# 👑 Application Layer (应用层) - 业务流程编排
# =======================================================

def run_business_logic(llm: BaseLLM):
    print("🚀 业务层开始编排流程...\n")
    
    # 1. 实例化数据契约 (而不是直接手写 dict)
    sys_msg = Message(role="system", content="你是一个暴躁的老哥。")
    usr_msg = Message(role="user", content="请解释依赖倒置原则。")
    
    # 2. 调用行为契约
    generator = llm.chat_stream([sys_msg, usr_msg])
    
    print("🤖 收到大模型流式回复：")
    for word in generator:
        print(word, end="", flush=True)


if __name__ == "__main__":
    # 依赖注入：在系统边缘，将具体实现注入给业务层
    my_llm = DeepSeekClient() 
    run_business_logic(my_llm)