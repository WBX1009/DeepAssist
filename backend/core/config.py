import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # 模型鉴权
    DEEPSEEK_API_KEY: str = Field(default="")
    DEEPSEEK_BASE_URL: str = Field(default="https://api.deepseek.com")
    
    # 模型名称定义
    LLM_CHAT_MODEL: str = "deepseek-chat"
    LLM_REASONER_MODEL: str = "deepseek-reasoner"
    
    # 路径定义 (默认指向服务器离线环境，可通过 .env 覆盖)
    EMBEDDING_MODEL_PATH: str = Field(default="/workspace/bge-m3")
    VECTOR_DB_PATH: str = Field(default="/workspace/my_vector_db")
    KEYWORD_DB_PATH: str = Field(default="/workspace/whoosh_index")
    
    # 检索配置
    RRF_K: int = 60
    RETRIEVAL_TOP_K: int = 5
    
    # Agent配置
    MAX_AGENT_STEPS: int = 10

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore"
    )

# 全局单例配置对象
settings = Settings()