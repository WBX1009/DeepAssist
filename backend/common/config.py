import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator

# 🌟 核心魔法：动态获取项目根目录
# 无论是在 Windows (E:\DeepAssist) 还是 Linux (/workspace/DeepAssist)
# 它都能精准定位，因为它是基于当前 config.py 文件的物理位置向上找 3 层
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    # === [模型鉴权] ===
    DEEPSEEK_API_KEY: str = Field(default="")
    DEEPSEEK_BASE_URL: str = Field(default="https://api.deepseek.com")
    
    # === [模型名称定义] ===
    LLM_CHAT_MODEL: str = "deepseek-chat"
    LLM_REASONER_MODEL: str = "deepseek-reasoner"
    
    # === [路径定义 (双端自适应兜底)] ===
    # 就算没有 .env 文件，这里的默认值也会根据 PROJECT_ROOT 自动生成正确的绝对路径
    EMBEDDING_MODEL_PATH: str = Field(default=str(PROJECT_ROOT / "data" / "models" / "bge-m3"))
    VECTOR_DB_PATH: str = Field(default=str(PROJECT_ROOT / "data" / "indexes" / "vector_store"))
    KEYWORD_DB_PATH: str = Field(default=str(PROJECT_ROOT / "data" / "indexes" / "keyword_store"))
    CONVERSATION_DB_PATH: str = Field(default=str(PROJECT_ROOT / "data" / "application_db" / "conversations.db"))
    USER_PROFILE_DB_PATH: str = Field(default=str(PROJECT_ROOT / "data" / "application_db" / "user_profiles.db"))
    INGEST_PROGRESS_PATH: str = Field(default=str(PROJECT_ROOT / "data" / "application_db" / "ingest_progress.json"))
    KB_MANIFEST_PATH: str = Field(default=str(PROJECT_ROOT / "data" / "application_db" / "knowledge_base_manifest.json"))
    VECTOR_HEALTH_REPORT_PATH: str = Field(default=str(PROJECT_ROOT / "data" / "application_db" / "vector_index_doctor_report.json"))
    
    # ===[检索配置] ===
    RRF_K: int = 60
    RETRIEVAL_TOP_K: int = 5
    
    # === [Agent配置] ===
    MAX_AGENT_STEPS: int = 10

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore"
    )

    # 🌟 核心魔法 2：路径拦截转换器
    @field_validator(
        "EMBEDDING_MODEL_PATH", 
        "VECTOR_DB_PATH", 
        "KEYWORD_DB_PATH", 
        "CONVERSATION_DB_PATH", 
        "USER_PROFILE_DB_PATH", 
        "INGEST_PROGRESS_PATH",
        "KB_MANIFEST_PATH",
        "VECTOR_HEALTH_REPORT_PATH",
        mode="after"
    )
    @classmethod
    def resolve_relative_path(cls, v: str) -> str:
        """
        如果 .env 文件中写的是 './data/xxx' 这种相对路径，
        这里会自动拦截，并将其与 PROJECT_ROOT 拼接成绝对路径。
        这彻底解决了 ChromaDB 和 SQLite 在不同终端目录下启动时找错路径的 Bug！
        """
        if v and v.startswith("./"):
            # 将 ./ 去掉，拼接到根目录后转为字符串
            return str((PROJECT_ROOT / v[2:]).resolve())
        return v

# 全局单例配置对象
settings = Settings()

# ==========================================
# 调试用：你可以在文件末尾加上这几行测试一下，确保路径正确
# ==========================================
if __name__ == "__main__":
    print(f"当前推断的项目根目录: {PROJECT_ROOT}")
    print(f"向量库最终加载路径: {settings.VECTOR_DB_PATH}")
    print(f"模型最终加载路径: {settings.EMBEDDING_MODEL_PATH}")
