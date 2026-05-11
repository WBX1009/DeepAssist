import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    # === [模型鉴权] ===
    DEEPSEEK_API_KEY: str = Field(default="")
    DEEPSEEK_BASE_URL: str = Field(default="https://api.deepseek.com")

    # === [模型名称定义] ===
    LLM_CHAT_MODEL: str = "deepseek-chat"
    LLM_REASONER_MODEL: str = "deepseek-reasoner"

    # ===[路径定义 (双端自适应兜底)] ===
    EMBEDDING_MODEL_PATH: str = Field(default=str(PROJECT_ROOT / "data" / "models" / "bge-m3"))
    VECTOR_DB_PATH: str = Field(default=str(PROJECT_ROOT / "data" / "indexes" / "vector_store"))
    KEYWORD_DB_PATH: str = Field(default=str(PROJECT_ROOT / "data" / "indexes" / "keyword_store"))
    CONVERSATION_DB_PATH: str = Field(default=str(PROJECT_ROOT / "data" / "application_db" / "conversations.db"))
    USER_PROFILE_DB_PATH: str = Field(default=str(PROJECT_ROOT / "data" / "application_db" / "user_profiles.db"))
    INGEST_PROGRESS_PATH: str = Field(default=str(PROJECT_ROOT / "data" / "application_db" / "ingest_progress.json"))
    KB_MANIFEST_PATH: str = Field(default=str(PROJECT_ROOT / "data" / "application_db" / "knowledge_base_manifest.json"))
    VECTOR_HEALTH_REPORT_PATH: str = Field(default=str(PROJECT_ROOT / "data" / "application_db" / "vector_index_doctor_report.json"))

    # === [检索配置] ===
    RRF_K: int = 60
    RETRIEVAL_TOP_K: int = 5
    ENABLE_LLM_REWRITE: bool = Field(default=True)
    
    # 🌟 新增：RAG 检索低相关性阈值
    # 在 RAG 模式下，如果检索结果的最高得分低于该阈值，则视为 "low_relevance" 并触发回退。
    # 默认值为 0.2，低于旧版固定的 0.35，更好地适配长句查询和复杂指令。
    RAG_LOW_RELEVANCE_THRESHOLD: float = Field(default=0.2)

    # === [Agent配置] ===
    MAX_AGENT_STEPS: int = 10
    ALLOWED_TOOL_CATEGORIES: str = "read_only,write_file,network,db_query"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

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
        if v and v.startswith("./"):
            return str((PROJECT_ROOT / v[2:]).resolve())
        return v

settings = Settings()