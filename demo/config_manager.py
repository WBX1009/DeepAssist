from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# 1. 定义强类型的配置类
class Settings(BaseSettings):
    # Field(..., ) 表示必填项，启动时如果读不到直接报错！
    DEEPSEEK_API_KEY: str = Field(..., description="DeepSeek 秘钥")
    # 提供默认值，就算 .env 没写，也不会报错
    DEEPSEEK_BASE_URL: str = Field(default="https://api.deepseek.com")
    # 自动将 .env 里的字符串 "3" 转为整数 3
    MAX_RETRIES: int = Field(default=3)

    # 告诉 Pydantic 去哪里读取环境变量
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore") # forbid 模式：如果 .env 里有，但 Settings 类里没有定义，就报错；ignore 模式：如果 .env 里有，但 Settings 类里没有定义，直接忽略（推荐）

# 2. 实例化为全局单例（全项目只需导入这个 settings 即可）
settings = Settings()

if __name__ == "__main__":
    print("✅ 配置加载成功！")
    print(f"Key 前缀: {settings.DEEPSEEK_API_KEY[:6]}...")
    print(f"最大重试次数 (类型 {type(settings.MAX_RETRIES)}): {settings.MAX_RETRIES}")