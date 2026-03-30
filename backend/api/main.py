import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 确保将项目根目录加入环境，以便正确执行相对导入
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.api.routers import chat_api, agent_api, kb_api
from backend.core.logger import get_logger

logger = get_logger("FastAPI_Main")

app = FastAPI(
    title="MediAsk Enterprise API",
    description="支持 RAG、Agent、流式输出的大模型应用系统接口",
    version="2.0.0"
)

# 配置跨域资源共享 (CORS) 
# 因为 Streamlit 运行在 8501 端口，FastAPI 运行在 8000 端口
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境请改为前端的实际地址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册所有路由模块
app.include_router(chat_api.router)
app.include_router(agent_api.router)
app.include_router(kb_api.router)

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "MediAsk Backend is running smoothly."}

if __name__ == "__main__":
    import uvicorn
    logger.info("启动 FastAPI 服务器...")
    uvicorn.run("backend.api.main:app", host="0.0.0.0", port=8000, reload=True)