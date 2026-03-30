from fastapi import APIRouter, Depends, UploadFile, File
from backend.application.kb_app import KnowledgeBaseApp
from backend.api.dependencies import get_kb_app

router = APIRouter(prefix="/api/kb", tags=["Knowledge Base"])

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    app: KnowledgeBaseApp = Depends(get_kb_app)
):
    """处理前端上传的文件，切分并入库"""
    content = await file.read()
    
    # 假设前端传来的都是 utf-8 的文本/Markdown
    text_content = content.decode("utf-8")
    
    # 调用应用层进行处理
    result = app.process_document(file_name=file.filename, content=text_content)
    return result