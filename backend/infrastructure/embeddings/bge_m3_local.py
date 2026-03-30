import os
import warnings
from typing import List

from backend.domain.interfaces.embedding import BaseEmbedding
from backend.core.config import settings
from backend.core.logger import get_logger

# 忽略一些底层框架的警告信息
warnings.filterwarnings("ignore")

logger = get_logger(__name__)

class BGEM3Local(BaseEmbedding):
    """
    BGE-M3 本地离线向量模型实现类。
    完全遵守 domain/interfaces/embedding.py 的契约。
    """
    def __init__(self):
        # ⚠️ 强制植入离线运行规范（对应你最初给出的核心约束）
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        
        # 推理设备逻辑：有GPU用GPU，没GPU退化到CPU（方便你服务器构建，本地笔记本检索）
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 延迟导入，加快应用启动速度，只在实例化时加载模型
        from sentence_transformers import SentenceTransformer
        
        model_path = settings.EMBEDDING_MODEL_PATH
        logger.info(f"⏳ 正在加载离线 Embedding 模型 (Device: {self.device}): {model_path}")
        
        try:
            # 加载离线模型文件夹
            self.model = SentenceTransformer(model_path, device=self.device)
            logger.info("✅ BGE-M3 本地模型加载成功！")
        except Exception as e:
            logger.error(f"❌ BGE-M3 模型加载失败，请检查路径 {model_path} 是否正确: {e}")
            raise e

    def embed_text(self, text: str) -> List[float]:
        """
        单条文本向量化（例如：用户在界面上提了一个问题，立刻转成向量去搜索）
        """
        if not text.strip():
            return[]
            
        # normalize_embeddings=True 强烈建议用于 RAG 余弦相似度计算
        vector = self.model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return vector.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量文本向量化（例如：服务器后台切分了一份 50 页的 PDF，批量计算向量写入数据库）
        """
        if not texts:
            return[]
            
        # 批量编码，开启进度条（在服务器端构建大量数据时，可以直观看到进度）
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        return embeddings.tolist()