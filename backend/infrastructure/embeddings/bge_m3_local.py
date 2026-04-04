import os
import warnings
from typing import List

from backend.domain.interfaces.embedding import BaseEmbedding
from backend.core.config import settings
from backend.core.logger import get_logger

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

class BGEM3Local(BaseEmbedding):
    def __init__(self):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        # 🟢 独占卡 2
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        from sentence_transformers import SentenceTransformer
        
        model_path = settings.EMBEDDING_MODEL_PATH
        logger.info(f"⏳ 正在加载离线 Embedding 模型 (Device: {self.device}): {model_path}")
        
        try:
            self.model = SentenceTransformer(model_path, device=self.device)
            
            # 🔥 最佳实践：限制在 512 Tokens，保证最高语义浓度！超出的部分向量化时丢弃，但数据库存全量。
            self.model.max_seq_length = 512
            logger.info("✅ BGE-M3 加载成功！单卡模式，上下文截断限制在 512 Tokens。")
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            raise e

    def embed_text(self, text: str) -> List[float]:
        if not text.strip(): return[]
        vector = self.model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return vector.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts: return[]
        
        # 🟢 纯单卡模式：长度缩短到 512 后，A6000 的显存可以轻松吃下 256 的 Batch
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=256 
        )
        return embeddings.tolist()