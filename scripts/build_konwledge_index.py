import os
import sys
import shutil
from pathlib import Path
from tqdm import tqdm

# 将项目根目录加入 sys.path，确保能正确导入 backend 模块
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# ⚠️ 强制植入离线环境变量（服务器端强管控）
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from backend.core.config import settings
from backend.core.logger import get_logger

# 导入底层具体实现 (Infrastructure)
from backend.infrastructure.embeddings.bge_m3_local import BGEM3Local
from backend.infrastructure.databases.chroma_store import ChromaStore
from backend.infrastructure.databases.whoosh_store import WhooshStore

# 导入核心服务 (Services)
from backend.services.rag.chunking import DocumentChunker

logger = get_logger("Server_Build_Index")

def clean_old_dbs():
    """清理旧的索引库，确保每次构建都是干净的"""
    logger.info("🧹 正在清理旧的向量库和关键词索引库...")
    if os.path.exists(settings.VECTOR_DB_PATH):
        shutil.rmtree(settings.VECTOR_DB_PATH)
    if os.path.exists(settings.KEYWORD_DB_PATH):
        shutil.rmtree(settings.KEYWORD_DB_PATH)
    logger.info("✅ 清理完毕。")

def main():
    print("🚀=" * 50)
    print("🌟 MediAsk 企业级脱网数据流水线启动 🌟")
    print("🚀=" * 50)

    # 1. 准备数据目录 (假设原始文档放在 data/raw_docs 下)
    raw_docs_dir = os.path.join(project_root, "data", "raw_docs")
    os.makedirs(raw_docs_dir, exist_ok=True)
    
    # 查找所有 markdown 和 txt 文件 (后续可扩展 PDF)
    files = list(Path(raw_docs_dir).rglob("*.md")) + list(Path(raw_docs_dir).rglob("*.txt"))
    if not files:
        logger.error(f"❌ 未在 {raw_docs_dir} 找到任何 .md 或 .txt 文档，请放入技术文档后重试！")
        return

    # 2. 清理历史数据 (可选：如果是增量更新请注释掉此行)
    clean_old_dbs()

    # 3. 初始化所有组件 (依赖注入准备)
    logger.info("⏳ 正在初始化基础设施组件...")
    embedding_model = BGEM3Local()
    vector_db = ChromaStore()
    keyword_db = WhooshStore()
    chunker = DocumentChunker()
    
    collection_name = "tech_docs_kb"  # 统一使用的知识库集合名称

    # 4. 读取并切分文档
    all_chunks =[]
    logger.info("📖 正在读取并切分文档...")
    for file_path in tqdm(files, desc="Parsing Docs"):
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            # 调用切分服务，保留层级结构
            chunks = chunker.split_markdown(content, source_name=file_path.name)
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"读取文件 {file_path.name} 失败: {e}")

    if not all_chunks:
        logger.warning("⚠️ 切分后没有得到任何有效的 Chunk。")
        return

    logger.info(f"✅ 文档切分完成，共生成 {len(all_chunks)} 个 Chunks。")

    # 5. 批处理与向量化 (防 OOM 策略)
    batch_size = 64
    logger.info(f"⚡ 开始批量向量化并写入双库 (Batch Size: {batch_size})...")
    
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Ingesting to DBs"):
        batch_chunks = all_chunks[i : i + batch_size]
        
        # 提取当前批次的文本内容用于计算向量
        batch_texts =[chunk.content for chunk in batch_chunks]
        
        # 调用 BGE-M3 离线计算 Embedding
        batch_embeddings = embedding_model.embed_documents(batch_texts)
        
        # 写入 Chroma 向量库
        vector_db.add_chunks(collection_name, batch_chunks, batch_embeddings)
        
        # 写入 Whoosh 关键词库 (全文索引不依赖向量，只需文本和ID)
        keyword_db.build_index(collection_name, batch_chunks)

    logger.info("🎉🎉🎉 混合检索库 (Chroma + Whoosh) 构建完成！")
    logger.info(f"📂 向量库路径: {settings.VECTOR_DB_PATH}")
    logger.info(f"📂 索引库路径: {settings.KEYWORD_DB_PATH}")
    logger.info("➡️ 现在你可以将 /workspace 目录下的数据同步到本地笔记本进行检索测试了！")

if __name__ == "__main__":
    main()