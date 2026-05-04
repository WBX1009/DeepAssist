import os
import sys
import json
import re
import hashlib
import signal
from pathlib import Path
from typing import Iterator, List, Tuple
from tqdm import tqdm
import pandas as pd

# 系统资源与环境配置
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path: sys.path.append(project_root)

from backend.common.config import settings
from backend.common.logger import get_logger
from backend.domain.entities.document import DocumentChunk
from backend.infrastructure.embeddings.bge_m3_local import BGEM3Local
from backend.infrastructure.databases.chroma_store import ChromaStore
from backend.infrastructure.databases.whoosh_store import WhooshStore
from backend.services.rag.chunking import DocumentChunker

logger = get_logger("Server_Build_Index")

# ==========================================
# 🎛️ 核心配置：每个原始文件的最大读取行数/样本数
# ==========================================
# 设置为 float('inf') 代表全量入库，设置为 50000 适合快速验证与开发。
MAX_SAMPLES_PER_FILE = 50000  

# ==========================================
# 🛑 全局中断信号监听 (Graceful Shutdown)
# ==========================================
SHUTDOWN_REQUESTED = False

def signal_handler(sig, frame):
    global SHUTDOWN_REQUESTED
    if not SHUTDOWN_REQUESTED:
        print("\n" + "!"*60)
        print("🛑 接收到中断信号 (Ctrl+C)！")
        print("⏳ 正在安全释放资源，请等待当前批次入库完成...")
        print("⚠️ 请不要再次按 Ctrl+C，否则可能导致数据库死锁！")
        print("!"*60 + "\n")
        SHUTDOWN_REQUESTED = True

signal.signal(signal.SIGINT, signal_handler)

def generate_structural_id(file_path: str, index: int, sub_type: str = "") -> str:
    raw_str = f"{file_path}_{index}_{sub_type}"
    return hashlib.md5(raw_str.encode('utf-8')).hexdigest()


def sanitize_metadata(metadata: dict) -> dict:
    sanitized = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        elif isinstance(value, (list, tuple)):
            sanitized[key] = [str(item) for item in value if item is not None]
        elif isinstance(value, dict):
            sanitized[key] = {
                str(inner_key): str(inner_value)
                for inner_key, inner_value in value.items()
                if inner_value is not None
            }
        else:
            sanitized[key] = str(value)
    return sanitized

class ProgressManager:
    def __init__(self):
        self.progress_file = Path(settings.INGEST_PROGRESS_PATH)
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        self.progress = self._load()

    def _load(self) -> dict:
        if self.progress_file.exists():
            with open(self.progress_file, "r", encoding="utf-8") as f: return json.load(f)
        return {}

    def get_last_index(self, file_path: str) -> int:
        return self.progress.get(str(file_path), -1)

    def save_index(self, file_path: str, index: int):
        self.progress[str(file_path)] = index
        with open(self.progress_file, "w", encoding="utf-8") as f:
            json.dump(self.progress, f, ensure_ascii=False)

# ==========================================
# 🧩 适配器 (加入上限截断)
# ==========================================
def adapter_medical(file_path: str, start_idx: int) -> Iterator[Tuple[DocumentChunk, int]]:
    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= MAX_SAMPLES_PER_FILE: break # 🟢 达到上限直接停止读取
            if idx <= start_idx: continue
            if not line.strip(): continue
            try:
                data = json.loads(line)
                content = f"Q: {data.get('instruction', '') + data.get('input', '')}\nA: {data.get('output', '')}"
                yield DocumentChunk(
                    id=generate_structural_id(file_path, idx),
                    content=content,
                    metadata=sanitize_metadata(
                        {"source_file": file_path, "domain": "medical", "type": "qa"}
                    ),
                ), idx
            except: continue

def adapter_legal(file_path: str, start_idx: int) -> Iterator[Tuple[DocumentChunk, int]]:
    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= MAX_SAMPLES_PER_FILE: break
            if idx <= start_idx: continue
            if not line.strip(): continue
            try:
                data = json.loads(line)
                content = f"Q: {data.get('input', '')}\nA: {data.get('output', '')}"
                yield DocumentChunk(
                    id=data.get("id", generate_structural_id(file_path, idx)),
                    content=content,
                    metadata=sanitize_metadata(
                        {"source_file": file_path, "domain": "legal", "type": "qa"}
                    ),
                ), idx
            except: continue

def adapter_financial(file_path: str, start_idx: int) -> Iterator[Tuple[DocumentChunk, int]]:
    with open(file_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)
        for idx, data in enumerate(data_list):
            if idx >= MAX_SAMPLES_PER_FILE: break
            if idx <= start_idx: continue
            instruction = data.get('instruction', '')
            input_text = data.get('input', '')
            raw_output = data.get('output', '')

            clean_output = re.sub(r'\[Calculator.*?\]', '', raw_output)

            content = f"Q: {instruction + input_text}\nA: {clean_output}"
            yield DocumentChunk(
                id=generate_structural_id(file_path, idx),
                content=content,
                metadata=sanitize_metadata(
                    {"source_file": file_path, "domain": "finance", "type": "qa"}
                ),
            ), idx

def adapter_enron(file_path: str, start_idx: int) -> Iterator[Tuple[DocumentChunk, int]]:
    chunker = DocumentChunker()
    df = pd.read_parquet(file_path)
    for row_idx, row in df.iterrows():
        if row_idx >= MAX_SAMPLES_PER_FILE: break
        if row_idx <= start_idx: continue
        email_content = row.get("email", "")
        if not email_content: continue
        questions, gold_answers = row.get("questions",[]), row.get("gold_answers",[])
        if isinstance(questions, (list, tuple)) and isinstance(gold_answers, (list, tuple)):
            for qa_idx, (q, a) in enumerate(zip(questions, gold_answers)):
                yield DocumentChunk(
                    id=generate_structural_id(file_path, row_idx, f"qa_{qa_idx}"),
                    content=f"Q: {q}\nA: {a}",
                    metadata=sanitize_metadata(
                        {"source_file": file_path, "domain": "office", "type": "qa"}
                    ),
                ), row_idx
        for chunk_idx, c in enumerate(chunker.split_markdown(email_content, source_name="enron")):
            c.id = generate_structural_id(file_path, row_idx, f"email_chunk_{chunk_idx}")
            c.metadata.update(
                sanitize_metadata(
                    {"source_file": file_path, "domain": "office", "type": "email"}
                )
            )
            yield c, row_idx

# ==========================================
# ⚙️ 串行安全引擎
# ==========================================
def ingest_file_to_db(collection_name: str, file_path: str, adapter_func, 
                      embedding_model: BGEM3Local, vector_db: ChromaStore, keyword_db: WhooshStore, 
                      progress_mgr: ProgressManager, batch_size: int = 1024):
    
    global SHUTDOWN_REQUESTED
    start_idx = progress_mgr.get_last_index(file_path)
    
    # 逻辑防御：如果之前已经跑到上限了，直接跳过，防止反复清空重建
    if start_idx >= MAX_SAMPLES_PER_FILE - 1:
        logger.info(f"⏭️ 文件 {Path(file_path).name} 已达到预设上限 ({MAX_SAMPLES_PER_FILE})，跳过...")
        return

    if start_idx == -1:
        logger.info(f"🧹 首次写入 {Path(file_path).name}，清理历史冗余数据...")
        vector_db.delete_by_source(collection_name, file_path)
        keyword_db.delete_by_source(collection_name, file_path)
    else:
        logger.info(f"♻️ 断点触发！文件 {Path(file_path).name} 将从第 {start_idx + 1} 行续传。")
        
    chunk_generator = adapter_func(file_path, start_idx)
    batch_chunks =[]
    last_row_idx = -1
    total_inserted = 0
    pbar = tqdm(desc=f"写入 {Path(file_path).name}", unit="条")
    
    for chunk, row_idx in chunk_generator:
        if SHUTDOWN_REQUESTED: break

        batch_chunks.append(chunk)
        last_row_idx = row_idx
        
        if len(batch_chunks) >= batch_size:
            texts = [c.content for c in batch_chunks]
            embeddings = embedding_model.embed_documents(texts)
            
            vector_db.add_chunks(collection_name, batch_chunks, embeddings)
            keyword_db.build_index(collection_name, batch_chunks)
            
            progress_mgr.save_index(file_path, last_row_idx)
            total_inserted += len(batch_chunks)
            pbar.update(len(batch_chunks))
            batch_chunks =[]
            
    if batch_chunks:
        texts =[c.content for c in batch_chunks]
        embeddings = embedding_model.embed_documents(texts)
        vector_db.add_chunks(collection_name, batch_chunks, embeddings)
        keyword_db.build_index(collection_name, batch_chunks)
        progress_mgr.save_index(file_path, last_row_idx)
        total_inserted += len(batch_chunks)
        pbar.update(len(batch_chunks))
        
    pbar.close()

# ==========================================
# 🚀 主程序
# ==========================================
def main():
    global SHUTDOWN_REQUESTED
    logger.info(f"⏳ 初始化单卡高稳定性流水线 (上限: {MAX_SAMPLES_PER_FILE} 条/文件)...")
    embedding_model = BGEM3Local()
    vector_db = ChromaStore()
    keyword_db = WhooshStore()
    progress_mgr = ProgressManager()
    
    sources_dir = Path(project_root) / "data" / "sources"
    
    tasks =[
        {"collection": "medical_kb", "path": sources_dir / "medical" / "finetune" / "train_zh_0.json", "adapter": adapter_medical},
        {"collection": "legal_kb", "path": sources_dir / "DISC-Law-SFT" / "DISC-Law-SFT-Pair-QA-released.jsonl", "adapter": adapter_legal},
        {"collection": "financial_kb", "path": sources_dir / "DISC-FIN-SFT" / "data" / "total.json", "adapter": adapter_financial},
        {"collection": "enron_kb", "path": sources_dir / "enron_qa_0922" / "data" / "train-00000-of-00002.parquet", "adapter": adapter_enron},
        {"collection": "enron_kb", "path": sources_dir / "enron_qa_0922" / "data" / "train-00001-of-00002.parquet", "adapter": adapter_enron}
    ]

    for task in tasks:
        if SHUTDOWN_REQUESTED: break
            
        logger.info("=" * 60)
        file_path_str = str(task["path"])
        if not task["path"].exists():
            logger.error(f"❌ 数据源不存在，跳过: {file_path_str}")
            continue
            
        ingest_file_to_db(
            collection_name=task["collection"],
            file_path=file_path_str,
            adapter_func=task["adapter"],
            embedding_model=embedding_model,
            vector_db=vector_db,
            keyword_db=keyword_db,
            progress_mgr=progress_mgr,
            batch_size=1024 
        )

    if SHUTDOWN_REQUESTED:
        logger.info("👋 任务已安全中止！进度已记录，下次运行将自动续传。")
    else:
        logger.info("🎉 恭喜！单卡极限稳定性流水线完美执行完毕！")

if __name__ == "__main__":
    main()
