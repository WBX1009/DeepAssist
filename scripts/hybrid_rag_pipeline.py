import os
import json
import shutil
import warnings
from typing import List, Dict

# 忽略一些三方库的警告信息，保持控制台清爽
warnings.filterwarnings("ignore")


# ==========================================
# 4. 领域层 (Domain): 统一的实体契约
# ==========================================
class DocumentChunk:
    def __init__(self, id: str, content: str, metadata: dict = None, score: float = 0.0):
        self.id = id
        self.content = content
        self.metadata = metadata or {}
        self.score = score


# ==========================================
# 5. 基础设施层 (Infrastructure): 外部依赖的具体实现
# ==========================================
from sentence_transformers import SentenceTransformer
import chromadb
from whoosh.index import create_in, open_dir, exists_in
from whoosh.fields import Schema, TEXT, ID, STORED
from whoosh.qparser import QueryParser
from jieba.analyse import ChineseAnalyzer


class BGEM3Local:
    """本地 BGE-M3 嵌入模型封装"""

    def __init__(self):
        print("⏳ 正在加载本地 bge-m3 嵌入模型 (首次运行会自动下载权重)...")
        # 实际项目中，这里填入你服务器上的绝对路径，如: /workspace/DeepAssist/data/models/bge-m3
        # 这里为了演示通用性，使用 HuggingFace 仓库 ID，会自动走缓存
        self.model = SentenceTransformer("BAAI/bge-m3", device="cpu")
        print("✅ 嵌入模型加载完成！")

    def embed_text(self, text: str) -> List[float]:
        """单条文本转向量（用于查询阶段）"""
        if not text.strip(): return []
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量文本转向量（用于建库阶段）"""
        if not texts: return []
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32).tolist()


class ChromaStore:
    """真实的 Chroma 向量库：接收外部计算好的向量，纯做 ANN 近似最近邻检索"""

    def __init__(self):
        self.db_path = "./real_chroma_db"
        # 强制使用本地持久化，落盘到真实目录
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(name="tech_kb")

    def _sanitize_metadata(self, metadata: dict) -> dict:
        """安全清洗机制"""
        sanitized = {}
        for k, v in metadata.items():
            if isinstance(v, (list, dict)):
                sanitized[k] = json.dumps(v, ensure_ascii=False)
            elif v is not None:
                sanitized[k] = v
        return sanitized

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]):
        """显式传入由 BGE-M3 算好的向量 (embeddings)"""
        if not chunks: return
        ids = [c.id for c in chunks]
        documents = [c.content for c in chunks]
        metadatas = [self._sanitize_metadata(c.metadata) for c in chunks]

        # 关键变化：直接传入 embeddings 列表，绕过 Chroma 默认模型
        self.collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    def search(self, query_vector: List[float], top_k: int) -> List[DocumentChunk]:
        """接收高维浮点数组，执行 ANN 检索"""
        if not query_vector: return []

        results = self.collection.query(
            query_embeddings=[query_vector],  # 直接传入浮点数组进行相似度计算
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        chunks = []
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                distance = results['distances'][0][i]
                chunks.append(DocumentChunk(
                    id=results['ids'][0][i],
                    content=results['documents'][0][i],
                    metadata=results['metadatas'][0][i],
                    score=1.0 / (1.0 + distance)  # 余弦距离转得分
                ))
        return chunks


class WhooshStore:
    """真实的 BM25 全文索引库"""

    def __init__(self):
        self.db_path = "./real_whoosh_db"
        os.makedirs(self.db_path, exist_ok=True)

        self.schema = Schema(
            id=ID(stored=True, unique=True),
            content=TEXT(stored=True, analyzer=ChineseAnalyzer()),
            metadata=STORED()
        )
        # 如果不存在则建库，存在则打开
        if not exists_in(self.db_path):
            self.ix = create_in(self.db_path, self.schema)
        else:
            self.ix = open_dir(self.db_path)

    def add_chunks(self, chunks: List[DocumentChunk]):
        writer = self.ix.writer()
        for c in chunks:
            writer.update_document(id=c.id, content=c.content, metadata=json.dumps(c.metadata))
        writer.commit()

    def search(self, query: str, top_k: int) -> List[DocumentChunk]:
        chunks = []
        try:
            with self.ix.searcher() as searcher:
                q = QueryParser("content", self.ix.schema).parse(query)
                results = searcher.search(q, limit=top_k)
                for hit in results:
                    chunks.append(DocumentChunk(
                        id=hit["id"],
                        content=hit["content"],
                        metadata=json.loads(hit["metadata"]),
                        score=hit.score
                    ))
        except Exception:
            pass
        return chunks


# ==========================================
# 3. 服务层 (Service): 业务算法核心 (RRF 融合)
# ==========================================
class HybridRetriever:
    """混合检索服务：统筹外部模型能力，执行 RRF 融合打分"""

    def __init__(self, vector_db: ChromaStore, keyword_db: WhooshStore, embedding_model: BGEM3Local):
        self.vector_db = vector_db
        self.keyword_db = keyword_db
        self.embedding_model = embedding_model
        self.rrf_k = 60  # RRF 平滑常数

    def retrieve(self, query: str, final_top_k: int = 3) -> List[DocumentChunk]:
        candidate_k = final_top_k * 2

        # 👑 核心逻辑：自己动手调模型获取向量，传给底层数据库
        query_vector = self.embedding_model.embed_text(query)

        # 双路并发召回
        vector_results = self.vector_db.search(query_vector, candidate_k)
        keyword_results = self.keyword_db.search(query, candidate_k)

        # RRF 融合重排算法
        score_map: Dict[str, float] = {}
        chunk_map: Dict[str, DocumentChunk] = {}

        for rank, chunk in enumerate(vector_results):
            score_map[chunk.id] = score_map.get(chunk.id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            chunk_map[chunk.id] = chunk

        for rank, chunk in enumerate(keyword_results):
            score_map[chunk.id] = score_map.get(chunk.id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            chunk_map[chunk.id] = chunk

        # 降序排列并截断
        sorted_ids = sorted(score_map.keys(), key=lambda x: score_map[x], reverse=True)
        final_results = []
        for cid in sorted_ids[:final_top_k]:
            c = chunk_map[cid]
            c.score = score_map[cid]
            final_results.append(c)

        return final_results


# ==========================================
# 2. 应用层 (Application): 编排检索流
# ==========================================
class RAGApplication:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def process_query(self, user_query: str):
        print(f"\n👤 [App 层] 收到用户检索指令: '{user_query}'")

        # 1. 触发底层级联检索
        docs = self.retriever.retrieve(user_query, final_top_k=2)

        # 2. 编排结果
        if not docs:
            print("🤖 [App 层拦截] 知识库中未查找到任何匹配项。")
            return

        print("💡 [App 层] RRF 融合后的优质参考上下文：")
        for i, d in enumerate(docs):
            print(f"[{i + 1}] (RRF 综合得分: {d.score:.4f}) -> {d.content}")


# ==========================================
# 独立测试区域：启动生命周期
# ==========================================
def init_test_data(v_db: ChromaStore, k_db: WhooshStore, embed_model: BGEM3Local):
    # 为防止测试污染，先清理旧的 Chroma 集合
    try:
        v_db.client.delete_collection("tech_kb")
        v_db.collection = v_db.client.create_collection("tech_kb")
    except Exception:
        pass

    test_chunks = [
        DocumentChunk("doc1", "Linux 服务器磁盘空间满了怎么清理？可以使用 du -sh * 命令排查大文件。"),
        DocumentChunk("doc2", "前端 Vue 组件通信方式有很多，比如 props、emit、EventBus 和 Vuex。"),
        DocumentChunk("doc3", "Redis 的持久化机制分为 RDB 和 AOF 两种，RDB 是快照，AOF 是追加日志。"),
        # 故意放一条包含专业英文缩写、依靠 BM25 优势召回的内容
        DocumentChunk("doc4", "执行 Docker 构建时，如果遇到 ERROR-255-EOF 异常，请检查 Dockerfile 文件层级。")
    ]

    # 模拟“业务建库”过程：应用层负责调用模型生成向量，然后传给数据库（双写一致性）
    print("\n🧱 正在执行数据向量化和落库...")
    texts = [c.content for c in test_chunks]
    embeddings = embed_model.embed_documents(texts)

    v_db.add_chunks(test_chunks, embeddings)
    k_db.add_chunks(test_chunks)
    print("✅ 知识库初始化成功！")


if __name__ == "__main__":
    print("===" * 15)
    print("🚀 DeepAssist V3 真·双路混合检索底座测试")
    print("===" * 15)

    # DI: 依赖注入
    embed_model = BGEM3Local()
    v_db = ChromaStore()
    k_db = WhooshStore()
    retriever = HybridRetriever(v_db, k_db, embed_model)
    app = RAGApplication(retriever)

    # 1. 模拟入库生命周期
    init_test_data(v_db, k_db, embed_model)

    # 2. 模拟真实检索测试
    app.process_query("我服务器报了一个叫 ERROR-255-EOF 的错误，怎么破？")  # BM25 的强项
    app.process_query("缓存数据库挂了，快照数据怎么恢复？")  # 向量语义的强项 (没直接搜Redis)