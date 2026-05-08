# DeepAssist

DeepAssist 是一个面向工程实践的本地化大模型应用 Demo，核心目标是构建一个具备 **普通对话、知识库问答、Agent 工具调用、多轮会话记忆、混合检索与流式输出能力** 的 LLM 应用系统。

项目采用接近 **Clean Architecture / DDD 分层架构** 的设计方式，将 API 接入层、应用编排层、领域契约层、基础设施适配层和业务服务层进行拆分，重点体现大模型应用开发中的工程化能力，而不是只停留在简单的 API 调用 Demo。

---

## 一、项目定位

DeepAssist 不是一个简单的聊天机器人，而是一个完整的 LLM 应用后端工程示例，主要用于展示以下能力：

- FastAPI 后端接口设计
- SSE 流式输出
- RAG 知识库问答
- 向量检索与关键词检索融合
- ChromaDB 本地向量库持久化
- Whoosh 中文关键词索引
- BGE-M3 本地 Embedding
- DeepSeek 大模型接入
- Agent 工具调用与任务编排
- 多轮会话管理
- 用户长期记忆与上下文压缩
- 知识库文件上传、删除、列表与健康检查
- 面向后续扩展的分层架构设计

---

## 二、核心功能

### 1. 普通对话模式

普通对话模式用于完成不依赖知识库的 LLM 问答。

主要能力包括：

- 多轮会话历史管理
- 上下文窗口规划
- 历史轮次控制
- 用户长期记忆开关
- SSE 流式返回
- 支持模型参数配置，例如 temperature、top_p、history_rounds

对应接口：

```text
POST /api/chat/stream
```

请求中的 `mode` 设置为：

```json
{
  "mode": "quick"
}
```

---

### 2. RAG 知识库问答模式

RAG 模式用于基于本地知识库进行问答。

系统会先从知识库中检索相关文档片段，再将检索结果组织成上下文，最后交给大模型生成回答。

主要能力包括：

- Markdown / UTF-8 文档上传
- 文档切块
- Embedding 向量化
- ChromaDB 向量检索
- Whoosh 关键词检索
- RRF 融合排序
- 检索诊断信息
- 引用片段追踪
- 防幻觉检查
- 检索失败时自动降级为普通对话

对应接口：

```text
POST /api/chat/stream
```

请求中的 `mode` 设置为：

```json
{
  "mode": "rag"
}
```

---

### 3. Auto 自动路由模式

Auto 模式会根据用户问题自动判断应该进入普通对话还是知识库问答。

适合前端只暴露一个输入框，由后端根据问题意图自动选择处理路径。

请求中的 `mode` 设置为：

```json
{
  "mode": "auto"
}
```

---

### 4. Agent 模式

Agent 模式用于处理需要多步骤推理、工具调用或任务编排的问题。

当前 Agent 系统支持：

- Supervisor 路由
- Chat Worker
- RAG Worker
- Tool Agent Worker
- Orchestrator Worker
- 工具调用事件流
- 工具结果事件流
- 任务恢复
- 中断快照
- 多智能体计划追踪
- 自我修正与失败恢复事件

对应接口：

```text
POST /api/agent/stream
```

---

### 5. 知识库管理

知识库模块提供文档上传、文件列表、集合列表、健康检查和删除能力。

主要接口：

```text
POST   /api/kb/upload
GET    /api/kb/files
GET    /api/kb/collections
GET    /api/kb/health
DELETE /api/kb/files/{source_file}
```

当前上传管线主要支持 UTF-8 文本或 Markdown 文件。

---

### 6. 运行时能力查询

前端可以通过运行时接口获取后端当前支持的模型、模式、知识库集合和工具列表。

对应接口：

```text
GET /api/runtime/capabilities
```

该接口适合用于 Streamlit、React、Vue 等前端动态渲染可用功能。

---

## 三、技术栈

### 后端框架

- FastAPI
- Uvicorn
- Pydantic v2
- pydantic-settings

### 大模型接入

- OpenAI SDK
- DeepSeek Chat
- DeepSeek Reasoner

### RAG 与检索

- sentence-transformers
- BGE-M3
- ChromaDB
- Whoosh
- jieba
- RRF 融合排序
- Lexical Overlap Reranker

### Agent 与编排

- LangChain Core
- LangGraph
- 自定义 ToolRegistry
- 自定义 AgentEngine
- 自定义 AgentSupervisor

### 前端展示

- Streamlit

---

## 四、项目架构

当前项目采用分层架构，核心目录结构如下：

```text
DeepAssist/
├── backend/
│   ├── api/
│   │   ├── main.py
│   │   ├── dependencies.py
│   │   └── routers/
│   │       ├── chat_api.py
│   │       ├── agent_api.py
│   │       ├── kb_api.py
│   │       └── runtime_api.py
│   │
│   ├── application/
│   │   ├── chat_app.py
│   │   ├── agent_app.py
│   │   ├── kb_app.py
│   │   └── runtime_app.py
│   │
│   ├── domain/
│   │   ├── entities/
│   │   └── interfaces/
│   │
│   ├── infrastructure/
│   │   ├── databases/
│   │   ├── embeddings/
│   │   ├── llms/
│   │   └── tools/
│   │
│   ├── services/
│   │   ├── agent/
│   │   ├── rag/
│   │   ├── session/
│   │   └── streaming/
│   │
│   └── common/
│       ├── config.py
│       ├── logger.py
│       └── event_bus.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 五、分层说明

### 1. API 层

API 层负责处理 HTTP 请求、参数校验、依赖注入和响应格式封装。

它不直接写业务逻辑，而是调用 Application 层完成实际任务。

典型文件：

```text
backend/api/main.py
backend/api/dependencies.py
backend/api/routers/chat_api.py
backend/api/routers/agent_api.py
backend/api/routers/kb_api.py
backend/api/routers/runtime_api.py
```

---

### 2. Application 层

Application 层负责任务级业务流程编排。

例如：

- ChatApplication 负责编排普通对话、RAG 问答、Auto 路由、上下文构建和流式输出。
- KnowledgeBaseApp 负责编排文档切块、Embedding、向量库写入、关键词索引写入和失败回滚。
- AgentApplication 负责编排 Agent 任务流、任务恢复、事件转换和结果持久化。
- RuntimeApplication 负责向前端暴露当前系统能力。

Application 层是系统的用例入口，不应该直接依赖具体数据库或模型 SDK，而应该通过接口和依赖注入使用底层能力。

---

### 3. Domain 层

Domain 层定义系统内部最稳定的实体和接口契约。

典型实体包括：

```text
DocumentChunk
Message
ToolCall
RetrievalResult
StreamEvent
TaskSnapshot
```

其中 `DocumentChunk` 是系统内部统一的文档块结构，用于屏蔽 ChromaDB、Whoosh 等底层存储差异。

典型结构：

```python
class DocumentChunk(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any]
    score: Optional[float] = None
```

Domain 层不应该依赖 FastAPI、ChromaDB、Whoosh、DeepSeek SDK 等外部技术实现。

---

### 4. Infrastructure 层

Infrastructure 层负责适配外部系统和第三方组件。

当前包括：

- DeepSeekClient：大模型客户端适配器
- BGEM3Local：本地 Embedding 模型适配器
- ChromaStore：ChromaDB 向量库适配器
- WhooshStore：Whoosh 中文关键词索引适配器
- SQLiteMemoryStore：本地会话与记忆存储
- file_ops / python_ops / weather_ops / sql_ops：Agent 工具实现

Infrastructure 层负责“怎么做”，Domain 和 Application 层负责“做什么”。

---

### 5. Services 层

Services 层包含可复用的业务能力组件。

例如：

- RAG 检索融合
- 文档切块
- 查询改写
- 查询规划
- 重排序
- 上下文打包
- 防幻觉检查
- 会话管理
- 长期记忆召回
- Agent 工具注册
- Agent 任务分解
- Agent Supervisor 路由

这些服务可以被 Application 层组合使用。

---

## 六、RAG 检索流程

DeepAssist 的 RAG 流程大致如下：

```text
用户问题
  ↓
QueryPlanner 查询规划
  ↓
向量检索 ChromaDB
  ↓
关键词检索 Whoosh
  ↓
Weighted RRF 融合
  ↓
Lexical Reranker 重排序
  ↓
ContextPacker 构造上下文
  ↓
LLM 生成答案
  ↓
AnswerGuard 防幻觉检查
  ↓
SSE 流式返回
```

当前检索系统不是单一路径，而是混合检索：

```text
Vector Search + Keyword Search + RRF Fusion + Rerank
```

这样做的目的是同时利用语义召回和关键词精确匹配，提升知识库问答的稳定性。

---

## 七、环境准备

### 1. 创建 Python 环境

推荐使用 Conda：

```bash
conda create -n llm-dev python=3.11
conda activate llm-dev
```

或者使用 venv：

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux / macOS：

```bash
python -m venv .venv
source .venv/bin/activate
```

---

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

---

### 3. 配置环境变量

在项目根目录创建 `.env` 文件：

```env
DEEPSEEK_API_KEY=你的 DeepSeek API Key
DEEPSEEK_BASE_URL=https://api.deepseek.com

LLM_CHAT_MODEL=deepseek-chat
LLM_REASONER_MODEL=deepseek-reasoner

EMBEDDING_MODEL_PATH=./data/models/bge-m3
VECTOR_DB_PATH=./data/indexes/vector_store
KEYWORD_DB_PATH=./data/indexes/keyword_store
CONVERSATION_DB_PATH=./data/application_db/conversations.db
USER_PROFILE_DB_PATH=./data/application_db/user_profiles.db
INGEST_PROGRESS_PATH=./data/application_db/ingest_progress.json
KB_MANIFEST_PATH=./data/application_db/knowledge_base_manifest.json
VECTOR_HEALTH_REPORT_PATH=./data/application_db/vector_index_doctor_report.json

RRF_K=60
RETRIEVAL_TOP_K=5
MAX_AGENT_STEPS=10
```

注意：

- `.env` 不应该提交到 GitHub。
- `data/` 目录通常包含模型、数据库和索引文件，也不应该提交到 GitHub。
- 本项目通过 `backend/common/config.py` 自动将 `./data/xxx` 解析为项目根目录下的绝对路径，避免不同启动目录导致路径错乱。

---

## 八、启动后端服务

在项目根目录执行：

```bash
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
```

启动成功后，可以访问：

```text
http://localhost:8000/health
```

如果返回类似下面内容，说明后端启动成功：

```json
{
  "status": "ok",
  "service": "deepassist-api",
  "message": "DeepAssist backend is running."
}
```

FastAPI 自动文档地址：

```text
http://localhost:8000/docs
```

---

## 九、接口示例

### 1. 普通对话流式接口

```bash
curl -X POST "http://localhost:8000/api/chat/stream" ^
  -H "Content-Type: application/json" ^
  -d "{\"session_id\":\"demo-session\",\"query\":\"你好，请介绍一下你自己\",\"mode\":\"quick\"}"
```

Linux / macOS：

```bash
curl -X POST "http://localhost:8000/api/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo-session","query":"你好，请介绍一下你自己","mode":"quick"}'
```

---

### 2. RAG 知识库问答

```bash
curl -X POST "http://localhost:8000/api/chat/stream" ^
  -H "Content-Type: application/json" ^
  -d "{\"session_id\":\"demo-session\",\"query\":\"根据知识库解释这个项目的架构\",\"mode\":\"rag\",\"collection_name\":\"tech_docs_kb\"}"
```

---

### 3. 上传知识库文档

```bash
curl -X POST "http://localhost:8000/api/kb/upload?collection_name=tech_docs_kb" ^
  -F "file=@docs/example.md"
```

---

### 4. 查看知识库集合

```bash
curl "http://localhost:8000/api/kb/collections"
```

---

### 5. 查看运行时能力

```bash
curl "http://localhost:8000/api/runtime/capabilities"
```

---

## 十、项目亮点

### 1. 分层架构清晰

项目不是把所有逻辑堆在一个脚本中，而是按照 API、Application、Domain、Infrastructure、Services 分层组织。

这种结构更接近真实后端工程，便于后续扩展、测试和维护。

---

### 2. 支持混合检索

系统同时使用：

```text
ChromaDB 向量检索
Whoosh 关键词检索
RRF 融合排序
Reranker 重排序
```

相比只使用向量数据库，混合检索对于中文知识库、专业术语、代码文档和精确关键词问题更加稳定。

---

### 3. 支持知识库管理闭环

系统不仅能检索，还支持：

- 上传文档
- 切块
- 写入向量库
- 写入关键词索引
- 文件级删除
- 集合列表
- 文件列表
- 索引健康检查
- Manifest 管理

这比单纯的 RAG Demo 更接近真实业务系统。

---

### 4. 支持 SSE 流式输出

普通对话、RAG 问答和 Agent 模式都支持流式输出，适合对接前端聊天界面。

---

### 5. 支持 Agent 工具调用

Agent 系统中已经接入工具注册机制，可以扩展本地文件操作、知识库检索、Python 执行、SQL 查询、天气查询等能力。

---

### 6. 支持上下文管理与长期记忆

系统支持：

- 会话历史保存
- 历史轮次控制
- 上下文窗口管理
- 长对话摘要压缩
- 长期记忆召回
- 用户画像提取

这为后续构建更个性化的 AI 助手打下基础。

---

## 十一、当前限制与后续优化方向

当前项目仍处于工程 Demo 和能力验证阶段，后续可以继续优化：

### 1. Python 工具沙箱安全

当前 Agent 中的 Python 执行工具需要进一步加强安全隔离。

后续可以考虑：

- RestrictedPython
- AST 白名单过滤
- Docker 沙箱
- Firecracker / gVisor 等更强隔离方案

---

### 2. 前端工程化

当前前端更适合使用 Streamlit 作为 Demo 展示。

如果后续要做正式产品，可以替换为：

- React
- Vue
- Next.js
- 独立前后端分离部署

---

### 3. RAG 质量评估

后续可以加入：

- 检索召回率评估
- MRR / Recall@K
- 答案 groundedness 评估
- 引用准确性评估
- 多知识库对比测试

---

### 4. 权限与多用户系统

当前项目更偏单机 Demo。

后续如果做成产品，需要补充：

- 用户登录
- Token 鉴权
- 多用户知识库隔离
- 权限控制
- 操作审计

---

### 5. 部署与运维

后续可以加入：

- Dockerfile
- docker-compose
- Nginx 反向代理
- 日志轮转
- 健康检查
- CI/CD
- 云服务器部署脚本

---

## 十二、适合展示的能力点

本项目适合用于展示以下 LLM 应用开发能力：

- 大模型 API 接入
- FastAPI 后端开发
- SSE 流式响应
- RAG 系统设计
- 向量库与关键词索引结合
- 本地知识库构建
- Agent 工具调用
- 多轮对话记忆
- Clean Architecture / DDD 分层意识
- 工程化配置管理
- 后端系统模块拆分
- 可扩展 AI 应用架构设计

---

## 十三、运行状态检查

后端健康检查：

```text
GET /health
```

知识库健康检查：

```text
GET /api/kb/health
```

运行时能力检查：

```text
GET /api/runtime/capabilities
```

---

## 十四、仓库说明

本仓库不包含以下内容：

- `.env` 环境变量文件
- DeepSeek API Key
- 本地 ChromaDB 数据
- Whoosh 索引数据
- BGE-M3 模型权重
- SQLite 会话数据库
- 日志文件
- 临时文件

这些内容需要在本地运行时自行配置或生成。

---

## 十五、项目目标

DeepAssist 的目标不是追求复杂 UI，而是围绕 LLM 应用开发岗位所需能力，构建一个结构清晰、可解释、可扩展 AI 应用系统。

项目重点在于：

```text
LLM 接入能力
RAG 工程能力
Agent 编排能力
后端接口能力
分层架构能力
工程化落地能力
```
