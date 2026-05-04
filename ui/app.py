import json
import os
import re
import uuid
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import quote

import requests
import streamlit as st


API_BASE_URL = os.getenv("DEEPASSIST_API_BASE", "http://localhost:8000/api").rstrip("/")
DEFAULT_COLLECTION = "tech_docs_kb"
RAG_COLLECTION = "__all__"
REQUEST_TIMEOUT = int(os.getenv("DEEPASSIST_UI_TIMEOUT", "180"))


st.set_page_config(page_title="DeepAssist", page_icon="D", layout="wide")

st.markdown(
    """
<style>
    #MainMenu, footer, .stAppDeployButton {visibility: hidden;}
    .block-container {padding: 1.4rem 2rem 1rem 2rem; max-width: 100%;}
    section[data-testid="stSidebar"] {background: #f7f8fb; border-right: 1px solid #e9edf3;}
    section[data-testid="stSidebar"] .block-container {padding: 1.2rem 1.1rem;}
    div[data-testid="stVerticalBlock"] {gap: 0.55rem;}
    .brand {display: flex; align-items: center; gap: .6rem; font-weight: 800; font-size: 1.25rem; color: #111827;}
    .brand-mark {width: 30px; height: 30px; border: 1px solid #111827; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-size: .78rem; font-weight: 800;}
    .muted {color: #6b7280; font-size: .9rem;}
    .tiny {color: #8b95a5; font-size: .78rem;}
    .page-title {font-size: 2rem; line-height: 1.15; font-weight: 850; color: #111827; margin: 0;}
    .page-subtitle {color: #6b7280; margin-top: .4rem;}
    .section-title {font-size: 1.18rem; font-weight: 800; color: #111827; margin: 0 0 .4rem 0;}
    .mode-card {border: 1px solid #e6eaf0; border-radius: 8px; padding: 1.35rem 1.45rem; background: #fff; transition: border-color .15s ease, box-shadow .15s ease;}
    .mode-card:hover {border-color: #9bbcff; box-shadow: 0 10px 26px rgba(36, 99, 235, .08);}
    .mode-row {display: flex; align-items: center; gap: 1rem;}
    .mode-icon {width: 52px; height: 52px; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: #0f172a; font-weight: 850; letter-spacing: 0;}
    .mode-title {font-size: 1.12rem; font-weight: 800; color: #111827; margin-bottom: .25rem;}
    .mode-desc {color: #6b7280; font-size: .95rem;}
    .panel {border: 1px solid #e6eaf0; border-radius: 8px; padding: 1.1rem 1.25rem; background: #fff;}
    .header-card {border: 1px solid #e6eaf0; border-radius: 8px; padding: 1.25rem 1.35rem; background: #fff;}
    .metric-pill {display:inline-flex; align-items:center; gap:.35rem; border:1px solid #e6eaf0; border-radius:999px; padding:.22rem .55rem; color:#526071; font-size:.78rem; margin-right:.35rem;}
    .trace-box {border-left: 3px solid #d6e4ff; padding-left: .75rem; color: #526071; font-size: .88rem;}
    .stButton button {border-radius: 8px; min-height: 2.35rem;}
    .stTextInput input, .stSelectbox div[data-baseweb="select"], .stNumberInput input {border-radius: 8px;}
    div[data-testid="stChatMessage"] {border-radius: 8px;}
</style>
""",
    unsafe_allow_html=True,
)


MODES = {
    "agent": {
        "title": "智能对话",
        "desc": "复杂任务智能体，支持工具调用、路由与自修正",
        "icon": "AI",
        "color": "#9b7cff",
        "welcome": "已进入智能对话模式。我会根据任务自动路由到聊天、RAG 或工具智能体。",
    },
    "rag": {
        "title": "知识问答",
        "desc": "显式调用知识库链路，输出引用与检索轨迹",
        "icon": "RAG",
        "color": "#1ed6a5",
        "welcome": "已进入知识问答模式。我会使用后端知识库检索、重排、引用打包和答案守卫。",
    },
    "quick": {
        "title": "快速开始",
        "desc": "纯 LLM 对话，适合日常问答和轻量生成",
        "icon": "LLM",
        "color": "#ff725f",
        "welcome": "已进入快速开始模式。这里不会显式调用知识库或工具链路。",
    },
}


def init_state() -> None:
    defaults = {
        "nav_tab": "assistant",
        "session_id": f"sess_{uuid.uuid4().hex[:8]}",
        "chat_mode": "quick",
        "messages": [],
        "show_home": True,
        "selected_model": "DeepSeek Chat",
        "temperature": 0.7,
        "top_p": 1.0,
        "history_rounds": 10,
        "use_user_memory": False,
        "rrf_weight": 1.0,
        "show_settings": False,
        "kb_collection": DEFAULT_COLLECTION,
        "kb_confirm_delete_all": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def api_url(path: str) -> str:
    return f"{API_BASE_URL}{path}"


def request_json(method: str, path: str, **kwargs) -> Dict[str, Any]:
    try:
        response = requests.request(method, api_url(path), timeout=REQUEST_TIMEOUT, **kwargs)
    except requests.RequestException as exc:
        return {"status": "error", "message": str(exc), "data": []}

    try:
        payload = response.json()
    except ValueError:
        payload = {"status": "error", "message": response.text, "data": []}
    if response.status_code >= 400:
        payload.setdefault("status", "error")
        payload.setdefault("message", f"HTTP {response.status_code}")
    return payload


def fetch_sessions() -> List[Dict[str, Any]]:
    payload = request_json("GET", "/chat/sessions")
    data = payload.get("data", [])
    return data if isinstance(data, list) else []


def fetch_history(session_id: str) -> List[Dict[str, Any]]:
    payload = request_json("GET", f"/chat/history/{quote(session_id, safe='')}")
    data = payload.get("data", [])
    return data if isinstance(data, list) else []


def delete_session(session_id: str) -> Dict[str, Any]:
    return request_json("DELETE", f"/chat/sessions/{quote(session_id, safe='')}")


def fetch_kb_collections() -> Dict[str, Any]:
    return request_json("GET", "/kb/collections")


def fetch_kb_files(collection_name: Optional[str] = None) -> Dict[str, Any]:
    return request_json(
        "GET",
        "/kb/files",
        params={"collection_name": collection_name or st.session_state.kb_collection},
    )


def fetch_kb_health(refresh: bool = False) -> Dict[str, Any]:
    return request_json(
        "GET",
        "/kb/health",
        params={"refresh": refresh},
    )


def delete_kb_file(source_file: str, collection_name: Optional[str] = None) -> Dict[str, Any]:
    encoded = quote(source_file, safe="")
    return request_json(
        "DELETE",
        f"/kb/files/{encoded}",
        params={"collection_name": collection_name or st.session_state.kb_collection},
    )


def upload_kb_file(uploaded_file, collection_name: Optional[str] = None) -> Dict[str, Any]:
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "text/markdown",
        )
    }
    return request_json(
        "POST",
        "/kb/upload",
        params={"collection_name": collection_name or st.session_state.kb_collection},
        files=files,
    )


def create_new_chat(mode: Optional[str] = None) -> None:
    st.session_state.session_id = f"sess_{uuid.uuid4().hex[:8]}"
    st.session_state.messages = []
    st.session_state.show_home = True
    if mode:
        st.session_state.chat_mode = mode


def enter_mode(mode: str) -> None:
    st.session_state.chat_mode = mode
    st.session_state.nav_tab = "assistant"
    st.session_state.show_home = False
    if not st.session_state.messages:
        st.session_state.messages = [
            {"role": "assistant", "content": MODES[mode]["welcome"], "events": []}
        ]


def load_session(session_id: str) -> None:
    history = fetch_history(session_id)
    st.session_state.session_id = session_id
    st.session_state.messages = history
    st.session_state.chat_mode = infer_mode_from_history(history)
    st.session_state.show_home = False
    st.session_state.nav_tab = "history"


def infer_mode_from_history(messages: Iterable[Dict[str, Any]]) -> str:
    text_parts: List[str] = []
    for message in messages:
        if message.get("role") == "tool" or message.get("tool_calls"):
            return "agent"
        if message.get("role") == "assistant":
            text_parts.append(str(message.get("content", "")))
    text = "\n".join(text_parts)
    if re.search(r"\[C\d+\]", text):
        return "rag"
    return st.session_state.get("chat_mode", "quick")


def summarize(text: str, length: int = 34) -> str:
    text = " ".join((text or "").split())
    if not text:
        return "新会话"
    return text if len(text) <= length else f"{text[:length]}..."


def mode_label(mode: str) -> str:
    return MODES.get(mode, MODES["quick"])["title"]


def backend_model_name() -> str:
    selected = st.session_state.get("selected_model", "DeepSeek Chat")
    if selected == "DeepSeek Reasoner":
        return "deepseek-reasoner"
    if selected == "DeepSeek Chat":
        return "deepseek-chat"
    return "deepseek-chat"


def runtime_payload() -> Dict[str, Any]:
    return {
        "model_name": backend_model_name(),
        "temperature": float(st.session_state.temperature),
        "top_p": float(st.session_state.top_p),
        "history_rounds": int(st.session_state.history_rounds),
        "use_user_memory": bool(st.session_state.use_user_memory),
    }


def render_brand() -> None:
    st.markdown(
        """
<div class="brand">
  <div class="brand-mark">DA</div>
  <div>DeepAssist</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    with st.sidebar:
        render_brand()
        st.divider()

        tab_cols = st.columns(3)
        tabs = [("assistant", "智能体助手"), ("history", "聊天记录"), ("kb", "知识库")]
        for col, (key, label) in zip(tab_cols, tabs):
            if col.button(
                label,
                use_container_width=True,
                type="primary" if st.session_state.nav_tab == key else "secondary",
            ):
                st.session_state.nav_tab = key
                st.rerun()

        st.divider()

        if st.session_state.nav_tab in {"assistant", "history"}:
            if st.button("新聊天", use_container_width=True, type="primary"):
                create_new_chat()
                st.rerun()

        if st.session_state.nav_tab == "assistant":
            render_sidebar_agent_controls()
        elif st.session_state.nav_tab == "history":
            render_sidebar_history()
        else:
            render_sidebar_kb_v2()


def render_sidebar_agent_controls() -> None:
    st.caption("当前驱动模型")
    st.selectbox(
        "选择模型",
        ["DeepSeek Chat", "DeepSeek Reasoner", "智谱 GLM-4.6"],
        key="selected_model",
        label_visibility="collapsed",
    )
    if st.session_state.selected_model.startswith("智谱"):
        st.caption("智谱模型入口已预留，当前后端仍使用 DeepSeek 适配器。")

    with st.expander("智能体参数", expanded=True):
        st.slider("Temperature", 0.0, 2.0, key="temperature", step=0.1)
        st.slider("Top P", 0.0, 1.0, key="top_p", step=0.05)
        st.slider("历史窗口", 1, 30, key="history_rounds")
        st.checkbox("启用长期记忆画像", key="use_user_memory")

    with st.expander("可用工具", expanded=False):
        st.markdown("- Python 沙箱\n- 只读 SQL 探查\n- 本地文件读写\n- 天气查询\n- 知识库检索")


def render_sidebar_history() -> None:
    sessions = fetch_sessions()
    st.caption("历史会话")
    if not sessions:
        st.markdown('<div class="muted">暂无聊天记录</div>', unsafe_allow_html=True)
        return

    for session in sessions:
        session_id = session.get("session_id", "")
        title = summarize(session.get("title", ""), 18)
        cols = st.columns([0.78, 0.22])
        if cols[0].button(title, key=f"open_{session_id}", use_container_width=True):
            load_session(session_id)
            st.rerun()
        if cols[1].button("删", key=f"delete_{session_id}", use_container_width=True):
            result = delete_session(session_id)
            if result.get("status") == "success":
                if st.session_state.session_id == session_id:
                    create_new_chat()
                st.rerun()
            else:
                st.error(result.get("message", "删除失败"))


def render_sidebar_kb() -> None:
    collections_payload = fetch_kb_collections()
    collections = collections_payload.get("data", [])
    collections = collections if isinstance(collections, list) else []
    collection_names = [item.get("collection_name") for item in collections if isinstance(item, dict)]
    collection_names = [name for name in collection_names if name]
    if st.session_state.kb_collection not in collection_names and collection_names:
        st.session_state.kb_collection = collection_names[0]

    selected = st.session_state.kb_collection
    selected_meta = next(
        (item for item in collections if item.get("collection_name") == selected),
        {},
    )

    st.caption("知识库索引")
    if collection_names:
        st.selectbox(
            "当前管理集合",
            collection_names,
            key="kb_collection",
            label_visibility="collapsed",
        )
        selected = st.session_state.kb_collection
        selected_meta = next(
            (item for item in collections if item.get("collection_name") == selected),
            {},
        )
    st.markdown(
        f"""
<div class="panel">
  <div style="font-weight:800;">{selected}</div>
  <div class="tiny">{selected_meta.get('file_count', 0)} 个文件，{selected_meta.get('chunk_count', 0)} 个文本块</div>
  <div style="margin-top:.45rem;">
    <span class="metric-pill">{" + ".join(selected_meta.get('stores', [])) or "未连接"}</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.caption("问答默认使用全部知识库检索；这里仅用于维护单个集合。")
    if st.button("新建知识库", use_container_width=True):
        st.info("多知识库命名空间入口已预留；当前可管理已有集合和默认 tech_docs_kb。")


def render_sidebar_kb_v2() -> None:
    collections_payload = fetch_kb_collections()
    health_payload = fetch_kb_health(refresh=False)

    collections = collections_payload.get("data", [])
    collections = collections if isinstance(collections, list) else []
    health_report = health_payload.get("data", {})
    health_report = health_report if isinstance(health_report, dict) else {}
    health_collections = health_report.get("collections", [])
    health_collections = health_collections if isinstance(health_collections, list) else []

    collection_names = [
        item.get("collection_name")
        for item in collections
        if isinstance(item, dict) and item.get("collection_name")
    ]
    if st.session_state.kb_collection not in collection_names and collection_names:
        st.session_state.kb_collection = collection_names[0]

    st.caption("Knowledge Base")
    if collection_names:
        st.selectbox(
            "Current collection",
            collection_names,
            key="kb_collection",
            label_visibility="collapsed",
        )

    selected = st.session_state.kb_collection
    selected_meta = next(
        (item for item in collections if item.get("collection_name") == selected),
        {},
    )
    selected_health = next(
        (item for item in health_collections if item.get("collection_name") == selected),
        {},
    )

    stores = " + ".join(selected_meta.get("stores", [])) or "Unavailable"
    file_count = int(selected_meta.get("file_count", 0))
    chunk_count = int(selected_meta.get("chunk_count", 0))
    health_status = "Healthy" if selected_health.get("healthy") else "Needs attention"
    checked_at = health_report.get("checked_at", "--")
    errors = selected_health.get("errors", []) or []

    st.markdown(
        f"""
<div class="panel">
  <div style="font-weight:800;">{selected or "No collection selected"}</div>
  <div class="tiny">{file_count} files, {chunk_count} chunks</div>
  <div style="margin-top:.45rem;">
    <span class="metric-pill">{stores}</span>
    <span class="metric-pill">Health: {health_status}</span>
  </div>
  <div class="tiny" style="margin-top:.45rem;">Checked: {checked_at}</div>
  <div class="tiny">Health issues: {len(errors)}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    refresh_disabled = not bool(collection_names)
    if st.button("Refresh KB health", use_container_width=True, disabled=refresh_disabled):
        refreshed = fetch_kb_health(refresh=True)
        if refreshed.get("status") in {"success", "partial_success"}:
            st.success("Knowledge-base health report updated.")
        else:
            st.error(refreshed.get("message", "Knowledge-base health refresh failed."))
        st.rerun()

    with st.expander("Health details", expanded=False):
        st.caption("RAG and Agent retrieval search across all collections by default.")
        if selected_health:
            st.json(selected_health)
        elif collection_names:
            st.info("No cached health data for the selected collection yet.")
        else:
            st.info("No knowledge-base collections found.")


def render_home() -> None:
    st.markdown('<div style="height: 10vh;"></div>', unsafe_allow_html=True)
    st.markdown('<p class="page-title" style="text-align:center;">开始新的对话</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle" style="text-align:center;">选择一个工作模式，或直接在底部输入消息创建新会话</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<div style="height: 1.6rem;"></div>', unsafe_allow_html=True)

    _, center, _ = st.columns([1, 1.35, 1])
    with center:
        for mode in ("agent", "rag", "quick"):
            spec = MODES[mode]
            st.markdown(
                f"""
<div class="mode-card">
  <div class="mode-row">
    <div class="mode-icon" style="background:{spec['color']};">{spec['icon']}</div>
    <div>
      <div class="mode-title">{spec['title']}</div>
      <div class="mode-desc">{spec['desc']}</div>
    </div>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
            cols = st.columns([0.7, 0.3])
            if cols[1].button(f"进入{spec['title']}", key=f"enter_{mode}", use_container_width=True):
                enter_mode(mode)
                st.rerun()
            st.write("")


def render_chat_header() -> None:
    spec = MODES.get(st.session_state.chat_mode, MODES["quick"])
    cols = st.columns([0.62, 0.38])
    with cols[0]:
        st.markdown(
            f"""
<div class="header-card">
  <div class="section-title">{spec['title']}</div>
  <div class="muted">{spec['desc']}</div>
  <div style="margin-top:.55rem;">
    <span class="metric-pill">Session: {st.session_state.session_id}</span>
    <span class="metric-pill">Model: {st.session_state.selected_model}</span>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
    with cols[1]:
        button_cols = st.columns(3)
        if button_cols[0].button("参数", use_container_width=True):
            st.session_state.show_settings = not st.session_state.show_settings
        if button_cols[1].button("切换模式", use_container_width=True):
            st.session_state.show_home = True
            st.rerun()
        if button_cols[2].button("新聊天", use_container_width=True, type="primary"):
            create_new_chat()
            st.rerun()

    if st.session_state.show_settings:
        render_settings_panel()


def render_settings_panel() -> None:
    with st.container(border=True):
        st.markdown("#### 智能体助手设置")
        left, right = st.columns([0.35, 0.65])
        with left:
            st.radio(
                "设置分组",
                ["基础设置", "模型设置", "知识库设置", "工具调用"],
                label_visibility="collapsed",
            )
        with right:
            st.selectbox(
                "选择模型",
                ["DeepSeek Chat", "DeepSeek Reasoner", "智谱 GLM-4.6"],
                key="selected_model",
            )
            st.slider("Temperature", 0.0, 2.0, key="temperature", step=0.1)
            st.slider("Top P", 0.0, 1.0, key="top_p", step=0.05)
            st.slider("消息窗口长度", 1, 30, key="history_rounds")
            st.checkbox("启用长期记忆画像", key="use_user_memory")
            st.caption("这些参数会随每次请求透传给后端；智谱模型入口为预留位，当前会回退到 DeepSeek 适配器。")


def render_message(message: Dict[str, Any]) -> None:
    role = message.get("role")
    content = message.get("content", "")

    if role in {"user", "assistant"}:
        with st.chat_message(role):
            st.markdown(content or "")
            events = message.get("events") or []
            if events:
                render_event_trace_v2(events)
            tool_calls = message.get("tool_calls")
            if tool_calls:
                with st.expander("工具调用请求", expanded=False):
                    st.json(tool_calls)
    elif role == "tool":
        with st.chat_message("assistant"):
            with st.expander(f"工具结果：{message.get('name') or 'tool'}", expanded=False):
                st.markdown(content or "")


def render_event_trace(events: List[Dict[str, Any]]) -> None:
    with st.expander("运行轨迹", expanded=False):
        for event in events:
            event_name = event.get("event")
            data = event.get("data", {}) or {}
            if event_name == "supervisor_route":
                st.markdown(
                    f"- 路由：`{data.get('worker_kind') or data.get('worker')}`，"
                    f"意图 `{data.get('intent')}`，置信度 `{data.get('confidence')}`"
                )
            elif event_name == "retrieval_trace":
                st.markdown(
                    f"- 检索：命中 `{data.get('hit_count', 0)}`，"
                    f"候选 `{data.get('candidate_k', 0)}`，融合 `{data.get('fusion', 'n/a')}`"
                )
            elif event_name == "citation_trace":
                st.markdown(f"- 引用打包：`{len(data.get('citations', []))}` 条引用")
            elif event_name == "tool_call":
                st.markdown(f"- 工具调用：`{event.get('name')}`")
                st.json(event.get("args", {}))
            elif event_name == "tool_result":
                ok = data.get("success")
                st.markdown(f"- 工具结果：`{event.get('name')}`，success=`{ok}`")
                if data.get("error"):
                    st.caption(data.get("error"))
            elif event_name == "answer_guard":
                st.markdown(
                    f"- 答案守卫：grounded=`{data.get('grounded')}`，"
                    f"warnings=`{data.get('warnings', [])}`"
                )
            elif event_name == "self_correction":
                st.markdown(f"- 自修正：{event.get('message') or data.get('error')}")
            elif event_name == "reasoning":
                st.markdown("- 模型推理片段")
                st.code(event.get("content", ""), language="text")


def compact_event_summary(events: List[Dict[str, Any]]) -> str:
    if not events:
        return ""
    parts: List[str] = []
    for event in events[-8:]:
        name = event.get("event")
        data = event.get("data", {}) or {}
        if name == "supervisor_route":
            parts.append(f"route:{data.get('worker_kind') or data.get('worker')}")
        elif name == "retrieval_trace":
            parts.append(f"retrieval:{data.get('hit_count', 0)}")
        elif name == "citation_trace":
            parts.append(f"citations:{len(data.get('citations', []))}")
        elif name == "tool_call":
            parts.append(f"tool:{event.get('name')}")
        elif name == "tool_result":
            parts.append(f"result:{event.get('name')}")
        elif name == "answer_guard":
            parts.append("guard")
        elif name == "error":
            parts.append("error")
    return " -> ".join(parts)


def render_event_trace_v2(events: List[Dict[str, Any]]) -> None:
    with st.expander("Runtime Trace", expanded=False):
        for event in events:
            event_name = event.get("event")
            data = event.get("data", {}) or {}
            if event_name == "supervisor_route":
                st.markdown(
                    f"- Route: `{data.get('worker_kind') or data.get('worker')}` "
                    f"intent=`{data.get('intent')}` confidence=`{data.get('confidence')}`"
                )
            elif event_name == "context_window_trace":
                st.markdown(
                    f"- Context window: budget=`{data.get('budget', 0)}` "
                    f"selected_turns=`{data.get('selected_turn_count', 0)}` "
                    f"dropped_turns=`{data.get('dropped_turn_count', 0)}` "
                    f"memories=`{data.get('recalled_memory_count', 0)}`"
                )
                recalled_memories = data.get("recalled_memories", []) or []
                if recalled_memories:
                    st.markdown("Memory recall")
                    for memory in recalled_memories:
                        st.markdown(
                            f"- [{memory.get('category', 'memory')}] {memory.get('content', '')}"
                        )
                selected_turns = data.get("selected_turns", []) or []
                if selected_turns:
                    st.markdown("Selected turns")
                    for turn in selected_turns[:4]:
                        st.markdown(
                            f"- `{turn.get('turn_id')}` `{turn.get('priority_band')}` "
                            f"score=`{turn.get('priority_score')}` {turn.get('preview', '')}"
                        )
                dropped_turns = data.get("dropped_turns", []) or []
                if dropped_turns:
                    st.markdown("Dropped turns")
                    for turn in dropped_turns[:3]:
                        st.markdown(
                            f"- `{turn.get('turn_id')}` `{turn.get('priority_band')}` "
                            f"{turn.get('preview', '')}"
                        )
                summary = data.get("summary")
                if isinstance(summary, dict):
                    st.markdown(
                        f"- Summary injected: dropped_messages=`{summary.get('dropped_message_count', 0)}` "
                        f"dropped_turns=`{summary.get('dropped_turn_count', 0)}`"
                    )
            elif event_name == "retrieval_trace":
                st.markdown(
                    f"- Retrieval: hits=`{data.get('hit_count', 0)}` "
                    f"candidate_k=`{data.get('candidate_k', 0)}` fusion=`{data.get('fusion', 'n/a')}`"
                )
            elif event_name == "citation_trace":
                st.markdown(f"- Citations packed: `{len(data.get('citations', []))}`")
            elif event_name == "tool_call":
                st.markdown(f"- Tool call: `{event.get('name')}`")
                st.json(event.get("args", {}))
            elif event_name == "tool_result":
                ok = data.get("success")
                st.markdown(f"- Tool result: `{event.get('name')}` success=`{ok}`")
                if data.get("error"):
                    st.caption(data.get("error"))
            elif event_name == "answer_guard":
                st.markdown(
                    f"- Answer guard: grounded=`{data.get('grounded')}` "
                    f"warnings=`{data.get('warnings', [])}`"
                )
            elif event_name == "self_correction":
                st.markdown(f"- Self correction: {event.get('message') or data.get('error')}")
            elif event_name == "reasoning":
                st.markdown("- Reasoning snippet")
                st.code(event.get("content", ""), language="text")


def compact_event_summary_v2(events: List[Dict[str, Any]]) -> str:
    if not events:
        return ""
    parts: List[str] = []
    for event in events[-8:]:
        name = event.get("event")
        data = event.get("data", {}) or {}
        if name == "supervisor_route":
            parts.append(f"route:{data.get('worker_kind') or data.get('worker')}")
        elif name == "context_window_trace":
            parts.append(
                f"context:{data.get('selected_turn_count', 0)}/{data.get('budget', 0)}"
            )
        elif name == "retrieval_trace":
            parts.append(f"retrieval:{data.get('hit_count', 0)}")
        elif name == "citation_trace":
            parts.append(f"citations:{len(data.get('citations', []))}")
        elif name == "tool_call":
            parts.append(f"tool:{event.get('name')}")
        elif name == "tool_result":
            parts.append(f"result:{event.get('name')}")
        elif name == "answer_guard":
            parts.append("guard")
        elif name == "error":
            parts.append("error")
    return " -> ".join(parts)


def stream_backend_answer(prompt: str) -> Dict[str, Any]:
    if st.session_state.chat_mode == "agent":
        path = "/agent/stream"
        payload = {
            "session_id": st.session_state.session_id,
            "query": prompt,
            **runtime_payload(),
        }
    else:
        path = "/chat/stream"
        payload = {
            "session_id": st.session_state.session_id,
            "query": prompt,
            "mode": st.session_state.chat_mode,
            "collection_name": RAG_COLLECTION if st.session_state.chat_mode == "rag" else DEFAULT_COLLECTION,
            **runtime_payload(),
        }

    answer = ""
    events: List[Dict[str, Any]] = []
    text_box = st.empty()
    trace_box = st.empty()

    try:
        with requests.post(
            api_url(path),
            json=payload,
            stream=True,
            timeout=REQUEST_TIMEOUT,
        ) as response:
            if response.status_code != 200:
                text = f"后端请求失败：HTTP {response.status_code}\n\n{response.text[:800]}"
                text_box.error(text)
                return {"role": "assistant", "content": text, "events": events}

            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if not raw or raw == "[DONE]":
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                event_name = event.get("event")
                if event_name == "done":
                    break

                if event_name in {
                    "supervisor_route",
                    "context_window_trace",
                    "retrieval_trace",
                    "citation_trace",
                    "tool_call",
                    "tool_result",
                    "answer_guard",
                    "self_correction",
                    "reasoning",
                    "error",
                }:
                    events.append(event)

                if event_name == "message_delta":
                    answer += event.get("content", "")
                elif event_name == "final_answer":
                    answer = event.get("content", "") or answer
                elif event_name == "status" and not answer:
                    text_box.markdown(f"<span class='muted'>{event.get('message', '')}</span>", unsafe_allow_html=True)
                elif event_name == "error":
                    answer += f"\n\n**错误**：{event.get('message', '')}"

                if answer:
                    text_box.markdown(answer + "▌")
                summary = compact_event_summary_v2(events)
                if summary:
                    trace_box.markdown(f"<div class='trace-box'>{summary}</div>", unsafe_allow_html=True)

    except requests.RequestException as exc:
        answer = f"无法连接后端服务：{exc}"
        text_box.error(answer)

    text_box.markdown(answer or "没有收到模型输出。")
    trace_box.empty()
    return {"role": "assistant", "content": answer or "没有收到模型输出。", "events": events}


def render_chat_workspace() -> None:
    if st.session_state.show_home:
        render_home()
        return

    render_chat_header()
    st.divider()

    for message in st.session_state.messages:
        render_message(message)


def render_kb_workspace() -> None:
    collection_name = st.session_state.kb_collection
    payload = fetch_kb_files(collection_name)
    files = payload.get("data", []) if isinstance(payload.get("data", []), list) else []
    total_chunks = sum(int(item.get("chunk_count", 0)) for item in files if isinstance(item, dict))

    st.markdown('<p class="page-title">知识库管理</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">用于维护后端统一知识库矩阵。问答入口会由 RAG/Agent 自动调用这些知识。</p>',
        unsafe_allow_html=True,
    )

    top_left, top_right = st.columns([0.42, 0.58])
    with top_left:
        st.markdown(
            f"""
<div class="header-card">
  <div class="section-title">默认知识库矩阵</div>
  <div class="muted">Collection: {collection_name}</div>
  <div style="margin-top:.65rem;">
    <span class="metric-pill">{len(files)} 个文件</span>
    <span class="metric-pill">{total_chunks} 个文本块</span>
    <span class="metric-pill">Chroma + Whoosh</span>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
    with top_right:
        with st.container(border=True):
            st.markdown("#### 上传结构化 Markdown")
            uploaded_file = st.file_uploader(
                "选择 Markdown 文件",
                type=["md"],
                accept_multiple_files=False,
                label_visibility="collapsed",
            )
            upload_cols = st.columns([0.25, 0.75])
            if upload_cols[0].button("上传入库", type="primary", use_container_width=True, disabled=uploaded_file is None):
                result = upload_kb_file(uploaded_file, collection_name)
                if result.get("status") == "success":
                    st.success(result.get("message", "上传成功"))
                    st.rerun()
                else:
                    st.error(result.get("message", "上传失败"))
            upload_cols[1].caption("当前仅接收已清洗、已结构化的 Markdown 文档。")

    st.write("")
    with st.container(border=True):
        st.markdown("#### 文档列表")
        if not files:
            st.info("当前知识库还没有文档。")
        else:
            header = st.columns([0.45, 0.18, 0.2, 0.17])
            header[0].markdown("**文件名**")
            header[1].markdown("**文本块**")
            header[2].markdown("**双库状态**")
            header[3].markdown("**操作**")
            st.divider()
            for item in files:
                source_file = item.get("source_file", "")
                consistent = bool(item.get("consistent"))
                row = st.columns([0.45, 0.18, 0.2, 0.17])
                row[0].markdown(f"`{source_file}`")
                row[1].markdown(str(item.get("chunk_count", 0)))
                row[2].markdown("一致" if consistent else "需检查")
                if row[3].button("删除", key=f"kb_delete_{source_file}", use_container_width=True):
                    result = delete_kb_file(source_file, collection_name)
                    if result.get("status") == "success":
                        st.success(f"已删除 {source_file}")
                        st.rerun()
                    else:
                        st.error(result.get("message", "删除失败"))

        st.divider()
        danger_cols = st.columns([0.22, 0.78])
        if danger_cols[0].button("删除全部文档", use_container_width=True, disabled=not files):
            st.session_state.kb_confirm_delete_all = True
        if st.session_state.kb_confirm_delete_all and files:
            danger_cols[1].warning("再次点击确认会删除当前默认知识库内全部文档。")
            if danger_cols[1].button("确认删除全部", type="primary"):
                for item in files:
                    delete_kb_file(item.get("source_file", ""), collection_name)
                st.session_state.kb_confirm_delete_all = False
                st.rerun()

    with st.expander("新建知识库入口", expanded=False):
        st.text_input("知识库名称", value="future_collection", disabled=True)
        st.caption("多知识库命名空间入口已预留；问答请求默认使用 __all__ 让后端跨集合检索。")


def render_history_workspace() -> None:
    if not st.session_state.show_home:
        render_chat_workspace()
        return

    st.markdown('<p class="page-title">聊天记录</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">选择左侧历史会话恢复上下文，或新建一轮对话。</p>', unsafe_allow_html=True)
    sessions = fetch_sessions()
    if not sessions:
        st.info("暂无聊天记录。")
        return

    with st.container(border=True):
        for session in sessions:
            session_id = session.get("session_id", "")
            cols = st.columns([0.55, 0.25, 0.2])
            cols[0].markdown(f"**{summarize(session.get('title', ''), 56)}**")
            cols[1].caption(session.get("updated_at", ""))
            if cols[2].button("打开", key=f"history_open_{session_id}", use_container_width=True):
                load_session(session_id)
                st.rerun()


def handle_chat_input() -> None:
    if st.session_state.nav_tab == "kb":
        return

    placeholder = {
        "agent": "输入复杂任务，智能体会自动选择 RAG 或工具链路...",
        "rag": "输入需要知识库支撑的问题，回答会带引用...",
        "quick": "输入消息开始普通问答...",
    }.get(st.session_state.chat_mode, "输入消息开始对话...")

    prompt = st.chat_input(placeholder)
    if not prompt:
        return

    if st.session_state.show_home:
        st.session_state.show_home = False
        if st.session_state.nav_tab == "history":
            st.session_state.nav_tab = "assistant"
        if not st.session_state.chat_mode:
            st.session_state.chat_mode = "quick"

    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        assistant_message = stream_backend_answer(prompt)
    st.session_state.messages.append(assistant_message)


def main() -> None:
    init_state()
    render_sidebar()

    if st.session_state.nav_tab == "kb":
        render_kb_workspace()
    elif st.session_state.nav_tab == "history":
        render_history_workspace()
    else:
        render_chat_workspace()

    handle_chat_input()


if __name__ == "__main__":
    main()
