import streamlit as st
import requests
import uuid
import json

# === 1. 页面基础与全局 CSS 配置 ===
st.set_page_config(page_title="DeepAssist", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    .stAppDeployButton {display: none;}
    footer {visibility: hidden;}
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] { gap: 0.5rem; }
</style>
""", unsafe_allow_html=True)

API_BASE_URL = "http://localhost:8000/api"

# === 2. 初始化系统状态 ===
if "nav_tab" not in st.session_state:
    st.session_state.nav_tab = "agent"  
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "DeepSeek 智能体" 
if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = "quick" 
if "session_id" not in st.session_state:
    st.session_state.session_id = f"sess_{uuid.uuid4().hex[:8]}"
if "chat_display" not in st.session_state:
    st.session_state.chat_display =[]

# 🚀 核心状态重构：控制是否显示大卡片
if "show_welcome_cards" not in st.session_state:
    st.session_state.show_welcome_cards = True 

# === 3. 核心交互函数 ===
def fetch_sessions():
    try:
        res = requests.get(f"{API_BASE_URL}/chat/sessions")
        if res.status_code == 200:
            return res.json().get("data",[])
    except:
        pass
    return[]

def load_history(session_id):
    """点击历史记录：隐藏卡片，直接进入聊天室"""
    try:
        res = requests.get(f"{API_BASE_URL}/chat/history/{session_id}")
        if res.status_code == 200:
            st.session_state.chat_display = res.json().get("data",[])
            st.session_state.session_id = session_id
            st.session_state.show_welcome_cards = False
            st.session_state.nav_tab = "history"
    except Exception as e:
        st.error(f"加载历史失败: {e}")

def create_new_chat():
    """彻底重置，回到迎宾选模式状态"""
    st.session_state.session_id = f"sess_{uuid.uuid4().hex[:8]}"
    st.session_state.chat_display =[]
    st.session_state.show_welcome_cards = True
    st.session_state.nav_tab = "agent"

def enter_mode(mode: str):
    """🚀 新增：用户点击卡片后，进入专属模式房间并发送欢迎语"""
    st.session_state.chat_mode = mode
    st.session_state.show_welcome_cards = False
    
    # 根据不同模式，给一句专属欢迎语（完美复刻截图体验）
    greetings = {
        "quick": "我是 DeepSeek 最新版本模型，由深度求索公司创造的 AI 助手！😊\n\n目前处于 **[快速开始]** 模式，闪电响应，纯净无痕。有什么我可以帮你的吗？✨",
        "rag": "您已开启 **[知识问答]** 模式 📚。\n\n我将严格根据左侧【知识库】中上传的技术文档为您解答。请直接提问。",
        "agent": "您已开启 **[智能对话]** 模式 🧠。\n\n我具备完整的长记忆，并可以自主调动数据库查询、读写文件等工具帮您执行复杂任务。请下达指令。"
    }
    # 填入第一条 AI 欢迎消息
    st.session_state.chat_display =[{"role": "assistant", "content": greetings[mode]}]

def upload_to_backend(uploaded_file):
    url = f"{API_BASE_URL}/kb/upload"
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    try:
        res = requests.post(url, files=files)
        return res.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}

# === 4. 侧边栏构建 ===
with st.sidebar:
    st.markdown("### 👁️ DeepAssist")
    st.markdown("---")
    
    c1, c2, c3 = st.columns(3)
    if c1.button("🤖\n助手", use_container_width=True, type="primary" if st.session_state.nav_tab == "agent" else "secondary"):
        st.session_state.nav_tab = "agent"
        st.rerun()
    if c2.button("💬\n记录", use_container_width=True, type="primary" if st.session_state.nav_tab == "history" else "secondary"):
        st.session_state.nav_tab = "history"
        st.rerun()
    if c3.button("📚\n知识库", use_container_width=True, type="primary" if st.session_state.nav_tab == "kb" else "secondary"):
        st.session_state.nav_tab = "kb"
        st.rerun()
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.session_state.nav_tab in ["agent", "history"]:
        if st.button("➕ 新建智能体对话", type="primary", use_container_width=True):
            create_new_chat()
            st.rerun()
            
    if st.session_state.nav_tab == "agent":
        st.write(" ")
        st.caption("🎯 当前驱动模型")
        model = st.selectbox(
            "选择模型",["DeepSeek 智能体", "智谱 AI 智能体"],
            index=0 if st.session_state.selected_model == "DeepSeek 智能体" else 1,
            label_visibility="collapsed"
        )
        st.session_state.selected_model = model

    elif st.session_state.nav_tab == "history":
        st.write(" ")
        st.caption("🕒 历史会话记录")
        sessions = fetch_sessions()
        if not sessions:
            st.info("暂无聊天记录")
        else:
            for s in sessions:
                btn_label = s['title'] if len(s['title']) < 18 else s['title'][:15] + "..."
                if st.button(f"💬 {btn_label}", key=f"btn_{s['session_id']}", use_container_width=True):
                    load_history(s['session_id'])
                    st.rerun()

    elif st.session_state.nav_tab == "kb":
        st.write(" ")
        st.markdown("#### 📚 知识库管理")
        st.caption("上传文档 (仅限 Markdown / TXT)")
        uploaded_file = st.file_uploader("Drag and drop files here", type=["md", "txt"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            if st.button("开始上传并入库", type="primary", use_container_width=True):
                with st.spinner("🚀 正在解析文本、计算向量并双写入库..."):
                    res = upload_to_backend(uploaded_file)
                    if res.get("status") == "success":
                        st.success(f"✅ {uploaded_file.name} 处理成功！")
                    else:
                        st.error(f"❌ 失败: {res.get('message')}")

# === 5. 右侧主工作区 ===

# 路由 1：知识库页面
if st.session_state.nav_tab == "kb":
    st.markdown("<h1 style='margin-bottom: 20px;'>📚 知识库管理中心</h1>", unsafe_allow_html=True)
    st.info("👈 请在左侧边栏上传您的技术文档。文档将被自动切片，并同步写入 Chroma (向量) 和 Whoosh (关键词) 双库中。")
    st.markdown("---")
    st.write("### 已管理的知识资产")
    st.caption("（API 列表接口待开发，UI 占位呈现）")
    with st.container(border=True):
        col_doc1, col_doc2, col_doc3 = st.columns([3, 1, 1])
        col_doc1.markdown("**tech_architecture.md**")
        col_doc2.caption("24 个文本块")
        col_doc3.button("🗑️ 删除", key="del_mock", disabled=True)

# 路由 2：显示三大卡片的迎宾大厅
elif st.session_state.show_welcome_cards:
    st.markdown("<h1 style='text-align: center; margin-top: 5vh;'>开始新的对话</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray; margin-bottom: 40px;'>选择一个智能体助手开始聊天，或直接下方发送消息使用默认模式</p>", unsafe_allow_html=True)
    
    card_col1, card_col2, card_col3 = st.columns(3)
    with card_col1:
        with st.container(border=True):
            st.markdown("### 🤖 智能对话")
            st.caption("与 AI 进行多轮交互，自主调度工具。")
            st.write("")
            if st.button("进入智能对话", key="sel_agent", use_container_width=True):
                enter_mode("agent")
                st.rerun()
                
    with card_col2:
        with st.container(border=True):
            st.markdown("### 📚 知识问答")
            st.caption("基于左侧上传的知识库文档获取准确信息。")
            st.write("")
            if st.button("进入知识问答", key="sel_rag", use_container_width=True):
                enter_mode("rag")
                st.rerun()
                
    with card_col3:
        with st.container(border=True):
            st.markdown("### ⚡ 快速开始")
            st.caption("无记忆负担，高速纯净的直接对话。")
            st.write("")
            if st.button("进入快速开始", key="sel_quick", use_container_width=True):
                enter_mode("quick")
                st.rerun()

# 路由 3：聊天界面（专属房间）
else:
    # 🚀 还原截图的精美顶部标题栏与返回按钮
    if st.session_state.nav_tab == "history":
        st.markdown(f"### 📜 历史会话记录")
        st.caption(f"继续追问会话 ID: `{st.session_state.session_id}`")
    else:
        titles = {
            "quick": "⚡ 快速开始模式 (Quick - 无痕)",
            "rag": "📚 知识问答模式 (RAG - 严谨)",
            "agent": "🧠 智能对话模式 (Agent - 全能)"
        }
        st.markdown(f"### {titles.get(st.session_state.chat_mode, '')}")
        
        # 逃生舱：一键返回重选模式
        if st.button("🔙 返回重选模式", type="secondary"):
            create_new_chat()
            st.rerun()
            
    st.markdown("---")

    # 渲染消息记录
    for msg in st.session_state.chat_display:
        role = msg.get("role")
        if role in ["user", "assistant"]:
            with st.chat_message(role):
                st.markdown(msg.get("content", ""))

# === 6. 底部输入框 (聊天全局可用) ===
if st.session_state.nav_tab != "kb":
    
    ph_map = {
        "quick": "输入消息开始纯净对话...",
        "rag": "基于知识库提问，例如：系统如何部署？...",
        "agent": "执行任务，例如：帮我查一下今天天气..."
    }
    placeholder = ph_map.get(st.session_state.chat_mode, "输入消息开始对话...")

    if prompt := st.chat_input(placeholder):
        # 如果是在迎宾页直接输入，默认进入当前绑定的模式隐藏卡片
        if st.session_state.show_welcome_cards:
            st.session_state.show_welcome_cards = False
            
        st.session_state.chat_display.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            url = f"{API_BASE_URL}/chat/stream"
            payload = {"session_id": st.session_state.session_id, "query": prompt, "mode": st.session_state.chat_mode}
            if st.session_state.chat_mode == "agent":
                url = f"{API_BASE_URL}/agent/stream"
                payload = {"session_id": st.session_state.session_id, "query": prompt, "use_user_memory": False}
            
            try:
                with requests.post(url, json=payload, stream=True) as r:
                    for line in r.iter_lines():
                        if line:
                            decoded_line = line.decode('utf-8')
                            if decoded_line.startswith("data:"):
                                data_str = decoded_line[5:].strip()
                                if data_str == "[DONE]": break
                                try:
                                    data_json = json.loads(data_str)
                                    if "error" in data_json:
                                        full_response += f"\n❌ **错误**: {data_json['error']}"
                                    elif "content" in data_json:
                                        full_response += data_json["content"]
                                    response_placeholder.markdown(full_response + "▌")
                                except json.JSONDecodeError: pass
                response_placeholder.markdown(full_response)
                st.session_state.chat_display.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"连接后端失败: {e}")