import streamlit as st
import requests
import json
import os
import chromadb
import time
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
MODEL_NAME = "Qwen/Qwen3-8B"
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
EMBED_MODEL = "BAAI/bge-m3"
EMBED_URL = "https://api.siliconflow.cn/v1/embeddings"
DISTANCE_THRESHOLD = 0.85

if not API_KEY:
    st.error("未读取到 API_KEY，请检查 .env 文件是否配置正确。")
    st.stop()

def split_text(text, chunk_size=200, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks

def embed_text(text):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }   
    payload = {
        "model": EMBED_MODEL,
        "input": text
    }
    response = requests.post(EMBED_URL, headers=headers, json=payload)
    data = response.json()
    return data["data"][0]["embedding"]

def build_collection(content):
    chunks = split_text(content)
    embeddings = [embed_text(chunk) for chunk in chunks]
    client = chromadb.EphemeralClient()          # 内存客户端
    collection_name = f"my_docs_{int(time.time() * 1000)}"   # 唯一名字
    collection = client.create_collection(collection_name, embedding_function=None)
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings
    )
    return collection


def ask_ai(question, content, history):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    messages = [{"role": "system", "content": f"你是一个文档问答助手，只根据下面的参考资料回答。\n\n参考资料： \n{content}"}]
    messages.extend(history)
    messages.append({"role": "user", "content": question})

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": True
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload, stream=True, timeout=60)
        for chunk in response.iter_lines():
            if not chunk:
                continue
            chunk_str = chunk.decode("utf-8").replace("data: ", "")
            if chunk_str == "[DONE]":
                break
            try:
                chunk_data = json.loads(chunk_str)
                delta = chunk_data["choices"][0].get("delta", {})
                content_piece = delta.get("content", "")
                if content_piece:
                    yield content_piece
            except (KeyError, IndexError, json.JSONDecodeError):
                continue
    except requests.exceptions.Timeout:
        yield "\n\n[请求超时喵, 请检查网络后重试捏！]"
    except requests.exceptions.RequestException:
        yield "\n\n[网络出问题了喵, 请检查网络后重试捏！]"


def main():
    st.set_page_config(page_title="文档问答助手", page_icon="📄", layout="centered")
    st.title("📄 文档问答助手")

    st.markdown("""
    <style>
    /* 用户消息：靠右、浅蓝背景、圆角 */
    .stChatMessage:has([data-testid="stChatMessageAvatarUser"]) {
        display: flex !important;
        flex-direction: row-reverse !important;
        background-color: #e3f2fd !important;
        border-radius: 18px !important;
        padding: 12px 16px !important;
        margin: 8px 0 8px auto !important;
        max-width: 80% !important;
    }
   

    /* AI 消息：靠左、浅灰背景、圆角 */
    .stChatMessage:has([data-testid="stChatMessageAvatarAssistant"]) {
        background-color: #f5f5f5 !important;
        border-radius: 18px !important;
        padding: 12px 16px !important;
        margin: 8px auto 8px 0 !important;
        max-width: 80% !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ===== 侧边栏：上传和控制 =====
    with st.sidebar:
        st.header("📂 文档设置")
        uploadfile_name = st.file_uploader("上传一个txt文件", type=["txt"])

        if uploadfile_name is None:
            st.info("请先上传一个txt文件")
            content = None
            collection = None
        else:
            content = uploadfile_name.read().decode("utf-8")

            if "messages" not in st.session_state:
                st.session_state.messages = []
            if "current_file" not in st.session_state:
                st.session_state.current_file = uploadfile_name.name

            if st.session_state.current_file != uploadfile_name.name:
                with st.spinner("正在处理文档，切块并向量化......"):
                    st.session_state.collection = build_collection(content)
                st.session_state.messages = []
                st.session_state.current_file = uploadfile_name.name

            if "collection" not in st.session_state:
                with st.spinner("正在处理文档，切块并向量化..."):
                    st.session_state.collection = build_collection(content)

            collection = st.session_state.collection

            if st.button("🗑️ 清空对话", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

            st.divider()
            st.caption(f"当前文件：{uploadfile_name.name}")
            st.caption(f"模型：{MODEL_NAME}")

    # ===== 主区域：聊天 =====
    if collection is None:
        st.info("👈 请先在左侧上传一个 txt 文件，然后就可以开始提问了。")
    else:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"].strip())

    question = st.chat_input(
        "请先上传文件..." if collection is None else "请输入您的问题...",
        disabled=(collection is None)
    )

    if question:
        clean_question = question.strip() 

        if not clean_question:
            st.warning("问题不能为空，请勿输入纯空格。")
        else:
            with st.chat_message("user"):
                st.markdown(clean_question)

            with st.chat_message("assistant"):
                  # 判断是不是总结类问题
                summary_keywords = ["主要讲", "总结", "概括", "摘要", "核心内容", "讲了什么"]
                is_summary = any(kw in clean_question for kw in summary_keywords)

                if is_summary:
                    # 总结类：直接用全文
                    context = content
                else:
                    # 具体问题：检索
                    with st.spinner("正在检索文档..."):
                        question_vector = embed_text(clean_question)
                        results = collection.query(
                            query_embeddings=[question_vector],
                            n_results=3
                        )
                        min_distance = results["distances"][0][0]
                    if min_distance > DISTANCE_THRESHOLD:
                        st.markdown("文档中没有找到相关内容，请换个问题或上传相关文档。")
                        context = None
                    else:
                        context = "\n\n".join(results["documents"][0])
                if context:
                    with st.spinner("AI 正在思考..."):
                        full_answer = st.write_stream(
                            ask_ai(clean_question, context, st.session_state.messages)
                        )

                    if "[请求超时" in full_answer or "[网络出问题" in full_answer:
                        st.error("本次请求失败，请重试。")
                    else:
                        st.session_state.messages.append({"role": "user", "content": clean_question})
                        st.session_state.messages.append({"role": "assistant", "content": full_answer.strip()})


main()