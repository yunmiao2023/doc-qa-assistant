import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
MODEL_NAME = "Qwen/Qwen3-8B"
API_URL = "https://api.siliconflow.cn/v1/chat/completions"

if not API_KEY:
    st.error("未读取到 API_KEY，请检查 .env 文件是否配置正确。")
    st.stop()


def ask_ai(question, content, history):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    messages = [{"role": "system", "content": f"你是一个文档问答助手，帮助用户从文档中获取信息。\n\n文档内容: \n{content}"}]
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
    .stChatMessage:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
        text-align: right !important;
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
        else:
            content = uploadfile_name.read().decode("utf-8")

            if "messages" not in st.session_state:
                st.session_state.messages = []
            if "current_file" not in st.session_state:
                st.session_state.current_file = uploadfile_name.name
            if st.session_state.current_file != uploadfile_name.name:
                st.session_state.messages = []
                st.session_state.current_file = uploadfile_name.name

            if st.button("🗑️ 清空对话", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

            st.divider()
            st.caption(f"当前文件：{uploadfile_name.name}")
            st.caption(f"模型：{MODEL_NAME}")

    # ===== 主区域：聊天 =====
    if content is None:
        st.info("👈 请先在左侧上传一个 txt 文件，然后就可以开始提问了。")
    else:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"].strip())

    question = st.chat_input(
        "请先上传文件..." if content is None else "请输入您的问题...",
        disabled=(content is None)
    )

    if question:
        clean_question = question.strip()

        if not clean_question:
            st.warning("问题不能为空，请勿输入纯空格。")
        else:
            with st.chat_message("user"):
                st.markdown(clean_question)

            with st.chat_message("assistant"):
                with st.spinner("AI 正在思考..."):
                    full_answer = st.write_stream(
                        ask_ai(clean_question, content, st.session_state.messages)
                    )

            if "[请求超时" in full_answer or "[网络出问题" in full_answer:
                st.error("本次请求失败，请重试。")
            else:
                st.session_state.messages.append({"role": "user", "content": clean_question})
                st.session_state.messages.append({"role": "assistant", "content": full_answer.strip()})


main()