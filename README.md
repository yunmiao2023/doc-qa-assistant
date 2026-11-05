# 文档问答助手

基于 Streamlit 和大模型 API 的本地文档问答工具。

## 功能
- 上传 txt 文件，围绕文档内容提问
- 流式输出，回答逐字显示
- 多轮对话记忆
- 换文件自动清空历史，支持手动清空
- API 异常友好提示

## 技术栈
Python、Streamlit、requests、OpenAI 兼容 API（硅基流动 Qwen3-8B）

## 运行方式
1. 安装依赖：`pip install streamlit requests python-dotenv`
2. 在项目根目录创建 `.env`，写入：`API_KEY=硅基流动Key`
3. 启动：`streamlit run app.py`