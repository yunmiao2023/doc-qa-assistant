# 文档问答助手

基于 Streamlit 和大模型 API 的本地文档问答工具。

## 功能
- 上传 txt 文件，围绕文档内容提问
- **RAG 检索增强**：文档切块、向量化、存入 Chroma，提问时先检索最相关的 3 块再生成
- **距离阈值过滤**：检索结果与问题差异过大时，提示“文档中没有找到相关内容”，避免模型乱答
- 流式输出，回答逐字显示
- 多轮对话记忆
- 换文件自动清空历史，支持手动清空
- API 异常友好提示

## 技术栈
Python、Streamlit、requests、ChromaDB、硅基流动 API（Qwen3-8B 对话 + bge-m3 向量化）

## 运行方式
1. 安装依赖：`pip install streamlit requests python-dotenv`
2. 在项目根目录创建 `.env`，写入：`API_KEY=硅基流动Key`
3. 启动：`streamlit run app.py`