# RAG Pro 🎓

一个专为学术研究写作设计的本地知识库 RAG（检索增强生成）工具。

## 🌟 核心特性

- **多模式学术写作**：内置“内容论述”、“文献综述”、“概念梳理”等多种专业风格，严格遵守学术脚注规范。
- **三轮动态权重检索**：独创的检索算法，根据用户赋予文献的权重等级（1-3级），动态调整检索碎片的配额与深度。
- **混合检索系统**：结合了 Chroma 向量检索（语义）与 BM25 检索（关键词），并支持 HyDE（假设性文档嵌入）增强。
- **本地化管理**：支持 PDF、Word、TXT 及网页 URL 抓取，且URL抓取支持递归递进，所有知识碎片本地向量化。
- **多模型支持**：兼容 DeepSeek、OpenAI、Gemini、xAI、通义千问等主流大模型 API。
- **可视化 GUI**：基于 CustomTkinter 构建，支持多知识库切换、文件权重实时调节。

 🚀 快速开始

 1. 环境准备
确保你已安装 Python 3.10 或以上版本。

 2. 克隆仓库
git clone https://github.com/你的用户名/PhilosophyRAG.git
cd PhilosophyRAG

 3. 安装依赖
pip install -r requirements.txt

 4.运行程序
python main.py

 5.配置说明
启动程序后，在API设置的选项卡中配置你的模型提供商和 API Key。
在导入数据模块中添加你的研究内容。
在文件管理中，你可以通过右侧的 Lvl 下拉菜单为重要文献设置更高的权重（1-3级），级别越高权重越大。

 6. 技术架构
GUI: CustomTkinter
Vector DB: ChromaDB
Embedding: BAAI/bge-small-zh-v1.5 (本地运行)
RAG Framework: LangChain

 7. 开源协议
MIT License