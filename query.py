import json
import os
import pickle
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from model import ModelManager

# 💡 导入你朋友写的学术 Prompt 处理器
import academic_prompt 

def get_llm(provider, api_key, base_url, model):
    """根据提供商获取 LLM 对象，并设置较长的超时时间以应对长文生成"""
    common_params = {
        "model": model,
        "api_key": api_key,
        "temperature": 0.3,
    }
    
    if provider == "Gemini":
        return ChatGoogleGenerativeAI(
            model=model or "gemini-1.5-pro",
            google_api_key=api_key,
            temperature=0.3
        )
    elif provider == "Qwen":
        return ChatOpenAI(
            **common_params,
            base_url=base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    elif provider == "xAI":
        return ChatOpenAI(
            **common_params,
            base_url="https://api.x.ai/v1"
        )
    else: # DeepSeek, OpenAI, etc.
        return ChatOpenAI(
            **common_params,
            base_url=base_url
        )

def query_rag(question, api_key, base_url, provider, model, 
              selected_files=None, hyde=True, db_path="./chroma_db", answer_style="哲学论述"):
    """
    完整的 RAG 查询函数
    answer_style 对应 academic_prompt.py 中的模式名称
    """
    
    # 1. 初始化 Embedding 和向量库
    embeddings = ModelManager.get_embeddings()
    if embeddings is None:
        return "模型正在初始化中，请稍后再试...", [], {"total_tokens": 0}
        
    vectordb = Chroma(persist_directory=db_path, embedding_function=embeddings, collection_name="rag_collection")

    # 2. 构建混合检索器 (Hybrid Retriever)
    # A. 密集检索 (Chroma)
    search_kwargs = {"k": 150}
    if selected_files:
        search_kwargs["filter"] = {"source": {"$in": selected_files}}
    dense_retriever = vectordb.as_retriever(search_kwargs=search_kwargs)

    # B. 稀疏检索 (BM25)
    pkl_path = os.path.join(db_path, "chunks.pkl")
    if not os.path.exists(pkl_path):
        return f"错误：在该路径找不到索引文件 {pkl_path}，请重新导入数据。", [], {"total_tokens": 0}
    
    with open(pkl_path, "rb") as f:
        all_chunks = pickle.load(f)
    
    # 过滤属于选中文件的 chunks
    if selected_files:
        filtered_chunks = [c for c in all_chunks if c.metadata.get("source") in selected_files]
        if not filtered_chunks:
            return "错误：选中的文献在索引库中不存在，请重新刷新列表或导入。", [], {"total_tokens": 0}
    else:
        filtered_chunks = all_chunks

    bm25_retriever = BM25Retriever.from_documents(filtered_chunks)
    bm25_retriever.k = 40

    # C. 混合 (权重分配：向量 0.6 + 文本 0.4)
    hybrid_retriever = EnsembleRetriever(retrievers=[dense_retriever, bm25_retriever], weights=[0.6, 0.4])

    # 3. 初始化模型
    llm = get_llm(provider, api_key, base_url, model)

    # 4. 执行 HyDE (假设性回答增强检索)
    retrieval_query = question
    if hyde:
        hyde_prompt = ChatPromptTemplate.from_template(
            "请针对以下问题，提供一个简短的假设性答案框架，用作检索扩充。\n问题：{question}\n假设答案："
        )
        try:
            hypothetical = (hyde_prompt | llm).invoke({"question": question}).content
            retrieval_query = f"{question}\n{hypothetical}"
        except Exception as e:
            print(f"HyDE 增强失败，回退到原始搜索: {e}")

    # 5. 执行检索
    docs = hybrid_retriever.invoke(retrieval_query)
    
    weights_path = os.path.join(db_path, "file_weights.json")
    file_weights = {}
    if os.path.exists(weights_path):
        with open(weights_path, "r", encoding="utf-8") as f:
            file_weights = json.load(f)

    final_docs = []
    source_counts = {}
    lang_distribution = {"EN": 0, "CN": 0}
    
    # 将 150 个原始碎片按 50 个一组分为三轮
    rounds = [docs[0:50], docs[50:100], docs[100:150]]

    for i, round_docs in enumerate(rounds):
        round_num = i + 1 # 1, 2, 3 轮
        
        for doc in round_docs:
            src = doc.metadata.get("source")
            parent = doc.metadata.get("parent") # URL 母站
            lookup_key = parent if parent else src
            
            # 获取用户赋予的权重等级 (默认 1)
            weight = int(file_weights.get(lookup_key, 1))
            
            # 动态判断当前轮次的准入限制 (Limit)
            current_limit = 999 # 默认不加限制 (Unlimited)
            
            if weight == 1:
                if round_num == 2: current_limit = 12
                if round_num == 3: current_limit = 6
            elif weight == 2:
                # 2级权重：前两轮不限，第三轮限制为 12
                if round_num == 3: current_limit = 12
            elif weight == 3:
                # 3级权重：所有轮次均不加限制
                current_limit = 999

            # 检查该文件已录入的总数是否超过当前轮次的限制
            if source_counts.get(src, 0) < current_limit:
                # 查重：避免同一片段在不同轮次重复录入
                if doc.page_content not in [d.page_content for d in final_docs]:
                    final_docs.append(doc)
                    source_counts[src] = source_counts.get(src, 0) + 1


    # 6. 💡 核心修改：按照学术 Prompt 的要求格式化 Context
    # 学术 Prompt 极其依赖 "Cite as" 标签来生成页码和脚注
    formatted_context = ""
    for doc in final_docs:
        source_name = doc.metadata.get("name") or os.path.basename(doc.metadata.get("source", "未知来源"))
        page = doc.metadata.get("page", "？") # PDF 可能会有页码信息
        formatted_context += f"Cite as: ({source_name}, p. {page})\nContent: {doc.page_content}\n----------------\n"

    # 7. 💡 核心修改：调用学术 Prompt 模板
    final_full_prompt = academic_prompt.build_prompt(
        question=question,
        context=formatted_context,
        answer_style=answer_style
    )

    # 8. 向 LLM 请求生成回答
    # 注意：学术回答通常很长，确保前端能等待
    try:
        # 因为 academic_prompt 生成的是带指令的全量字符串，直接丢给模型即可
        result = llm.invoke(final_full_prompt)
        answer = result.content if hasattr(result, 'content') else str(result)
    except Exception as e:
        return f"请求 LLM 失败: {e}", [], {"total_tokens": 0}

    # 9. 提取参考路径 (用于前端 UI 变色高亮)
    source_paths = list(set([doc.metadata.get("source") for doc in docs]))
    
    # 统计 (简单估算)
    token_usage = {"total_tokens": (len(answer) + len(final_full_prompt)) // 2}

    return answer, source_paths, token_usage