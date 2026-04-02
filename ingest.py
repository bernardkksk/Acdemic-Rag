import argparse
import os
import pickle
import re
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, WebBaseLoader, RecursiveUrlLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


from model import ModelManager

def detect_language(text):
    """检测文本语言：通过统计英文字符比例判定"""
    if not text: return "CN"
    # 只取前1000个字符进行判定
    sample = text[:1000]
    # 统计字母数量
    en_count = len(re.findall(r'[a-zA-Z]', sample))
    total_count = len(sample)
    # 如果英文字符占比超过 30%，判定为英文资料
    return "EN" if (en_count / total_count) > 0.3 else "CN"

def load_documents(docs_dir: str = None, urls: List[str] = None, recursive_web: bool = False, max_depth: int = 3):
    docs = []
    os.environ.setdefault("USER_AGENT", "Mozilla/5.0 (compatible; PhilosophyRAG/1.0)")

    # 1. 遍历本地文件夹，支持 PDF, TXT, DOCX
    if docs_dir and os.path.exists(docs_dir):
        for root, _, files in os.walk(docs_dir):
            for file in files:
                file_path = os.path.join(root, file)
                ext = file.lower().split('.')[-1]
                
                try:
                    if ext == "pdf":
                        loader = PyPDFLoader(file_path)
                        file_docs = loader.load()
                        doc_type = "PDF"
                    elif ext == "txt":
                        loader = TextLoader(file_path, encoding='utf-8')
                        file_docs = loader.load()
                        doc_type = "TXT"
                    elif ext in ["doc", "docx"]:
                        loader = Docx2txtLoader(file_path)
                        file_docs = loader.load()
                        doc_type = "Word"
                    else:
                        continue
                    
                    # 规范化 metadata
                    for doc in file_docs:
                        lang = detect_language(doc.page_content)
                        doc.metadata.update({
                            "source": file_path,
                            "name": file,
                            "type": doc_type,
                            "lang": lang,
                        })
                    docs.extend(file_docs)
                except Exception as e:
                    print(f"加载文件失败 {file_path}: {e}")

        # 模拟一个真实的浏览器头
    custom_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
    }
    # 2. 处理 URLs
    if urls:
        for url in urls:
            print(f"正在尝试连接: {url}") 
            try:
                if recursive_web:
                    loader = RecursiveUrlLoader(
                        url=url, 
                        max_depth=max_depth, 
                        prevent_outside=True,
                        use_async=True,
                        # 增加超时和头部信息
                        timeout=30,
                        headers=custom_headers,
                    )
                else:
                    loader = WebBaseLoader(                        
                        web_paths=[url],
                        requests_kwargs={
                            "timeout": 30,  # 把超时时间增加到 30 秒
                            "headers": custom_headers,
                        }
                            )
                
                web_docs = loader.load()
                for doc in web_docs:
                    lang = detect_language(doc.page_content)
                    actual_src = doc.metadata.get("source", url)
                    doc.metadata.update({
                        "source": actual_src,
                        "parent": url, # 💡 记录母站 URL，用于后续折叠显示
                        "name": actual_src,
                        "lang": lang, 
                        "type": "URL"
                    })
                docs.extend(web_docs)
            except Exception as e:
                print(f"抓取 URL 失败 {url}: {e}")
            
                
    return docs


def run_ingest(docs_dir=None, urls=None, mode="append", chunk_size=1200, chunk_overlap=250, 
               recursive_web=False, progress_callback=None, db_path="./chroma_db"):
    if progress_callback: progress_callback(f"🌱 正在向 [{os.path.basename(db_path)}] 导入素材...")

    documents = load_documents(docs_dir, urls, recursive_web)
    if not documents:
        if progress_callback: progress_callback("❌ 未找到有效文档或网页。")
        return 0

    if progress_callback: progress_callback(f"已加载 {len(documents)} 页内容，开始切分...")

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)

    if progress_callback: progress_callback(f"切分完成，共 {len(chunks)} 个片段，开始向量化...")

    embeddings = ModelManager.get_embeddings() # 使用统一加载器
    vectordb = Chroma(persist_directory=db_path, embedding_function=embeddings, collection_name="rag_collection")

    pkl_path = os.path.join(db_path, "chunks.pkl")
    
    if mode == "new":
        if progress_callback: progress_callback("🧹 正在清空旧数据库...")
        vectordb.delete_collection()
        vectordb = Chroma(persist_directory=db_path, embedding_function=embeddings, collection_name="rag_collection")
        all_chunks = chunks
    else:
        # 追加模式：读取旧的 + 新的
        existing_chunks = [] 
        if os.path.exists(pkl_path) and os.path.getsize(pkl_path) > 0:
            try:
                with open(pkl_path, "rb") as f:
                    existing_chunks = pickle.load(f)
            except (EOFError, pickle.UnpicklingError):
                # 💡 如果读取失败（Ran out of input），则视为没有旧数据
                if progress_callback: progress_callback("⚠️ 发现损坏的索引文件，已自动跳过旧数据。")
                existing_chunks = []
        all_chunks = existing_chunks + chunks
           
    # 存入 Chroma
    batch_size = 200
    total = len(chunks)
    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        vectordb.add_documents(chunks[i:end])
        if progress_callback: progress_callback(f"正在存入 Chroma 向量库... {end}/{total}")

    # 保存完整的 Chunks 供 BM25 使用
    os.makedirs(db_path, exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump(all_chunks, f)

    if progress_callback: progress_callback(f"✅ Ingest 完成！位置: {db_path}")
    return total