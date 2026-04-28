import os
import pickle
import re
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    RecursiveUrlLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from model import ModelManager


def detect_language(text: str) -> str:
    if not text:
        return "CN"
    sample = text[:1000]
    english_chars = len(re.findall(r"[a-zA-Z]", sample))
    return "EN" if sample and english_chars / len(sample) > 0.3 else "CN"


def load_documents(
    docs_dir: Optional[str] = None,
    urls: Optional[List[str]] = None,
    recursive_web: bool = False,
    max_depth: int = 3,
):
    docs = []
    os.environ.setdefault("USER_AGENT", "Mozilla/5.0 (compatible; AcademicRAG/1.0)")

    if docs_dir and os.path.exists(docs_dir):
        for root, _, files in os.walk(docs_dir):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                suffix = os.path.splitext(file_name)[1].lower()
                try:
                    if suffix == ".pdf":
                        loader = PyPDFLoader(file_path)
                        doc_type = "PDF"
                    elif suffix == ".txt":
                        loader = TextLoader(file_path, encoding="utf-8")
                        doc_type = "TXT"
                    elif suffix in {".doc", ".docx"}:
                        loader = Docx2txtLoader(file_path)
                        doc_type = "WORD"
                    else:
                        continue

                    loaded_docs = loader.load()
                    for doc in loaded_docs:
                        doc.metadata.update(
                            {
                                "source": file_path,
                                "name": file_name,
                                "type": doc_type,
                                "lang": detect_language(doc.page_content),
                            }
                        )
                    docs.extend(loaded_docs)
                except Exception as exc:
                    print(f"加载文件失败 {file_path}: {exc}")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    for url in urls or []:
        try:
            if recursive_web:
                loader = RecursiveUrlLoader(
                    url=url,
                    max_depth=max_depth,
                    prevent_outside=True,
                    use_async=True,
                    timeout=30,
                    headers=headers,
                )
            else:
                loader = WebBaseLoader(
                    web_paths=[url],
                    requests_kwargs={"timeout": 30, "headers": headers},
                )

            loaded_docs = loader.load()
            for doc in loaded_docs:
                actual_source = doc.metadata.get("source", url)
                doc.metadata.update(
                    {
                        "source": actual_source,
                        "parent": url,
                        "name": actual_source,
                        "type": "URL",
                        "lang": detect_language(doc.page_content),
                    }
                )
            docs.extend(loaded_docs)
        except Exception as exc:
            print(f"抓取 URL 失败 {url}: {exc}")

    return docs


def run_ingest(
    docs_dir=None,
    urls=None,
    mode="append",
    chunk_size=1200,
    chunk_overlap=250,
    recursive_web=False,
    progress_callback=None,
    db_path="./chroma_db",
):
    os.makedirs(db_path, exist_ok=True)
    if progress_callback:
        progress_callback(f"正在向 [{os.path.basename(db_path)}] 导入资料…")

    documents = load_documents(docs_dir, urls, recursive_web)
    if not documents:
        if progress_callback:
            progress_callback("未找到可导入的本地文档或网页内容。")
        return 0

    if progress_callback:
        progress_callback(f"已载入 {len(documents)} 份文档，开始切分…")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    if progress_callback:
        progress_callback(f"共生成 {len(chunks)} 个片段，开始向量化…")

    embeddings = ModelManager.get_embeddings()
    if embeddings is None:
        raise RuntimeError(ModelManager.get_last_error() or "Embedding 模型尚未完成初始化。")

    vectordb = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_name="rag_collection",
    )

    pkl_path = os.path.join(db_path, "chunks.pkl")
    if mode == "new":
        if progress_callback:
            progress_callback("正在清空旧知识库…")
        try:
            vectordb.delete_collection()
        except Exception:
            pass
        vectordb = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings,
            collection_name="rag_collection",
        )
        all_chunks = chunks
    else:
        existing_chunks = []
        if os.path.exists(pkl_path) and os.path.getsize(pkl_path) > 0:
            try:
                with open(pkl_path, "rb") as handle:
                    existing_chunks = pickle.load(handle)
            except Exception:
                existing_chunks = []
        all_chunks = existing_chunks + chunks

    batch_size = 200
    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        vectordb.add_documents(chunks[start:end])
        if progress_callback:
            progress_callback(f"正在写入向量库… {end}/{len(chunks)}")

    with open(pkl_path, "wb") as handle:
        pickle.dump(all_chunks, handle)

    if progress_callback:
        progress_callback(f"导入完成：{db_path}")
    return len(chunks)
