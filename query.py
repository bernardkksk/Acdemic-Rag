from __future__ import annotations

import json
import math
import os
import pickle
import re
import threading
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from rank_bm25 import BM25Okapi

import academic_prompt
from model import ModelManager


RRF_K = 60
ROUND_RATIOS = (0.2, 0.3, 0.5)
NONE_VALUE = "（无）"
_LIBRARY_CACHE: Dict[str, "LibraryIndex"] = {}
_CACHE_LOCK = threading.Lock()


@dataclass
class QuestionExpansion:
    expanded_question: str
    search_queries: List[str]
    keywords: List[str]
    hyde_text: str = ""


@dataclass
class Candidate:
    key: str
    doc: Document
    library_name: str
    library_path: str
    weight: int
    score: float = 0.0


@dataclass
class LibraryIndex:
    path: str
    name: str
    vector_store: Chroma
    chunks: List[Document]
    bm25: BM25Okapi
    weights: Dict[str, int]
    sources_by_weight: Dict[int, set[str]]


def get_llm(provider: str, api_key: str, base_url: str, model: str):
    common_params = {
        "model": model,
        "api_key": api_key,
        "temperature": 0.2,
    }

    if provider == "Gemini":
        return ChatGoogleGenerativeAI(
            model=model or "gemini-1.5-pro",
            google_api_key=api_key,
            temperature=0.2,
        )
    if provider == "Qwen":
        return ChatOpenAI(
            **common_params,
            base_url=base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    if provider == "xAI":
        return ChatOpenAI(
            **common_params,
            base_url=base_url or "https://api.x.ai/v1",
        )
    return ChatOpenAI(
        **common_params,
        base_url=base_url,
    )


def invalidate_library_cache(path: str) -> None:
    with _CACHE_LOCK:
        _LIBRARY_CACHE.pop(os.path.abspath(path), None)


def preload_libraries(
    libraries: Sequence[dict],
    progress_callback: Optional[Callable[[str], None]] = None,
) -> None:
    for library in libraries:
        if progress_callback:
            progress_callback(f"正在载入知识库：{library['name']}")
        _get_library_index(library["path"], library["name"])


def query_rag(
    question: str,
    api_key: str,
    base_url: str,
    provider: str,
    model: str,
    primary_library: dict,
    secondary_libraries: Optional[Sequence[dict]] = None,
    answer_style: str = academic_prompt.STYLE_PHILOSOPHER,
    progress_callback: Optional[Callable[[str, str], None]] = None,
):
    if ModelManager.get_embeddings() is None:
        raise RuntimeError(ModelManager.get_last_error() or "模型仍在初始化中。")

    llm = get_llm(provider, api_key, base_url, model)
    selected_libraries = [primary_library] + list(secondary_libraries or [])

    _emit(progress_callback, "expand", "正在进行问题扩写…")
    expansion = _expand_question(question, llm)

    _emit(progress_callback, "expand", "正在生成 HyDE 假设答案…")
    expansion.hyde_text = _generate_hyde(question, expansion.expanded_question, llm)

    _emit(progress_callback, "retrieve", "正在预热已选知识库…")
    preload_libraries(selected_libraries)

    retrieval_target = academic_prompt.get_retrieval_target(answer_style)
    library_quotas = _build_library_quotas(primary_library, secondary_libraries or [], retrieval_target)

    _emit(progress_callback, "retrieve", "正在执行多库三轮混合检索…")
    collected: List[Candidate] = []
    collected_keys: set[str] = set()

    for library in selected_libraries:
        quota = library_quotas.get(library["path"], 0)
        if quota <= 0:
            continue

        index = _get_library_index(library["path"], library["name"])
        library_candidates = _retrieve_from_library(index, expansion, quota)
        for candidate in library_candidates:
            if candidate.key in collected_keys:
                continue
            collected.append(candidate)
            collected_keys.add(candidate.key)

    if not collected:
        return "当前所选知识库中未检索到相关材料。请调整问题、提高权重，或先完成该库的导入。", [], {"total_tokens": 0}

    _emit(progress_callback, "rerank", "正在使用同一 embedding 模型进行离线 rerank…")
    reranked = _rerank_candidates(question, expansion, collected, retrieval_target)

    context = _build_context(reranked)
    prompt = academic_prompt.build_prompt(
        question=question,
        context=context,
        answer_style=answer_style,
    )

    _emit(progress_callback, "generate", "正在生成学术回答…")
    result = llm.invoke(prompt)
    answer = result.content if hasattr(result, "content") else str(result)
    sources = list(dict.fromkeys(candidate.doc.metadata.get("source", "") for candidate in reranked if candidate.doc.metadata.get("source")))
    token_usage = {"total_tokens": (len(prompt) + len(answer)) // 2}
    return answer, sources, token_usage


def _get_library_index(path: str, name: str) -> LibraryIndex:
    abs_path = os.path.abspath(path)
    with _CACHE_LOCK:
        cached = _LIBRARY_CACHE.get(abs_path)
        if cached is not None:
            return cached

    pkl_path = os.path.join(abs_path, "chunks.pkl")
    if not os.path.exists(pkl_path):
        raise RuntimeError(f"知识库 {name} 尚未完成导入：{pkl_path} 不存在。")

    with open(pkl_path, "rb") as handle:
        chunks = pickle.load(handle)

    if not chunks:
        raise RuntimeError(f"知识库 {name} 为空，请先导入文档。")

    weights = _load_weights(abs_path)
    sources_by_weight = {1: set(), 2: set(), 3: set()}
    tokenized_corpus: List[List[str]] = []

    for doc in chunks:
        doc.metadata.setdefault("library_name", name)
        doc.metadata.setdefault("library_path", abs_path)
        source = str(doc.metadata.get("source", ""))
        weight = _lookup_weight(doc, weights)
        sources_by_weight.setdefault(weight, set()).add(source)
        tokens = _tokenize_text(doc.page_content)
        tokenized_corpus.append(tokens or ["_"])

    bm25 = BM25Okapi(tokenized_corpus)
    vector_store = Chroma(
        persist_directory=abs_path,
        embedding_function=ModelManager.get_embeddings(),
        collection_name="rag_collection",
    )

    index = LibraryIndex(
        path=abs_path,
        name=name,
        vector_store=vector_store,
        chunks=chunks,
        bm25=bm25,
        weights=weights,
        sources_by_weight=sources_by_weight,
    )
    with _CACHE_LOCK:
        _LIBRARY_CACHE[abs_path] = index
    return index


def _expand_question(question: str, llm) -> QuestionExpansion:
    prompt = f"""
请根据下面的问题做学术检索预处理，并且只返回 JSON。

要求：
1. `expanded_question`：将原问题扩写成更适合检索的完整学术问题，保留核心概念。
2. `search_queries`：给出 3 条彼此不同、适合向量检索的短查询。
3. `keywords`：给出 6 到 10 个检索关键词，尽量包含中英文术语。

返回格式：
{{
  "expanded_question": "...",
  "search_queries": ["...", "...", "..."],
  "keywords": ["...", "..."]
}}

原问题：
{question}
"""
    fallback = QuestionExpansion(
        expanded_question=question,
        search_queries=[question],
        keywords=_heuristic_keywords(question),
    )
    try:
        raw = llm.invoke(prompt)
        text = raw.content if hasattr(raw, "content") else str(raw)
        data = _extract_json(text)
        expanded = str(data.get("expanded_question") or question).strip()
        search_queries = [str(item).strip() for item in data.get("search_queries", []) if str(item).strip()]
        keywords = [str(item).strip() for item in data.get("keywords", []) if str(item).strip()]
        if not search_queries:
            search_queries = [expanded]
        if not keywords:
            keywords = _heuristic_keywords(question)
        return QuestionExpansion(
            expanded_question=expanded,
            search_queries=_dedupe_preserve_order([question, expanded] + search_queries)[:4],
            keywords=_dedupe_preserve_order(keywords)[:10],
        )
    except Exception:
        return fallback


def _generate_hyde(question: str, expanded_question: str, llm) -> str:
    prompt = f"""
请围绕下面的问题写一段 150 到 220 字的假设性学术回答，用于检索增强，不要写脚注，也不要解释任务。

原问题：
{question}

扩写问题：
{expanded_question}
"""
    try:
        raw = llm.invoke(prompt)
        text = raw.content if hasattr(raw, "content") else str(raw)
        return text.strip()
    except Exception:
        return ""


def _retrieve_from_library(index: LibraryIndex, expansion: QuestionExpansion, quota: int) -> List[Candidate]:
    round_targets = _split_round_targets(quota)
    collected: List[Candidate] = []
    collected_keys: set[str] = set()

    round_specs = [
        ({3}, round_targets[0]),
        ({2, 3}, round_targets[1]),
        ({1, 2, 3}, round_targets[2]),
    ]

    for allowed_weights, round_target in round_specs:
        if round_target <= 0:
            continue
        round_candidates = _retrieve_round(index, expansion, allowed_weights, round_target)
        for candidate in round_candidates:
            if candidate.key in collected_keys:
                continue
            collected.append(candidate)
            collected_keys.add(candidate.key)
            if len(collected) >= quota:
                return collected

    if len(collected) < quota:
        fallback = _retrieve_round(index, expansion, {1, 2, 3}, quota)
        for candidate in fallback:
            if candidate.key in collected_keys:
                continue
            collected.append(candidate)
            collected_keys.add(candidate.key)
            if len(collected) >= quota:
                break

    return collected


def _retrieve_round(
    index: LibraryIndex,
    expansion: QuestionExpansion,
    allowed_weights: set[int],
    round_target: int,
) -> List[Candidate]:
    allowed_sources = set()
    for weight in allowed_weights:
        allowed_sources.update(index.sources_by_weight.get(weight, set()))
    if not allowed_sources:
        return []

    raw_k = min(max(round_target * 3, 30), 180)
    dense_queries = _dedupe_preserve_order(
        [
            expansion.search_queries[0] if expansion.search_queries else expansion.expanded_question,
            expansion.expanded_question,
            expansion.hyde_text,
        ]
    )

    ranked_lists: List[Tuple[List[Candidate], float]] = []
    for idx, dense_query in enumerate(query for query in dense_queries if query):
        docs_with_scores = index.vector_store.similarity_search_with_score(
            dense_query,
            k=raw_k,
            filter={"source": {"$in": sorted(allowed_sources)}},
        )
        weight = 1.0 if idx == 0 else 0.85
        ranked_lists.append((_wrap_dense_results(index, docs_with_scores), weight))

    sparse_query = " ".join(_dedupe_preserve_order([expansion.expanded_question] + expansion.keywords))
    sparse_candidates = _run_sparse_search(index, sparse_query, allowed_sources, raw_k)
    if sparse_candidates:
        ranked_lists.append((sparse_candidates, 0.95))

    fused = _rrf_fuse(ranked_lists)
    return fused[:round_target]


def _wrap_dense_results(index: LibraryIndex, docs_with_scores: Sequence[Tuple[Document, float]]) -> List[Candidate]:
    wrapped: List[Candidate] = []
    seen: set[str] = set()
    for doc, distance in docs_with_scores:
        key = _candidate_key(index.path, doc)
        if key in seen:
            continue
        seen.add(key)
        wrapped.append(
            Candidate(
                key=key,
                doc=doc,
                library_name=index.name,
                library_path=index.path,
                weight=_lookup_weight(doc, index.weights),
                score=1.0 / (1.0 + max(float(distance), 0.0)),
            )
        )
    return wrapped


def _run_sparse_search(
    index: LibraryIndex,
    query_text: str,
    allowed_sources: set[str],
    limit: int,
) -> List[Candidate]:
    tokens = _tokenize_text(query_text)
    if not tokens:
        return []

    scores = index.bm25.get_scores(tokens)
    ranked_indices = np.argsort(scores)[::-1]
    candidates: List[Candidate] = []

    for idx in ranked_indices:
        if len(candidates) >= limit:
            break
        score = float(scores[idx])
        if score <= 0:
            continue
        doc = index.chunks[int(idx)]
        source = str(doc.metadata.get("source", ""))
        if source not in allowed_sources:
            continue
        candidates.append(
            Candidate(
                key=_candidate_key(index.path, doc),
                doc=doc,
                library_name=index.name,
                library_path=index.path,
                weight=_lookup_weight(doc, index.weights),
                score=score,
            )
        )
    return candidates


def _rrf_fuse(ranked_lists: Sequence[Tuple[List[Candidate], float]]) -> List[Candidate]:
    totals: Dict[str, Candidate] = {}
    score_map: Dict[str, float] = {}

    for candidates, list_weight in ranked_lists:
        for rank, candidate in enumerate(candidates, start=1):
            if candidate.key not in totals:
                totals[candidate.key] = candidate
                score_map[candidate.key] = 0.0
            score_map[candidate.key] += list_weight / (RRF_K + rank)
            score_map[candidate.key] += candidate.score * 0.02

    fused = list(totals.values())
    fused.sort(key=lambda item: score_map[item.key], reverse=True)
    for candidate in fused:
        candidate.score = score_map[candidate.key]
    return fused


def _rerank_candidates(
    question: str,
    expansion: QuestionExpansion,
    candidates: Sequence[Candidate],
    retrieval_target: int,
) -> List[Candidate]:
    rerank_query = "\n".join(
        part for part in [question, expansion.expanded_question, expansion.hyde_text] if part
    )
    query_vector = ModelManager.encode_texts([rerank_query])[0]
    doc_vectors = ModelManager.encode_texts(
        [_build_rerank_text(candidate.doc) for candidate in candidates],
        batch_size=24,
    )
    scores = doc_vectors @ query_vector

    reranked = list(candidates)
    for candidate, score in zip(reranked, scores.tolist()):
        candidate.score = float(score) + candidate.score * 0.1

    reranked.sort(key=lambda item: item.score, reverse=True)
    return reranked[:retrieval_target]


def _build_context(candidates: Sequence[Candidate]) -> str:
    blocks: List[str] = []
    for idx, candidate in enumerate(candidates, start=1):
        doc = candidate.doc
        source_path = str(doc.metadata.get("source", "未知来源"))
        source_name = str(doc.metadata.get("name") or os.path.basename(source_path) or source_path)
        page = doc.metadata.get("page") or doc.metadata.get("page_number") or "?"
        excerpt = _normalize_excerpt(doc.page_content)
        blocks.append(
            "\n".join(
                [
                    f"[材料 {idx}]",
                    f"Library: {candidate.library_name}",
                    f"Cite as: ({source_name}, p. {page})",
                    f"Source: {source_path}",
                    f"Weight: {candidate.weight}",
                    f"Content: {excerpt}",
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


def _build_library_quotas(primary_library: dict, secondary_libraries: Sequence[dict], total: int) -> Dict[str, int]:
    quotas: Dict[str, int] = {}
    if not secondary_libraries:
        quotas[primary_library["path"]] = total
        return quotas

    primary_quota = max(1, int(round(total * 0.8)))
    remaining = max(0, total - primary_quota)
    quotas[primary_library["path"]] = primary_quota

    if len(secondary_libraries) == 1:
        quotas[secondary_libraries[0]["path"]] = remaining
    else:
        share = remaining // len(secondary_libraries)
        extra = remaining % len(secondary_libraries)
        for idx, library in enumerate(secondary_libraries):
            quotas[library["path"]] = share + (1 if idx < extra else 0)

    return quotas


def _split_round_targets(total: int) -> Tuple[int, int, int]:
    first = max(1, int(math.floor(total * ROUND_RATIOS[0])))
    second = max(1, int(math.floor(total * ROUND_RATIOS[1])))
    third = max(0, total - first - second)
    if total == 1:
        return 1, 0, 0
    if total == 2:
        return 1, 1, 0
    return first, second, max(1, third)


def _load_weights(db_path: str) -> Dict[str, int]:
    path = os.path.join(db_path, "file_weights.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {str(key): _coerce_weight(value) for key, value in raw.items()}


def _lookup_weight(doc: Document, weights: Dict[str, int]) -> int:
    parent = str(doc.metadata.get("parent", "") or "")
    source = str(doc.metadata.get("source", "") or "")
    return weights.get(parent, weights.get(source, 1))


def _coerce_weight(value) -> int:
    try:
        weight = int(value)
    except Exception:
        return 1
    return 1 if weight < 1 else 3 if weight > 3 else weight


def _heuristic_keywords(question: str) -> List[str]:
    tokens = _tokenize_text(question)
    return _dedupe_preserve_order(tokens)[:8] or [question]


def _tokenize_text(text: str) -> List[str]:
    lowered = str(text or "").lower()
    latin_tokens = re.findall(r"[a-z0-9_]+", lowered)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", lowered)
    cjk_bigrams = ["".join(cjk_chars[idx : idx + 2]) for idx in range(len(cjk_chars) - 1)]
    return [token for token in latin_tokens + cjk_chars + cjk_bigrams if token.strip()]


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("未找到 JSON。")
    return json.loads(match.group(0))


def _candidate_key(library_path: str, doc: Document) -> str:
    source = str(doc.metadata.get("source", ""))
    page = str(doc.metadata.get("page") or doc.metadata.get("page_number") or "")
    snippet = _normalize_excerpt(doc.page_content, limit=120)
    return f"{library_path}::{source}::{page}::{hash(snippet)}"


def _normalize_excerpt(text: str, limit: int = 420) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    return cleaned[:limit] + ("…" if len(cleaned) > limit else "")


def _build_rerank_text(doc: Document) -> str:
    source = str(doc.metadata.get("name") or doc.metadata.get("source") or "")
    page = str(doc.metadata.get("page") or doc.metadata.get("page_number") or "")
    return f"{source} {page} {_normalize_excerpt(doc.page_content, limit=600)}"


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _emit(progress_callback: Optional[Callable[[str, str], None]], stage: str, message: str) -> None:
    if progress_callback:
        progress_callback(stage, message)
