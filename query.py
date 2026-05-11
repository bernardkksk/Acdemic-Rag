from __future__ import annotations

import json
import math
import os
import pickle
import re
import threading
from dataclasses import dataclass
from hashlib import sha1
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import requests
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
CITE_PATCH_REPAIR_ROUNDS = 2
_LIBRARY_CACHE: Dict[str, "LibraryIndex"] = {}
_CACHE_LOCK = threading.Lock()


def _read_library_meta(path: str) -> dict:
    meta_path = os.path.join(path, "library_meta.json")
    if not os.path.exists(meta_path):
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


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


def _normalize_text_for_id(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _build_doc_id(metadata: dict) -> str:
    source = str(metadata.get("source") or "")
    parent = str(metadata.get("parent") or "")
    doc_type = str(metadata.get("type") or "")
    page = str(metadata.get("page") or metadata.get("page_number") or "")
    name = str(metadata.get("name") or "")
    raw = "||".join([source, parent, doc_type, page, name])
    return f"doc_{sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def _build_citation_label(metadata: dict) -> str:
    source_name = str(metadata.get("name") or metadata.get("source") or "Unknown Source")
    page = metadata.get("page") or metadata.get("page_number")
    if page in (None, ""):
        return source_name
    return f"{source_name}, p. {page}"


def _ensure_chunk_metadata(doc: Document) -> None:
    metadata = doc.metadata
    metadata["doc_id"] = str(metadata.get("doc_id") or _build_doc_id(metadata))
    metadata["page_label"] = str(metadata.get("page_label") or metadata.get("page") or metadata.get("page_number") or "?")
    metadata["citation_label"] = str(metadata.get("citation_label") or _build_citation_label(metadata))
    if "chunk_index" not in metadata:
        metadata["chunk_index"] = 0
    if "chunk_total" not in metadata:
        metadata["chunk_total"] = 1
    if not metadata.get("chunk_id"):
        content_hash = sha1(_normalize_text_for_id(doc.page_content).encode("utf-8")).hexdigest()[:16]
        metadata["chunk_id"] = f"{metadata['doc_id']}::chunk::{int(metadata['chunk_index']):04d}::{content_hash}"


def get_llm(provider: str, api_key: str, base_url: str, model: str):
    normalized_base_url = _normalize_openai_compatible_base_url(base_url)
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
            base_url=normalized_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    if provider == "xAI":
        return ChatOpenAI(
            **common_params,
            base_url=normalized_base_url or "https://api.x.ai/v1",
        )
    if provider == "OpenAI" and _requires_responses_api_model(model):
        return ResponsesAPIModel(
            api_key=api_key,
            base_url=normalized_base_url or "https://api.openai.com/v1",
            model=model,
            temperature=0.2,
        )
    return ChatOpenAI(
        **common_params,
        base_url=normalized_base_url,
    )


def _normalize_openai_compatible_base_url(base_url: str) -> str:
    """LangChain expects the OpenAI-compatible API root, not a concrete endpoint."""
    normalized = str(base_url or "").strip().rstrip("/")
    for suffix in ("/chat/completions", "/responses", "/completions", "/embeddings"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized


def _requires_responses_api_model(model: str) -> bool:
    normalized = str(model or "").strip().lower()
    return normalized == "gpt-5.5" or normalized.startswith("gpt-5.5-")


@dataclass
class LLMResult:
    content: str


class ResponsesAPIModel:
    def __init__(self, api_key: str, base_url: str, model: str, temperature: float = 0.2, timeout: int = 240):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def invoke(self, prompt: str) -> LLMResult:
        response = requests.post(
            f"{self.base_url}/responses",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": [{"role": "user", "content": str(prompt)}],
            },
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise RuntimeError(_format_responses_error(response))

        try:
            payload = response.json()
        except Exception as exc:
            raise RuntimeError(f"Responses API 返回了非 JSON 内容：{response.text[:500]}") from exc

        content = _extract_responses_text(payload)
        if not content:
            raise RuntimeError(f"Responses API 未返回可解析文本：{str(payload)[:800]}")
        return LLMResult(content=content)


def _format_responses_error(response) -> str:
    try:
        payload = response.json()
    except Exception:
        payload = response.text[:1000]
    return f"Responses API 请求失败：HTTP {response.status_code} - {payload}"


def _extract_responses_text(response) -> str:
    if isinstance(response, dict):
        output_text = response.get("output_text")
        if output_text:
            return str(output_text)

        chunks: List[str] = []
        for output_item in response.get("output", []) or []:
            if not isinstance(output_item, dict):
                continue
            for content_item in output_item.get("content", []) or []:
                if not isinstance(content_item, dict):
                    continue
                text = content_item.get("text")
                if text:
                    chunks.append(str(text))
        if chunks:
            return "\n".join(chunks).strip()

        choices = response.get("choices", []) or []
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message") or {}
            content = message.get("content") if isinstance(message, dict) else None
            if content:
                return str(content)
        return ""

    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)

    chunks: List[str] = []
    for output_item in getattr(response, "output", []) or []:
        for content_item in getattr(output_item, "content", []) or []:
            text = getattr(content_item, "text", None)
            if text:
                chunks.append(str(text))
    if chunks:
        return "\n".join(chunks).strip()
    return str(response)


def invalidate_library_cache(path: str) -> None:
    with _CACHE_LOCK:
        _LIBRARY_CACHE.pop(os.path.abspath(path), None)


def preload_libraries(
    libraries: Sequence[dict],
    progress_callback: Optional[Callable[[str], None]] = None,
) -> List[LibraryIndex]:
    loaded_indexes: List[LibraryIndex] = []
    for library in libraries:
        if progress_callback:
            progress_callback(f"正在载入知识库：{library['name']}")
        loaded_indexes.append(_get_library_index(library["path"], library["name"]))
    return loaded_indexes


def query_rag(
    question: str,
    api_key: str,
    base_url: str,
    provider: str,
    model: str,
    primary_library: dict,
    secondary_libraries: Optional[Sequence[dict]] = None,
    answer_style: str = academic_prompt.STYLE_PHILOSOPHER,
    target_document: Optional[Dict[str, str]] = None,
    progress_callback: Optional[Callable[[str, str], None]] = None,
):
    if ModelManager.get_embeddings() is None:
        raise RuntimeError(ModelManager.get_last_error() or "模型仍在初始化中。")

    llm = get_llm(provider, api_key, base_url, model)
    selected_libraries = [primary_library] + list(secondary_libraries or [])
    retrieval_question = _build_retrieval_question(question, target_document, answer_style)
    target_mode = bool(target_document and academic_prompt.requires_target_document(answer_style))

    if target_mode:
        _emit(progress_callback, "expand", "正在按目标原文分段构造检索问题…")
        expansion = _build_target_document_expansion(target_document or {})
    else:
        _emit(progress_callback, "expand", "正在进行问题扩写…")
        expansion = _expand_question(retrieval_question, llm)

        _emit(progress_callback, "expand", "正在生成 HyDE 假设答案…")
        expansion.hyde_text = _generate_hyde(retrieval_question, expansion.expanded_question, llm)

    _emit(progress_callback, "retrieve", "正在预热已选知识库…")
    loaded_indexes = preload_libraries(selected_libraries)

    retrieval_target = academic_prompt.get_retrieval_target(answer_style)
    if target_mode:
        retrieval_target = max(retrieval_target, min(220, max(100, len(expansion.search_queries) * 8)))
    library_quotas = _build_library_quotas(primary_library, secondary_libraries or [], retrieval_target)
    indexes_by_path = {index.path: index for index in loaded_indexes}

    _emit(progress_callback, "retrieve", "正在执行多库三轮混合检索…")
    collected: List[Candidate] = []
    collected_keys: set[str] = set()
    forced_candidate_keys: set[str] = set()
    target_ref_ids = _extract_target_ref_ids(str((target_document or {}).get("text") or "")) if target_mode else set()

    for library in selected_libraries:
        quota = library_quotas.get(library["path"], 0)
        if quota <= 0:
            continue

        index = indexes_by_path.get(os.path.abspath(library["path"])) or _get_library_index(library["path"], library["name"])
        for candidate in _collect_refid_candidates(index, target_ref_ids, limit=max(8, quota)):
            if candidate.key in collected_keys:
                continue
            collected.append(candidate)
            collected_keys.add(candidate.key)
            forced_candidate_keys.add(candidate.key)

        candidate_target = _build_library_candidate_target(index, retrieval_target, quota)
        library_candidates = _retrieve_from_library(index, expansion, candidate_target)
        library_candidates = _rerank_candidates(
            retrieval_question,
            expansion,
            library_candidates,
            min(quota, len(library_candidates)),
        )
        for candidate in library_candidates:
            if candidate.key in collected_keys:
                continue
            collected.append(candidate)
            collected_keys.add(candidate.key)

    if not collected:
        return "当前所选知识库中未检索到相关材料。请调整问题、提高权重，或先完成该库的导入。", [], {"total_tokens": 0}

    _emit(progress_callback, "rerank", "正在使用同一 embedding 模型进行离线 rerank…")
    reranked = _rerank_candidates(retrieval_question, expansion, collected, retrieval_target)
    if forced_candidate_keys:
        forced_candidates = [candidate for candidate in collected if candidate.key in forced_candidate_keys]
        reranked = _dedupe_candidates(forced_candidates + reranked)[:retrieval_target]
    context_candidates = _compress_candidates_for_context(reranked, max_per_source=18 if target_mode else 3)

    context = _build_context(context_candidates)
    prompt = academic_prompt.build_prompt(
        question=question,
        context=context,
        answer_style=answer_style,
        target_document=target_document,
    )

    _emit(progress_callback, "generate", "正在生成学术回答…")
    result = llm.invoke(prompt)
    answer = result.content if hasattr(result, "content") else str(result)
    if academic_prompt.normalize_answer_style(answer_style) == academic_prompt.STYLE_CITE_PATCH and target_document:
        answer, context_candidates = _repair_cite_patch_answer(
            answer=answer,
            selected_libraries=selected_libraries,
            indexes_by_path=indexes_by_path,
            library_quotas=library_quotas,
            retrieval_target=retrieval_target,
            existing_context_candidates=context_candidates,
            llm=llm,
            progress_callback=progress_callback,
        )
        answer = _normalize_cite_patch_output(answer)
    else:
        answer = _normalize_cite_patch_output(answer)

    sources = list(
        dict.fromkeys(
            candidate.doc.metadata.get("source", "")
            for candidate in context_candidates
            if candidate.doc.metadata.get("source")
        )
    )
    token_usage = {
        "total_tokens": (len(prompt) + len(answer)) // 2,
        "references": _serialize_references(context_candidates),
    }
    return answer, sources, token_usage


def _build_retrieval_question(question: str, target_document: Optional[Dict[str, str]], answer_style: str) -> str:
    if not target_document or not academic_prompt.requires_target_document(answer_style):
        return question
    target_name = str(target_document.get("name") or "").strip()
    target_text = _extract_target_body_for_retrieval(str(target_document.get("text") or ""))
    excerpt = _normalize_text_for_id(target_text)[:9000]
    return (
        f"目标文章：{target_name}\n"
        f"请仅围绕以下目标文章内容检索相关证据、页码和可用引用来源：\n{excerpt}"
    ).strip()


def _build_target_document_expansion(target_document: Dict[str, str]) -> QuestionExpansion:
    target_name = str(target_document.get("name") or "").strip()
    body_text = _extract_target_body_for_retrieval(str(target_document.get("text") or ""))
    segments = _split_target_text_for_retrieval(body_text)
    expanded = f"围绕目标文章《{target_name or '未命名文档'}》中的具体论断检索可用于正文夹注、页码核验和审稿判断的材料。"
    keywords = _heuristic_keywords(body_text)[:12]
    return QuestionExpansion(
        expanded_question=expanded,
        search_queries=segments or [expanded],
        keywords=keywords,
        hyde_text="",
    )


def _extract_target_body_for_retrieval(text: str) -> str:
    body = str(text or "")
    body = re.sub(r"(?is)^\s*user\s+.*?\n\s*[^\n]{0,80}\n(?=##|\S)", "", body, count=1)
    for marker in ("## 脚注", "## 参考文献", "## 未能补注的位置", "参考来源："):
        if marker in body:
            body = body.split(marker, 1)[0]
    body = re.sub(r"(?m)^\s*RefID:.*$", "", body)
    return body.strip()


def _split_target_text_for_retrieval(text: str, max_segments: int = 36, segment_chars: int = 800) -> List[str]:
    paragraphs = [para.strip() for para in re.split(r"\n\s*\n+", str(text or "")) if para.strip()]
    segments: List[str] = []
    current = ""

    for paragraph in paragraphs:
        normalized = _normalize_text_for_id(paragraph)
        if not normalized:
            continue
        if len(normalized) > segment_chars:
            if current:
                segments.append(current)
                current = ""
            for start in range(0, len(normalized), segment_chars):
                chunk = normalized[start : start + segment_chars].strip()
                if chunk:
                    segments.append(chunk)
                if len(segments) >= max_segments:
                    return segments
            continue
        if current and len(current) + len(normalized) + 1 > segment_chars:
            segments.append(current)
            current = normalized
        else:
            current = f"{current}\n{normalized}".strip() if current else normalized
        if len(segments) >= max_segments:
            return segments

    if current and len(segments) < max_segments:
        segments.append(current)
    return segments[:max_segments]


def _extract_target_ref_ids(text: str) -> set[str]:
    ref_ids: set[str] = set()
    for match in re.finditer(r"RefID\s*[:：]\s*([^\n。]+)", str(text or ""), re.I):
        raw = match.group(1)
        for ref_id in re.findall(r"doc_[A-Za-z0-9]+(?:::[A-Za-z]+::[0-9]{4}(?:::[A-Za-z0-9]+)?)?", raw):
            ref_ids.add(ref_id)
    return ref_ids


def _collect_refid_candidates(index: LibraryIndex, ref_ids: set[str], limit: int) -> List[Candidate]:
    if not ref_ids:
        return []
    collected: List[Candidate] = []
    seen: set[str] = set()
    for doc in index.chunks:
        _ensure_chunk_metadata(doc)
        chunk_id = str(doc.metadata.get("chunk_id") or "")
        doc_id = str(doc.metadata.get("doc_id") or "")
        matched = False
        for ref_id in ref_ids:
            if ref_id == chunk_id or ref_id == doc_id or chunk_id.startswith(f"{ref_id}::"):
                matched = True
                break
        if not matched:
            continue
        key = _candidate_key(index.path, doc)
        if key in seen:
            continue
        seen.add(key)
        collected.append(
            Candidate(
                key=key,
                doc=doc,
                library_name=index.name,
                library_path=index.path,
                weight=_lookup_weight(doc, index.weights),
                score=20.0,
            )
        )
        if len(collected) >= limit:
            break
    return collected


def _repair_cite_patch_answer(
    answer: str,
    selected_libraries: Sequence[dict],
    indexes_by_path: Dict[str, LibraryIndex],
    library_quotas: Dict[str, int],
    retrieval_target: int,
    existing_context_candidates: Sequence[Candidate],
    llm,
    progress_callback: Optional[Callable[[str, str], None]] = None,
) -> Tuple[str, List[Candidate]]:
    current_answer = answer
    context_candidates = list(existing_context_candidates)
    context_keys = {candidate.key for candidate in context_candidates}

    for round_index in range(1, CITE_PATCH_REPAIR_ROUNDS + 1):
        unresolved_snippets = _extract_unresolved_cite_snippets(current_answer)
        if not unresolved_snippets:
            break

        _emit(
            progress_callback,
            "retrieve",
            f"正在进行第 {round_index} 轮补注修补检索，待补位置 {len(unresolved_snippets)} 处…",
        )
        expansion = _build_unresolved_snippet_expansion(unresolved_snippets)
        additional_candidates = _retrieve_additional_candidates_for_expansion(
            selected_libraries=selected_libraries,
            indexes_by_path=indexes_by_path,
            library_quotas=library_quotas,
            retrieval_target=retrieval_target,
            expansion=expansion,
        )
        new_candidates = [candidate for candidate in additional_candidates if candidate.key not in context_keys]
        if not new_candidates:
            break

        context_candidates = _dedupe_candidates(list(context_candidates) + new_candidates)
        context_candidates = _compress_candidates_for_context(context_candidates, max_per_source=24)
        context_keys = {candidate.key for candidate in context_candidates}
        additional_context = _build_context(new_candidates[: min(len(new_candidates), 48)])

        _emit(progress_callback, "generate", f"正在进行第 {round_index} 轮补注修补生成…")
        repair_prompt = academic_prompt.build_cite_patch_repair_prompt(
            previous_answer=current_answer,
            unresolved_snippets=unresolved_snippets,
            additional_context=additional_context,
        )
        repaired = llm.invoke(repair_prompt)
        current_answer = repaired.content if hasattr(repaired, "content") else str(repaired)

    return current_answer, context_candidates


def _extract_unresolved_cite_snippets(answer: str, limit: int = 10) -> List[str]:
    text = str(answer or "")
    if "## 未能补注的位置" not in text:
        return []
    section = text.split("## 未能补注的位置", 1)[1]
    if "\n## " in section:
        section = section.split("\n## ", 1)[0]
    if re.search(r"(无|没有|全部补齐|无需补注)", section) and len(section.strip()) < 80:
        return []

    snippets: List[str] = []
    for line in section.splitlines():
        cleaned = re.sub(r"^\s*(?:[-*]|\d+[\.、])\s*", "", line).strip()
        if not cleaned:
            continue
        quoted = re.findall(r"[“\"']([^”\"']{8,260})[”\"']", cleaned)
        if quoted:
            snippets.extend(quoted)
        else:
            cleaned = re.sub(r"原因[:：].*$", "", cleaned).strip()
            snippets.append(cleaned[:260])
        if len(snippets) >= limit:
            break
    return _dedupe_preserve_order(snippets)[:limit]


def _build_unresolved_snippet_expansion(snippets: Sequence[str]) -> QuestionExpansion:
    query_text = "\n".join(_normalize_text_for_id(snippet) for snippet in snippets if str(snippet).strip())
    return QuestionExpansion(
        expanded_question=f"为以下尚未补注的原文片段检索可用引用、页码和来源：\n{query_text}",
        search_queries=[_normalize_text_for_id(snippet) for snippet in snippets if str(snippet).strip()],
        keywords=_heuristic_keywords(query_text)[:12],
        hyde_text="",
    )


def _retrieve_additional_candidates_for_expansion(
    selected_libraries: Sequence[dict],
    indexes_by_path: Dict[str, LibraryIndex],
    library_quotas: Dict[str, int],
    retrieval_target: int,
    expansion: QuestionExpansion,
) -> List[Candidate]:
    collected: List[Candidate] = []
    collected_keys: set[str] = set()
    retrieval_query = "\n".join([expansion.expanded_question] + expansion.search_queries)

    for library in selected_libraries:
        quota = library_quotas.get(library["path"], 0)
        if quota <= 0:
            continue
        index = indexes_by_path.get(os.path.abspath(library["path"])) or _get_library_index(library["path"], library["name"])
        candidate_target = min(
            max(len(index.chunks), 1),
            max(quota * 2, len(expansion.search_queries) * 12, 40),
        )
        candidates = _retrieve_from_library(index, expansion, candidate_target)
        candidates = _rerank_candidates(
            retrieval_query,
            expansion,
            candidates,
            min(max(quota, len(expansion.search_queries) * 8), retrieval_target),
        )
        for candidate in candidates:
            if candidate.key in collected_keys:
                continue
            collected.append(candidate)
            collected_keys.add(candidate.key)

    return _rerank_candidates(retrieval_query, expansion, collected, min(len(collected), retrieval_target)) if collected else []


def _normalize_cite_patch_output(answer: str) -> str:
    text = str(answer or "").replace("\r\n", "\n").strip()
    reference_match = re.search(r"(?m)^##\s*(?:脚注|参考文献)\s*$", text)
    if not reference_match:
        return re.sub(r"\[(\d+)\]", "", _strip_refids_from_text(text)).strip()

    body_part = text[: reference_match.start()]
    reference_tail = text[reference_match.end() :]
    reference_part = reference_tail
    unresolved_part = ""
    unresolved_match = re.search(r"(?m)^##\s*未能补注的位置\s*$", reference_part)
    if unresolved_match:
        unresolved_part = reference_part[unresolved_match.end() :].strip()
        reference_part = reference_part[: unresolved_match.start()]

    numbered_references = _parse_footnote_block(reference_part)
    ordered_references: List[str] = []
    seen_references: set[str] = set()

    def remember_reference(reference_text: str) -> str:
        cleaned_reference = _strip_reference_number(_strip_refids_from_text(reference_text))
        if not cleaned_reference:
            cleaned_reference = "出处缺失，需人工核验。"
        reference_key = _normalize_text_for_id(cleaned_reference).lower()
        if reference_key not in seen_references:
            ordered_references.append(cleaned_reference)
            seen_references.add(reference_key)
        return cleaned_reference

    def replace_marker(match):
        old_number = match.group(1)
        reference_text = numbered_references.get(old_number, "")
        inline_reference = remember_reference(reference_text)
        return f"（{inline_reference}）"

    normalized_body = re.sub(r"\[(\d+)\]", replace_marker, body_part)
    normalized_body = _strip_refids_from_text(normalized_body)

    if not numbered_references:
        for reference_text in _parse_reference_lines(reference_part):
            remember_reference(reference_text)
    else:
        for number in sorted(numbered_references, key=lambda item: int(item) if item.isdigit() else item):
            reference_text = numbered_references[number]
            reference_key = _normalize_text_for_id(_strip_reference_number(_strip_refids_from_text(reference_text))).lower()
            if reference_key not in seen_references:
                remember_reference(reference_text)

    output_parts = [normalized_body]
    if ordered_references:
        output_parts.append("## 参考文献\n" + "\n".join(ordered_references))
    if unresolved_part:
        output_parts.append("## 未能补注的位置\n" + _strip_refids_from_text(unresolved_part))
    return "\n\n".join(part for part in output_parts if part.strip()).strip()


def _parse_footnote_block(footnote_text: str) -> Dict[str, str]:
    footnotes: Dict[str, str] = {}
    matches = list(re.finditer(r"(?m)^\s*\[(\d+)\]\s*", str(footnote_text or "")))
    for index, match in enumerate(matches):
        number = match.group(1)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(footnote_text)
        footnotes[number] = footnote_text[start:end].strip()
    return footnotes


def _parse_reference_lines(reference_text: str) -> List[str]:
    references: List[str] = []
    for line in str(reference_text or "").splitlines():
        cleaned = _strip_reference_number(_strip_refids_from_text(line))
        if cleaned:
            references.append(cleaned)
    return references


def _strip_reference_number(text: str) -> str:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^\s*(?:[-*]|\d+[\.、])\s*", "", cleaned)
    cleaned = re.sub(r"^\s*\[\d+\]\s*", "", cleaned)
    return cleaned.strip()


def _strip_refids_from_text(text: str) -> str:
    cleaned = str(text or "")
    cleaned = re.sub(r"\s*RefID\s*[:：]\s*[^\n。；;]*[。；;]?", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*(?:DocID|ChunkID)\s*[:：]\s*[^\n。；;]*[。；;]?", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*\(RefID\s*[:：][^)]+\)", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*[\(（\[]\s*doc_[A-Za-z0-9_:.-]+\s*[\)）\]]", "", cleaned)
    cleaned = re.sub(r"\s*[\(（\[]\s*chunk[_:][A-Za-z0-9_:.-]+\s*[\)）\]]", "", cleaned, flags=re.I)
    cleaned = "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in cleaned.splitlines())
    return cleaned.strip()


def _dedupe_candidates(candidates: Sequence[Candidate]) -> List[Candidate]:
    deduped: List[Candidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.key in seen:
            continue
        deduped.append(candidate)
        seen.add(candidate.key)
    return deduped


def _get_library_index(path: str, name: str) -> LibraryIndex:
    abs_path = os.path.abspath(path)
    with _CACHE_LOCK:
        cached = _LIBRARY_CACHE.get(abs_path)
        if cached is not None:
            return cached

    pkl_path = os.path.join(abs_path, "chunks.pkl")
    if not os.path.exists(pkl_path):
        raise RuntimeError(f"知识库 {name} 尚未完成导入：{pkl_path} 不存在。")

    try:
        with open(pkl_path, "rb") as handle:
            chunks = pickle.load(handle)
    except Exception as exc:
        raise RuntimeError(f"知识库 {name} 的 chunks.pkl 无法读取，请重新导入或重建该库：{exc}") from exc

    if not chunks:
        raise RuntimeError(f"知识库 {name} 为空，请先导入文档。")

    library_meta = _read_library_meta(abs_path)
    stored_embedding_model = str(library_meta.get("embedding_model") or "").strip()
    current_embedding_model = str(ModelManager.EMBEDDING_MODEL_NAME or "").strip()
    if stored_embedding_model and current_embedding_model and stored_embedding_model != current_embedding_model:
        raise RuntimeError(
            f"知识库 {name} 当前仍使用旧的向量模型 [{stored_embedding_model}]，"
            f"而程序已经切换为 [{current_embedding_model}]。"
            "请在文件管理页点击“重建索引”，或重新导入该知识库，以完成多语言向量升级。"
        )

    weights = _load_weights(abs_path)
    sources_by_weight = {1: set(), 2: set(), 3: set()}
    tokenized_corpus: List[List[str]] = []

    for doc in chunks:
        _ensure_chunk_metadata(doc)
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
3. `keywords`：给出 6 到 10 个检索关键词，尽量覆盖中文、英文、法文或材料中实际出现的其他语言术语。

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
请围绕下面的问题写一段 150 到 220 字的假设性学术回答，用于检索增强，不要写引用标注，也不要解释任务。

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


def _retrieve_from_library(index: LibraryIndex, expansion: QuestionExpansion, candidate_target: int) -> List[Candidate]:
    round_targets = _split_round_targets(candidate_target)
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
            if len(collected) >= candidate_target:
                return collected

    if len(collected) < candidate_target:
        fallback = _retrieve_round(index, expansion, {1, 2, 3}, candidate_target)
        for candidate in fallback:
            if candidate.key in collected_keys:
                continue
            collected.append(candidate)
            collected_keys.add(candidate.key)
            if len(collected) >= candidate_target:
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
            *expansion.search_queries[:12],
            expansion.hyde_text,
        ]
    )

    ranked_lists: List[Tuple[List[Candidate], float]] = []
    for idx, dense_query in enumerate(query for query in dense_queries if query):
        try:
            docs_with_scores = index.vector_store.similarity_search_with_score(
                dense_query,
                k=raw_k,
                filter={"source": {"$in": sorted(allowed_sources)}},
            )
        except Exception as exc:
            error_text = str(exc)
            if "Error finding id" in error_text or "Internal error" in error_text:
                raise RuntimeError(
                    f"知识库 [{index.name}] 的向量索引与 chunks.pkl 可能不一致，请在文件管理页点击“重建索引”后重试。原始错误：{error_text}"
                ) from exc
            raise
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


def _compute_lexical_overlap(query_tokens: set[str], doc_text: str) -> float:
    doc_tokens = set(_tokenize_text(doc_text))
    if not query_tokens or not doc_tokens:
        return 0.0
    shared_tokens = query_tokens.intersection(doc_tokens)
    return len(shared_tokens) / max(len(query_tokens), 1)


def _compress_candidates_for_context(candidates: Sequence[Candidate], max_per_source: int = 3) -> List[Candidate]:
    compressed: List[Candidate] = []
    seen_excerpt_keys: set[str] = set()
    source_counts: Dict[str, int] = {}

    for candidate in candidates:
        doc = candidate.doc
        _ensure_chunk_metadata(doc)
        parent = str(doc.metadata.get("parent") or "")
        source = str(doc.metadata.get("source") or "")
        source_group = parent or source
        excerpt_key = _normalize_excerpt(doc.page_content, limit=220).lower()
        if excerpt_key in seen_excerpt_keys:
            continue
        if source_counts.get(source_group, 0) >= max_per_source:
            continue

        compressed.append(candidate)
        seen_excerpt_keys.add(excerpt_key)
        source_counts[source_group] = source_counts.get(source_group, 0) + 1

    return compressed


def _serialize_references(candidates: Sequence[Candidate]) -> List[Dict[str, str]]:
    serialized: List[Dict[str, str]] = []
    seen_chunk_ids: set[str] = set()

    for candidate in candidates:
        doc = candidate.doc
        _ensure_chunk_metadata(doc)
        chunk_id = str(doc.metadata.get("chunk_id") or "")
        if chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)
        serialized.append(
            {
                "ref_id": chunk_id,
                "library_name": str(candidate.library_name),
                "source": str(doc.metadata.get("source") or ""),
                "name": str(doc.metadata.get("name") or ""),
                "type": str(doc.metadata.get("type") or ""),
                "parent": str(doc.metadata.get("parent") or ""),
                "page": str(doc.metadata.get("page_label") or doc.metadata.get("page") or doc.metadata.get("page_number") or ""),
                "citation_label": str(doc.metadata.get("citation_label") or ""),
            }
        )

    return serialized


def _rerank_candidates(
    question: str,
    expansion: QuestionExpansion,
    candidates: Sequence[Candidate],
    retrieval_target: int,
) -> List[Candidate]:
    query_texts = _dedupe_preserve_order(
        [question, expansion.expanded_question, expansion.hyde_text] + list(expansion.search_queries[:36])
    )
    query_vectors = ModelManager.encode_texts(query_texts or [question])
    query_tokens = set(_tokenize_text("\n".join(query_texts)))
    doc_vectors = ModelManager.encode_texts(
        [_build_rerank_text(candidate.doc) for candidate in candidates],
        batch_size=24,
    )
    score_matrix = doc_vectors @ query_vectors.T
    scores = np.max(score_matrix, axis=1)

    reranked = list(candidates)
    for candidate, score in zip(reranked, scores.tolist()):
        lexical_overlap = _compute_lexical_overlap(query_tokens, candidate.doc.page_content)
        weight_bonus = candidate.weight * 0.03
        candidate.score = float(score) + candidate.score * 0.1 + lexical_overlap * 0.2 + weight_bonus

    reranked.sort(key=lambda item: item.score, reverse=True)
    return reranked[:retrieval_target]


def _build_context(candidates: Sequence[Candidate]) -> str:
    blocks: List[str] = []
    for idx, candidate in enumerate(candidates, start=1):
        doc = candidate.doc
        _ensure_chunk_metadata(doc)
        source_path = str(doc.metadata.get("source", "未知来源"))
        source_name = str(doc.metadata.get("name") or os.path.basename(source_path) or source_path)
        page = str(doc.metadata.get("page_label") or doc.metadata.get("page") or doc.metadata.get("page_number") or "?")
        parent_url = str(doc.metadata.get("parent") or "")
        chunk_id = str(doc.metadata.get("chunk_id") or "")
        citation_label = str(doc.metadata.get("citation_label") or source_name)
        chunk_index = int(doc.metadata.get("chunk_index") or 0) + 1
        chunk_total = int(doc.metadata.get("chunk_total") or 1)
        excerpt = _normalize_excerpt(doc.page_content, limit=320)
        blocks.append(
            "\n".join(
                [
                    f"[材料 {idx}]",
                    f"RefID: {chunk_id}",
                    f"Library: {candidate.library_name}",
                    f"Type: {doc.metadata.get('type', 'UNKNOWN')}",
                    f"Cite as: ({citation_label})",
                    f"Source: {source_path}",
                    f"Parent URL: {parent_url or '-'}",
                    f"Page: {page}",
                    f"Chunk: {chunk_index}/{chunk_total}",
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


def _build_library_candidate_target(index: LibraryIndex, retrieval_target: int, quota: int) -> int:
    total_chunks = max(1, len(index.chunks))
    base_target = max(retrieval_target * 3, quota * 4, 60)
    return min(total_chunks, base_target)


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
    latin_tokens = re.findall(r"[a-z0-9_à-öø-ÿœæ]+", lowered)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", lowered)
    cjk_bigrams = ["".join(cjk_chars[idx : idx + 2]) for idx in range(len(cjk_chars) - 1)]
    return [token for token in latin_tokens + cjk_chars + cjk_bigrams if token.strip()]


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("未找到 JSON。")
    return json.loads(match.group(0))


def _candidate_key(library_path: str, doc: Document) -> str:
    _ensure_chunk_metadata(doc)
    chunk_id = str(doc.metadata.get("chunk_id") or "")
    if chunk_id:
        return f"{library_path}::{chunk_id}"
    source = str(doc.metadata.get("source", ""))
    page = str(doc.metadata.get("page") or doc.metadata.get("page_number") or "")
    snippet = _normalize_excerpt(doc.page_content, limit=120)
    stable_hash = sha1(snippet.encode("utf-8")).hexdigest()[:16]
    return f"{library_path}::{source}::{page}::{stable_hash}"


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
