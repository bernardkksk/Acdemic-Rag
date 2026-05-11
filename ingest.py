import os
import pickle
import re
import json
import io
import ctypes
import inspect
import logging
import threading
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from hashlib import sha1
from typing import Dict, List, Optional, Tuple

import numpy as np
from langchain_chroma import Chroma
from langchain_core.documents import Document

PDF_STATUS_FILE = "ingest_status.json"
OCR_CACHE_DIR = ".ocr_cache"
PDF_SAMPLE_PAGES = 3
PDF_TEXT_THRESHOLD = 80
OCR_DPI = 150
OCR_MAX_WORKERS = 4
_OCR_ENGINE = None
_OCR_ENGINE_LOCK = threading.Lock()
_OCR_CALL_LOCK = threading.Lock()
_OCR_DLL_PATHS_READY = False
_OCR_DLL_HANDLES = []


@contextmanager
def _suppress_pdf_console_noise():
    suppressed_stdout = io.StringIO()
    suppressed_stderr = io.StringIO()
    pypdf_logger = logging.getLogger("pypdf")
    pdfminer_logger = logging.getLogger("pdfminer")
    previous_pypdf_level = pypdf_logger.level
    previous_pdfminer_level = pdfminer_logger.level
    pypdf_logger.setLevel(logging.CRITICAL)
    pdfminer_logger.setLevel(logging.CRITICAL)
    try:
        with redirect_stdout(suppressed_stdout), redirect_stderr(suppressed_stderr):
            yield
    finally:
        pypdf_logger.setLevel(previous_pypdf_level)
        pdfminer_logger.setLevel(previous_pdfminer_level)


def detect_language(text: str) -> str:
    if not text:
        return "CN"
    sample = str(text or "")[:1500].lower()
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", sample)
    if len(cjk_chars) >= max(10, len(sample) * 0.08):
        return "CN"

    french_markers = re.findall(r"[àâæçéèêëîïôœùûüÿ]", sample)
    french_words = re.findall(
        r"\b(le|la|les|de|des|du|une|un|et|est|dans|pour|avec|sur|par|que|qui|ce|cette|ces|ainsi|comme|mais|ou|donc|car)\b",
        sample,
    )
    english_words = re.findall(
        r"\b(the|and|is|are|in|on|for|with|that|this|these|those|from|into|about|because|therefore|however)\b",
        sample,
    )
    latin_words = re.findall(r"\b[a-zà-öø-ÿœæ]+\b", sample)

    if french_markers or (len(french_words) >= 3 and len(french_words) >= len(english_words)):
        return "FR"
    if latin_words and len(english_words) >= 2:
        return "EN"
    if latin_words:
        return "LATIN"
    return "CN"


def _get_model_manager():
    from model import ModelManager

    return ModelManager


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


def _assign_chunk_metadata(chunks) -> None:
    chunk_totals = defaultdict(int)
    chunk_counters = defaultdict(int)

    for chunk in chunks:
        chunk.metadata["doc_id"] = str(chunk.metadata.get("doc_id") or _build_doc_id(chunk.metadata))
        chunk_totals[chunk.metadata["doc_id"]] += 1

    for chunk in chunks:
        metadata = chunk.metadata
        doc_id = metadata["doc_id"]
        chunk_index = chunk_counters[doc_id]
        chunk_counters[doc_id] += 1
        content_hash = sha1(_normalize_text_for_id(chunk.page_content).encode("utf-8")).hexdigest()[:16]
        metadata["chunk_index"] = chunk_index
        metadata["chunk_total"] = chunk_totals[doc_id]
        metadata["page_label"] = str(metadata.get("page") or metadata.get("page_number") or "?")
        metadata["citation_label"] = _build_citation_label(metadata)
        metadata["chunk_id"] = f"{doc_id}::chunk::{chunk_index:04d}::{content_hash}"


def _dedupe_chunks_by_id(chunks):
    deduped = []
    seen_chunk_ids = set()
    for chunk in chunks:
        chunk_id = str(chunk.metadata.get("chunk_id") or "")
        if not chunk_id or chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)
        deduped.append(chunk)
    return deduped


def _write_library_meta(db_path: str) -> None:
    ModelManager = _get_model_manager()
    meta_path = os.path.join(db_path, "library_meta.json")
    payload = {
        "embedding_model": ModelManager.EMBEDDING_MODEL_NAME,
        "embedding_device": ModelManager.get_device(),
    }
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _read_library_meta(db_path: str) -> dict:
    meta_path = os.path.join(db_path, "library_meta.json")
    if not os.path.exists(meta_path):
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def get_ingest_status_path(db_path: str) -> str:
    return os.path.join(db_path, PDF_STATUS_FILE)


def load_ingest_status(db_path: str) -> Dict[str, dict]:
    path = get_ingest_status_path(db_path)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def save_ingest_status(db_path: str, payload: Dict[str, dict]) -> None:
    with open(get_ingest_status_path(db_path), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def remove_ingest_status_entries(db_path: str, sources: List[str]) -> None:
    manifest = load_ingest_status(db_path)
    changed = False
    for source in sources:
        if source in manifest:
            manifest.pop(source, None)
            changed = True
    if changed:
        save_ingest_status(db_path, manifest)


def _raise_if_embedding_model_mismatch(db_path: str, mode: str, has_existing_chunks: bool) -> None:
    if mode == "new" or not has_existing_chunks:
        return
    ModelManager = _get_model_manager()
    library_meta = _read_library_meta(db_path)
    stored_embedding_model = str(library_meta.get("embedding_model") or "").strip()
    current_embedding_model = str(ModelManager.EMBEDDING_MODEL_NAME or "").strip()
    if stored_embedding_model and current_embedding_model and stored_embedding_model != current_embedding_model:
        raise RuntimeError(
            f"当前知识库仍绑定旧的向量模型 [{stored_embedding_model}]，"
            f"而程序现在使用的是 [{current_embedding_model}]。"
            "请改用“清空当前主库后重建”，或先点击“重建索引”完成升级后再追加导入。"
        )


def _pdf_cache_root(db_path: str) -> str:
    root = os.path.join(db_path, OCR_CACHE_DIR)
    os.makedirs(root, exist_ok=True)
    return root


def _file_fingerprint(file_path: str) -> str:
    stat = os.stat(file_path)
    raw = f"{os.path.abspath(file_path)}|{stat.st_size}|{int(stat.st_mtime)}"
    return sha1(raw.encode("utf-8")).hexdigest()[:20]


def _pdf_ocr_cache_dir(db_path: str, file_path: str) -> str:
    cache_dir = os.path.join(_pdf_cache_root(db_path), _file_fingerprint(file_path))
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _page_cache_path(db_path: str, file_path: str, page_number: int) -> str:
    return os.path.join(_pdf_ocr_cache_dir(db_path, file_path), f"page_{page_number:04d}.json")


def _repaired_pdf_path(db_path: str, file_path: str, variant: str) -> str:
    return os.path.join(_pdf_ocr_cache_dir(db_path, file_path), f"repaired_{variant}.pdf")


def _make_pdf_status(
    file_path: str,
    file_name: str,
    pdf_type: str,
    status: str,
    *,
    total_pages: int = 0,
    ocr_pages: int = 0,
    failed_pages: int = 0,
    skipped_pages: int = 0,
    error_reason: str = "",
) -> dict:
    return {
        "source": file_path,
        "name": file_name,
        "type": "PDF",
        "lang": "CN",
        "pdf_type": pdf_type,
        "status": status,
        "ocr_pages": int(ocr_pages),
        "failed_pages": int(failed_pages),
        "skipped_pages": int(skipped_pages),
        "total_pages": int(total_pages),
        "error_reason": str(error_reason or ""),
        "skip_reason": "",
    }


def _normalize_page_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _is_low_value_text(text: str) -> Tuple[bool, str]:
    normalized = _normalize_page_text(text)
    if len(normalized) < 4:
        return True, "blank"
    dotted = normalized.count("....") + normalized.count("……")
    digit_lines = len(re.findall(r"\b\d{1,4}\b", normalized))
    if dotted >= 2 or (dotted >= 1 and digit_lines >= 6):
        return True, "toc"
    lowered = normalized.lower()
    if any(keyword in lowered for keyword in ("isbn", "版权所有", "版权", "publisher", "published by", "出版社")):
        return True, "copyright"
    return False, ""


def _is_low_value_image(image) -> Tuple[bool, str]:
    from PIL import ImageStat

    grayscale = image.convert("L")
    stat = ImageStat.Stat(grayscale)
    if stat.stddev and stat.stddev[0] < 3:
        return True, "blank"
    if stat.mean and stat.mean[0] > 250:
        return True, "blank"
    return False, ""


def _format_reason_counter(reason_counter: Dict[str, int], label_map: Dict[str, str]) -> str:
    parts = []
    for key, count in sorted(reason_counter.items(), key=lambda item: (-item[1], item[0])):
        if count <= 0:
            continue
        parts.append(f"{label_map.get(key, key)} {count}页")
    return "，".join(parts)


def _configure_windows_gpu_dll_paths() -> None:
    global _OCR_DLL_PATHS_READY, _OCR_DLL_HANDLES

    if _OCR_DLL_PATHS_READY or os.name != "nt":
        return

    site_packages = os.path.join(sys.prefix, "Lib", "site-packages")
    candidate_dirs = []
    nvidia_root = os.path.join(site_packages, "nvidia")
    if os.path.isdir(nvidia_root):
        for name in sorted(os.listdir(nvidia_root)):
            bin_dir = os.path.join(nvidia_root, name, "bin")
            if os.path.isdir(bin_dir):
                candidate_dirs.append(bin_dir)

    paddle_libs = os.path.join(site_packages, "paddle", "libs")
    if os.path.isdir(paddle_libs):
        candidate_dirs.append(paddle_libs)

    normalized_existing = {
        os.path.normcase(os.path.normpath(path))
        for path in os.environ.get("PATH", "").split(os.pathsep)
        if path
    }
    prepend_paths = []
    for directory in candidate_dirs:
        normalized = os.path.normcase(os.path.normpath(directory))
        if normalized not in normalized_existing:
            prepend_paths.append(directory)
        try:
            os.add_dll_directory(directory)
        except Exception:
            pass

    if prepend_paths:
        os.environ["PATH"] = os.pathsep.join(prepend_paths + [os.environ.get("PATH", "")])

    preload_relpaths = [
        ("cuda_runtime", "bin", "cudart64_12.dll"),
        ("nvjitlink", "bin", "nvJitLink_120_0.dll"),
        ("cublas", "bin", "cublasLt64_12.dll"),
        ("cublas", "bin", "cublas64_12.dll"),
        ("cusparse", "bin", "cusparse64_12.dll"),
        ("cusolver", "bin", "cusolver64_11.dll"),
        ("curand", "bin", "curand64_10.dll"),
        ("cufft", "bin", "cufft64_11.dll"),
        ("cudnn", "bin", "cudnn64_9.dll"),
    ]
    for parts in preload_relpaths:
        dll_path = os.path.join(nvidia_root, *parts)
        if not os.path.exists(dll_path):
            continue
        try:
            _OCR_DLL_HANDLES.append(ctypes.CDLL(dll_path))
        except Exception:
            continue

    _OCR_DLL_PATHS_READY = True


def _paddle_gpu_available(paddle_module=None) -> bool:
    try:
        paddle = paddle_module
        if paddle is None:
            import paddle as paddle_module_imported

            paddle = paddle_module_imported

        is_compiled = bool(getattr(paddle.device, "is_compiled_with_cuda", lambda: False)())
        device_count_getter = getattr(getattr(paddle.device, "cuda", None), "device_count", None)
        device_count = int(device_count_getter()) if callable(device_count_getter) else 0
        return is_compiled and device_count > 0
    except Exception:
        return False


def _build_ocr_init_variants(paddle_module=None) -> List[dict]:
    preferred_device = str(os.environ.get("ACADEMIC_RAG_OCR_DEVICE", "") or "").strip().lower()
    use_gpu = preferred_device in {"gpu", "cuda", "1", "true", "yes"}
    if not preferred_device:
        use_gpu = _paddle_gpu_available(paddle_module)

    variants: List[dict] = []
    if use_gpu:
        variants.extend(
            [
                {"use_angle_cls": False, "lang": "ch", "device": "gpu"},
                {"use_angle_cls": False, "lang": "ch", "use_gpu": True},
                {"lang": "ch", "device": "gpu"},
                {"lang": "ch", "use_gpu": True},
            ]
        )

    variants.extend(
        [
            {"use_angle_cls": False, "lang": "ch", "device": "cpu"},
            {"use_angle_cls": False, "lang": "ch", "use_gpu": False},
            {"lang": "ch", "device": "cpu"},
            {"lang": "ch", "use_gpu": False},
        ]
    )
    return variants


def _filter_kwargs_for_callable(func, kwargs: dict) -> dict:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return dict(kwargs)

    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return dict(kwargs)

    allowed_names = set(signature.parameters.keys())
    return {key: value for key, value in kwargs.items() if key in allowed_names}


def _retry_paddle_ocr_init(PaddleOCR, kwargs: dict):
    pending_kwargs = dict(kwargs)
    last_error = None
    for _ in range(max(1, len(kwargs) + 1)):
        try:
            with _suppress_pdf_console_noise():
                return PaddleOCR(**pending_kwargs)
        except Exception as exc:
            last_error = exc
            match = re.search(r"Unknown argument:\s*([A-Za-z_][A-Za-z0-9_]*)", str(exc))
            if not match:
                break
            unknown_key = match.group(1)
            if unknown_key not in pending_kwargs:
                break
            pending_kwargs.pop(unknown_key, None)
    raise last_error or RuntimeError("PaddleOCR 初始化失败。")


def _get_paddle_ocr():
    global _OCR_ENGINE

    if _OCR_ENGINE is None:
        with _OCR_ENGINE_LOCK:
            if _OCR_ENGINE is not None:
                return _OCR_ENGINE

            _configure_windows_gpu_dll_paths()
            import paddle
            from paddleocr import PaddleOCR

            init_variants = _build_ocr_init_variants(paddle)
            last_error = None
            engine = None
            for kwargs in init_variants:
                try:
                    filtered_kwargs = _filter_kwargs_for_callable(PaddleOCR.__init__, kwargs)
                    engine = _retry_paddle_ocr_init(PaddleOCR, filtered_kwargs)
                    break
                except TypeError as exc:
                    last_error = exc
                    continue
                except Exception as exc:
                    last_error = exc
                    continue
            if engine is None:
                raise RuntimeError(last_error or "PaddleOCR 初始化失败。")
            _OCR_ENGINE = engine

    return _OCR_ENGINE


def _ensure_ocr_available() -> None:
    try:
        _get_paddle_ocr()
    except Exception as exc:
        raise RuntimeError(
            "PaddleOCR 初始化失败。请确认已安装 paddleocr 及其依赖（通常还需要 paddlepaddle），"
            f"原始错误：{exc}"
        ) from exc


def _supports_keyword(func, keyword: str) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return False

    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == keyword:
            return True
    return False


def _call_ocr_engine(ocr, ocr_input):
    last_error = None

    with _OCR_CALL_LOCK:
        if hasattr(ocr, "ocr"):
            ocr_kwargs = {}
            if _supports_keyword(ocr.ocr, "cls"):
                ocr_kwargs["cls"] = False
            elif _supports_keyword(ocr.ocr, "use_cls"):
                ocr_kwargs["use_cls"] = False
            try:
                return ocr.ocr(ocr_input, **ocr_kwargs)
            except Exception as exc:
                last_error = exc

        if hasattr(ocr, "predict"):
            predict_kwargs = {}
            if _supports_keyword(ocr.predict, "cls"):
                predict_kwargs["cls"] = False
            elif _supports_keyword(ocr.predict, "use_cls"):
                predict_kwargs["use_cls"] = False
            try:
                return ocr.predict(ocr_input, **predict_kwargs)
            except Exception as exc:
                last_error = exc

    raise RuntimeError(last_error or "PaddleOCR 调用失败。")


def _strip_pdf_leading_garbage(file_path: str, db_path: str) -> Optional[str]:
    repaired_path = _repaired_pdf_path(db_path, file_path, "header")
    if os.path.exists(repaired_path) and os.path.getsize(repaired_path) > 0:
        return repaired_path
    try:
        with open(file_path, "rb") as handle:
            raw = handle.read()
        marker = raw.find(b"%PDF")
        if marker <= 0:
            return None
        cleaned = raw[marker:]
        with open(repaired_path, "wb") as handle:
            handle.write(cleaned)
        return repaired_path
    except Exception:
        return None


def _rewrite_pdf_with_pypdf(file_path: str, db_path: str) -> Optional[str]:
    repaired_path = _repaired_pdf_path(db_path, file_path, "rewrite")
    if os.path.exists(repaired_path) and os.path.getsize(repaired_path) > 0:
        return repaired_path
    try:
        from pypdf import PdfReader, PdfWriter  # type: ignore

        with _suppress_pdf_console_noise():
            reader = PdfReader(file_path, strict=False)
            writer = PdfWriter()
            for page in reader.pages:
                writer.add_page(page)
            with open(repaired_path, "wb") as handle:
                writer.write(handle)
        return repaired_path if os.path.getsize(repaired_path) > 0 else None
    except Exception:
        return None


def _render_pdf_page_to_image(file_path: str, page_index: int, db_path: str, dpi: int = OCR_DPI):
    import pypdfium2 as pdfium

    last_error = None
    candidate_paths = [file_path]
    header_fixed = _strip_pdf_leading_garbage(file_path, db_path)
    if header_fixed and header_fixed not in candidate_paths:
        candidate_paths.append(header_fixed)
    rewritten = _rewrite_pdf_with_pypdf(file_path, db_path)
    if rewritten and rewritten not in candidate_paths:
        candidate_paths.append(rewritten)

    for candidate_path in candidate_paths:
        try:
            with _suppress_pdf_console_noise():
                document = pdfium.PdfDocument(candidate_path)
                page = document[page_index]
                scale = max(1.0, dpi / 72.0)
                bitmap = page.render(scale=scale)
                return bitmap.to_pil()
        except Exception as exc:
            last_error = exc
            continue

    raise RuntimeError(last_error or "PDF 页面渲染失败。")


def _get_pdf_page_count_via_pdfium(file_path: str, db_path: str) -> int:
    import pypdfium2 as pdfium

    last_error = None
    candidate_paths = [file_path]
    header_fixed = _strip_pdf_leading_garbage(file_path, db_path)
    if header_fixed and header_fixed not in candidate_paths:
        candidate_paths.append(header_fixed)
    rewritten = _rewrite_pdf_with_pypdf(file_path, db_path)
    if rewritten and rewritten not in candidate_paths:
        candidate_paths.append(rewritten)

    for candidate_path in candidate_paths:
        try:
            with _suppress_pdf_console_noise():
                document = pdfium.PdfDocument(candidate_path)
                return len(document)
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(last_error or "PDF 页数读取失败。")


def _ocr_page_worker(file_path: str, page_number: int, db_path: str) -> dict:
    cache_path = _page_cache_path(db_path, file_path, page_number)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as handle:
                cached = json.load(handle)
            if isinstance(cached, dict):
                return cached
        except Exception:
            pass

    image = _render_pdf_page_to_image(file_path, page_number - 1, db_path=db_path, dpi=OCR_DPI)
    skip_image, skip_reason = _is_low_value_image(image)
    if skip_image:
        payload = {"text": "", "status": "skipped", "skip_reason": skip_reason}
    else:
        ocr = _get_paddle_ocr()
        with _suppress_pdf_console_noise():
            ocr_input = np.array(image)
            result = _call_ocr_engine(ocr, ocr_input)
        lines = []
        for block in result or []:
            for item in block or []:
                try:
                    lines.append(str(item[1][0]).strip())
                except Exception:
                    continue
        text = "\n".join(line for line in lines if line)
        skip_text, text_reason = _is_low_value_text(text)
        payload = {
            "text": "" if skip_text else text,
            "status": "skipped" if skip_text else "ocr",
            "skip_reason": text_reason if skip_text else "",
        }

    with open(cache_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    return payload


def _classify_pdf_pages(file_path: str) -> Tuple[str, object, List[Tuple[int, str]]]:
    from pypdf import PdfReader  # type: ignore

    with _suppress_pdf_console_noise():
        reader = PdfReader(file_path, strict=False)
    sample_results: List[Tuple[int, str]] = []
    extractable = 0
    sample_limit = min(PDF_SAMPLE_PAGES, len(reader.pages))
    for page_number in range(1, sample_limit + 1):
        text = ""
        try:
            with _suppress_pdf_console_noise():
                text = reader.pages[page_number - 1].extract_text() or ""
        except Exception:
            text = ""
        normalized = _normalize_page_text(text)
        sample_results.append((page_number, normalized))
        if len(normalized) >= PDF_TEXT_THRESHOLD:
            extractable += 1

    if sample_limit <= 0:
        return "BROKEN", reader, sample_results
    if extractable == sample_limit:
        return "TEXT", reader, sample_results
    if extractable == 0:
        return "SCANNED", reader, sample_results
    return "MIXED", reader, sample_results


def _build_pdf_documents(
    file_path: str,
    file_name: str,
    db_path: str,
    progress_callback=None,
) -> Tuple[List[Document], dict]:
    try:
        loaded_docs = _load_pdf_documents(file_path, file_name)
    except Exception as exc:
        status_meta = _make_pdf_status(
            file_path,
            file_name,
            "BROKEN",
            "failed",
            error_reason=str(exc) or "PDF 无法读取文本。",
        )
        if progress_callback:
            progress_callback(f"PDF {file_name} · 失败 · 原因：{status_meta['error_reason']}")
        return [], status_meta

    documents = []
    for doc in loaded_docs:
        if not str(doc.page_content or "").strip():
            continue
        documents.append(doc)

    if not documents:
        status_meta = _make_pdf_status(
            file_path,
            file_name,
            "BROKEN",
            "failed",
            error_reason="未读取到可用文本。该 PDF 可能尚未经过 OCR，请先完成 OCR 后再导入。",
        )
        if progress_callback:
            progress_callback(f"PDF {file_name} · 失败 · 原因：{status_meta['error_reason']}")
        return [], status_meta

    status_meta = _make_pdf_status(
        file_path,
        file_name,
        "TEXT",
        "success",
        total_pages=len(documents),
    )
    for doc in documents:
        doc.metadata.update(status_meta)
        doc.metadata["lang"] = detect_language(doc.page_content)
    if progress_callback:
        progress_callback(f"PDF {file_name} · 正常读取")
    return documents, status_meta


def _load_pdf_documents(file_path: str, file_name: str):
    suppressed_stderr = io.StringIO()
    pypdf_logger = logging.getLogger("pypdf")
    previous_level = pypdf_logger.level
    pypdf_logger.setLevel(logging.ERROR)

    try:
        with redirect_stderr(suppressed_stderr):
            try:
                from pypdf import PdfReader  # type: ignore

                reader = PdfReader(file_path, strict=False)
                documents = []
                for page_number, page in enumerate(reader.pages, start=1):
                    page_text = page.extract_text() or ""
                    if not page_text.strip():
                        continue
                    documents.append(
                        Document(
                            page_content=page_text,
                            metadata={
                                "source": file_path,
                                "name": file_name,
                                "type": "PDF",
                                "page": page_number,
                            },
                        )
                    )
                if documents:
                    return documents
            except Exception:
                pass

            from langchain_community.document_loaders import PyPDFLoader

            loader = PyPDFLoader(file_path)
            return loader.load()
    finally:
        pypdf_logger.setLevel(previous_level)


def load_documents(
    docs_dir: Optional[str] = None,
    urls: Optional[List[str]] = None,
    recursive_web: bool = False,
    max_depth: int = 3,
    progress_callback=None,
    db_path: str = "./chroma_db",
):
    docs = []
    status_entries: Dict[str, dict] = {}
    os.environ.setdefault("USER_AGENT", "Mozilla/5.0 (compatible; AcademicRAG/1.0)")

    if docs_dir and os.path.exists(docs_dir):
        for root, _, files in os.walk(docs_dir):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                suffix = os.path.splitext(file_name)[1].lower()
                try:
                    if suffix == ".pdf":
                        loaded_docs, status_entry = _build_pdf_documents(
                            file_path,
                            file_name,
                            db_path=db_path,
                            progress_callback=progress_callback,
                        )
                        status_entries[file_path] = status_entry
                        doc_type = "PDF"
                    elif suffix == ".txt":
                        from langchain_community.document_loaders import TextLoader

                        loader = TextLoader(file_path, encoding="utf-8")
                        doc_type = "TXT"
                    elif suffix in {".doc", ".docx"}:
                        from langchain_community.document_loaders import Docx2txtLoader

                        loader = Docx2txtLoader(file_path)
                        doc_type = "WORD"
                    else:
                        continue

                    if suffix != ".pdf":
                        loaded_docs = loader.load()
                        status_entries[file_path] = {
                            "source": file_path,
                            "name": file_name,
                            "type": doc_type,
                            "status": "success",
                        }
                    for doc in loaded_docs:
                        doc.metadata.setdefault("page", doc.metadata.get("page_number"))
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
                    if suffix == ".pdf":
                        status_entries[file_path] = _make_pdf_status(
                            file_path,
                            file_name,
                            "BROKEN",
                            "failed",
                            error_reason=str(exc),
                        )

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
                from langchain_community.document_loaders import RecursiveUrlLoader

                loader = RecursiveUrlLoader(
                    url=url,
                    max_depth=max_depth,
                    prevent_outside=True,
                    use_async=True,
                    timeout=30,
                    headers=headers,
                )
            else:
                from langchain_community.document_loaders import WebBaseLoader

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
                        "status": "success",
                    }
                )
            docs.extend(loaded_docs)
        except Exception as exc:
            print(f"抓取 URL 失败 {url}: {exc}")
    return docs, status_entries


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
    failed_pdfs: List[dict] = []
    os.makedirs(db_path, exist_ok=True)
    if progress_callback:
        progress_callback(f"正在向 [{os.path.basename(db_path)}] 导入资料…")

    documents, status_entries = load_documents(
        docs_dir,
        urls,
        recursive_web,
        progress_callback=progress_callback,
        db_path=db_path,
    )
    manifest = load_ingest_status(db_path)
    manifest.update(status_entries)
    save_ingest_status(db_path, manifest)
    failed_pdfs = [
        {
            "name": str(meta.get("name") or ""),
            "reason": str(meta.get("error_reason") or ""),
        }
        for meta in status_entries.values()
        if str(meta.get("type") or "").upper() == "PDF" and str(meta.get("status") or "").lower() == "failed"
    ]
    if not documents:
        if progress_callback:
            progress_callback("未找到可导入的本地文档或网页内容。")
        return {"chunk_count": 0, "failed_pdfs": failed_pdfs}

    if progress_callback:
        progress_callback(f"已载入 {len(documents)} 份文档，开始切分…")

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    _assign_chunk_metadata(chunks)

    if progress_callback:
        progress_callback(f"共生成 {len(chunks)} 个片段，开始向量化…")

    ModelManager = _get_model_manager()
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
        _raise_if_embedding_model_mismatch(
            db_path=db_path,
            mode=mode,
            has_existing_chunks=bool(existing_chunks),
        )
        existing_chunk_ids = {
            str(chunk.metadata.get("chunk_id") or "")
            for chunk in existing_chunks
            if getattr(chunk, "metadata", None)
        }
        chunks = [chunk for chunk in chunks if str(chunk.metadata.get("chunk_id") or "") not in existing_chunk_ids]
        all_chunks = _dedupe_chunks_by_id(existing_chunks + chunks)

    if not chunks:
        if progress_callback:
            progress_callback("未发现新的有效片段，已跳过重复写入。")
        with open(pkl_path, "wb") as handle:
            pickle.dump(all_chunks, handle)
        _write_library_meta(db_path)
        return {"chunk_count": 0, "failed_pdfs": failed_pdfs}

    batch_size = 200
    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        batch_docs = chunks[start:end]
        batch_ids = [str(doc.metadata["chunk_id"]) for doc in batch_docs]
        try:
            vectordb.add_documents(batch_docs, ids=batch_ids)
        except Exception as exc:
            error_text = str(exc)
            if "expecting embedding with dimension" in error_text:
                raise RuntimeError(
                    "当前知识库中的 Chroma 向量维度与现用 embedding 模型不一致。"
                    "这通常说明该库是用旧模型建立的。请改用“清空当前主库后重建”，"
                    "或先点击“重建索引”完成升级后再追加导入。"
                ) from exc
            raise
        if progress_callback:
            progress_callback(f"正在写入向量库… {end}/{len(chunks)}")

    with open(pkl_path, "wb") as handle:
        pickle.dump(all_chunks, handle)

    _write_library_meta(db_path)

    if progress_callback:
        progress_callback(f"导入完成：{db_path}")
    return {"chunk_count": len(chunks), "failed_pdfs": failed_pdfs}


def rebuild_library_index(
    db_path: str,
    progress_callback=None,
) -> int:
    os.makedirs(db_path, exist_ok=True)
    pkl_path = os.path.join(db_path, "chunks.pkl")
    if not os.path.exists(pkl_path):
        raise RuntimeError("当前知识库缺少 chunks.pkl，无法重建索引。请先重新导入资料。")

    try:
        with open(pkl_path, "rb") as handle:
            chunks = pickle.load(handle)
    except Exception as exc:
        raise RuntimeError(f"chunks.pkl 无法读取，无法重建索引：{exc}") from exc

    if not chunks:
        raise RuntimeError("当前知识库没有可重建的 chunk 数据。")

    _assign_chunk_metadata(chunks)
    chunks = _dedupe_chunks_by_id(chunks)

    ModelManager = _get_model_manager()
    embeddings = ModelManager.get_embeddings()
    if embeddings is None:
        raise RuntimeError(ModelManager.get_last_error() or "Embedding 模型尚未完成初始化。")

    if progress_callback:
        progress_callback(f"正在为 [{os.path.basename(db_path)}] 重建 chunk_id 与索引...")

    vectordb = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_name="rag_collection",
    )
    try:
        vectordb.delete_collection()
    except Exception:
        pass

    vectordb = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_name="rag_collection",
    )

    batch_size = 200
    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        batch_docs = chunks[start:end]
        batch_ids = [str(doc.metadata["chunk_id"]) for doc in batch_docs]
        vectordb.add_documents(batch_docs, ids=batch_ids)
        if progress_callback:
            progress_callback(f"正在重建向量索引... {end}/{len(chunks)}")

    with open(pkl_path, "wb") as handle:
        pickle.dump(chunks, handle)

    _write_library_meta(db_path)

    if progress_callback:
        progress_callback("重建完成。旧库已补齐 chunk_id，并重新写入索引。")
    return len(chunks)
