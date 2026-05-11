import json
import os
import shutil
import threading
import time
from typing import Any, Callable, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from langchain_core.embeddings import Embeddings
from transformers import AutoModel, AutoTokenizer


class LocalSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, manager: "ModelManager"):
        self.manager = manager

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.manager.encode_texts(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.manager.encode_texts([text])[0].tolist()


class ModelManager:
    EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    DOWNLOAD_RETRY_COUNT = 4
    DOWNLOAD_RETRY_BASE_DELAY = 2.0
    OBSOLETE_MODEL_DIRS = (
        "BAAI__bge-m3",
        "BAAI__bge-small-zh-v1.5",
    )
    OBSOLETE_CACHE_DIRS = (
        "models--BAAI--bge-m3",
        "models--BAAI--bge-small-zh-v1.5",
    )

    _model: Optional[Any] = None
    _tokenizer: Optional[Any] = None
    _embeddings: Optional[LocalSentenceTransformerEmbeddings] = None
    _loading = False
    _lock = threading.Lock()
    _ready_event = threading.Event()
    _last_error: Optional[str] = None

    @classmethod
    def get_cache_dir(cls) -> str:
        cache_dir = os.path.join(os.getcwd(), "model_cache")
        os.makedirs(cache_dir, exist_ok=True)
        os.environ.setdefault("USER_AGENT", "AcademicRAG/1.0")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)
        os.environ.setdefault("HF_HOME", cache_dir)
        return cache_dir

    @classmethod
    def get_model_dir(cls) -> str:
        model_dir = os.path.join(cls.get_cache_dir(), "local_models", cls.EMBEDDING_MODEL_NAME.replace("/", "__"))
        os.makedirs(model_dir, exist_ok=True)
        return model_dir

    @classmethod
    def get_marker_path(cls) -> str:
        return os.path.join(cls.get_model_dir(), ".model_ready.json")

    @classmethod
    def get_device(cls) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def get_embeddings(cls) -> Optional[Embeddings]:
        if cls._embeddings is None:
            cls.ensure_model_ready()
        return cls._embeddings

    @classmethod
    def get_sentence_transformer(cls) -> Optional[Any]:
        if cls._model is None:
            cls.ensure_model_ready()
        return cls._model

    @classmethod
    def get_last_error(cls) -> Optional[str]:
        return cls._last_error

    @classmethod
    def _has_any_existing_model_files(cls, model_dir: str) -> bool:
        for relative_path in (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "sentencepiece.bpe.model",
            "model.safetensors",
            "model.safetensors.index.json",
        ):
            if os.path.exists(os.path.join(model_dir, relative_path)):
                return True
        return False

    @classmethod
    def _is_complete_local_model(cls, model_dir: str) -> bool:
        required_any = [
            ("tokenizer.json", "sentencepiece.bpe.model"),
        ]
        required_single = ["config.json", "tokenizer_config.json"]
        for relative_path in required_single:
            full_path = os.path.join(model_dir, relative_path)
            if not os.path.exists(full_path) or os.path.getsize(full_path) <= 0:
                return False
        safetensors_path = os.path.join(model_dir, "model.safetensors")
        if not os.path.exists(safetensors_path) or os.path.getsize(safetensors_path) <= 0:
            return False
        for candidates in required_any:
            if not any(
                os.path.exists(os.path.join(model_dir, relative_path))
                and os.path.getsize(os.path.join(model_dir, relative_path)) > 0
                for relative_path in candidates
            ):
                return False
        return True

    @classmethod
    def _write_ready_marker(cls, model_dir: str) -> None:
        payload = {
            "model_name": cls.EMBEDDING_MODEL_NAME,
            "device": cls.get_device(),
        }
        with open(cls.get_marker_path(), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    @classmethod
    def _clear_incomplete_local_model(cls, model_dir: str) -> None:
        for entry in os.listdir(model_dir):
            path = os.path.join(model_dir, entry)
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
            except Exception:
                pass

    @classmethod
    def _cleanup_obsolete_model_caches(cls, status_callback: Optional[Callable[[str], None]] = None) -> None:
        cache_dir = cls.get_cache_dir()
        local_models_dir = os.path.join(cache_dir, "local_models")
        lock_dir = os.path.join(cache_dir, ".locks")

        removed_paths = []
        for dirname in cls.OBSOLETE_MODEL_DIRS:
            path = os.path.join(local_models_dir, dirname)
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
                removed_paths.append(path)

        for dirname in cls.OBSOLETE_CACHE_DIRS:
            path = os.path.join(cache_dir, dirname)
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
                removed_paths.append(path)
            lock_path = os.path.join(lock_dir, dirname)
            if os.path.isdir(lock_path):
                shutil.rmtree(lock_path, ignore_errors=True)
                removed_paths.append(lock_path)

        if removed_paths and status_callback:
            status_callback("已清理旧版 embedding 模型缓存，后续将只保留当前多语言模型。")

    @classmethod
    def _download_model_snapshot(cls, model_dir: str, status_callback: Optional[Callable[[str], None]] = None) -> None:
        from huggingface_hub import snapshot_download

        last_error: Optional[Exception] = None
        for attempt in range(1, cls.DOWNLOAD_RETRY_COUNT + 1):
            try:
                if status_callback:
                    status_callback(
                        f"本地模型不完整，正在下载并固化到本地目录… "
                        f"(第 {attempt}/{cls.DOWNLOAD_RETRY_COUNT} 次)"
                    )

                snapshot_download(
                    repo_id=cls.EMBEDDING_MODEL_NAME,
                    local_dir=model_dir,
                    allow_patterns=[
                        "*.json",
                        "*.model",
                        "*.safetensors",
                        "*.txt",
                    ],
                    ignore_patterns=[
                        "*.bin",
                        "*.h5",
                        "*.ot",
                        "*.msgpack",
                    ],
                    max_workers=2,
                )

                if not cls._is_complete_local_model(model_dir):
                    raise RuntimeError("模型下载结束后，本地目录仍不完整。")
                return
            except Exception as exc:
                last_error = exc
                cls._clear_incomplete_local_model(model_dir)
                if attempt >= cls.DOWNLOAD_RETRY_COUNT:
                    break
                delay_seconds = cls.DOWNLOAD_RETRY_BASE_DELAY * attempt
                if status_callback:
                    status_callback(
                        f"模型下载失败：{exc}。"
                        f" 将在 {delay_seconds:.0f} 秒后自动重试。"
                    )
                time.sleep(delay_seconds)

        raise RuntimeError(
            f"模型下载连续失败 {cls.DOWNLOAD_RETRY_COUNT} 次，请检查网络、代理或 HuggingFace 连接状态。"
            f"最后一次错误：{last_error}"
        )

    @classmethod
    def _load_from_local_dir(cls, model_dir: str, device: str) -> tuple[Any, Any]:
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=False,
        )
        model = AutoModel.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
        )
        model.to(device)
        model.eval()
        return tokenizer, model

    @classmethod
    def initialize_model(
        cls,
        callback: Optional[Callable[[], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        with cls._lock:
            if cls._loading:
                return
            if cls._model is not None and cls._embeddings is not None and cls._tokenizer is not None:
                if callback:
                    callback()
                return
            cls._loading = True
            cls._ready_event.clear()

        def load() -> None:
            try:
                cache_dir = cls.get_cache_dir()
                model_dir = cls.get_model_dir()
                device = cls.get_device()
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                os.environ["HF_HUB_OFFLINE"] = "1"
                cls._cleanup_obsolete_model_caches(status_callback=status_callback)

                if status_callback:
                    status_callback("正在检查本地模型目录…")

                local_complete = cls._is_complete_local_model(model_dir)
                if not local_complete:
                    if cls._has_any_existing_model_files(model_dir):
                        if status_callback:
                            status_callback("检测到未完成的本地模型缓存，正在清理后重新下载…")
                        cls._clear_incomplete_local_model(model_dir)

                    os.environ["TRANSFORMERS_OFFLINE"] = "0"
                    os.environ["HF_HUB_OFFLINE"] = "0"
                    cls._download_model_snapshot(model_dir, status_callback=status_callback)
                    os.environ["TRANSFORMERS_OFFLINE"] = "1"
                    os.environ["HF_HUB_OFFLINE"] = "1"

                if status_callback:
                    status_callback("正在从本地 safetensors 目录加载 embedding 模型，并按模型最大长度自动裁切文本…")

                cls._tokenizer, cls._model = cls._load_from_local_dir(model_dir, device)
                cls._embeddings = LocalSentenceTransformerEmbeddings(cls)
                cls._write_ready_marker(model_dir)
                cls._last_error = None

                if status_callback:
                    mode = "离线本地目录"
                    if not local_complete:
                        mode = "本地目录已初始化"
                    status_callback(f"模型就绪：{cls.EMBEDDING_MODEL_NAME} ({mode})")
            except Exception as exc:
                cls._last_error = str(exc)
                cls._model = None
                cls._tokenizer = None
                cls._embeddings = None
            finally:
                with cls._lock:
                    cls._loading = False
                    cls._ready_event.set()
                if callback:
                    callback()

        threading.Thread(target=load, daemon=True).start()

    @classmethod
    def ensure_model_ready(
        cls,
        timeout: Optional[float] = None,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> Any:
        if cls._model is not None and cls._embeddings is not None and cls._tokenizer is not None:
            return cls._model

        if cls._loading:
            if status_callback:
                status_callback("正在等待 embedding 模型加载完成…")
        else:
            cls.initialize_model(status_callback=status_callback)

        if not cls._ready_event.wait(timeout):
            raise RuntimeError("Embedding 模型初始化超时，请稍后重试。")

        if cls._model is None or cls._embeddings is None or cls._tokenizer is None:
            raise RuntimeError(cls._last_error or "Embedding 模型尚未完成初始化。")
        return cls._model

    @classmethod
    def _mean_pool(cls, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked = last_hidden_state * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    @classmethod
    def encode_texts(
        cls,
        texts: Iterable[str],
        normalize_embeddings: bool = True,
        batch_size: int = 32,
    ) -> np.ndarray:
        cls.ensure_model_ready()
        tokenizer = cls._tokenizer
        model = cls._model
        if tokenizer is None or model is None:
            raise RuntimeError(cls._last_error or "Embedding 模型尚未完成初始化。")

        text_list = [str(text or "") for text in texts]
        if not text_list:
            return np.zeros((0, 0), dtype=np.float32)

        device = cls.get_device()
        vectors: list[np.ndarray] = []
        tokenizer_limit = getattr(tokenizer, "model_max_length", 512) or 512
        config_limit = getattr(getattr(model, "config", None), "max_position_embeddings", 512) or 512
        try:
            tokenizer_limit = int(tokenizer_limit)
        except Exception:
            tokenizer_limit = 512
        try:
            config_limit = int(config_limit)
        except Exception:
            config_limit = 512

        if tokenizer_limit <= 0 or tokenizer_limit > 100000:
            tokenizer_limit = config_limit
        max_length = max(8, min(tokenizer_limit, config_limit, 512))

        for start in range(0, len(text_list), batch_size):
            batch_texts = text_list[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}

            with torch.no_grad():
                outputs = model(**encoded)
                pooled = cls._mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
                if normalize_embeddings:
                    pooled = F.normalize(pooled, p=2, dim=1)
                vectors.append(pooled.cpu().numpy().astype(np.float32))

        return np.vstack(vectors)
