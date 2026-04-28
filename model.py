import os
import threading
from typing import Callable, Iterable, Optional

import numpy as np
import torch
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


class LocalSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, manager: "ModelManager"):
        self.manager = manager

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.manager.encode_texts(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.manager.encode_texts([text])[0].tolist()


class ModelManager:
    EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

    _model: Optional[SentenceTransformer] = None
    _embeddings: Optional[LocalSentenceTransformerEmbeddings] = None
    _loading = False
    _lock = threading.Lock()
    _last_error: Optional[str] = None

    @classmethod
    def get_cache_dir(cls) -> str:
        cache_dir = os.path.join(os.getcwd(), "model_cache")
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    @classmethod
    def get_device(cls) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def get_embeddings(cls) -> Optional[Embeddings]:
        if cls._embeddings is None and not cls._loading:
            cls.initialize_model()
        return cls._embeddings

    @classmethod
    def get_sentence_transformer(cls) -> Optional[SentenceTransformer]:
        if cls._model is None and not cls._loading:
            cls.initialize_model()
        return cls._model

    @classmethod
    def get_last_error(cls) -> Optional[str]:
        return cls._last_error

    @classmethod
    def initialize_model(
        cls,
        callback: Optional[Callable[[], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        with cls._lock:
            if cls._loading:
                return
            if cls._model is not None and cls._embeddings is not None:
                if callback:
                    callback()
                return
            cls._loading = True

        def load() -> None:
            try:
                if status_callback:
                    status_callback("正在检查本地模型缓存…")

                cache_dir = cls.get_cache_dir()
                device = cls.get_device()
                local_only = True

                try:
                    os.environ["TRANSFORMERS_OFFLINE"] = "1"
                    os.environ["HF_HUB_OFFLINE"] = "1"
                    cls._model = SentenceTransformer(
                        cls.EMBEDDING_MODEL_NAME,
                        cache_folder=cache_dir,
                        device=device,
                        local_files_only=True,
                    )
                except Exception:
                    local_only = False
                    if status_callback:
                        status_callback("本地缓存不存在，首次启动将下载模型…")
                    os.environ["TRANSFORMERS_OFFLINE"] = "0"
                    os.environ["HF_HUB_OFFLINE"] = "0"
                    cls._model = SentenceTransformer(
                        cls.EMBEDDING_MODEL_NAME,
                        cache_folder=cache_dir,
                        device=device,
                        local_files_only=False,
                    )

                cls._embeddings = LocalSentenceTransformerEmbeddings(cls)
                cls._last_error = None

                if status_callback:
                    mode = "离线缓存" if local_only else "本地缓存已初始化"
                    status_callback(f"模型就绪：{cls.EMBEDDING_MODEL_NAME} ({mode})")
            except Exception as exc:
                cls._last_error = str(exc)
            finally:
                with cls._lock:
                    cls._loading = False
                if callback:
                    callback()

        threading.Thread(target=load, daemon=True).start()

    @classmethod
    def encode_texts(
        cls,
        texts: Iterable[str],
        normalize_embeddings: bool = True,
        batch_size: int = 32,
    ) -> np.ndarray:
        model = cls.get_sentence_transformer()
        if model is None:
            raise RuntimeError(cls._last_error or "Embedding 模型尚未完成初始化。")

        text_list = [str(text or "") for text in texts]
        if not text_list:
            return np.zeros((0, 0), dtype=np.float32)

        vectors = model.encode(
            text_list,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)
