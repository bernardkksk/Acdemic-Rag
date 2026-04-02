import os
import sys
import threading
from langchain_huggingface import HuggingFaceEmbeddings
import torch

class ModelManager:
    _embeddings = None
    _loading = False

    @classmethod
    def get_embeddings(cls):
        if cls._embeddings is None:
            # 如果正在加载中，抛出异常或返回 None 并在 UI 处理
            if cls._loading:
                return None
            cls.initialize_model()
        return cls._embeddings

    @classmethod
    def initialize_model(cls, callback=None):
        def load():
            cls._loading = True
            import os
            # 💡 强力手段 1：设置环境变量，强制 transformers 和 huggingface 进入离线模式
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
            
            cache_dir = os.path.join(os.getcwd(), "model_cache")
            if not os.path.exists(cache_dir): os.makedirs(cache_dir)
            
            print("🚀 正在加载本地 Embedding 模型...")
            try:
                cls._embeddings = HuggingFaceEmbeddings(
                    model_name="BAAI/bge-small-zh-v1.5",
                    cache_folder=cache_dir,
                    # 💡 强力手段 2：明确告诉插件只从本地读取
                    model_kwargs={
                        'device': 'cuda', # 如果报错显存不足请改回 'cpu'
                        'local_files_only': True 
                    }
                )
            except Exception as e:
                print(f"⚠️ 离线加载失败，尝试联网初始化: {e}")
                # 如果本地还没下载过模型，则临时开启网络下载
                os.environ["TRANSFORMERS_OFFLINE"] = "0"
                os.environ["HF_HUB_OFFLINE"] = "0"
                cls._embeddings = HuggingFaceEmbeddings(
                    model_name="BAAI/bge-small-zh-v1.5",
                    cache_folder=cache_dir,
                    model_kwargs={'device': 'cuda', 'local_files_only': False}
                )

            cls._loading = False
            if callback: callback()
            print("✅ 模型就绪")

        thread = threading.Thread(target=load, daemon=True)
        thread.start()