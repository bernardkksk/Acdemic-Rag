import gc
import json
import os
import pickle
import shutil
import threading
import time
from tkinter import filedialog, messagebox

import customtkinter as ctk
from langchain_chroma import Chroma

import ingest
import query
from model import ModelManager
from ui_layout import (
    ACCENT,
    ACCENT_SOFT,
    BG,
    EMPTY_OPTION,
    ERROR,
    IN_PROGRESS,
    SUCCESS,
    TEXT,
    AppUIBuilder,
)


ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

CONFIG_FILE = "config.json"
LIBRARIES_FILE = "libraries.json"


class AcademicRAGApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Academic RAG")
        self.geometry("1500x930")
        self.configure(fg_color=BG)

        self.libraries = self.load_libraries()
        self.configs = self.load_configs()
        self.current_library = self.libraries[0]
        self.current_config = self.configs[0] if self.configs else None
        self.active_retrieval_libraries = []
        self.file_checkboxes = {}
        self.file_id_map = {}
        self.file_delete_sources = {}
        self.file_weight_targets = {}
        self.file_tree_state = {"groups": {}, "urls": {}}
        self.file_tree_payload = None
        self.is_busy = False
        self.latest_answer_text = ""
        self.placeholder_text = "模型预加载完成后，请先选择主库与次库并点击“确认并加载知识库”。"
        self.current_stage = "model"

        self.ui = AppUIBuilder(self)
        self.ui.setup_sidebar()
        self.ui.create_tabs()
        self.ui.refresh_sidebar()
        self.ui.refresh_chat_selectors()
        self._set_answer_text(self.placeholder_text)

        self.set_pipeline_state("model", "模型预加载中", "正在检查本地缓存…", completed=False)
        ModelManager.initialize_model(
            callback=self.on_model_ready,
            status_callback=self.on_model_status,
        )

    def load_libraries(self):
        if os.path.exists(LIBRARIES_FILE):
            with open(LIBRARIES_FILE, "r", encoding="utf-8") as handle:
                libraries = json.load(handle)
                if libraries:
                    return libraries
        return [{"name": "默认库", "path": "./chroma_db"}]

    def save_libraries(self):
        with open(LIBRARIES_FILE, "w", encoding="utf-8") as handle:
            json.dump(self.libraries, handle, ensure_ascii=False, indent=2)

    def load_configs(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r", encoding="utf-8") as handle:
                return json.load(handle)
        return []

    def save_configs(self):
        with open(CONFIG_FILE, "w", encoding="utf-8") as handle:
            json.dump(self.configs, handle, ensure_ascii=False, indent=2)

    def on_provider_change(self, choice):
        presets = {
            "DeepSeek": ("https://api.deepseek.com/v1", "deepseek-chat"),
            "OpenAI": ("https://api.openai.com/v1", "gpt-4o"),
            "xAI": ("https://api.x.ai/v1", "grok-beta"),
            "Gemini": ("", "gemini-1.5-pro"),
            "Qwen": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus"),
        }
        url, model_name = presets.get(choice, ("", ""))
        self.url_entry.delete(0, "end")
        self.url_entry.insert(0, url)
        self.model_entry.delete(0, "end")
        self.model_entry.insert(0, model_name)

    def on_chat_config_change(self, selected_name):
        for config in self.configs:
            if config["name"] == selected_name:
                self.current_config = config
                self.set_pipeline_state("library", "模型配置已切换", f"当前配置：{selected_name}", completed=True)
                return
        self.current_config = None

    def save_new_config(self):
        name = self.new_name.get().strip()
        if not name:
            messagebox.showwarning("提示", "请输入配置名。")
            return

        config = {
            "name": name,
            "provider": self.provider_var.get(),
            "model": self.model_entry.get().strip(),
            "api_key": self.key_entry.get().strip(),
            "base_url": self.url_entry.get().strip(),
        }

        replaced = False
        for idx, existing in enumerate(self.configs):
            if existing["name"] == name:
                self.configs[idx] = config
                replaced = True
                break
        if not replaced:
            self.configs.append(config)

        self.save_configs()
        self.current_config = config
        self.ui.refresh_chat_selectors()
        messagebox.showinfo("成功", f"配置已{'更新' if replaced else '保存'}。")

    def on_model_status(self, message):
        self.after(0, lambda: self.set_pipeline_state("model", "模型预加载中", message, completed=False))

    def on_model_ready(self):
        def finish():
            error = ModelManager.get_last_error()
            if error:
                self.set_pipeline_error("模型初始化失败", error)
            else:
                self.set_pipeline_state("model", "模型加载完毕", "请选择主库与次库后点击确认。", completed=True)

        self.after(0, finish)

    def set_pipeline_state(self, active_step, summary, detail="", completed=False):
        self.current_stage = active_step
        display_text = summary if not detail else f"{summary} · {detail}"
        self.status_text.configure(text=display_text, text_color=TEXT if not completed else SUCCESS)
        self.progress_bar.configure(progress_color=SUCCESS if completed else IN_PROGRESS)
        self.progress_bar.set(1.0 if completed else 0.35)

    def set_pipeline_error(self, summary, detail=""):
        display_text = summary if not detail else f"{summary} · {detail}"
        self.status_text.configure(text=display_text, text_color=ERROR)
        self.progress_bar.configure(progress_color=ERROR)
        self.progress_bar.set(1.0)

    def preview_library(self, library):
        self.current_library = library
        self.ui.refresh_sidebar()
        self.ui.refresh_chat_selectors()
        self.refresh_file_list()

    def select_file_library(self, selected_name):
        library = self.get_library_by_name(selected_name)
        if not library:
            return
        self.current_library = library
        self.ui.refresh_sidebar()
        self.ui.refresh_chat_selectors()
        self.refresh_file_list()

    def create_new_library(self):
        name = ctk.CTkInputDialog(text="请输入新知识库名称：", title="新建知识库").get_input()
        if not name:
            return
        db_path = f"./db_{int(time.time())}"
        os.makedirs(db_path, exist_ok=True)
        self.libraries.append({"name": name.strip(), "path": db_path})
        self.save_libraries()
        self.ui.refresh_sidebar()
        self.ui.refresh_chat_selectors()

    def rename_library(self, library):
        new_name = ctk.CTkInputDialog(text="请输入新的库名：", title="重命名知识库").get_input()
        if not new_name:
            return
        for item in self.libraries:
            if item["path"] == library["path"]:
                item["name"] = new_name.strip()
                break
        self.save_libraries()
        if self.current_library["path"] == library["path"]:
            self.current_library = next(item for item in self.libraries if item["path"] == library["path"])
        self.ui.refresh_sidebar()
        self.ui.refresh_chat_selectors()

    def delete_library(self, library):
        if library["path"] == "./chroma_db":
            messagebox.showwarning("提示", "默认库不能删除。")
            return
        confirmed = messagebox.askyesno(
            "确认删除",
            f"确定要删除知识库 [{library['name']}] 吗？\n这会删除对应目录和其中的向量数据。",
        )
        if not confirmed:
            return

        if self.current_library["path"] == library["path"]:
            self.current_library = self.libraries[0]

        self.active_retrieval_libraries = [
            item for item in self.active_retrieval_libraries if item["path"] != library["path"]
        ]

        self.libraries = [item for item in self.libraries if item["path"] != library["path"]]
        self.save_libraries()
        query.invalidate_library_cache(library["path"])
        self.ui.refresh_sidebar()
        self.ui.refresh_chat_selectors()
        threading.Thread(target=self._force_delete_folder, args=(library["path"],), daemon=True).start()

    def _force_delete_folder(self, path):
        try:
            for _ in range(3):
                gc.collect()
                time.sleep(0.1)
            if os.path.isdir(path):
                shutil.rmtree(path)
        except Exception:
            pass

    def apply_library_selection(self):
        primary = self.get_library_by_name(self.primary_library_var.get())
        if not primary:
            messagebox.showwarning("提示", "请选择主库。")
            return

        secondaries = []
        for name in [self.secondary_library_var_1.get(), self.secondary_library_var_2.get()]:
            library = self.get_library_by_name(name)
            if not library or library["path"] == primary["path"]:
                continue
            if all(existing["path"] != library["path"] for existing in secondaries):
                secondaries.append(library)

        self.current_library = primary
        self.active_retrieval_libraries = [primary] + secondaries
        library_names = " / ".join(item["name"] for item in self.active_retrieval_libraries)
        self.ui.refresh_sidebar()
        self.ui.refresh_chat_selectors()
        self.set_pipeline_state("library", "知识库加载中", f"正在载入：{library_names}", completed=False)

        def run():
            try:
                query.preload_libraries(self.active_retrieval_libraries)
                self.after(0, self.refresh_file_list)
                self.after(
                    0,
                    lambda: self.set_pipeline_state(
                        "library",
                        "知识库加载完毕",
                        f"当前检索库：{library_names}",
                        completed=True,
                    ),
                )
            except Exception as exc:
                error_text = str(exc)
                self.after(0, lambda error_text=error_text: self.set_pipeline_error("知识库加载失败", error_text))

        threading.Thread(target=run, daemon=True).start()

    def get_library_by_name(self, name):
        if not name or name == EMPTY_OPTION:
            return None
        for library in self.libraries:
            if library["name"] == name:
                return library
        return None

    def get_weight_file(self):
        return os.path.join(self.current_library["path"], "file_weights.json")

    def load_all_weights(self):
        path = self.get_weight_file()
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        return {}

    def update_file_weight(self, source, weight):
        weights = self.load_all_weights()
        weights[source] = int(weight)
        os.makedirs(self.current_library["path"], exist_ok=True)
        with open(self.get_weight_file(), "w", encoding="utf-8") as handle:
            json.dump(weights, handle, ensure_ascii=False, indent=2)
        if self.file_tree_payload:
            self.file_tree_payload["weights"] = dict(weights)
        query.invalidate_library_cache(self.current_library["path"])

    def rerender_file_tree(self):
        if not self.file_tree_payload:
            return
        self.ui.render_file_tree(
            self.file_tree_payload["metas"],
            self.file_tree_payload["ids"],
            self.file_tree_payload["count"],
            self.file_tree_payload["weights"],
        )

    def refresh_file_list(self):
        self.file_tree_payload = None
        if not self.current_library:
            self.ui.render_empty_files("请先在内容编辑页选择一个知识库。")
            return

        self.ui.render_empty_files(f"正在读取知识库 [{self.current_library['name']}] 的内容...")

        def run():
            try:
                db_path = self.current_library["path"]
                pkl_path = os.path.join(db_path, "chunks.pkl")
                if not os.path.exists(pkl_path):
                    self.after(0, lambda: self.ui.render_empty_files("当前知识库还没有导入内容。"))
                    return

                vectordb = Chroma(
                    persist_directory=db_path,
                    embedding_function=ModelManager.get_embeddings(),
                    collection_name="rag_collection",
                )
                try:
                    total = vectordb._collection.count()
                except Exception:
                    total = 0

                if total <= 0:
                    self.after(0, lambda: self.ui.render_empty_files("当前知识库里还没有可编辑的内容。"))
                    return

                batch_size = 1000
                metas = []
                ids = []
                for offset in range(0, total, batch_size):
                    batch = vectordb.get(include=["metadatas"], limit=batch_size, offset=offset)
                    metas.extend(batch["metadatas"])
                    ids.extend(batch["ids"])

                weights = self.load_all_weights()
                self.file_tree_payload = {"metas": metas, "ids": ids, "count": total, "weights": dict(weights)}
                self.after(0, self.rerender_file_tree)
            except Exception as exc:
                error_text = str(exc)
                self.after(0, lambda error_text=error_text: self.ui.render_empty_files(f"读取知识库失败：{error_text}"))

        threading.Thread(target=run, daemon=True).start()

    def select_all_files(self):
        for item in self.file_checkboxes.values():
            item["var"].set(True)

    def deselect_all_files(self):
        for item in self.file_checkboxes.values():
            item["var"].set(False)

    def delete_selected_file(self):
        if not self.current_library:
            return
        selected_keys = [source for source, data in self.file_checkboxes.items() if data["var"].get()]
        if not selected_keys:
            messagebox.showwarning("提示", "请先勾选需要删除的条目。")
            return
        confirmed = messagebox.askyesno("确认删除", f"确定删除当前知识库中的 {len(selected_keys)} 个条目吗？")
        if not confirmed:
            return

        try:
            vectordb = Chroma(
                persist_directory=self.current_library["path"],
                embedding_function=ModelManager.get_embeddings(),
                collection_name="rag_collection",
            )
            ids_to_delete = []
            delete_sources = set()
            delete_parents = set()
            for key in selected_keys:
                ids_to_delete.extend(self.file_id_map.get(key, []))
                delete_meta = self.file_delete_sources.get(key, {})
                for source in delete_meta.get("sources", []):
                    delete_sources.add(source)
                parent = delete_meta.get("parent")
                if parent:
                    delete_parents.add(parent)

            ids_to_delete = list(dict.fromkeys(ids_to_delete))
            for start in range(0, len(ids_to_delete), 500):
                vectordb.delete(ids=ids_to_delete[start : start + 500])

            pkl_path = os.path.join(self.current_library["path"], "chunks.pkl")
            if os.path.exists(pkl_path):
                with open(pkl_path, "rb") as handle:
                    all_chunks = pickle.load(handle)
                new_chunks = []
                for chunk in all_chunks:
                    chunk_source = chunk.metadata.get("source")
                    chunk_parent = chunk.metadata.get("parent")
                    if chunk_source in delete_sources or chunk_parent in delete_parents:
                        continue
                    new_chunks.append(chunk)
                with open(pkl_path, "wb") as handle:
                    pickle.dump(new_chunks, handle)

            weights = self.load_all_weights()
            changed = False
            for key in list(delete_sources) + list(delete_parents):
                if key in weights:
                    weights.pop(key, None)
                    changed = True
            if changed:
                with open(self.get_weight_file(), "w", encoding="utf-8") as handle:
                    json.dump(weights, handle, ensure_ascii=False, indent=2)

            query.invalidate_library_cache(self.current_library["path"])
            self.refresh_file_list()
        except Exception as exc:
            messagebox.showerror("错误", f"删除失败：{exc}")

    def clear_all(self):
        if not self.current_library:
            return
        confirmed = messagebox.askyesno("确认清空", f"确定清空当前主库 [{self.current_library['name']}] 吗？")
        if not confirmed:
            return

        try:
            vectordb = Chroma(
                persist_directory=self.current_library["path"],
                embedding_function=ModelManager.get_embeddings(),
                collection_name="rag_collection",
            )
            try:
                vectordb.delete_collection()
            except Exception:
                pass

            for file_name in ["chunks.pkl", "file_weights.json"]:
                file_path = os.path.join(self.current_library["path"], file_name)
                if os.path.exists(file_path):
                    os.remove(file_path)

            query.invalidate_library_cache(self.current_library["path"])
            self.refresh_file_list()
            messagebox.showinfo("完成", "当前主库已清空。")
        except Exception as exc:
            messagebox.showerror("错误", f"清空失败：{exc}")

    def select_folder_ingest(self):
        if not self.current_library:
            messagebox.showwarning("提示", "请先确认主库。")
            return
        folder = filedialog.askdirectory()
        if not folder:
            return
        mode = "new" if messagebox.askyesno("导入模式", "是否清空当前主库后重建？") else "append"
        threading.Thread(target=self.run_ingest_thread, args=(folder, None, mode, False), daemon=True).start()

    def input_url_ingest(self):
        if not self.current_library:
            messagebox.showwarning("提示", "请先确认主库。")
            return
        dialog = ctk.CTkInputDialog(text="请输入网页 URL：", title="抓取网页")
        url = dialog.get_input()
        if not url:
            return
        recursive = messagebox.askyesno("递归抓取", "是否递归抓取该站点的子页面？")
        mode = "new" if messagebox.askyesno("导入模式", "是否清空当前主库后重建？") else "append"
        threading.Thread(target=self.run_ingest_thread, args=(None, [url], mode, recursive), daemon=True).start()

    def run_ingest_thread(self, docs_dir, urls, mode, recursive_web):
        self._append_progress("开始导入…")

        def callback(message):
            self.after(0, lambda: self._append_progress(message))

        try:
            ingest.run_ingest(
                docs_dir=docs_dir,
                urls=urls,
                mode=mode,
                recursive_web=recursive_web,
                progress_callback=callback,
                db_path=self.current_library["path"],
            )
            query.invalidate_library_cache(self.current_library["path"])
            self.after(0, self.refresh_file_list)
        except Exception as exc:
            error_text = str(exc)
            self.after(0, lambda error_text=error_text: self._append_progress(f"导入失败：{error_text}"))

    def _append_progress(self, message):
        self.progress_box.configure(state="normal")
        self.progress_box.insert("end", message + "\n")
        self.progress_box.see("end")
        self.progress_box.configure(state="disabled")

    def ask_question(self):
        if self.is_busy:
            return "break"
        if not self.current_config:
            messagebox.showwarning("提示", "请先在配置页保存配置，并在问答页选中它。")
            return "break"
        if not self.active_retrieval_libraries:
            messagebox.showwarning("提示", "请先确认主库和次库。")
            return "break"

        question = self.question_box.get("1.0", "end").strip()
        if not question:
            return "break"

        self.question_box.delete("1.0", "end")
        self.question_box.focus_set()

        primary = self.active_retrieval_libraries[0]
        secondaries = self.active_retrieval_libraries[1:]
        model_name = self.current_config.get("model", "assistant")

        self.is_busy = True
        self.ask_button.configure(state="disabled")
        self._set_answer_text(self._build_dialog_text(question, "正在处理…", model_name))
        self.set_pipeline_state("expand", "问题扩写中", "正在准备检索问题与 HyDE 内容…", completed=False)

        def progress(stage, message):
            summary_map = {
                "expand": "问题扩写中",
                "retrieve": "混合检索中",
                "rerank": "结果重排中",
                "generate": "答案生成中",
                "library": "知识库加载中",
                "model": "模型预加载中",
            }
            summary = summary_map.get(stage, "处理中")
            self.after(0, lambda: self.set_pipeline_state(stage, summary, message, completed=False))

        def run():
            try:
                answer, sources, tokens = query.query_rag(
                    question=question,
                    api_key=self.current_config.get("api_key", ""),
                    base_url=self.current_config.get("base_url", ""),
                    provider=self.current_config.get("provider", "DeepSeek"),
                    model=model_name,
                    primary_library=primary,
                    secondary_libraries=secondaries,
                    answer_style=self.answer_style_var.get(),
                    progress_callback=progress,
                )

                source_line = "\n\n参考来源：\n" + "\n".join(f"- {item}" for item in sources[:20]) if sources else ""
                final_text = self._build_dialog_text(question, answer + source_line, model_name)
                token_count = tokens.get("total_tokens", 0)
                self.after(0, lambda: self._set_answer_text(final_text))
                self.after(0, lambda: self.highlight_sources(sources))
                self.after(
                    0,
                    lambda: self.set_pipeline_state(
                        "generate",
                        "答案生成完成",
                        f"估算 token：{token_count}",
                        completed=True,
                    ),
                )
            except Exception as exc:
                error_text = str(exc)
                self.after(
                    0,
                    lambda error_text=error_text: self._set_answer_text(
                        self._build_dialog_text(question, f"请求失败：{error_text}", model_name)
                    ),
                )
                self.after(0, lambda error_text=error_text: self.set_pipeline_error("问答失败", error_text))
            finally:
                self.after(0, self._finish_question)

        threading.Thread(target=run, daemon=True).start()
        return "break"

    def _build_dialog_text(self, question, answer, model_name):
        return f"user\n{question}\n\n{model_name}\n{answer}"

    def _finish_question(self):
        self.is_busy = False
        self.ask_button.configure(state="normal")
        self.question_box.focus_set()

    def _set_answer_text(self, text):
        self.latest_answer_text = text
        self.answer_display.configure(state="normal")
        self.answer_display.delete("1.0", "end")
        self.answer_display.insert("1.0", text)
        self.answer_display.see("1.0")
        self.answer_display.configure(state="disabled")

    def save_answer(self):
        content = self.latest_answer_text.strip()
        if not content:
            messagebox.showwarning("提示", "当前没有可保存的回答。")
            return

        file_path = filedialog.asksaveasfilename(
            title="保存回答",
            initialfile="rag_anwser.txt",
            defaultextension=".txt",
            filetypes=[
                ("Text Files", "*.txt"),
                ("Markdown Files", "*.md"),
                ("All Files", "*.*"),
            ],
        )
        if not file_path:
            return

        try:
            with open(file_path, "w", encoding="utf-8") as handle:
                handle.write(content)
            messagebox.showinfo("成功", f"回答已保存到：{file_path}")
        except Exception as exc:
            messagebox.showerror("错误", f"保存失败：{exc}")

    def highlight_sources(self, active_sources):
        for item in self.file_checkboxes.values():
            item["widget"].configure(text_color=TEXT)

        for source_path, item in self.file_checkboxes.items():
            for active_source in active_sources:
                if active_source in source_path or source_path in active_source:
                    item["widget"].configure(text_color=SUCCESS)
                    break


if __name__ == "__main__":
    app = AcademicRAGApp()
    app.mainloop()
