import pickle
import shutil
import gc
import customtkinter as ctk
from tkinter import filedialog, scrolledtext, messagebox
import threading
import json
import os

from langchain_chroma import Chroma
import ingest
from model import ModelManager
import query
import time

# 美化主题设置
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

import sys

def get_resource_path(relative_path):
    """获取程序运行时的绝对路径（兼容开发环境和打包后的环境）"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

CONFIG_FILE = "config.json"

class PhilosophyRAGApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("RAG Pro - 本地知识库")
        self.geometry("1450x880")

        # 1. 先加载库数据
        self.libraries_file = "libraries.json"
        self.libraries = self.load_libraries()
        self.current_library = self.libraries[0] if self.libraries else {"name": "默认库", "path": "./chroma_db"}

        # 核心数据
        self.configs = self.load_configs()
        self.current_config = None
        self.file_checkboxes = {}   # 保存 source -> Checkbox 对象的映射
        self.file_id_map = {}       # 保存 source -> document_ids 的映射

        # 2. 调用侧边栏 (必须在 create_tabs 之前)
        self.setup_sidebar()

        self.create_tabs()
        self.set_status("loading", "🚀 正在初始化 AI 模型与本地向量库...")
             
        ModelManager.initialize_model(callback=self.on_startup_complete)

        self.libraries_file = "libraries.json"
        self.libraries = self.load_libraries() # 格式: [{"name": "默认库", "path": "./chroma_db"}]
        self.current_library = self.libraries[0] if self.libraries else {"name": "默认库", "path": "./chroma_db"}
        
        self.cleanup_orphan_folders() # 启动时清理

    def on_startup_complete(self):
            
        self.after(0, lambda: self.set_status("loading", "📦 模型加载成功，正在扫描向量库..."))
        self.after(500, self.refresh_file_list)   

    # ==================== 添加库 ====================
    def load_libraries(self):
        if os.path.exists(self.libraries_file):
            with open(self.libraries_file, "r", encoding="utf-8") as f: return json.load(f)
        return [{"name": "默认库", "path": "./chroma_db"}]

    def save_libraries(self):
        with open(self.libraries_file, "w", encoding="utf-8") as f:
            json.dump(self.libraries, f, ensure_ascii=False, indent=2)

    def load_configs(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def save_configs(self):
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(self.configs, f, ensure_ascii=False, indent=2)

    # --- 新增：权重管理逻辑 ---
    def get_weight_file(self):
        return os.path.join(self.current_library["path"], "file_weights.json")

    def load_all_weights(self):
        path = self.get_weight_file()
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f: return json.load(f)
        return {}

    def update_file_weight(self, source, weight):
        weights = self.load_all_weights()
        weights[source] = weight
        with open(self.get_weight_file(), "w", encoding="utf-8") as f:
            json.dump(weights, f, ensure_ascii=False, indent=2)


    # 在 create_tabs 之前调用
    def setup_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.pack(side="left", fill="y", padx=0, pady=0)
        
        ctk.CTkLabel(self.sidebar, text="📚 知识库列表", font=("Microsoft YaHei", 16, "bold")).pack(pady=20)
        
        self.lib_list_frame = ctk.CTkScrollableFrame(self.sidebar, fg_color="transparent")
        self.lib_list_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        ctk.CTkButton(self.sidebar, text="+ 新建知识库", command=self.create_new_library).pack(pady=10, padx=10)
        self.refresh_sidebar()

    def refresh_sidebar(self):
        for widget in self.lib_list_frame.winfo_children(): widget.destroy()
        for lib in self.libraries:
            btn_frame = ctk.CTkFrame(self.lib_list_frame, fg_color="transparent")
            btn_frame.pack(fill="x", pady=2)
            
            # 切换按钮
            color = "#1f538d" if lib["path"] == self.current_library["path"] else "gray25"
            btn = ctk.CTkButton(btn_frame, text=lib["name"], fg_color=color, anchor="w",
                                command=lambda l=lib: self.switch_library(l))
            btn.pack(side="left", fill="x", expand=True, padx=(0, 5))
            
            # 删除按钮
            if lib["path"] != "./chroma_db": # 保护默认库
                ctk.CTkButton(btn_frame, text="×", width=30, fg_color="#A52A2A", 
                            command=lambda l=lib: self.delete_library(l)).pack(side="right")

    def create_tabs(self):
        
        # 增加整体 padding 让界面透气
        self.tabview = ctk.CTkTabview(self, corner_radius=20)
        self.tabview.pack(side="right", fill="both", expand=True, padx=20, pady=20)

        tab_chat = self.tabview.add("💬 聊天提问")
        tab_files = self.tabview.add("📂 文件/URL管理")
        tab_ingest = self.tabview.add("📥 导入数据")
        tab_settings = self.tabview.add("⚙️ API 设置")

        self.create_chat_tab(tab_chat)
        self.create_files_tab(tab_files)
        self.create_ingest_tab(tab_ingest)
        self.create_settings_tab(tab_settings)
        
        self.tabview.set("💬 聊天提问")

    # ==================== UI 构建: 设置 ====================

    def on_provider_change(self, choice):
        # 预设各个提供商的默认 URL 和 模型名
        presets = {
            "DeepSeek": ("https://api.deepseek.com/v1", "deepseek-chat"),
            "OpenAI": ("https://api.openai.com/v1", "gpt-4o"),
            "xAI": ("https://api.x.ai/v1", "grok-beta"),
            "Gemini": ("N/A (SDK自动处理)", "gemini-1.5-flash"),
            "Qwen": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus")
        }
        url, model = presets.get(choice, ("", ""))
        self.url_entry.delete(0, "end")
        self.url_entry.insert(0, url)
        self.model_entry.delete(0, "end")
        self.model_entry.insert(0, model)

    def create_settings_tab(self, tab):

        frame = ctk.CTkFrame(tab, corner_radius=10)
        frame.pack(pady=40, padx=100, fill="both", expand=True)

        ctk.CTkLabel(frame, text="模型 API 配置", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=(30, 20))

        top_row = ctk.CTkFrame(frame, fg_color="transparent")
        top_row.pack(fill="x", padx=60, pady=10)
        
        self.config_combo = ctk.CTkComboBox(top_row, values=[c["name"] for c in self.configs] or ["无配置"],
                                             width=300)
        self.config_combo.pack(side="left", padx=(0, 20))
        ctk.CTkButton(top_row, text="加载此配置", command=self.load_selected_config, fg_color="#2E8B57", 
                        hover_color="#3CB371").pack(side="left")

        ctk.CTkFrame(frame, height=2, fg_color="gray30").pack(fill="x", padx=60, pady=20) # 分割线

        form_frame = ctk.CTkFrame(frame, fg_color="transparent")
        form_frame.pack(fill="x", padx=60)

        def add_field(parent, label_text, default_val="", is_pwd=False):
            row = ctk.CTkFrame(parent, fg_color="transparent")
            row.pack(fill="x", pady=8)
            ctk.CTkLabel(row, text=label_text, width=100, anchor="w").pack(side="left")
            entry = ctk.CTkEntry(row, show="*" if is_pwd else "")
            entry.pack(side="left", fill="x", expand=True)
            if default_val: entry.insert(0, default_val)
            return entry

        self.new_name = add_field(form_frame, "配置名称:", "My-Config")

        # --- 修改提供商下拉菜单 ---
        row_prov = ctk.CTkFrame(form_frame, fg_color="transparent")
        row_prov.pack(fill="x", pady=8)
        ctk.CTkLabel(row_prov, text="提供商:", width=100, anchor="w").pack(side="left")
        self.provider_var = ctk.StringVar(value="DeepSeek")

        ctk.CTkOptionMenu(row_prov, 
                        values=["DeepSeek", "OpenAI", "Gemini", "xAI", "Qwen"], 
                        variable=self.provider_var,
                        command=self.on_provider_change # 增加联动逻辑
                        ).pack(side="left", fill="x", expand=True)

        # --- 增加模型名称输入框 ---
        self.model_entry = add_field(form_frame, "模型名称:", "deepseek-chat") 
        self.key_entry = add_field(form_frame, "API Key:", "", is_pwd=True)
        self.url_entry = add_field(form_frame, "Base URL:", "https://api.deepseek.com/v1")

        save_btn = ctk.CTkButton(
            frame, 
            text="💾 保存为新配置方案", 
            command=self.save_new_config, 
            height=45, 
            font=("Microsoft YaHei", 14, "bold"),
            fg_color="#1f538d"
        )
        save_btn.pack(pady=(20, 30))
        
    def save_new_config(self):
        name = self.new_name.get().strip()
        if not name: return messagebox.showwarning("警告", "请输入配置名称")
        config = {
            "name": name, 
            "provider": self.provider_var.get(),
            "model": self.model_entry.get().strip(),
            "api_key": self.key_entry.get().strip(),
            "base_url": self.url_entry.get().strip()
        }
        self.configs.append(config)
        self.save_configs()
        messagebox.showinfo("成功", "配置已保存！")
        self.config_combo.configure(values=[c["name"] for c in self.configs])

         
    # ==================== UI 构建: Ingest ====================
    def create_ingest_tab(self, tab):

        top_frame = ctk.CTkFrame(tab, fg_color="transparent")
        top_frame.pack(fill="x", padx=40, pady=20)

        left_card = ctk.CTkFrame(top_frame, corner_radius=10)
        left_card.pack(side="left", fill="both", expand=True, padx=(0, 10))
        ctk.CTkLabel(left_card, text="📄 导入本地文件", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=15)
        ctk.CTkLabel(left_card, text="支持 PDF, Word (.docx), TXT 格式\n将遍历选择的文件夹下的所有符合条件的文件", text_color="gray60").pack(pady=5)
        ctk.CTkButton(left_card, text="选择文件夹并解析", command=self.select_folder_ingest, height=40).pack(pady=20)

        right_card = ctk.CTkFrame(top_frame, corner_radius=10)
        right_card.pack(side="left", fill="both", expand=True, padx=(10, 0))
        ctk.CTkLabel(right_card, text="🌐 抓取网页内容", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=15)
        ctk.CTkLabel(right_card, text="输入 URL\n可选是否深度递归抓取子页面", text_color="gray60").pack(pady=5)
        ctk.CTkButton(right_card, text="输入 URL 并解析", command=self.input_url_ingest, height=40).pack(pady=20)

        # 日志区美化
        ctk.CTkLabel(tab, text="终端执行日志", anchor="w").pack(fill="x", padx=40, pady=(10,0))
        self.progress_box = scrolledtext.ScrolledText(tab, height=12, font=("Consolas", 12), bg="#1e1e1e", fg="#00ff00")
        self.progress_box.pack(fill="both", expand=True, padx=40, pady=(5, 20))

    # ==================== UI 构建: 文件管理 ====================
    def create_files_tab(self, tab):

        toolbar = ctk.CTkFrame(tab, corner_radius=10)
        toolbar.pack(fill="x", padx=20, pady=10)

        # 工具栏：全选、删除等
        ctk.CTkButton(toolbar, text="全选", width=80, command=self.select_all_files).pack(side="left", padx=10, pady=10)
        ctk.CTkButton(toolbar, text="取消全选", width=80, command=self.deselect_all_files, fg_color="gray40", hover_color="gray50").pack(side="left", padx=10)
        ctk.CTkButton(toolbar, text="🔄 刷新列表", width=100, command=self.refresh_file_list, fg_color="#3a7ebf", hover_color="#1f538d").pack(side="left", padx=20)
        
        ctk.CTkButton(toolbar, text="🗑️ 清空全部数据", width=120, command=self.clear_all, fg_color="#8B0000", hover_color="#CD5C5C").pack(side="right", padx=10)
        ctk.CTkButton(toolbar, text="🗑️ 删除选中项", width=120, command=self.delete_selected_file, fg_color="#c93434", hover_color="#e05151").pack(side="right", padx=10)

        # 列表区
        self.file_frame = ctk.CTkScrollableFrame(tab, corner_radius=10)
        self.file_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))

    # ==================== UI 构建: 聊天 ====================
    def create_chat_tab(self, tab):
        # 1. 强力清除 Tab 的默认 grid 布局干扰，建立一个纯净的 base_frame
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        base_container = ctk.CTkFrame(tab, fg_color="transparent")
        base_container.grid(row=0, column=0, sticky="nsew")

        style_frame = ctk.CTkFrame(base_container, fg_color="transparent")
        style_frame.pack(fill="x", padx=20, pady=5)
        
        ctk.CTkLabel(style_frame, text="✍️ 回答风格:", font=("Microsoft YaHei", 12)).pack(side="left")
        
        # 对应 academic_prompt.py 中的中文名称
        self.answer_style_var = ctk.StringVar(value="内容论述")
        style_menu = ctk.CTkOptionMenu(
            style_frame, 
            values=["内容论述", "文献综述", "概念梳理", "引文补注", "盲审审稿"],
            variable=self.answer_style_var,
            width=150,
            fg_color="#3a7ebf",
            button_color="#1f538d"
        )
        style_menu.pack(side="left", padx=10)

        # 2. 聊天记录区域 (用 pack)
        self.chat_display = scrolledtext.ScrolledText(
            base_container, font=("Segoe UI", 11), wrap="word", 
            bg="#1e1e1e", fg="#ffffff", insertbackground="white",
            padx=15, pady=15, borderwidth=0
        )
        self.chat_display.pack(fill="both", expand=True, padx=20, pady=(20, 5))

        welcome_text = (
            "💡 欢迎使用本地学术知识库！\n\n"
            "请在选择完config后在文件管理面板中勾选你需要查询的资料。\n"
            "你可以基于这些语料进行深度提问...\n\n"
            + "="*60 + "\n"
        )
        self.chat_display.insert("0.0", welcome_text)
        
        # 3. 重点：创建一个专门的状态 Frame (status_frame)
        # 这样加载进度条和状态文字就可以在里面并排 pack 了
        status_frame = ctk.CTkFrame(base_container, fg_color="transparent", height=30)
        status_frame.pack(fill="x", padx=20)

        # 上层文字：初始隐形或显示初始化
        self.status_label = ctk.CTkLabel(status_frame, text="", font=("Microsoft YaHei", 12))
        self.status_label.pack(side="top", expand=True)

        # 这里的父容器也必须是 status_frame 
        self.loading_bar = ctk.CTkProgressBar(status_frame, width=500, height=8, mode="indeterminate")
        self.loading_bar.set(0)
        # 既然 status_frame 内部使用 pack，这里就安全了
        # 注意：如果原来这行报错，检查 self.loading_bar 的第一个参数是不是 status_frame

        # 4. 输入区域
        input_frame = ctk.CTkFrame(base_container, fg_color="transparent")
        input_frame.pack(fill="x", padx=20, pady=(5, 20))

        self.input_entry = ctk.CTkEntry(
            input_frame, placeholder_text="输入问题...", 
            height=50, font=("Microsoft YaHei", 12)
        )
        self.input_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        send_btn = ctk.CTkButton(
            input_frame, text="发送", width=100, height=50,
            command=self.ask_question, fg_color="#1f538d"
        )
        send_btn.pack(side="right")

    def set_status(self, mode: str, text: str = ""):
        """
        mode: "loading" (转圈), "success" (变绿打勾), "idle" (完全隐藏), "error" (变红)
        """
        self.status_label.configure(text=text)
        
        if mode == "loading":
            self.status_label.configure(text_color="#FFD700") # 金色文字表示思考中
            self.loading_bar.pack(side="top", pady=5)
            self.loading_bar.configure(mode="indeterminate", progress_color="#1f538d")
            self.loading_bar.start()
            
        elif mode == "success":
            self.loading_bar.stop()
            self.loading_bar.configure(mode="determinate", progress_color="#2ea043") # 绿色
            self.loading_bar.set(1.0) # 装满
            self.status_label.configure(text=f"✅ {text}", text_color="#2ea043")
            # 保持 5 秒后自动变回灰色或者隐形（可选）
            # self.after(5000, lambda: self.status_label.configure(text_color="gray50"))

        elif mode == "error":
            self.loading_bar.stop()
            self.loading_bar.configure(mode="determinate", progress_color="#a52a2a") # 红色
            self.loading_bar.set(1.0)
            self.status_label.configure(text=f"❌ {text}", text_color="#a52a2a")

        elif mode == "idle":
            self.loading_bar.pack_forget()
            self.status_label.configure(text="")

    def append_chat(self, role, message):
        import re
        tag = "user" if role == "USER" else "bot"
        
        # 定义样式标签
        self.chat_display.tag_config("user", foreground="#58a6ff", font=("Segoe UI", 11, "bold"))
        self.chat_display.tag_config("bot", foreground="#2ea043", font=("Segoe UI", 11, "bold"))
        self.chat_display.tag_config("bold_gold", foreground="#FFD700", font=("Segoe UI", 11, "bold"))
        self.chat_display.tag_config("heading", foreground="#ffffff", font=("Segoe UI", 13, "bold"))

        self.chat_display.insert("end", f"\n● {role}\n", tag)
        
        # 简单的 Markdown 解析逻辑
        lines = message.split('\n')
        for line in lines:
            if line.startswith('## '): # 处理二级标题
                self.chat_display.insert("end", line + "\n", "heading")
            else:
                # 处理 **加粗**
                parts = re.split(r'(\*\*.*?\*\*)', line)
                for part in parts:
                    if part.startswith("**") and part.endswith("**"):
                        self.chat_display.insert("end", part[2:-2], "bold_gold")
                    else:
                        self.chat_display.insert("end", part)
                self.chat_display.insert("end", "\n")
        
        self.chat_display.insert("end", "—" * 30 + "\n")
        self.chat_display.see("end")

    def ask_question(self):
        if not self.current_config:
            messagebox.showwarning("配置缺失", "请先在设置中加载 API 配置")
            return
            
        question = self.input_entry.get().strip()
        if not question: return
        
        selected_sources = [src for src, data in self.file_checkboxes.items() if data["var"].get()]
        if not selected_sources:
            messagebox.showwarning("无参考源", "请在文件管理中至少勾选一个文档")
            return
        
        style = self.answer_style_var.get()
        self.set_status("loading", f"🤖 正在以 [{style}] 模式检索资料并深度思考...")

        self.append_chat("USER", question)
        self.input_entry.delete(0, "end")

        def run():
            try:
                # 传入整个 config 字典，方便在 query.py 中判断 provider
                answer, sources, tokens = query.query_rag(
                    question=question,
                    api_key=self.current_config["api_key"],
                    base_url=self.current_config["base_url"],
                    provider=self.current_config["provider"],
                    model=self.current_config.get("model", "deepseek-chat"),
                    selected_files=selected_sources,
                    db_path=self.current_library["path"],
                    answer_style=style,
                )

                used_tokens = tokens.get('total_tokens', 0)
                msg = f"生成完毕 | 消耗 {used_tokens} tokens | 参考了 {len(sources)} 处资料"
                
                self.after(0, lambda: self.set_status("success", msg))
                self.after(0, lambda: self.highlight_sources(sources)) 
                self.after(0, lambda: self.append_chat("AI", answer))

            except Exception as e:
                self.after(0, lambda: self.set_status("error", f"请求失败: {str(e)[:50]}..."))
                self.after(0, lambda: self.append_chat("SYSTEM", f"错误详情: {e}"))

        threading.Thread(target=run, daemon=True).start()
    # ==================== 逻辑方法补充 ====================
    # (保留原有 save_new_config, load_selected_config 等)
    def save_new_config(self):
        name = self.new_name.get().strip()
        if not name: return messagebox.showwarning("警告", "请输入配置名称")
        config = {
            "name": name, 
            "provider": self.provider_var.get(),
            "model": self.model_entry.get().strip(),
            "api_key": self.key_entry.get().strip(),
            "base_url": self.url_entry.get().strip()
        }
        self.configs.append(config)
        self.save_configs()
        messagebox.showinfo("成功", "配置已保存！")
        self.config_combo.configure(values=[c["name"] for c in self.configs])

    def load_selected_config(self):
        name = self.config_combo.get()
        for c in self.configs:
            if c["name"] == name:
                self.current_config = c
                messagebox.showinfo("已加载", f"已切换到配置：{name}")
                return
        messagebox.showwarning("错误", "未找到配置")

    # Ingest 相关保持原逻辑框架，增加 UI 联动
    def select_folder_ingest(self):
        folder = filedialog.askdirectory()
        if folder:
            mode = "new" if messagebox.askyesno("模式选择", "是否清空原有数据库重新构建？\n(选否将追加到现有库)") else "append"
            threading.Thread(target=self.run_ingest_thread, args=(folder, None, mode, False), daemon=True).start()

    def input_url_ingest(self):
        dialog = ctk.CTkInputDialog(text="请输入网页 URL：", title="抓取网页")
        url = dialog.get_input()
        if url:
            recursive = messagebox.askyesno("递归抓取", "是否启用递归抓取子页面？\n(适合整个百科站，但耗时较长)")
            mode = "new" if messagebox.askyesno("模式选择", "是否清空原有数据库重新构建？\n(选否将追加)") else "append"
            threading.Thread(target=self.run_ingest_thread, args=(None, [url], mode, recursive), daemon=True).start()

    def run_ingest_thread(self, pdf_dir, urls, mode, recursive_web):
        self.progress_box.delete(1.0, "end")
        
        def cb(msg):
            self.progress_box.insert("end", msg + "\n")
            self.progress_box.see("end")
        
        try:
            ingest.run_ingest(docs_dir=pdf_dir, urls=urls, mode=mode, recursive_web=recursive_web, 
                              progress_callback=cb, db_path=self.current_library["path"])
            self.after(500, self.refresh_file_list) # 刷新UI列表
        except Exception as e:
            cb(f"❌ 运行报错: {e}")

    # ===== 文件刷新与分类展示 =====
    def refresh_file_list(self):
        # 1. 初始化清理 UI
        for widget in self.file_frame.winfo_children():
            widget.destroy()
        
        self.status_label.configure(text="正在读取数据库索引...", text_color="orange")
       
        def run_query():
            try:
                if ModelManager.get_embeddings() is None:
                    return
                
                # 获取数据库连接
                vectordb = Chroma(
                    persist_directory=self.current_library["path"], 
                    embedding_function=ModelManager.get_embeddings(), 
                    collection_name="rag_collection"
                )
                
                # 获取总记录数，准备分页加载
                collection_count = vectordb._collection.count()
                if collection_count == 0:
                    self.after(0, lambda: self._render_empty_state())
                    return

                all_metas = []
                all_ids = []
                
                # 💡 核心逻辑：分批抓取 (每批取 1000 条元数据)
                # 这样即使有 10 万条数据，也不会触发 "too many SQL variables" 报错
                batch_size = 1000 
                for i in range(0, collection_count, batch_size):
                    batch_data = vectordb.get(
                        include=["metadatas"], 
                        limit=batch_size, 
                        offset=i
                    )
                    all_metas.extend(batch_data["metadatas"])
                    all_ids.extend(batch_data["ids"])

                # 2. 构建层级化数据树 (Tree Structure)                

                del vectordb
                gc.collect()
                current_weights = self.load_all_weights()
                self.after(0, lambda: self._render_file_tree(all_metas, all_ids, collection_count, current_weights))

            except Exception as e:
                # 给出具体的错误提示
                import traceback
                print(traceback.format_exc())
                self.after(0, lambda: self.status_label.configure(text=f"⚠️ 读取失败: {e}", text_color="red".pack(pady=20)))

        threading.Thread(target=run_query, daemon=True).start()
    
    def _render_file_tree(self, metas, ids, count, current_weights):
        self.file_checkboxes.clear()
        self.file_id_map.clear()
        self.type_colors = {
            "PDF": "#e55353", "WORD": "#4382c4", "TXT": "#808080", "URL": "#4caf50", "其他": "gray"
        }
        tree = {"PDF": {}, "WORD": {}, "TXT": {}, "URL": {}, "其他": {}}

        for idx, m in enumerate(metas):
            if not m: continue
            src = m.get("source", "未知来源")
            raw_tp = str(m.get("type", "其他")).upper()
            
            # 统一类型
            if raw_tp in ["DOC", "DOCX", "WORD"]: tp = "WORD"
            elif raw_tp == "PDF": tp = "PDF"
            elif raw_tp == "TXT": tp = "TXT"
            elif raw_tp == "URL": tp = "URL"
            else: tp = "其他"

            if tp == "URL":
                parent = m.get("parent", src) # 获取母网页 URL
                if parent not in tree["URL"]: tree["URL"][parent] = {}
                if src not in tree["URL"][parent]: tree["URL"][parent][src] = []
                tree["URL"][parent][src].append(ids[idx])
            else:
                if src not in tree[tp]: tree[tp][src] = []
                tree[tp][src].append(ids[idx])

        # 3. 渲染 UI (带彩色折叠功能)
        for cat_name, content in tree.items():
            if not content: continue
            # 调用你之前的彩色折叠辅助函数
            self.create_collapsible_category(cat_name, content, current_weights)

        msg = f"当前库: {self.current_library['name']} | 共 {count} 个知识片段"
        self.set_status("success", msg)

    def add_single_checkbox(self, container, src, name, doc_ids, color, cat_name, current_weights):
        row = ctk.CTkFrame(container, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=1)
        
        display_name = (name[:120] + "...") if len(name) > 53 else name
        var = ctk.BooleanVar(value=True)

        w_val = str(current_weights.get(src, "1"))
        w_menu = ctk.CTkOptionMenu(row, values=["1", "2", "3"], width=55, height=22,
                                   command=lambda v, s=src: self.update_file_weight(s, v))
        w_menu.set(w_val)
        w_menu.pack(side="right", padx=5)
        
        ctk.CTkLabel(row, text="Lvl:", font=("", 10), text_color="gray60").pack(side="right")

        # 💡 增加 cat_name 标识用于批量控制
        cb = ctk.CTkCheckBox(row, text=display_name, variable=var, text_color=color, font=("Microsoft YaHei", 12))
        cb.pack(side="left", fill="x", expand=True, padx=5)

        self.file_checkboxes[src] = {"var": var, "widget": cb, "cat_name": cat_name}
        self.file_id_map[src] = doc_ids

    def create_collapsible_category(self, cat_name, content, current_weights):
        cat_color = self.type_colors.get(cat_name, "gray")
        
        # 分类标题栏 (Header)
        header = ctk.CTkFrame(self.file_frame, fg_color="gray20", corner_radius=5)
        header.pack(fill="x", pady=2, padx=5)
       
        cat_var = ctk.BooleanVar(value=True)
        def toggle_cat():
            is_checked = cat_var.get()
            # 遍历 tree 数据，找到属于这个分类的所有 source 并设置 var
            for src, data in self.file_checkboxes.items():
                # 这里需要你在存储时多存一个 type 字段
                if data.get("cat_name") == cat_name:
                    data["var"].set(is_checked)

        cat_cb = ctk.CTkCheckBox(header, text="", variable=cat_var, width=20, command=toggle_cat)
        cat_cb.pack(side="left", padx=5)
      
        # 内容容器 (默认收起 URL 这种可能有很多子项的内容)
        sub_frame = ctk.CTkFrame(self.file_frame, fg_color="transparent")
        is_initially_visible = (cat_name != "URL") # PDF/WORD 默认展开，URL 默认收起
        if is_initially_visible:
            sub_frame.pack(fill="x", padx=20)

        def toggle():
            if sub_frame.winfo_viewable():
                sub_frame.pack_forget()
                btn.configure(text="▶")
            else:
                sub_frame.pack(fill="x", padx=20, after=header)
                btn.configure(text="▼")

        btn = ctk.CTkButton(header, text="▼" if is_initially_visible else "▶", width=30, fg_color="transparent", command=toggle)
        btn.pack(side="left")
        
        ctk.CTkLabel(header, text=f"📂 {cat_name} 知识库 ({len(content)} 项)", 
                    font=("Microsoft YaHei", 13, "bold"), text_color=cat_color).pack(side="left", padx=5)

        # 4. 根据类型分流渲染
        if cat_name == "URL":
            for parent_url, children in content.items():
                self.create_url_nested_group(sub_frame, parent_url, children, cat_color, current_weights)
        else:
            for src_path, doc_ids in content.items():
                display_name = os.path.basename(src_path)
                self.add_single_checkbox(sub_frame, src_path, display_name, doc_ids, cat_color, cat_name, current_weights)
    
    def create_url_nested_group(self, container, parent_url, children, color, current_weights):
        group = ctk.CTkFrame(container, fg_color="transparent")
        group.pack(fill="x", pady=1)

        # 母 URL 行 (Parent Row)
        p_row = ctk.CTkFrame(group, fg_color="gray18", corner_radius=5)
        p_row.pack(fill="x")
        
        # 子页面容器
        c_box = ctk.CTkFrame(group, fg_color="transparent")
        
        # 汇总该母站点下的所有 ID (方便一次性操作)
        all_child_ids = []
        for c_ids in children.values(): all_ids_to_add = all_child_ids.extend(c_ids)

        display_url = f"🌐 站点: {parent_url}"
        display_url = (display_url[:45] + "...") if len(display_url) > 48 else display_url

        # 母 URL 的 Checkbox
        p_var = ctk.BooleanVar(value=True)
        p_cb = ctk.CTkCheckBox(p_row, text=f"🌐 站点: {parent_url[:50]}...", variable=p_var, text_color=color, font=("Bold", 12))
        p_cb.pack(side="left", padx=5, pady=5, fill="x", expand=True)
        
        # --- 新增：URL母条目权重控制 ---
        current_weights = self.load_all_weights()
        w_val = str(current_weights.get(parent_url, "1"))
        w_menu = ctk.CTkOptionMenu(p_row, values=["1", "2", "3"], width=55, height=22,
                                   command=lambda v, s=parent_url: self.update_file_weight(s, v))
        w_menu.set(w_val)
        w_menu.pack(side="right", padx=5)   
        
        # 核心映射：保存母站
        self.file_checkboxes[parent_url] = {"var": p_var, "widget": p_cb}
        self.file_id_map[parent_url] = all_child_ids

        # 如果有子页面，增加一个内部折叠按钮
        if len(children) > 1:
            def toggle_sub():
                if c_box.winfo_viewable(): c_box.pack_forget()
                else: c_box.pack(fill="x", padx=40)
            
            ctk.CTkButton(p_row, text="子页面..", width=60, height=20, font=("", 10), command=toggle_sub).pack(side="right", padx=10)
            
            for c_url, c_ids in children.items():
                if c_url == parent_url: continue # 跳过母页本身
                c_var = ctk.BooleanVar(value=True)
                c_cb = ctk.CTkCheckBox(c_box, text=f"└─ {c_url.replace(parent_url, '')}", variable=c_var, text_color="#aaaaaa", font=("Consolas", 10))
                c_cb.pack(anchor="w", pady=1)
                # 核心映射：保存子页
                self.file_checkboxes[c_url] = {"var": c_var, "widget": c_cb}
                self.file_id_map[c_url] = c_ids

    def add_file_checkbox(self, container, src, display_name, doc_ids):
        var = ctk.BooleanVar(value=True)
        cb = ctk.CTkCheckBox(container, text=display_name, variable=var)
        cb.pack(anchor="w", pady=2)
        self.file_checkboxes[src] = {"var": var, "widget": cb}
        self.file_id_map[src] = doc_ids
    
    # ===== 库的调整逻辑 =====
    def create_new_library(self):
        name = ctk.CTkInputDialog(text="请输入新知识库名称:", title="新建库").get_input()
        if name:
            db_path = f"./db_{int(time.time())}" # 生成唯一路径
            self.libraries.append({"name": name, "path": db_path})
            self.save_libraries()
            self.refresh_sidebar()

    def switch_library(self, lib):
        self.current_library = lib
        self.refresh_sidebar()
        self.refresh_file_list()
        # 聊天框提示切换
        self.append_chat("SYSTEM", f"已切换到知识库：{lib['name']}")
    
    def rename_library(self, lib):
        new_name = ctk.CTkInputDialog(text="请输入库的新名称:", title="重命名").get_input()
        if new_name:
            for l in self.libraries:
                if l["path"] == lib["path"]:
                    l["name"] = new_name
            self.save_libraries()
            self.refresh_sidebar()

    def delete_library(self, lib):
        if lib["path"] == "./chroma_db":
            messagebox.showwarning("提示", "默认知识库是系统的根基，不能删除。")
            return

        if messagebox.askyesno("确认删除", f"确定要永久删除知识库 [{lib['name']}] 吗？\n此操作将清空该库下的所有文件和向量数据。"):
            
            # 1. 核心逻辑：如果删除的是当前库，先切换到默认库
            if self.current_library["path"] == lib["path"]:
                self.switch_library(self.libraries[0])
            
            # 2. 从列表中移除并保存配置文件
            self.libraries = [l for l in self.libraries if l["path"] != lib["path"]]
            self.save_libraries()
            self.refresh_sidebar()

            # 3. 开启后台线程执行“强力删除”，避免阻塞 UI 导致卡死
            threading.Thread(target=self._force_delete_folder, args=(lib["path"],), daemon=True).start()

    def _force_delete_folder(self, path):

        temp_path = f"{path}_to_be_deleted"
        
        try:
            # 1. 多次回收
            for _ in range(3):
                gc.collect()
                time.sleep(0.2)
            
            # 2. 尝试改名 (Windows 系统的妙招)
            if os.path.exists(path):
                os.rename(path, temp_path)
                shutil.rmtree(temp_path)
                print(f"🗑️ 已成功物理删除: {path}")
        except Exception as e:
            print(f"⚠️ 物理删除失败(文件仍占用)，将在下次启动时自动清理: {e}")

    def cleanup_orphan_folders(self):
        """物理删除不在列表中的 db_ 文件夹"""
        import shutil
        valid_paths = [lib["path"] for lib in self.libraries]
        valid_paths.append("./chroma_db") # 保护默认库
        
        current_dir = os.getcwd()
        for item in os.listdir(current_dir):
            item_path = os.path.join(current_dir, item)
            # 识别 db_ 开头或 chroma_db 之外的文件夹
            if os.path.isdir(item_path) and (item.startswith("db_") or item == "chroma_db"):
                # 如果这个路径不在 libraries.json 里，直接删掉
                normalized_path = f"./{item}"
                if normalized_path not in valid_paths:
                    try:
                        shutil.rmtree(item_path)
                        print(f"🧹 已清理残留文件夹: {item}")
                    except:
                        pass

    # ===== 文件操作按钮逻辑 =====
    def select_all_files(self):
        for data in self.file_checkboxes.values():
            data["var"].set(True)

    def deselect_all_files(self):
        for data in self.file_checkboxes.values():
            data["var"].set(False)

    def delete_selected_file(self):
        selected_sources = [src for src, data in self.file_checkboxes.items() if data["var"].get()]
        if not selected_sources: 
            return messagebox.showwarning("提示", "请先在下方勾选要删除的文件/URL")

        if messagebox.askyesno("确认删除", f"确定要从知识库中彻底移除选中的 {len(selected_sources)} 项内容吗？"):
            ids_to_delete = []
            for src in selected_sources:
                ids_to_delete.extend(self.file_id_map.get(src, []))
            
            try:  
                vectordb = Chroma(persist_directory=self.current_library["path"], embedding_function=ModelManager.get_embeddings(), collection_name="rag_collection")
                batch_size = 500
                for i in range(0, len(ids_to_delete), batch_size):
                    batch = ids_to_delete[i : i + batch_size]
                    vectordb.delete(ids=batch)

                # 2. 同步更新 chunks.pkl (关键优化)
                if os.path.exists("chunks.pkl"):
                    with open("chunks.pkl", "rb") as f:
                        all_chunks = pickle.load(f)
                    # 只保留不在删除列表中的 chunks
                    new_chunks = [c for c in all_chunks if c.metadata.get("source") not in selected_sources]
                    with open("chunks.pkl", "wb") as f:
                        pickle.dump(new_chunks, f)
           
                self.refresh_file_list()
            except Exception as e:
                messagebox.showerror("错误", f"删除失败: {e}")

    def clear_all(self):
        if messagebox.askyesno("危险操作", "确定要清空知识库所有内容吗？此操作不可逆！"):
            try:
                vectordb = Chroma(persist_directory="./chroma_db", embedding_function=None, collection_name="rag_collection")
                vectordb.delete_collection()
                if os.path.exists("chunks.pkl"): os.remove("chunks.pkl")
                self.refresh_file_list()
                messagebox.showinfo("成功", "已全部清空")
            except Exception as e:
                messagebox.showerror("错误", f"清空失败: {e}")

    

    def highlight_sources(self, active_sources):
        """
        active_sources: AI 返回的来源列表 (可能是文件名或路径)
        """
        # 先重置所有颜色
        for item in self.file_checkboxes.values():
            item["widget"].configure(text_color="white")

        # 高亮匹配到的项
        for source_path, item in self.file_checkboxes.items():
            # 判断 source_path 是否包含在 AI 返回的 source 名中
            for active_s in active_sources:
                if active_s in source_path or source_path in active_s:
                    item["widget"].configure(text_color="#FFD700") # 金色高亮
                    # 还可以自动滚动到可见区域（可选）

if __name__ == "__main__":
    app = PhilosophyRAGApp()
    app.mainloop()