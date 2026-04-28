import customtkinter as ctk

import academic_prompt


EMPTY_OPTION = "（无）"

BG = "#f4f4f4"
SURFACE = "#ffffff"
SOFT = "#ededed"
BORDER = "#d7d7d7"
TEXT = "#111111"
MUTED = "#666666"
ACCENT = "#111111"
ACCENT_SOFT = "#2f2f2f"
SUCCESS = "#1f7a3d"
ERROR = "#b42318"
IN_PROGRESS = "#2f7dd1"


class AppUIBuilder:
    def __init__(self, app):
        self.app = app

    def setup_sidebar(self):
        self.app.sidebar = ctk.CTkFrame(
            self.app,
            width=230,
            corner_radius=0,
            fg_color=SURFACE,
            border_width=1,
            border_color=BORDER,
        )
        self.app.sidebar.pack(side="left", fill="y")
        self.app.sidebar.pack_propagate(False)

        ctk.CTkLabel(
            self.app.sidebar,
            text="知识库",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=TEXT,
        ).pack(anchor="w", padx=18, pady=(20, 8))

        ctk.CTkLabel(
            self.app.sidebar,
            text="管理库名、删除库，并在问答页选择主库和次库。",
            text_color=MUTED,
            justify="left",
            wraplength=180,
        ).pack(anchor="w", padx=18, pady=(0, 14))

        self.app.lib_list_frame = ctk.CTkScrollableFrame(
            self.app.sidebar,
            fg_color="transparent",
        )
        self.app.lib_list_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        ctk.CTkButton(
            self.app.sidebar,
            text="新建知识库",
            command=self.app.create_new_library,
            height=36,
            fg_color=ACCENT,
            hover_color=ACCENT_SOFT,
        ).pack(fill="x", padx=14, pady=(0, 16))

    def refresh_sidebar(self):
        for widget in self.app.lib_list_frame.winfo_children():
            widget.destroy()

        for library in self.app.libraries:
            row = ctk.CTkFrame(
                self.app.lib_list_frame,
                fg_color=SOFT if library["path"] == self.app.current_library["path"] else SURFACE,
                border_width=1,
                border_color=BORDER,
                corner_radius=8,
            )
            row.pack(fill="x", pady=4, padx=2)

            select_btn = ctk.CTkButton(
                row,
                text=library["name"],
                anchor="w",
                fg_color="transparent",
                hover_color=SOFT,
                text_color=TEXT,
                command=lambda item=library: self.app.preview_library(item),
            )
            select_btn.pack(side="left", fill="x", expand=True, padx=(6, 4), pady=6)

            ctk.CTkButton(
                row,
                text="r",
                width=26,
                height=26,
                fg_color=SOFT,
                hover_color="#dddddd",
                text_color=TEXT,
                command=lambda item=library: self.app.rename_library(item),
            ).pack(side="right", padx=(0, 6), pady=6)

            if library["path"] != "./chroma_db":
                ctk.CTkButton(
                    row,
                    text="x",
                    width=26,
                    height=26,
                    fg_color="#f8e7e7",
                    hover_color="#f2d1d1",
                    text_color=ERROR,
                    command=lambda item=library: self.app.delete_library(item),
                ).pack(side="right", padx=(0, 6), pady=6)

    def create_tabs(self):
        self.app.tabview = ctk.CTkTabview(self.app, corner_radius=12, fg_color=BG)
        self.app.tabview.pack(side="right", fill="both", expand=True, padx=16, pady=16)

        chat_tab = self.app.tabview.add("问答")
        files_tab = self.app.tabview.add("文件管理")
        ingest_tab = self.app.tabview.add("导入")
        settings_tab = self.app.tabview.add("配置")

        self.create_chat_tab(chat_tab)
        self.create_files_tab(files_tab)
        self.create_ingest_tab(ingest_tab)
        self.create_settings_tab(settings_tab)
        self.app.tabview.set("问答")

    def create_chat_tab(self, tab):
        container = ctk.CTkFrame(tab, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=10, pady=10)

        control = ctk.CTkFrame(
            container,
            width=138,
            fg_color=SURFACE,
            border_width=1,
            border_color=BORDER,
            corner_radius=10,
        )
        control.pack(side="left", fill="y", padx=(0, 14))
        control.pack_propagate(False)

        ctk.CTkLabel(
            control,
            text="回答控制",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=TEXT,
        ).pack(anchor="w", padx=14, pady=(16, 12))

        self.app.chat_config_var = ctk.StringVar(
            value=self.app.current_config["name"] if self.app.current_config else "未配置"
        )
        self.app.answer_style_var = ctk.StringVar(value=academic_prompt.STYLE_PHILOSOPHER)
        self.app.primary_library_var = ctk.StringVar(value=self.app.current_library["name"])
        self.app.secondary_library_var_1 = ctk.StringVar(value=EMPTY_OPTION)
        self.app.secondary_library_var_2 = ctk.StringVar(value=EMPTY_OPTION)

        self.app.chat_config_menu = self._add_option_field(
            control,
            "模型",
            self.app.chat_config_var,
            ["未配置"],
            self.app.on_chat_config_change,
        )
        self.app.answer_style_menu = self._add_option_field(
            control,
            "风格",
            self.app.answer_style_var,
            academic_prompt.get_supported_styles(),
            None,
        )
        self.app.primary_library_menu = self._add_option_field(
            control,
            "主库",
            self.app.primary_library_var,
            [self.app.current_library["name"]],
            None,
        )
        self.app.secondary_library_menu_1 = self._add_option_field(
            control,
            "次库1",
            self.app.secondary_library_var_1,
            [EMPTY_OPTION],
            None,
        )
        self.app.secondary_library_menu_2 = self._add_option_field(
            control,
            "次库2",
            self.app.secondary_library_var_2,
            [EMPTY_OPTION],
            None,
        )

        ctk.CTkButton(
            control,
            text="确认知识库",
            command=self.app.apply_library_selection,
            height=36,
            fg_color=ACCENT,
            hover_color=ACCENT_SOFT,
        ).pack(fill="x", padx=14, pady=(12, 0))

        content = ctk.CTkFrame(container, fg_color="transparent")
        content.pack(side="left", fill="both", expand=True)

        answer_card = ctk.CTkFrame(
            content,
            fg_color=SURFACE,
            border_width=1,
            border_color=BORDER,
            corner_radius=10,
        )
        answer_card.pack(fill="both", expand=True)

        ctk.CTkLabel(
            answer_card,
            text="回答",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=TEXT,
        ).pack(anchor="w", padx=16, pady=(14, 8))

        self.app.answer_display = ctk.CTkTextbox(
            answer_card,
            font=("Microsoft YaHei", 14),
            wrap="word",
            border_width=0,
        )
        self.app.answer_display.pack(fill="both", expand=True, padx=16, pady=(0, 14))
        self.app.answer_display.configure(state="disabled")

        bottom_bar = ctk.CTkFrame(
            content,
            fg_color=SURFACE,
            border_width=1,
            border_color=BORDER,
            corner_radius=10,
            height=156,
        )
        bottom_bar.pack(fill="x", pady=(12, 0))
        bottom_bar.pack_propagate(False)

        top_row = ctk.CTkFrame(bottom_bar, fg_color="transparent")
        top_row.pack(fill="x", pady=(10, 8))

        top_inner = ctk.CTkFrame(top_row, fg_color="transparent", width=1074, height=44)
        top_inner.pack(anchor="center")
        top_inner.pack_propagate(False)

        status_block = ctk.CTkFrame(top_inner, fg_color="transparent", width=944, height=44)
        status_block.pack(side="left", padx=(0, 12))
        status_block.pack_propagate(False)

        self.app.status_text = ctk.CTkLabel(
            status_block,
            text="模型预加载中 · 正在检查本地缓存…",
            text_color=MUTED,
            anchor="center",
            justify="center",
            font=ctk.CTkFont(size=11),
        )
        self.app.status_text.pack(fill="x", pady=(0, 6))

        self.app.progress_bar = ctk.CTkProgressBar(
            status_block,
            height=8,
            progress_color=IN_PROGRESS,
        )
        self.app.progress_bar.pack(fill="x", padx=12)
        self.app.progress_bar.set(0.35)

        ctk.CTkButton(
            top_inner,
            text="保存回答",
            command=self.app.save_answer,
            width=118,
            height=44,
            fg_color=SOFT,
            hover_color="#dddddd",
            text_color=TEXT,
        ).pack(side="left", anchor="center")

        input_row = ctk.CTkFrame(bottom_bar, fg_color="transparent")
        input_row.pack(fill="x", pady=(0, 12))

        input_inner = ctk.CTkFrame(input_row, fg_color="transparent", width=1074, height=74)
        input_inner.pack(anchor="center")
        input_inner.pack_propagate(False)

        label_box = ctk.CTkFrame(input_inner, fg_color="transparent", width=52, height=74)
        label_box.pack(side="left", padx=(0, 12))
        label_box.pack_propagate(False)

        ctk.CTkLabel(
            label_box,
            text="提问",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=TEXT,
        ).place(relx=0.5, rely=0.5, anchor="center")

        self.app.question_box = ctk.CTkTextbox(
            input_inner,
            width=880,
            height=74,
            font=("Microsoft YaHei", 13),
            wrap="word",
        )
        self.app.question_box.pack(side="left", padx=(0, 12))
        self.app.question_box.bind("<Control-Return>", lambda _event: self.app.ask_question())

        self.app.ask_button = ctk.CTkButton(
            input_inner,
            text="发送",
            command=self.app.ask_question,
            width=118,
            height=74,
            fg_color=ACCENT,
            hover_color=ACCENT_SOFT,
        )
        self.app.ask_button.pack(side="left")

    def _add_option_field(self, parent, label_text, variable, values, command):
        wrapper = ctk.CTkFrame(parent, fg_color="transparent")
        wrapper.pack(fill="x", padx=14, pady=6)
        ctk.CTkLabel(wrapper, text=label_text, anchor="w", text_color=TEXT).pack(anchor="w")
        menu = ctk.CTkOptionMenu(
            wrapper,
            values=values,
            variable=variable,
            command=command,
            height=32,
            fg_color=ACCENT,
            button_color=ACCENT_SOFT,
            button_hover_color="#4b4b4b",
        )
        menu.pack(fill="x", pady=(6, 0))
        return menu

    def create_files_tab(self, tab):
        container = ctk.CTkFrame(tab, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=12, pady=12)

        toolbar = ctk.CTkFrame(
            container,
            fg_color=SURFACE,
            border_width=1,
            border_color=BORDER,
            corner_radius=10,
        )
        toolbar.pack(fill="x", pady=(0, 12))

        header = ctk.CTkFrame(toolbar, fg_color="transparent")
        header.pack(fill="x", padx=12, pady=(10, 6))

        ctk.CTkLabel(
            header,
            text="内容编辑",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=TEXT,
        ).pack(side="left")
        ctk.CTkLabel(
            header,
            text="先在这里选择库，再做勾选、赋权和删除。URL 默认按母链接折叠。",
            text_color=MUTED,
        ).pack(side="left", padx=(10, 0))

        controls = ctk.CTkFrame(toolbar, fg_color="transparent")
        controls.pack(fill="x", padx=12, pady=(0, 10))

        ctk.CTkLabel(controls, text="编辑库", text_color=TEXT).pack(side="left", padx=(0, 8))
        self.app.file_library_var = ctk.StringVar(value=self.app.current_library["name"])
        self.app.file_library_menu = ctk.CTkOptionMenu(
            controls,
            values=[self.app.current_library["name"]],
            variable=self.app.file_library_var,
            command=self.app.select_file_library,
            width=220,
            fg_color=ACCENT,
            button_color=ACCENT_SOFT,
            button_hover_color="#4b4b4b",
        )
        self.app.file_library_menu.pack(side="left", padx=(0, 14))

        ctk.CTkButton(controls, text="全选", width=70, command=self.app.select_all_files, fg_color=SOFT, hover_color="#dddddd", text_color=TEXT).pack(side="left", padx=(0, 8))
        ctk.CTkButton(controls, text="清空选择", width=90, command=self.app.deselect_all_files, fg_color=SOFT, hover_color="#dddddd", text_color=TEXT).pack(side="left", padx=(0, 8))
        ctk.CTkButton(controls, text="刷新列表", width=90, command=self.app.refresh_file_list, fg_color=ACCENT, hover_color=ACCENT_SOFT).pack(side="left", padx=(0, 8))
        ctk.CTkButton(controls, text="删除选中", width=100, command=self.app.delete_selected_file, fg_color="#f8e7e7", hover_color="#f2d1d1", text_color=ERROR).pack(side="right", padx=(8, 0))
        ctk.CTkButton(controls, text="清空当前库", width=110, command=self.app.clear_all, fg_color="#f8e7e7", hover_color="#f2d1d1", text_color=ERROR).pack(side="right")

        self.app.file_frame = ctk.CTkScrollableFrame(
            container,
            fg_color=SURFACE,
            border_width=1,
            border_color=BORDER,
            corner_radius=10,
        )
        self.app.file_frame.pack(fill="both", expand=True)

    def create_ingest_tab(self, tab):
        container = ctk.CTkFrame(tab, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=12, pady=12)

        cards = ctk.CTkFrame(container, fg_color="transparent")
        cards.pack(fill="x", pady=(0, 12))

        left = ctk.CTkFrame(cards, fg_color=SURFACE, border_width=1, border_color=BORDER, corner_radius=10)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))
        right = ctk.CTkFrame(cards, fg_color=SURFACE, border_width=1, border_color=BORDER, corner_radius=10)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))

        ctk.CTkLabel(left, text="导入本地文件", font=ctk.CTkFont(size=18, weight="bold"), text_color=TEXT).pack(anchor="w", padx=16, pady=(16, 8))
        ctk.CTkLabel(left, text="支持 PDF、DOCX、TXT。确认主库后再执行导入。", text_color=MUTED, justify="left").pack(anchor="w", padx=16)
        ctk.CTkButton(left, text="选择文件夹并导入", command=self.app.select_folder_ingest, height=38, fg_color=ACCENT, hover_color=ACCENT_SOFT).pack(fill="x", padx=16, pady=16)

        ctk.CTkLabel(right, text="抓取网页", font=ctk.CTkFont(size=18, weight="bold"), text_color=TEXT).pack(anchor="w", padx=16, pady=(16, 8))
        ctk.CTkLabel(right, text="适合单篇页面或递归抓取站点。导入目标仍为当前主库。", text_color=MUTED, justify="left").pack(anchor="w", padx=16)
        ctk.CTkButton(right, text="输入 URL 并导入", command=self.app.input_url_ingest, height=38, fg_color=ACCENT, hover_color=ACCENT_SOFT).pack(fill="x", padx=16, pady=16)

        log_card = ctk.CTkFrame(container, fg_color=SURFACE, border_width=1, border_color=BORDER, corner_radius=10)
        log_card.pack(fill="both", expand=True)

        ctk.CTkLabel(log_card, text="导入日志", font=ctk.CTkFont(size=16, weight="bold"), text_color=TEXT).pack(anchor="w", padx=16, pady=(14, 8))
        self.app.progress_box = ctk.CTkTextbox(log_card, font=("Consolas", 12))
        self.app.progress_box.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self.app.progress_box.configure(state="disabled")

    def create_settings_tab(self, tab):
        frame = ctk.CTkFrame(
            tab,
            fg_color=SURFACE,
            border_width=1,
            border_color=BORDER,
            corner_radius=10,
        )
        frame.pack(fill="both", expand=True, padx=60, pady=30)

        ctk.CTkLabel(
            frame,
            text="模型配置",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=TEXT,
        ).pack(anchor="w", padx=24, pady=(24, 8))

        ctk.CTkLabel(
            frame,
            text="这里保存可直接在问答页切换的 config。问答页只负责选择，新增与修改在这里完成。",
            text_color=MUTED,
            justify="left",
        ).pack(anchor="w", padx=24, pady=(0, 20))

        self.app.provider_var = ctk.StringVar(value="DeepSeek")
        self.app.new_name = self._add_entry_field(frame, "配置名", "My-Config")
        self.app.model_entry = self._add_entry_field(frame, "模型名", "deepseek-chat")
        self.app.key_entry = self._add_entry_field(frame, "API Key", "", is_password=True)
        self.app.url_entry = self._add_entry_field(frame, "Base URL", "https://api.deepseek.com/v1")

        row = ctk.CTkFrame(frame, fg_color="transparent")
        row.pack(fill="x", padx=24, pady=8)
        ctk.CTkLabel(row, text="提供商", width=90, anchor="w", text_color=TEXT).pack(side="left")
        ctk.CTkOptionMenu(
            row,
            values=["DeepSeek", "OpenAI", "Gemini", "xAI", "Qwen"],
            variable=self.app.provider_var,
            command=self.app.on_provider_change,
            fg_color=ACCENT,
            button_color=ACCENT_SOFT,
            button_hover_color="#4b4b4b",
        ).pack(side="left", fill="x", expand=True)

        ctk.CTkButton(
            frame,
            text="保存配置",
            command=self.app.save_new_config,
            height=40,
            fg_color=ACCENT,
            hover_color=ACCENT_SOFT,
        ).pack(anchor="w", padx=24, pady=(16, 24))

    def _add_entry_field(self, parent, label_text, default_value="", is_password=False):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=24, pady=8)
        ctk.CTkLabel(row, text=label_text, width=90, anchor="w", text_color=TEXT).pack(side="left")
        entry = ctk.CTkEntry(row, show="*" if is_password else "")
        entry.pack(side="left", fill="x", expand=True)
        if default_value:
            entry.insert(0, default_value)
        return entry

    def refresh_chat_selectors(self):
        config_names = [config["name"] for config in self.app.configs] or ["未配置"]
        library_names = [library["name"] for library in self.app.libraries]
        secondary_options = [EMPTY_OPTION] + library_names

        self.app.chat_config_menu.configure(values=config_names)
        if self.app.current_config:
            self.app.chat_config_var.set(self.app.current_config["name"])
        else:
            self.app.chat_config_var.set("未配置")

        self.app.primary_library_menu.configure(values=library_names)
        self.app.secondary_library_menu_1.configure(values=secondary_options)
        self.app.secondary_library_menu_2.configure(values=secondary_options)

        if self.app.current_library["name"] not in library_names and library_names:
            self.app.current_library = self.app.libraries[0]
        if library_names:
            if self.app.primary_library_var.get() not in library_names:
                self.app.primary_library_var.set(self.app.current_library["name"])
            if hasattr(self.app, "file_library_menu"):
                self.app.file_library_menu.configure(values=library_names)
                self.app.file_library_var.set(self.app.current_library["name"])

    def render_empty_files(self, message):
        for widget in self.app.file_frame.winfo_children():
            widget.destroy()
        self.app.file_checkboxes.clear()
        self.app.file_id_map.clear()
        self.app.file_delete_sources.clear()
        self.app.file_weight_targets.clear()
        ctk.CTkLabel(self.app.file_frame, text=message, text_color=MUTED).pack(anchor="w", padx=16, pady=16)

    def _toggle_group(self, group_name):
        current = self.app.file_tree_state["groups"].get(group_name, False)
        self.app.file_tree_state["groups"][group_name] = not current
        self.app.rerender_file_tree()

    def _toggle_url_group(self, parent_url):
        current = self.app.file_tree_state["urls"].get(parent_url, False)
        self.app.file_tree_state["urls"][parent_url] = not current
        self.app.rerender_file_tree()

    def _normalize_file_type(self, raw_type):
        upper = str(raw_type or "OTHER").upper()
        if upper in {"DOC", "DOCX", "WORD"}:
            return "WORD"
        if upper in {"PDF", "WORD", "TXT", "URL"}:
            return upper
        return "OTHER"

    def _short_label(self, source, is_url=False, limit=88):
        if is_url:
            label = str(source or "")
        else:
            label = str(source or "").rsplit("\\", 1)[-1].rsplit("/", 1)[-1] or str(source or "")
        return label if len(label) <= limit else label[: limit - 3] + "..."

    def render_file_tree(self, metas, ids, count, current_weights):
        for widget in self.app.file_frame.winfo_children():
            widget.destroy()
        self.app.file_checkboxes.clear()
        self.app.file_id_map.clear()
        self.app.file_delete_sources.clear()
        self.app.file_weight_targets.clear()

        groups = {"PDF": {}, "WORD": {}, "TXT": {}, "URL": {}, "OTHER": {}}
        colors = {"PDF": "#b42318", "WORD": "#1d4ed8", "TXT": MUTED, "URL": SUCCESS, "OTHER": MUTED}
        titles = {"PDF": "PDF", "WORD": "Word", "TXT": "Text", "URL": "URL", "OTHER": "Other"}

        for idx, meta in enumerate(metas):
            if not meta:
                continue
            source = str(meta.get("source") or "未知来源")
            file_type = self._normalize_file_type(meta.get("type"))
            if file_type == "URL":
                parent_url = str(meta.get("parent") or source)
                entry = groups["URL"].setdefault(parent_url, {"ids": [], "children": {}})
                entry["ids"].append(ids[idx])
                entry["children"].setdefault(source, []).append(ids[idx])
            else:
                groups[file_type].setdefault(source, []).append(ids[idx])

        group_order = ["PDF", "WORD", "TXT", "URL", "OTHER"]
        for group_name in group_order:
            items = groups[group_name]
            if not items:
                continue
            expanded = self.app.file_tree_state["groups"].get(group_name, False)
            card = ctk.CTkFrame(self.app.file_frame, fg_color=SURFACE, border_width=1, border_color=BORDER, corner_radius=8)
            card.pack(fill="x", padx=10, pady=6)

            header = ctk.CTkButton(
                card,
                text=f"{'v' if expanded else '>'} {titles[group_name]} ({len(items)})",
                anchor="w",
                command=lambda name=group_name: self._toggle_group(name),
                fg_color="transparent",
                hover_color=SOFT,
                text_color=colors.get(group_name, TEXT),
                font=ctk.CTkFont(size=15, weight="bold"),
            )
            header.pack(fill="x", padx=8, pady=(8, 4))

            if not expanded:
                continue

            body = ctk.CTkFrame(card, fg_color="transparent")
            body.pack(fill="x", padx=6, pady=(0, 8))

            if group_name == "URL":
                for parent_url, payload in sorted(items.items(), key=lambda item: item[0].lower()):
                    self._render_url_parent_block(body, parent_url, payload, current_weights)
            else:
                for source, doc_ids in sorted(items.items(), key=lambda item: item[0].lower()):
                    self._render_file_row(
                        body,
                        key=source,
                        source=source,
                        doc_ids=doc_ids,
                        color=colors.get(group_name, TEXT),
                        current_weights=current_weights,
                        weight_target=source,
                        delete_sources=[source],
                        parent=None,
                        is_url=False,
                        indent=0,
                    )

        self.app.set_pipeline_state(
            "library",
            "内容编辑已就绪",
            f"当前知识库 [{self.app.current_library['name']}] 共 {count} 个片段。",
            completed=True,
        )

    def _render_url_parent_block(self, container, parent_url, payload, current_weights):
        box = ctk.CTkFrame(container, fg_color="#fafafa", border_width=1, border_color=BORDER, corner_radius=8)
        box.pack(fill="x", padx=4, pady=4)

        child_sources = sorted(payload["children"].items(), key=lambda item: item[0].lower())
        nested_sources = [(source, doc_ids) for source, doc_ids in child_sources if source != parent_url]
        expanded = self.app.file_tree_state["urls"].get(parent_url, False)

        self._render_file_row(
            box,
            key=f"url-parent::{parent_url}",
            source=parent_url,
            doc_ids=payload["ids"],
            color=SUCCESS,
            current_weights=current_weights,
            weight_target=parent_url,
            delete_sources=[source for source, _ in child_sources],
            parent=parent_url,
            is_url=True,
            indent=0,
            suffix=f"母 URL · {len(child_sources)} 页",
            toggle=(parent_url if nested_sources else None),
            toggle_expanded=expanded,
            toggle_count=len(nested_sources),
        )

        if expanded and nested_sources:
            child_wrap = ctk.CTkFrame(box, fg_color="transparent")
            child_wrap.pack(fill="x", padx=6, pady=(0, 6))
            for source, doc_ids in nested_sources:
                self._render_file_row(
                    child_wrap,
                    key=f"url-child::{parent_url}::{source}",
                    source=source,
                    doc_ids=doc_ids,
                    color=TEXT,
                    current_weights=current_weights,
                    weight_target=source,
                    delete_sources=[source],
                    parent=parent_url,
                    is_url=True,
                    indent=18,
                    suffix="子 URL",
                )

    def _render_file_row(
        self,
        container,
        key,
        source,
        doc_ids,
        color,
        current_weights,
        weight_target,
        delete_sources,
        parent,
        is_url=False,
        indent=0,
        suffix="",
        toggle=None,
        toggle_expanded=False,
        toggle_count=0,
    ):
        row = ctk.CTkFrame(container, fg_color="transparent")
        row.pack(fill="x", padx=(10 + indent, 10), pady=4)

        left = ctk.CTkFrame(row, fg_color="transparent")
        left.pack(side="left", fill="x", expand=True)

        var = ctk.BooleanVar(value=False)
        checkbox = ctk.CTkCheckBox(
            left,
            text=self._short_label(source, is_url=is_url),
            variable=var,
            text_color=color,
        )
        checkbox.pack(side="left", fill="x", expand=True)

        if suffix:
            ctk.CTkLabel(left, text=suffix, text_color=MUTED).pack(side="left", padx=(8, 0))

        if toggle:
            label = f"{'收起' if toggle_expanded else '展开'}子 URL"
            if toggle_count:
                label += f" ({toggle_count})"
            ctk.CTkButton(
                row,
                text=label,
                width=108,
                height=28,
                command=lambda item=toggle: self._toggle_url_group(item),
                fg_color=SOFT,
                hover_color="#dddddd",
                text_color=TEXT,
            ).pack(side="right", padx=(8, 0))

        weight_value = str(current_weights.get(weight_target, current_weights.get(source, 1)))
        weight_selector = ctk.CTkSegmentedButton(
            row,
            values=["1", "2", "3"],
            width=96,
            height=28,
            command=lambda value, item=weight_target: self.app.update_file_weight(item, value),
            fg_color=SOFT,
            selected_color=ACCENT,
            selected_hover_color=ACCENT_SOFT,
            unselected_color=SOFT,
            unselected_hover_color="#dcdcdc",
            text_color=TEXT,
        )
        weight_selector.set(weight_value)
        weight_selector.pack(side="right")

        ctk.CTkLabel(row, text="权重", text_color=MUTED).pack(side="right", padx=(0, 6))

        self.app.file_checkboxes[key] = {"var": var, "widget": checkbox}
        self.app.file_id_map[key] = list(doc_ids)
        self.app.file_delete_sources[key] = {"sources": list(delete_sources), "parent": parent}
        self.app.file_weight_targets[key] = weight_target
