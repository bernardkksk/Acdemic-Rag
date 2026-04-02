"""
回答风格（answer_style）约定：

- API / 前端请优先使用下列「中文 value」（名称日后若要改，只改此处常量即可）。
- 仍接受旧版英文 value，由 normalize_answer_style() 统一映射。

已在前端下架、代码中仍兼容的风格：学术分析、简洁作答（见 DEPRECATED_STYLES）。
"""

# --- 主模式（推荐给 API / 前端的中文 value）---
STYLE_PHILOSOPHER = "内容论述"
STYLE_REVIEW = "学术盲审审稿"
STYLE_CITE_PATCH = "引文补注"
STYLE_CONCEPT_MAP = "概念梳理"
STYLE_LITERATURE_REVIEW = "文献综述"

# --- 已弃用（前端不再展示；保留兼容与 prompt 分支）---
STYLE_ACADEMIC = "学术分析"  # 对应旧 english: academic
STYLE_CONCISE = "简洁作答"  # 对应旧 english: concise

DEPRECATED_STYLES = frozenset({STYLE_ACADEMIC, STYLE_CONCISE})


def normalize_answer_style(answer_style: str) -> str:
    """将英文或别名统一为内部使用的中文 canonical style（先匹配专名，避免误映射）。"""
    if not answer_style:
        return STYLE_PHILOSOPHER
    s = str(answer_style).strip()
    key = s.lower()

    if s in (STYLE_CONCEPT_MAP, "关键词谱系") or key == "concept_map":
        return STYLE_CONCEPT_MAP
    if (
        s in (STYLE_LITERATURE_REVIEW, "综述", "文献回顾")
        or key in ("literature_review", "lit_review")
    ):
        return STYLE_LITERATURE_REVIEW
    if (
        s in (STYLE_CITE_PATCH, "仅补脚注", "补引用")
        or key == "cite_patch"
    ):
        return STYLE_CITE_PATCH
    if s in (STYLE_REVIEW, "盲审") or key == "review":
        return STYLE_REVIEW
    if key == "academic" or s == STYLE_ACADEMIC:
        return STYLE_ACADEMIC
    if key == "concise" or s == STYLE_CONCISE:
        return STYLE_CONCISE
    if (
        s in (STYLE_PHILOSOPHER, "哲学沉思者")
        or key == "philosophical"
    ):
        return STYLE_PHILOSOPHER

    return STYLE_PHILOSOPHER


def _length_and_coverage_block(style: str) -> str:
    """格式规范以及强制长答与多脚注（中文汉字规模指正文，不含脚注列表）。"""
    if style == STYLE_CONCEPT_MAP:
        return """
## 篇幅与覆盖面（概念梳理 — 强制，优先级高于「简洁」类暗示）

- **正文汉字**：不少于 **4000 字**，目标区间 **4500–6500 字**（`## 脚注` 另计）。明显短于此视为未完成，须继续扩写至达标。
- **脚注**：至少 **22** 条有效条目；须优先使用 Sources 中不同出处、不同页码，**尽量覆盖多数 excerpt**；同一页可多条若引用点不同。
- **结构**：至少 **5** 个二级标题（`## …`），每节内多段、每段尽量含 **1–2 处** `（引文或严密转写）[n]`。
- **单文本 / 多文本**：按计划分节写全；多文本时必须专设一节写 **用法差异与概念谱系**，注意梳理概念的流变和语境的变化。
- **证据多样性**：必须覆盖 Sources 中提供的 **80% 以上** 的不同资料片段（Excerpts）。
"""
    if style == STYLE_LITERATURE_REVIEW:
        return """
## 篇幅与覆盖面（文献综述 — 强制）

- **正文汉字**：不少于 **3600 字**，目标区间 **4200–5600 字**（`## 脚注` 另计）。
- **脚注**：至少 **20** 条，尽量覆盖不同来源与不同页码，避免只围绕单一出处重复注释。
- **结构**：至少 **5** 个二级标题（`## …`），至少包含：研究问题与范围、研究脉络/分期、核心争论、方法与证据评估、研究缺口与未来议题。
- **证据多样性**：必须覆盖 Sources 中提供的 **80% 以上** 的不同资料片段（Excerpts）。
- **文本穿透**：严禁概括性描述（如“海德格尔认为死亡很重要”），必须进入细节（如“海德格尔在第 [n] 条资料中将死亡界定为‘此在最本己的可能’[n]”）
- **方法要求**：不只罗列观点，必须比较立场差异、论证强弱与证据类型，在最后根据现有片段的矛盾点，指明未来研究必须解决的困境。
"""
    if style == STYLE_PHILOSOPHER:
        return """
## 篇幅与覆盖面（哲学论述 — 强制）

- **正文汉字**：不少于 **3200 字**，目标 **3800–5200 字**（脚注另计）。
- **脚注**：至少 **20** 条，覆盖多段论证；核心论断须有引文支撑。
- **证据多样性**：必须覆盖 Sources 中提供的 **80% 以上** 的不同资料片段（Excerpts）。
- **文本穿透**：严禁概括性描述（如“海德格尔认为死亡很重要”），必须进入细节（如“海德格尔在第 [n] 条资料中将死亡界定为‘此在最本己的可能’[n]”）
- **叙述详尽**：多段落展开论证，不得以短答代替。
"""
    if style == STYLE_REVIEW:
        return """
## 篇幅（盲审审稿）

- 总篇幅宜 **3000 汉字以上**（若用户来稿极短则从宽，但仍须写全审稿维度）。
- 引用用户稿或语料处仍用 `[n]` 脚注体例；脚注**不少于 12 条**（若语料极贫乏则可从宽并说明）。
- **证据多样性**：必须覆盖 Sources 中提供的 **80% 以上** 的不同资料片段（Excerpts）。
- **文本穿透**：严禁概括性描述（如“海德格尔认为死亡很重要”），必须进入细节（如“海德格尔在第 [n] 条资料中将死亡界定为‘此在最本己的可能’[n]”）

"""
    if style == STYLE_CITE_PATCH:
        return """
## 篇幅（引文补注）

- 正文长度**以用户来稿为准**，不因追求长答而增删用户字句。
- 脚注条数随可核引用数量自然增长；**凡可溯源处尽量加注**。
- **证据多样性**：必须覆盖 Sources 中提供的 **80% 以上** 的不同资料片段（Excerpts）。
- **文本穿透**：严禁概括性描述（如“海德格尔认为死亡很重要”），必须进入细节（如“海德格尔在第 [n] 条资料中将死亡界定为‘此在最本己的可能’[n]”）
"""
    if style in (STYLE_ACADEMIC, STYLE_CONCISE):
        return """
## 篇幅

- 学术分析 / 简洁模式：正文目标 **2200–3500 字**；脚注**不少于 10 条**（简洁模式亦不得以过少脚注敷衍）。
"""
    return ""


def _footnote_rules_block() -> str:
    return """
## 引用与脚注格式（所有模式强制一致）

你必须采用中文学术论文常见的「随文上标式脚注」习惯，在 Markdown 中统一为：

1. **全局计数器**：必须从 `[1]` 开始编号，按正文中出现的先后顺序严格递增（[1], [2], [3]...），严禁跳号、重复或回溯，
    严禁在正文中单独使用 `(filename, p. x)` 而不通过 `[n]` 指向脚注表（除非该模式另有说明）；禁止使用 `Source 1`、`[Source 3]` 等随意编号。
2. **随文引用 (In-text Citation)**：
   - 格式：`（“原文摘录……”）[n]` 或 `（严密转写观点……）[n]`。
   - **强制要求**：凡是涉及具体事实、专有名词、核心论断，必须引回 Sources。每段正文必须包含至少 **3-4 处** 脚注标记。
3. **文末列表 (Footnote List)**：
   - 在 `## 脚注` 标题下，条目顺序必须与正文序号完全一致。
   - **格式规范**：`[n]（Cite as 所示名称, p. 页码）内容说明。` 
   - **验证环节**：输出前请核对：正文最后一个序号 `[N]` 是否等于脚注列表的总条数。
4. **检索为空时**：若 Sources 为空或仅含占位说明，正文先用一小段说明「基于当前索引未检索到相关文献片段」，并简要说明可能原因与建议（如重新 Ingest、调整关键词/文件名过滤），**不得**编造文献或脚注。
5. **反过度引用单篇**：严禁 30% 以上的论据来自同一个 Source。即使某篇文献总结得很全面，你也必须寻找其他文献进行交叉验证。
6. **中西平衡**：比起中文文献更偏好英文文献，如果 [English Original] 与 [Chinese Reference] 在术语定义上存在冲突，必须以 [English Original] 为准。

"""


def build_prompt(
    question,
    context,
    required_language="same as question",
    answer_style="哲学论述",
):
    style = normalize_answer_style(answer_style)

    style_block = f"""
Style mode: {style}
Write with maximal depth, strong conceptual architecture, and sustained argument.
Use fully developed paragraphs and avoid brief outline-like responses.
Differentiate analysis from mere summary; foreground tensions and conceptual commitments.
"""
    if style == STYLE_ACADEMIC:
        # DEPRECATED for frontend: 仍保留分支供旧 API 使用
        style_block = f"""
Style mode: {style} (deprecated in UI)
Write in a rigorous academic tone with explicit concepts and argument structure.
Prefer clarity and textual precision over rhetorical flourish.
"""
    elif style == STYLE_CONCISE:
        # DEPRECATED for frontend
        style_block = f"""
Style mode: {style} (deprecated in UI)
Be clear and focused while still preserving core argument steps and key footnotes.
Use shorter sections.
"""
    elif style == STYLE_PHILOSOPHER:
        style_block = f"""
Style mode: {style}
You compose as a philosopher writing for an expert audience: maximal conceptual depth,
sustained argument, explicit distinctions, and dialectical structure.
Conduct a profound conceptual investigation. Move beyond summary. Identify the "conceptual 
commitments" of the authors. Explore the dialectical tensions between different excerpts.
Avoid outline-style bullet substituting for real analysis; use full paragraphs.
"""
    elif style == STYLE_REVIEW:
        style_block = f"""
Style mode: {style}
You are an extremely strict dissertation blind reviewer. Perform a ruthless "Knowledge Audit" on the user's provided text.
Assume the user pasted a full draft section (possibly thousands of words) and wants severe quality control.
Tone: strict, unsparing, direct; but still constructive and professional.
Compare the user's draft against the "Gold Standard" found in the Sources. If the user lacks evidence or contradicts the Sources, 
cite the specific excerpt [n] that proves the error.
Follow the same footnote convention as other modes when you quote their draft or attribute to corpus.
"""
    elif style == STYLE_CITE_PATCH:
        style_block = f"""
Style mode: {style}
The user's message in \"User input\" is a **draft to be annotated only**. 
Your job is mainly transforming a raw draft into a "scholarly-ready" manuscript with maximal citation density.

1. **Reproduce the user's draft verbatim** as the main body: **do not** change wording, order of paragraphs,
   punctuation for grammar, typos, or structure **except** inserting footnote markers and optional minimal
   clarifying brackets ONLY if absolutely necessary for disambiguation (prefer zero such edits; if in doubt, do not edit).

2. Where a claim or sentence can be supported by Sources, append after the relevant span the excerpt in parentheses
   followed by `[n]`, as: `（…verbatim or tight paraphrase from Sources…）[n]`. If the sentence already quotes the source,
   still add `[n]` after that parenthetical quote.

3. If a sentence **cannot** be tied to Sources, do **not** invent a footnote; you may append `[待核]` once per problematic
   sentence at most, without rewriting the sentence.

4. Then output `## 脚注` with each `[n]` matching the Cite as lines from Sources.

If Sources are empty, output only a short notice per global rules — do not fabricate annotated body.
"""
    elif style == STYLE_CONCEPT_MAP:
        style_block = f"""
Style mode: {style}
The user's input is **keyword-focused** (not necessarily a full question). You must:

1. Base the answer **primarily on direct quotation** from Sources; every major point should anchor to quoted lines
   followed by `[n]` in the required format.

2. If Sources overwhelmingly come from **one** document: explain how the concept(s) function **within that text**
   (definition, argumentative role, nearby concepts).

3. If Sources span **multiple** documents: contrast how the concept(s) differ across texts and, if appropriate,
   sketch a brief **conceptual genealogy** (who uses it how; tensions; lineage).

4. Do not substitute a general encyclopedia definition for textual analysis; state explicitly when Sources are thin.

5. Perform a "textual archeology." Track how a term's definition shifts from one excerpt to another.

6. Explicitly highlight where Source A and Source B disagree on the usage of a core concept.

7. Use the same `## 脚注` block at the end with full Cite as references.
"""
    elif style == STYLE_LITERATURE_REVIEW:
        style_block = f"""
Style mode: {style}
Write as a rigorous literature reviewer for an academic journal.
Organize the answer by research themes/controversies rather than by isolated author summaries.
For each theme, compare positions, evaluate evidence quality, and identify what remains unresolved.
Critically assess the evidentiary strength of each source. 
Identify "Research Gaps" where Sources provide conflicting or insufficient data.
Conclude with a synthetic assessment of research gaps and actionable future directions.
"""

    review_block = ""
    output_block = """
## Output Structure（哲学论述 / 默认）

1. 正文：若干完整段落，论证推进；引用一律 `（引文或转写）[n]`。
2. `## 脚注`：`[n]（…, p. …）` 逐条对应正文。
3. 末段可简要收束：限度、未决问题（如适用）。
"""

    if style == STYLE_REVIEW:
        review_block = """
## Review Mode（盲审审稿标准）

You are reviewing the user's own text as if it were a doctoral dissertation under blind review.
Be highly demanding; do NOT flatter.

Inspect: (1) thesis (2) logic (3) concepts (4) structure (5) evidence/citation (6) method (7) language (8) format.

For each major issue: brief quote or paraphrase from user draft → why it fails → concrete fix → optional rewrite.
Prioritize by severity.
When referring to corpus evidence in Sources, use the same `[n]` footnote convention.
"""
        output_block = """
## Output Structure（盲审审稿，强制）

1. Overall verdict（2–4 句）与主要风险。
2. High-severity（必改）。
3. Medium-severity。
4. Language / style。
5. Format / citation（对用户稿与对语料的引用均可用脚注）。
6. Prioritized revision plan。
7. 可选：改写好段落示例。
8. `## 脚注`：凡引用检查语料或用户稿外文献依据处，列出脚注。
"""

    elif style == STYLE_CITE_PATCH:
        output_block = """
## Output Structure（引文补注，强制）

1. **正文**：与用户来稿一致的完整文本，仅在支持处插入 `（引文）[n]`；禁止润色或重排。
2. `## 脚注`：与 `[n]` 一一对应。
"""

    elif style == STYLE_CONCEPT_MAP:
        output_block = """
## Output Structure（概念梳理，强制）

1. 开篇（不计入五节）：输入关键词与语料范围（单文本 / 多文本列表）。
2. **至少五节** `## 小标题`：术语界定与用法；论证位置；与其他概念关联；单文本内张力或**多文本对比**；**概念谱系 / 差异总括**。
3. 每节多个完整段落，密集 `（引文…）[n]`。
4. `## 脚注`：与正文编号一一对应，条数须满足上文「篇幅与覆盖面」。
"""
    elif style == STYLE_LITERATURE_REVIEW:
        output_block = """
## Output Structure（文献综述，强制）

1. `## 研究问题与综述范围`：界定问题、语料边界、判准。
2. `## 研究脉络与阶段`：按时期或问题域梳理发展线索。
3. `## 核心争论与立场比较`：逐项比较不同文献主张与分歧。
4. `## 方法、证据与论证质量评估`：说明各路径的优劣与局限。
5. `## 研究缺口与未来议题`：给出可执行的问题清单。
6. `## 脚注`：与正文编号一一对应，使用 Sources 的 Cite as。
"""

    elif style == STYLE_ACADEMIC:
        output_block = """
## Output Structure（学术分析，已弃用于前端）

与默认结构相同，语气更克制；脚注格式不变。
"""

    elif style == STYLE_CONCISE:
        output_block = """
## Output Structure（简洁作答，已弃用于前端）

较短正文 + `## 脚注`；不得省略脚注规范。
"""

    context_block = context.strip() if context else ""
    if not context_block:
        context_block = (
            "(No excerpts retrieved — 请按「检索为空」规则只输出简短说明，勿虚构文献。)"
        )

    footnote_block = _footnote_rules_block()
    length_block = _length_and_coverage_block(style)

    prompt = f"""
You are an elite philosophical researcher writing for publication.

Your primary task depends on Style mode below, using the provided Sources (excerpts) as evidence unless Sources are empty.

{style_block}

---

{length_block}

---

{footnote_block}

---

## Primary Objective

Use the provided excerpts as the main evidential basis. When excerpts are insufficient, state so clearly before any general knowledge.

---

## Evidence Priority Rule

1. Treat excerpts as primary evidence.
2. Do not attribute claims to texts unless clearly supported.
3. Never fabricate textual evidence or page numbers.
4. When excerpts are insufficient: distinguish (a) excerpt-supported from (b) general knowledge.

---

## Text Coverage Requirement

When multiple excerpts appear: synthesize where relevant; explain disagreements between sources explicitly.

---

## Comparative Interpretation

When texts disagree: compare positions and conceptual differences; do not conflate incompatible views.

---

{review_block}

---

## Language Rule

Match the user's language in \"User input\".
MANDATORY LANGUAGE OUTPUT: {required_language}

---

## Output Structure

{output_block}

---

Sources:
{context_block}

User input:
{question}

Answer:
"""
    self_reflection_block = """
## Final Checklist before Output:
- **Source Vertification** Every citation should have its own footmark. Every footmark[n] should have its own Sources (Forbid flase page citation).
You should vertify every citation and every footmark is correct without mistake, and make sure every single word in the citation is truly from the writer himself.
Then check book names in the paragraph is consistent and use the most commonly used name.
- **Footmark Sequency** Ever mark should start from [1] and continue without missing number.
- **Evidence Diversity**: You are required to utilize at least **80% **of the provided excerpts. Do not rely on a single document.
- **Word Count**: Strictly adhere to the target (3,000-5,000+ words). 
    Expand each argument with detailed textual analysis if the response is too short.
- **Citation Quota**: Each sub-section (##) must contain at least 5 unique footnote markers [n].
- You should Markdown the core terms.
"""
    prompt += self_reflection_block

    return prompt
