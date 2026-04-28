from __future__ import annotations

from typing import Dict, List


STYLE_PHILOSOPHER = "内容论述"
STYLE_LITERATURE_REVIEW = "文献综述"
STYLE_CONCEPT_MAP = "概念梳理"
STYLE_CITE_PATCH = "引文补注"
STYLE_REVIEW = "学术批评审稿"

STYLE_SPECS: Dict[str, Dict[str, object]] = {
    STYLE_PHILOSOPHER: {
        "min_citations": 22,
        "retrieval_multiplier": 5,
        "min_chars": 3200,
        "tone": "以学术论文正文的口吻展开论证，强调概念界定、论证层次与段落衔接。",
        "sections": ["问题界定", "文本分析", "概念推进", "异议与回应", "结论"],
    },
    STYLE_LITERATURE_REVIEW: {
        "min_citations": 20,
        "retrieval_multiplier": 5,
        "min_chars": 3600,
        "tone": "以严格综述写法组织材料，围绕问题域、研究脉络、核心分歧与研究缺口展开。",
        "sections": ["研究问题", "研究脉络", "核心争论", "方法与证据", "研究缺口"],
    },
    STYLE_CONCEPT_MAP: {
        "min_citations": 22,
        "retrieval_multiplier": 5,
        "min_chars": 3400,
        "tone": "以术语考释和概念考古的方式写作，比较概念在不同文本中的用法与张力。",
        "sections": ["术语界定", "局部语境", "关联概念", "文本差异", "概念谱系"],
    },
    STYLE_CITE_PATCH: {
        "min_citations": 12,
        "retrieval_multiplier": 4,
        "min_chars": 1800,
        "tone": "优先保持用户原文结构，只在必要处补充引文与脚注，不擅自润色。",
        "sections": ["正文补注", "脚注"],
    },
    STYLE_REVIEW: {
        "min_citations": 16,
        "retrieval_multiplier": 5,
        "min_chars": 2600,
        "tone": "以严格匿名审稿口吻指出论证、结构、证据和格式问题，同时给出可执行修改建议。",
        "sections": ["总体判断", "重大问题", "中度问题", "语言与格式", "修改建议"],
    },
}

LEGACY_STYLE_MAP = {
    "academic": STYLE_PHILOSOPHER,
    "philosophical": STYLE_PHILOSOPHER,
    "literature_review": STYLE_LITERATURE_REVIEW,
    "lit_review": STYLE_LITERATURE_REVIEW,
    "concept_map": STYLE_CONCEPT_MAP,
    "cite_patch": STYLE_CITE_PATCH,
    "review": STYLE_REVIEW,
}


def get_supported_styles() -> List[str]:
    return list(STYLE_SPECS.keys())


def normalize_answer_style(answer_style: str) -> str:
    if not answer_style:
        return STYLE_PHILOSOPHER
    raw = str(answer_style).strip()
    lower = raw.lower()
    if raw in STYLE_SPECS:
        return raw
    return LEGACY_STYLE_MAP.get(lower, STYLE_PHILOSOPHER)


def get_style_spec(answer_style: str) -> Dict[str, object]:
    style = normalize_answer_style(answer_style)
    return dict(STYLE_SPECS[style], style=style)


def get_retrieval_target(answer_style: str) -> int:
    spec = get_style_spec(answer_style)
    return int(spec["min_citations"]) * int(spec["retrieval_multiplier"])


def _build_style_block(style: str) -> str:
    spec = get_style_spec(style)
    sections = "、".join(spec["sections"])
    return (
        f"写作模式：{spec['style']}\n"
        f"最低正文长度：不少于 {spec['min_chars']} 个汉字。\n"
        f"最低脚注数量：不少于 {spec['min_citations']} 条。\n"
        f"推荐结构：{sections}。\n"
        f"写作要求：{spec['tone']}"
    )


def build_prompt(
    question: str,
    context: str,
    required_language: str = "same as question",
    answer_style: str = STYLE_PHILOSOPHER,
) -> str:
    style = normalize_answer_style(answer_style)
    spec = get_style_spec(style)
    context_block = context.strip() or "当前检索结果为空。请先简要说明未检索到足够材料，再给出谨慎的处理建议。"

    return f"""
你是一名从事哲学与人文学术写作的研究者。你必须仅以给定材料为主要证据来源，不得捏造引文、页码、书名或作者观点。

## 写作任务
{_build_style_block(style)}

## 证据规则
1. 只要涉及具体判断、概念定义、文本归属、研究结论或比较结论，就必须使用脚注。
2. 脚注统一使用 `[n]`，从 `[1]` 连续编号，正文与文末 `## 脚注` 一一对应，在文中的引用也从[1]开始，顺序向下，并和脚注对应。
3. 文末脚注格式统一为：`[n]（书名或文献名, p. 页码）说明。`
4. 如果某条材料无法支持某个判断，就明确说明证据不足，不要用常识填补。
5. 尽量覆盖多数材料，不要把整篇回答建立在单一文献上。

## 文风规则
1. 语言必须与用户提问保持一致。输出语言：{required_language}。
2. 采用学术汉语常见的长段论述方式，段内推进要自然、连贯、克制。
3. 避免空泛总结、模板化排比、口语化赞叹和明显的 AI 腔。
4. 避免机械重复“不是……而是……”“并非……而是……”这类生硬对照句式。
5. 不要把回答写成提纲式口号；除非当前模式要求，否则以完整段落为主。
6. 对相互冲突的材料要明示分歧点、概念前提和论证后果。

## 输出格式
1. 先给出完整正文。
2. 正文结束后另起一节，标题必须为 `## 脚注`。
3. `## 脚注` 下逐条列出全部脚注，数量不得少于 {spec['min_citations']} 条；若材料不足，也必须明确说明不足原因。

## 用户问题
{question}

## 可用材料
{context_block}

## 输出前自检
1. 检查 `[n]` 是否连续。
2. 检查每条脚注是否都能在材料中找到对应出处。
3. 检查段落之间是否存在重复表述或空洞转折。
4. 检查正文是否真正回应了用户问题，而不是泛泛复述材料。
5. 检查文中引用的序号是否顺序向下，并且与脚注一一对应。

请直接开始写正文。
"""
