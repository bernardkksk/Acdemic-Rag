from __future__ import annotations

import re
from typing import Dict, List


STYLE_PHILOSOPHER = "内容论述"
STYLE_LITERATURE_REVIEW = "文献综述"
STYLE_CONCEPT_MAP = "概念梳理"
STYLE_CITE_PATCH = "引文补注"
STYLE_REVIEW = "学术批评审稿"


STYLE_SPECS: Dict[str, Dict[str, object]] = {
    STYLE_PHILOSOPHER: {
        "min_citations": 24,
        "retrieval_multiplier": 6,
        "min_chars": 5600,
        "purpose": "围绕用户提出的问题生成总论性质的学术论证。它不是概念词条，也不是文献清单，而是要把问题置于多重材料、多个立场和多层反驳之中，形成一篇可继续扩展为论文的论证性正文。",
        "must_do": [
            "界定问题的理论位置、争议焦点和隐含前提。",
            "从不同材料中提取论据，说明每个论据的有效范围和限制。",
            "组织正面论证、反思性限定、可能反驳与再论证。",
            "在结尾形成有条件的中层判断，而不是只给资料汇编。",
        ],
        "avoid": [
            "不要写成文献综述式的作者罗列。",
            "不要写成概念解释词条。",
            "不要把问题简化为几个并列观点。",
        ],
        "tone": "以正式学术论文的总论口吻展开，要求论证推进清楚、反驳环节充分、材料评述有判断力。",
        "sections": ["问题的理论位置", "论据谱系与材料评述", "正面论证", "反思与反驳", "再论证与结论"],
    },
    STYLE_LITERATURE_REVIEW: {
        "min_citations": 26,
        "retrieval_multiplier": 6,
        "min_chars": 4800,
        "purpose": "生成研究史和争议地图。它的重点不是回答用户自己的理论问题，而是说明某一议题在既有研究中如何被提出、分化、争论、推进和遗留问题。",
        "must_do": [
            "按问题域、理论阵营、方法路径或时间线组织研究，不得只按检索顺序罗列。",
            "概括每组文献的核心命题、材料依据、方法偏好和局限。",
            "明确指出研究之间的继承、修正、分歧和盲点。",
            "最后提出可供用户继续写作的研究缺口和选题入口。",
        ],
        "avoid": [
            "不要写成总论式立场文章。",
            "不要只解释单个概念的含义。",
            "不要给每篇文献一两句摘要后结束。",
        ],
        "tone": "以规范文献综述的口吻写作，强调研究谱系、争议结构、方法差异和研究缺口。",
        "sections": ["综述对象与范围", "研究谱系", "主要争议", "方法与材料差异", "研究缺口与推进方向"],
    },
    STYLE_CONCEPT_MAP: {
        "min_citations": 22,
        "retrieval_multiplier": 5,
        "min_chars": 4200,
        "purpose": "生成面向单个术语或概念簇的概念谱系。它的任务是把一个名词从定义、语境、译名、相邻概念、反概念和使用风险等方面拆开，而不是替用户完成总论或综述。",
        "must_do": [
            "先判断用户给出的是否为单个概念、术语或概念簇；若问题过大，应收束为核心概念。",
            "说明概念的基本含义、语源或译名差异、关键使用场景。",
            "列出相邻概念、对立概念、容易混淆的概念，并说明差异。",
            "给出该概念在论文写作中的可用命题、风险表述和可继续追问的问题。",
        ],
        "avoid": [
            "不要写成文献综述。",
            "不要发展成长篇总论式论证。",
            "不要只给词典式定义。",
        ],
        "tone": "以概念史、术语学和理论辨析的方式写作，强调概念边界、语义层次和写作可用性。",
        "sections": ["概念入口", "语义层次", "相邻概念与反概念", "使用场景与误用风险", "写作命题"],
    },
    STYLE_CITE_PATCH: {
        "min_citations": 12,
        "retrieval_multiplier": 4,
        "min_chars": 1800,
        "purpose": "对用户提供的目标原文进行引文补注。重点是识别原文中需要证据的位置，并从三个知识库中找到可支撑或可限定的材料。",
        "must_do": [
            "保持目标文本原本的问题意识、结构和论证方向。",
            "在目标文本中补充必要的括号夹注、证据说明和材料限定。",
            "标出无法补证的位置。",
        ],
        "avoid": [
            "不要把目标文本重写成另一篇论文。",
            "不要为目标文本没有提出的论断添加虚假支撑。",
        ],
        "tone": "优先保持原问题的结构与表达目的，在不扩张结论边界的前提下补足证据、引文与参考文献。",
        "sections": ["正文补注", "证据补强", "参考文献"],
    },
    STYLE_REVIEW: {
        "min_citations": 16,
        "retrieval_multiplier": 5,
        "min_chars": 2600,
        "purpose": "对用户提供的目标原文进行学术审稿。重点是判断其论证能否成立、证据是否充分、结构是否清楚、文献覆盖是否可靠。",
        "must_do": [
            "指出目标文本中的核心贡献和主要风险。",
            "区分重大问题、次要问题和可操作修改建议。",
            "用知识库材料校验原文的理论归属、文献覆盖和论据强度。",
        ],
        "avoid": [
            "不要只做语言润色。",
            "不要脱离目标文本泛泛谈主题。",
        ],
        "tone": "以学术审稿与批评写作口吻指出论证漏洞、证据问题、结构缺陷与表达风险，并提出可执行修改建议。",
        "sections": ["总体判断", "主要问题", "次要问题", "证据与结构", "修改建议"],
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


def requires_target_document(answer_style: str) -> bool:
    style = normalize_answer_style(answer_style)
    return style in {STYLE_CITE_PATCH, STYLE_REVIEW}


def _build_style_block(style: str) -> str:
    spec = get_style_spec(style)
    sections = "\n".join(f"- {item}" for item in spec["sections"])
    must_do = "\n".join(f"- {item}" for item in spec.get("must_do", []))
    avoid = "\n".join(f"- {item}" for item in spec.get("avoid", []))
    if requires_target_document(style):
        volume_rule = "输出规模：根据目标原文和可用材料决定，不设置最低正文长度或最低引用数量。"
    else:
        volume_rule = (
            f"最低正文长度：不少于 {spec['min_chars']} 个汉字。\n"
            f"最低正文夹注数量：不少于 {spec['min_citations']} 处。"
        )
    return (
        f"写作模式：{spec['style']}\n"
        f"模式目标：{spec['purpose']}\n"
        f"{volume_rule}\n"
        f"建议一级结构：\n{sections}\n"
        f"必须完成：\n{must_do}\n"
        f"必须避免：\n{avoid}\n"
        f"写作口吻：{spec['tone']}"
    )


def _build_gbt7714_block() -> str:
    return """
## GB/T 7714-2015 引文与参考文献格式
1. 正文不使用 `[1]`、`[2]` 这类脚注编号；引用出处直接放在相关句子或段落末尾的圆括号中。
2. 专著尽量写作：作者. 题名[M]. 出版地: 出版者, 年份: 页码.
3. 期刊论文尽量写作：作者. 题名[J]. 刊名, 年份, 卷(期): 页码.
4. 论文集析出文献尽量写作：作者. 析出题名[C]//编者. 文集名. 出版地: 出版者, 年份: 页码.
5. 学位论文尽量写作：作者. 题名[D]. 保存地: 保存单位, 年份.
6. 网页或在线材料尽量写作：作者. 题名[EB/OL]. URL.
7. 材料缺少字段时，只保留可确认字段；不得补造出版地、出版社、年份、期卷期号或页码。
8. 文末统一设置 `## 参考文献`，列出正文已经使用的材料；参考文献条目不加 `[n]` 编号。
9. `RefID` 只用于内部核验，最终输出的正文夹注和参考文献中不要显示 `RefID`。
"""


def _build_cite_patch_prompt(question: str, context_block: str, target_name: str, target_text: str) -> str:
    target_body = _extract_cite_patch_body(target_text)
    return f"""
你是一名严谨的学术引文编辑。当前任务不是写新文章，而是对用户提供的目标原文做“保真引文补注”。

## 核心任务
1. 必须原封不动保留目标原文的正文内容、段落顺序、句子结构和措辞。
2. 只能在需要引用、可被文献支持或需要来源说明的位置插入括号夹注，例如 `（作者. 题名[M]. 出版地: 出版者, 年份: 页码）`。
3. 不得改写、扩写、润色、删减、重组目标原文。
4. 不得新增 AI 自己写出的论述段落，不得在正文中加入解释性旁白。
5. 输出的正文应该看起来像“原文 + 括号夹注”，而不是一篇重新生成的文章。
6. 不设置最低字数，不设置最低引用数量；有多少可靠依据就标多少，没有依据的位置不要硬标。

## 标注原则
1. 只有当可用材料能够支持目标原文中的某个判断、概念、作者归属、理论定位、文本解释、比较或历史事实时，才插入括号夹注。
2. 不得使用 `[1]`、`[2]` 这类脚注编号；不得输出 `## 脚注` 模块。
3. 如果目标正文中已经存在 `[n]` 形式的旧脚注编号，应把它们替换为对应的括号出处，不保留编号。
4. 同一句如果需要多个来源，可在同一个括号中综合列出多个来源，但最终输出中不要显示 `RefID`。
5. 若某处明显需要补注但材料不足，请不要编造引用；可在文末 `## 未能补注的位置` 中简要列出原句片段和原因。
6. 若材料有页码，括号夹注和参考文献中必须保留页码；若 `Page` 为 `?` 或材料未提供页码，不得编造页码。

{_build_gbt7714_block()}

## 输出结构
1. 第一部分标题为 `## 补注后正文`，其下放置保真后的目标原文，只允许新增括号夹注。
2. 第二部分标题为 `## 参考文献`，逐条列出正文实际使用过的材料，条目前不要加 `[n]` 编号。
3. 如存在材料不足但值得提示的位置，第三部分标题为 `## 未能补注的位置`。
4. 不要输出任务解释、方法说明或多余总结。

## 用户指令
{question}

## 目标原始文本
文件名：{target_name or "未命名 Word 文档"}

{target_body}

## 可用材料
{context_block}

## 输出前自检
1. 目标原文是否被原封不动保留，除了插入括号夹注之外没有改写。
2. 正文中是否没有 `[n]` 这类脚注编号。
3. 文末是否只有 `## 参考文献` 而不是 `## 脚注`。
4. 每条括号夹注和参考文献是否已去除内部 `RefID`。
5. 是否没有编造页码、来源、作者观点或文献信息。

请直接输出 `## 补注后正文`。
"""


def build_cite_patch_repair_prompt(
    previous_answer: str,
    unresolved_snippets: List[str],
    additional_context: str,
) -> str:
    snippets = "\n".join(f"{idx}. {snippet}" for idx, snippet in enumerate(unresolved_snippets, start=1))
    return f"""
你是一名学术引文编辑。现在需要在上一轮补注结果的基础上，专门处理“未能补注的位置”。

## 修补目标
1. 保留上一轮 `## 补注后正文` 中的正文内容，不得改写原文。
2. 只尝试为下列未补注片段增加括号夹注依据。
3. 如果新增引用，必须在正文相应位置加入括号夹注，并在 `## 参考文献` 中补入完整 GB/T 7714-2015 条目。
4. 已经正确存在的括号夹注和参考文献不要删除。
5. 如果新增材料仍不足以支撑某个片段，请继续把它留在 `## 未能补注的位置`，并写明缺少何种证据。
6. 不要输出解释性废话，只输出修订后的完整结果。

## 待修补片段
{snippets or "无"}

{_build_gbt7714_block()}

## 新增可用材料
{additional_context.strip() or "没有新增材料。"}

## 上一轮补注结果
{previous_answer}

## 输出要求
1. 输出完整的 `## 补注后正文`。
2. 输出完整的 `## 参考文献`，条目前不要加 `[n]` 编号。
3. 如仍有无法补注的位置，输出 `## 未能补注的位置`；若全部补齐，则不要输出该节。
4. 正文只允许新增或调整括号夹注，不得改写原文句子。
5. 最终正文夹注和参考文献中不要显示 `RefID`；不得使用 `[n]` 脚注编号。

请直接输出修订后的完整结果。
"""


def _extract_cite_patch_body(text: str) -> str:
    body = str(text or "").strip()
    body = body.replace("\r\n", "\n")
    body = re.sub(r"(?is)^\s*user\s+.*?\n\s*[^\n]{0,80}\n(?=##|\S)", "", body, count=1)
    if "## 补注后正文" in body:
        body = body.split("## 补注后正文", 1)[1]
    for marker in ("## 脚注", "## 参考文献", "## 未能补注的位置", "参考来源："):
        if marker in body:
            body = body.split(marker, 1)[0]
    body = body.strip()
    body = body.removeprefix("user").strip()
    return body


def build_prompt(
    question: str,
    context: str,
    required_language: str = "auto",
    answer_style: str = STYLE_PHILOSOPHER,
    target_document: Dict[str, str] | None = None,
) -> str:
    style = normalize_answer_style(answer_style)
    spec = get_style_spec(style)
    context_block = context.strip() or "当前检索结果为空。请先说明材料不足，再给出谨慎而有限的处理建议。"
    target_document = target_document or {}
    target_name = str(target_document.get("name") or "").strip()
    target_text = str(target_document.get("text") or "").strip()
    if style == STYLE_CITE_PATCH:
        context_for_patch = context_block
        if not target_text:
            return _build_cite_patch_prompt(
                question=question,
                context_block=context_for_patch,
                target_name=target_name,
                target_text="未提供目标原始文本。无法执行保真引文补注。",
            )
        return _build_cite_patch_prompt(
            question=question,
            context_block=context_for_patch,
            target_name=target_name,
            target_text=target_text,
        )

    target_block = ""
    if target_text:
        target_block = f"""
## 目标原始文本
文件名：{target_name or "未命名 Word 文档"}

{target_text}
"""
    elif requires_target_document(style):
        target_block = """
## 目标原始文本
未提供目标原始文本。当前模式必须以用户提供的 Word 原文为主要依据；若缺失，应先说明无法完成严格的引文补注或学术批评审稿。
"""
    task_block = ""
    if style == STYLE_CITE_PATCH:
        task_block = """
## 引文补注专项要求
1. 目标原始文本是待补注正文，不得改写其核心论证方向。
2. 需要识别目标文本中缺少证据支撑的判断、概念界定、文献归属、比较性论断和历史性表述。
3. 补注应优先服务于目标文本原有论证，不得额外扩张为一篇全新的论文。
4. 若目标文本中的某一判断无法从三个知识库材料中找到支撑，必须明确标出“材料不足”，不得强行补造注释。
5. 输出应包含两个层次：一是可直接替换或插入的补注正文；二是说明每组补注为什么放在相应位置。
6. 补注不得改变目标文本的核心立场；若必须修正原判断，应明确说明“建议改写为”。
7. 对目标文本中已经有充分支撑的句子，不要为了增加引用而机械重复补注。
"""
    elif style == STYLE_REVIEW:
        task_block = """
## 学术批评审稿专项要求
1. 目标原始文本是被审稿对象，批评必须逐段或逐问题回到该文本的实际表述。
2. 需要区分论证问题、证据问题、结构问题、概念使用问题和文献覆盖问题。
3. 三个知识库材料用于校验目标文本的文献依据、理论归属和可补强方向。
4. 修改建议必须可执行，避免泛泛评价。
5. 输出应包含总体评价、主要问题、次要问题、可采纳修改方案和优先修改顺序。
6. 对每个主要问题都要说明：问题位置、为什么构成问题、材料依据、修改方向。
7. 审稿语气应严肃、清楚、可执行，不要使用情绪化贬低。
"""

    return f"""
你是一名面向哲学、人文学术研究与论文写作的高级研究助手。你的输出应接近可发表论文、研究札记或正式审稿意见的质量，而不是聊天回复、资料摘要或课堂提纲。你必须把用户问题、目标原文和检索材料组织成一篇有结构、有证据、有反思、有参考文献的学术文本。

## 任务规格
{_build_style_block(style)}

## 工作流
1. 先判断用户问题属于理论论证、概念辨析、研究史综述、引文补注还是审稿批评。
2. 再从可用材料中筛选与任务直接相关的证据，不得把所有材料平均摊开。
3. 将证据分成至少三类：正面支撑材料、限定或修正材料、可能构成反驳或张力的材料。
4. 正文写作时必须先给问题定位，再组织材料，再推进论证，最后形成有条件的判断。
5. 若材料不足，应说明“缺少何种材料”“影响哪个判断”“可暂时得出何种有限结论”。

## 证据边界
1. 所有重要判断必须由 `可用材料` 或 `目标原始文本` 支撑。
2. 不得捏造引文、页码、出处、出版信息、作者观点、文献标题或参考文献信息。
3. 不得把材料中的局部表述扩张为作者整体立场。
4. 不得把推断写成事实；推断必须用“可以理解为”“在这一语境中”“较为谨慎地说”等方式限定。
5. 不得用空泛常识替代材料分析。若无法从材料中得到支持，应明确标注材料不足。
6. 允许提出解释性判断，但必须说明其证据基础、推理路径和可能限制。

## RefID 与正文引文
1. 每则检索材料都附有 `RefID`、`Cite as`、`Source`、`Parent URL`、`Page`、`Chunk` 等信息。
2. 正文中的每个关键判断、定义、归属、比较、批评、材料转述或外文术语说明，都应在句末或段末直接使用括号夹注，例如 `（作者. 题名[M]. 出版地: 出版者, 年份: 页码）`。
3. 不得使用 `[1]`、`[2]` 这类脚注编号；不得输出 `## 脚注` 模块。
4. 文末必须设置 `## 参考文献`，只列出正文实际引用过的材料，条目前不要加 `[n]` 编号。
5. 每条正文夹注在生成时必须能回溯到至少一个 `RefID`，但最终输出中不要显示 `RefID`。
6. 若同一处综合多个材料，应在同一括号或同一段落中说明材料之间是互证、补充、限定还是冲突，不得只堆叠来源。
7. 若 `Page` 为 `?` 或材料没有页码，不得编造页码；正文夹注和参考文献中应保留这种不确定性。

{_build_gbt7714_block()}

## 文体规格
1. 使用正式、克制、密度较高的学术汉语；若用户明确要求其他语言，则服从用户要求。
2. 段落应有清晰功能：界定、转折、评述、反驳、综合、结论。不要让多个功能混在一个松散段落里。
3. 每个主体段落应包含一个中心判断、至少一项材料依据和一层分析推进。
4. 避免口语化表达、列表式堆砌、营销式形容词、空泛套话和明显 AI 腔。
5. 避免大量使用“首先、其次、最后”的机械串联；应通过概念关系和论证关系推进。
6. 不要分点分得过细，不要把回答写成碎片化清单；除必要的小标题外，尽可能生成完整、连贯、可直接进入论文草稿的长段落。
7. 可以保留英文、法文或德文关键术语，但必须解释其在当前论证中的作用。
8. 外文材料优先用于概念来源、理论归属和争议定位；中文材料可用于研究脉络、接受史和问题转译。
9. 不得把正文写成“我将从以下几方面分析”的说明书式开头，应直接进入学术论述。

## 论证深度
1. 必须区分材料事实、作者立场、你的解释和可争议推论。
2. 至少呈现一种可能反驳或限定，并说明它如何改变原判断。
3. 至少形成两个中层结论。中层结论应能连接材料分析与最终判断，而不是重复标题。
4. 若材料之间存在张力，必须指出张力发生在概念定义、规范立场、文本证据、历史语境还是方法论层面。
5. 结论不得只是摘要，应说明当前论证还能推进到哪里、边界在哪里。

{task_block}

## 输出结构
1. 先输出正文，不要输出“以下是答案”等元说明。
2. 正文必须使用小标题，但小标题应具有论文感，不要使用“第一部分”“第二部分”这类机械标题。
3. 正文结束后单独另起一节，标题为 `## 参考文献`。
4. `## 参考文献` 下逐条列出正文实际使用过的材料，条目前不要加 `[n]` 编号。
5. 若任务是 `引文补注`，正文应围绕目标文本给出补注版处理和补注说明。
6. 若任务是 `学术批评审稿`，正文应以审稿意见的形式组织，但语言仍保持学术正式性。

## 用户问题
{question}

{target_block}

## 可用材料
{context_block}

## 输出前自检
1. 是否严格执行当前写作模式，而不是混成其他模式。
2. 若当前模式设置了最低规模，是否达到不少于 {spec['min_chars']} 个汉字的正文规模；若当前模式依赖目标原文，则是否按原文和材料充分处理。
3. 若当前模式设置了最低引用数量，是否达到不少于 {spec['min_citations']} 处正文夹注；若当前模式依赖目标原文，则是否只在有可靠依据处加注。
4. 正文中是否已经去除 `[n]` 脚注编号，并改为括号出处。
5. 每条正文夹注和参考文献是否参照 GB/T 7714-2015，并已去除内部 `RefID`。
6. 是否存在没有证据支撑的重要判断。
7. 是否存在编造页码、出处、年份、作者观点或文献信息。
8. 是否呈现了反思、限定、反驳或张力，而不是只复述材料。
9. 文本整体是否接近正式学术论文或严肃审稿意见。

请直接开始正文。
"""
