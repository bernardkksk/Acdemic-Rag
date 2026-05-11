# Academic RAG

## 总论

Academic RAG 是一个面向哲学、人文学术写作、文献整理和论文生产的本地桌面 RAG 系统。它不是普通聊天机器人，也不是简单的 PDF 问答工具，而是围绕“知识库治理、证据优先检索、学术论证生成、引用格式控制、目标文本补注与审稿”构建的一套研究工作流。

这个项目的核心想法是：把用户自己的 PDF、Word、TXT、网页和专题资料整理成多个本地知识库，然后在提问、写作、综述、概念梳理、引文补注、论文审稿时，让大模型优先依据这些材料工作。系统会尽量减少“凭空发挥”，把检索到的材料、页码、来源、权重和引用格式纳入生成过程。

它尤其适合以下场景：

- 写论文前围绕一个问题做理论总论。
- 面对大量中外文文献时整理研究史和争议谱系。
- 对某个哲学、人文社科概念做概念史和术语辨析。
- 给一篇 Word 草稿补充引用出处。
- 对一篇论文草稿做学术批评和审稿式修改建议。
- 将中文、英文、法文材料统一放入本地库中检索。
- 把多个专题库合并参与同一次问答，但仍保留主库和次库权重差异。

本项目当前的输出规范已经从传统脚注编号改为“正文括号夹注 + 文末参考文献”。也就是说，系统不再要求正文中出现 `[1]`、`[2]` 这样的脚注号，而是直接在相关句子后放置出处，例如：

```text
责任在这里并不是一种可以被完全计算的规范义务，而是与无法回避的他者召唤相关（Derrida. The Gift of Death[M]. Chicago: University of Chicago Press, 1995: 41）。
```

文末则统一输出：

```text
## 参考文献
Derrida. The Gift of Death[M]. Chicago: University of Chicago Press, 1995: 41.
```

参考文献格式参照 GB/T 7714-2015。系统会要求模型不得编造作者、题名、年份、出版社和页码；如果材料缺少字段，只保留可确认的信息。

## 功能概览

### 多知识库管理

系统支持多个独立知识库。每个知识库都有自己的向量索引、切块文件、文件权重和元数据。你可以为不同研究主题建立不同库，例如：

- 独立专题库
- 法文文献库
- 中文论文库
- 论文草稿相关材料库
- 课程资料库
- 某个项目或章节专用库

支持的管理操作包括：

- 新建知识库。
- 重命名知识库。
- 删除知识库。
- 切换当前文件管理库。
- 选择问答主库。
- 选择两个次库。
- 清空当前主库。
- 重建当前知识库索引。

### 三库联合检索

问答页可以同时选择：

- 主库
- 次库 1
- 次库 2

当前逻辑是：加载时每个库都会完整载入，不会因为配额提前截断。真正问答时，系统会在每个库内部充分召回，再按照配额 rerank 截取。默认主库占总检索配额的约 80%，次库合计占约 20%。如果只选择一个次库，则该次库获得全部次库配额；如果选择两个次库，则两个次库平分次库配额。

这样设计的好处是：主库保证主题集中，次库提供补充视角。比如主库是德里达原著，次库是中文研究论文和法文研究论文，系统会优先围绕原著展开，同时吸收研究史和二级文献。

### 文件导入

支持导入：

- PDF
- DOCX
- TXT
- 单页 URL
- 递归 URL

PDF 当前要求具有可复制文本层。由于运行载荷限制，系统不再内置 OCR。如果 PDF 是扫描件，需要先用外部 OCR 工具识别成带文本层的 PDF 后再导入。

导入时可以选择：

- 清空当前主库后重建。
- 追加到当前主库。

导入完成后，系统会生成向量索引、保存切块、记录文件状态，并刷新文件管理页。

### 文件管理

文件管理页支持：

- 按类型折叠显示：PDF、Word、TXT、URL、Other。
- URL 母链接折叠显示：默认只显示母 URL，子 URL 需要展开后查看。
- 勾选文件。
- 全选文件。
- 取消全选。
- 删除选中文件。
- 为文件设置权重。
- 查看 PDF 读取状态。
- 重建索引。
- 清空库。

权重范围为 1 到 3。权重越高，该文件在检索和 rerank 中获得的加成越高。适合把核心原著、核心论文、用户最信任的资料设为高权重。

### 学术写作模式

当前内置五种写作模式：

```text
内容论述
文献综述
概念梳理
引文补注
学术批评审稿
```

内容论述用于生成总论性质的学术文章。它会围绕用户问题组织论据、评述、反思、反驳和再论证，目标是生成可继续扩展为论文正文的长段落。

文献综述用于梳理研究史、争议地图、方法差异和研究缺口。它不只是列文献摘要，而是强调文献之间的继承、分化、分歧和盲点。

概念梳理用于处理某个术语或概念簇。它会说明概念定义、语境、相邻概念、反概念、译名差异和写作风险。

引文补注用于处理用户提供的 Word 原文。系统会尽量保留原文，只在需要证据的位置补充括号夹注，并在文末生成参考文献。

学术批评审稿也需要目标 Word 原文。系统会根据目标文本和知识库材料，从论证、证据、结构、概念使用和文献覆盖等方面给出审稿意见。

### 目标 Word 文档

当选择“引文补注”或“学术批评审稿”时，界面会出现目标文章模块。用户需要选择一个 `.docx` 文件。系统会读取这篇 Word 原文，把它作为主要处理对象。

这解决了一个重要问题：几万字文章不适合直接塞进聊天框。现在用户只需选择 Word 文件，然后在问题框里写“给我补注”或“帮我审稿”，系统会用目标文档内容构造检索问题。

目标文档模式的检索逻辑是：

```text
目标 Word 原文
-> 分段构造检索查询
-> 从三个知识库中召回相关材料
-> 生成补注或审稿结果
-> 对未找到出处的位置做少量迭代修补
-> 输出正文夹注与参考文献
```

### 引用格式控制

系统内部仍会给模型提供 `RefID`、来源、页码、chunk 编号等信息，用于材料核验。但最终输出会清理这些内部 ID，不会把 `RefID` 暴露给用户。

当前最终可见输出遵循：

- 正文使用括号夹注。
- 不使用 `[1]`、`[2]` 脚注编号。
- 不输出 `## 脚注` 模块。
- 文末统一输出 `## 参考文献`。
- 参考文献条目前不加编号。
- 格式尽量参照 GB/T 7714-2015。
- 不编造缺失字段。

如果模型偶尔仍输出旧式脚注，`query.py` 会做一次后处理，把 `[n]` 转成括号出处，并把 `## 脚注` 改为 `## 参考文献`。

### 多语言能力

当前 embedding 模型为：

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

它支持中文、英文、法文等多语言材料统一检索。用户可以把法文论文、英文专著和中文研究论文放入同一个库或不同库中，系统会在同一检索流程中处理。

### 模型调用

系统支持多种模型配置：

- DeepSeek
- OpenAI-compatible
- Qwen
- xAI
- Gemini
- router.one 等中转服务

普通 OpenAI-compatible chat 模型走 `/v1/chat/completions`。`gpt-5.5` 这类必须走 Responses API 的模型已经单独适配，会自动请求 `/v1/responses`，不影响其他模型。

### 状态栏与日志

系统有状态栏和进度条，会显示：

- 模型预加载
- 知识库加载
- 导入
- 切块
- 向量化
- 问题扩写
- HyDE
- 检索
- rerank
- 生成
- 完成或失败

导入日志会尽量精简，不再把大量底层 PDF warning 刷屏。重复日志会被压缩，界面只保留最近部分信息，避免长任务时日志过载。

### 导出回答

问答结果可以导出为文本文件。默认导出文件名为：

```text
rag.txt
```

保存后，回答框会自动恢复为空白状态，方便开始下一轮问题。

## 架构

### 总体架构

系统大体分为五层：

```text
GUI 界面层
-> 应用编排层
-> 导入与索引层
-> 检索与生成层
-> 学术 prompt 层
```

这五层分别对应：

```text
ui_layout.py
main.py
ingest.py / model.py
query.py
academic_prompt.py
```

### `main.py`

`main.py` 是应用入口和编排中心。它负责：

- 启动 CustomTkinter 应用。
- 读取 `libraries.json`。
- 读取 `config.json`。
- 管理当前主库和次库。
- 管理模型配置。
- 触发文件导入。
- 触发 URL 导入。
- 触发问答。
- 触发重建索引。
- 调用 UI 刷新。
- 管理状态栏和进度条。
- 保存回答到 `rag.txt`。

它不直接实现复杂检索算法，而是把任务分发给 `ingest.py`、`query.py`、`model.py` 和 `ui_layout.py`。

### `ui_layout.py`

`ui_layout.py` 是界面布局层，负责创建：

- 左侧知识库栏。
- 问答页。
- 文件管理页。
- 导入页。
- 配置页。
- 目标 Word 文章选择模块。
- 文件树。
- 权重控件。
- 状态和日志区域。

它尽量只做布局，不承载核心业务逻辑。

### `model.py`

`model.py` 负责本地 embedding 模型。它提供：

- 本地模型缓存目录。
- 模型完整性检查。
- 下载重试。
- 不完整缓存清理。
- 旧模型缓存清理。
- CPU / CUDA 自动选择。
- LangChain 兼容 embedding 接口。
- `encode_texts()` 批量向量化。

首次运行时，如果本地没有模型，会从 HuggingFace 下载。下载完成后，后续启动会优先使用本地缓存。

### `ingest.py`

`ingest.py` 负责导入流程。它完成：

- PDF 文本读取。
- DOCX 文本读取。
- TXT 文本读取。
- URL 抓取。
- 递归 URL 抓取。
- 文档切块。
- 语言检测。
- `doc_id` 生成。
- `chunk_id` 生成。
- Chroma 写入。
- `chunks.pkl` 保存。
- `library_meta.json` 写入。
- `ingest_status.json` 写入。
- 旧库索引重建。

当前版本不再执行 OCR。PDF 如果没有文本层，会被记录为读取失败，并提示用户外部 OCR 后再导入。

### `query.py`

`query.py` 是检索与生成管线。它负责：

- 构造 LLM。
- 兼容 Chat Completions。
- 兼容 `gpt-5.5` Responses API。
- 问题扩写。
- HyDE 生成。
- 目标 Word 文档分段检索。
- 多库配额。
- Chroma dense retrieval。
- BM25 sparse retrieval。
- RRF 融合。
- 本地 embedding rerank。
- 文件权重加成。
- 上下文压缩。
- 引文补注迭代修补。
- 输出引用格式后处理。
- 生成参考文献。

多库检索并不是简单拼接结果，而是先在每个库里充分召回，再按配额进行筛选和 rerank。

### `academic_prompt.py`

`academic_prompt.py` 是学术任务与提示词控制层。它负责：

- 定义写作模式。
- 定义每个模式的目标。
- 定义每个模式必须完成和必须避免的事项。
- 控制最低正文规模。
- 控制最低正文夹注数量。
- 指定 GB/T 7714-2015 引用格式。
- 要求输出完整学术文段。
- 限制模型不要碎片化分点。
- 要求不得编造来源和页码。
- 要求最终输出 `## 参考文献`。

### 数据流

导入数据流：

```text
PDF / DOCX / TXT / URL
-> load_documents()
-> split_documents()
-> assign stable ids
-> embeddings
-> Chroma rag_collection
-> chunks.pkl
-> library_meta.json
```

问答数据流：

```text
question
-> expand
-> HyDE
-> retrieve from selected libraries
-> dense + sparse retrieval
-> RRF fusion
-> rerank
-> context compression
-> build prompt
-> LLM invoke
-> normalize references
-> final answer
```

目标文档数据流：

```text
target .docx
-> extract text
-> split into retrieval segments
-> retrieve evidence from libraries
-> generate cite patch or review
-> repair unresolved snippets
-> normalize references
```

## 目录结构

项目根目录主要文件如下：

```text
.
|- main.py
|- ui_layout.py
|- model.py
|- ingest.py
|- query.py
|- academic_prompt.py
|- libraries.json
|- config.json
|- requirements.txt
|- readme.md
|- chroma_db/
|- db_*/
`- model_cache/
```

### 源码文件

`main.py` 是应用入口。

`ui_layout.py` 定义界面布局。

`model.py` 管理本地 embedding 模型。

`ingest.py` 负责导入、切块、写入向量库。

`query.py` 负责检索、排序、生成和引用后处理。

`academic_prompt.py` 定义学术写作模式和引用规则。

### 配置文件

`config.json` 保存模型配置，例如模型名、API key、base URL。

`libraries.json` 保存知识库列表，例如库名和路径。

`requirements.txt` 保存 Python 依赖。

`.env` 可以用于保存环境变量，但当前主要配置仍由界面保存到 `config.json`。

### 知识库目录

每个知识库目录通常包含：

```text
chunks.pkl
file_weights.json
library_meta.json
ingest_status.json
Chroma 向量数据
```

`chunks.pkl` 保存切块后的 LangChain Document。

`file_weights.json` 保存每个文件或 URL 的权重。

`library_meta.json` 记录该库使用的 embedding 模型。

`ingest_status.json` 记录 PDF 读取状态。

Chroma 向量数据保存 dense retrieval 所需的向量索引。

### 模型缓存目录

`model_cache/` 保存本地 embedding 模型。模型下载完整后，后续启动会直接读取本地缓存，不应每次联网重新下载。

如果模型缓存损坏，系统会尝试清理不完整缓存并重新下载。

## 安装与使用

这一部分按“不懂技术也能照着做”的方式写。

### 第一步：准备 Python 环境

推荐安装 Anaconda 或 Miniconda。安装后打开：

```text
Anaconda Prompt
```

或者 Windows PowerShell。

建议创建一个专用环境：

```bash
conda create -n academic-rag python=3.11
conda activate academic-rag
```

如果你已经有自己的环境，例如 `pzy`，也可以直接使用：

```bash
conda activate pzy
```

### 第二步：进入项目目录

假设项目放在：

```text
D:\rag
```

在命令行输入：

```bash
cd /d D:\rag
```

如果你在 PowerShell 中，也可以输入：

```powershell
Set-Location D:\rag
```

### 第三步：安装依赖

在项目目录中运行：

```bash
pip install -r requirements.txt
```

主要依赖包括：

- `customtkinter`
- `langchain`
- `langchain-chroma`
- `langchain-openai`
- `langchain-google-genai`
- `sentence-transformers` 相关依赖
- `chromadb`
- `pypdf`
- `docx2txt`
- `beautifulsoup4`
- `requests`
- `rank_bm25`
- `torch`
- `numpy`

如果你以前为了 OCR 安装过 PaddleOCR，现在本项目已经不再使用，可以卸载：

```bash
pip uninstall paddleocr paddlex paddlepaddle paddlepaddle-gpu pypdfium2
```

### 第四步：启动程序

运行：

```bash
python main.py
```

如果你使用指定 Python 路径，也可以类似这样运行：

```bash
C:\Users\lenovo\anaconda3\envs\pzy\python.exe D:\rag\main.py
```

首次启动时，系统会检查 embedding 模型缓存。如果本地没有模型，会下载：

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

首次下载可能需要几分钟。下载完成后，后续启动会优先使用本地缓存。

### 第五步：创建知识库

程序打开后，先在左侧知识库区域创建一个库。建议库名清晰，例如：

```text
derrida_main
french_papers
cn_articles
chapter_1_sources
```

创建后，在问答页选择主库和次库，并点击：

```text
确认并加载知识库
```

### 第六步：导入资料

进入“导入”页。

如果导入本地文件，点击导入文件夹，选择包含 PDF、DOCX、TXT 的文件夹。

系统会提醒 PDF 需要具有文本层。如果 PDF 是扫描件，需要先外部 OCR。

导入时会询问：

```text
是否清空当前主库后重建？
```

选择“是”表示清空主库后重新导入。

选择“否”表示追加导入。

如果导入 URL，点击“输入 URL 并导入”，输入网页地址。系统会询问是否递归抓取。单页文章建议不递归，网站专题页可以尝试递归。

### 第七步：管理文件和权重

导入完成后，进入“文件管理”页。

你可以看到文件按类型折叠：

```text
PDF
Word
TXT
URL
Other
```

每个文件右侧可以设置权重：

```text
1 / 2 / 3
```

建议：

- 核心原著设为 3。
- 重要论文设为 2 或 3。
- 辅助材料设为 1。
- 质量不稳定的网页设为 1。

### 第八步：配置模型

进入“配置”页，填写：

```text
配置名
Provider
Model
API Key
Base URL
```

保存后，在问答页选择该配置。

常见配置示例：

```json
{
  "name": "deepseek",
  "provider": "DeepSeek",
  "model": "deepseek-chat",
  "api_key": "你的 key",
  "base_url": "https://api.deepseek.com/v1"
}
```

```json
{
  "name": "qwen",
  "provider": "Qwen",
  "model": "qwen-plus",
  "api_key": "你的 key",
  "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
}
```

```json
{
  "name": "gpt-5.5",
  "provider": "OpenAI",
  "model": "gpt-5.5",
  "api_key": "你的 router.one key",
  "base_url": "https://api.openai.com/v1"
}
```

注意：`base_url` 填根地址即可，不要写成 `/chat/completions` 或 `/responses`。

### 第九步：开始问答

回到问答页。

选择：

```text
模型配置
写作模式
主库
次库 1
次库 2
```

点击确认加载知识库。

在问题框输入问题，例如：

```text
德里达的责任概念为什么不能被简单理解为道德义务？
```

点击提问后，系统会经历：

```text
问题扩写
HyDE
多库检索
rerank
生成
引用后处理
```

最终输出会包含正文和 `## 参考文献`。

### 第十步：使用引文补注

选择写作模式：

```text
引文补注
```

界面会出现目标文章模块。点击选择 `.docx` 文件。

问题框可以输入：

```text
给我补注
```

或者：

```text
请尽可能为其中涉及德里达责任、他者、赠予和解构的论断补充出处。
```

系统会读取目标 Word 原文，并从所选知识库中检索可用材料。输出时尽量保留原文，只增加括号夹注和参考文献。

### 第十一步：使用学术批评审稿

选择写作模式：

```text
学术批评审稿
```

选择目标 `.docx` 文件。

问题框可以输入：

```text
请按照学术期刊审稿意见的方式指出这篇文章的问题。
```

系统会围绕目标文档和知识库材料，输出总体判断、主要问题、次要问题和修改建议。

### 第十二步：导出结果

生成后点击保存回答。默认文件名为：

```text
rag.txt
```

保存完成后，回答框会清空，方便继续下一轮任务。

## 配置与导入

### `config.json`

模型配置会保存到 `config.json`。每条配置通常包含：

```json
{
  "name": "配置名",
  "provider": "OpenAI",
  "model": "模型名",
  "api_key": "API key",
  "base_url": "https://example.com/v1"
}
```

`name` 是界面里显示的配置名。

`provider` 决定使用哪类模型适配。

`model` 是模型 ID。

`api_key` 是密钥。

`base_url` 是接口根地址。

安全提醒：当前项目会把 key 保存在本地 `config.json`。不要把带有真实 key 的 `config.json` 上传到公开仓库。

### Provider 说明

DeepSeek 默认地址：

```text
https://api.deepseek.com/v1
```

Qwen 默认地址：

```text
https://dashscope.aliyuncs.com/compatible-mode/v1
```

xAI 默认地址：

```text
https://api.x.ai/v1
```

OpenAI 官方地址：

```text
https://api.openai.com/v1
```

### GPT-5.5 专门适配

`openai/gpt-4.1` 可以走 chat completions。

但 `gpt-5.5` 当前要求走 Responses API。系统已经做了专门适配：

```text
model = gpt-5.5
-> POST {base_url}/responses
```

其他模型仍然走原来的 ChatOpenAI，不受影响。

如果出现：

```text
model 'gpt-5.5' must be called via /v1/responses
```

说明不能用 chat completions 调它，当前代码已经处理这个情况。


### `libraries.json`

知识库列表保存在 `libraries.json`。它记录每个库的名称和路径。一般不需要手动修改，建议通过界面创建和删除。

### 文件导入规则

PDF 规则：

- 只支持有文本层的 PDF。
- 扫描 PDF 需要外部 OCR。
- 损坏 PDF 会被记录为失败。
- 某些 PDF 可能能读取部分文本，但质量不稳定，建议检查导入日志。

DOCX 规则：

- 支持 `.docx`。
- 目标文章模块也只支持 `.docx`。
- 如果是 `.doc`，请先用 Word 另存为 `.docx`。

TXT 规则：

- 支持普通文本文件。
- 建议使用 UTF-8 编码。

URL 规则：

- 支持单页 URL。
- 支持递归 URL。
- 递归抓取适合结构清楚的网站。
- 对需要登录、反爬严格或动态渲染的网站，抓取效果可能不稳定。

### 导入模式

清空重建适合：

- 第一次导入。
- 想完全替换知识库内容。
- embedding 模型切换后重建。
- 旧库索引不一致。

追加导入适合：

- 已有库继续补充材料。
- 新增几篇论文。
- 新增少量网页。

如果当前库使用旧 embedding 模型，而程序已经切换新 embedding 模型，追加导入可能被阻止。这是为了避免同一个库里混合不同维度的向量。

### 模型缓存

模型缓存位于：

```text
model_cache/
```

如果模型每次启动都重新下载，通常说明缓存不完整或目录损坏。系统会做完整性检查和自动重试。

如果报：

```text
no file named model.safetensors
```

通常是模型下载不完整。可以等待系统自动重新下载，或手动清理对应模型缓存后重启。

## 项目管理

### 新建知识库

适合为不同主题建立不同库。推荐命名清楚，避免都叫 `chroma_db`。

示例：

```text
derrida_primary
derrida_cn_articles
derrida_fr_articles
levinas_sources
thesis_chapter_1
```

### 重命名知识库

重命名只改变显示名称，不应随意手动改动底层目录。建议通过界面操作。

### 删除知识库

删除库会删除对应目录和记录。这个操作不可轻易恢复。执行前请确认：

- 是否仍需要该库。
- 是否已经备份。
- 是否没有误选库。

### 清空当前主库

清空库会删除当前库的索引和辅助文件，包括：

```text
chunks.pkl
file_weights.json
library_meta.json
ingest_status.json
Chroma 数据
```

清空后可以重新导入。

### 删除文件

文件管理页可以勾选文件后删除。删除时系统会同时清理：

- Chroma 中对应向量。
- `chunks.pkl` 中对应 chunk。
- `file_weights.json` 中对应权重。
- PDF 状态记录中的对应项。

URL 删除时要注意母 URL 和子 URL 的关系。母 URL 默认折叠展示，展开后可以看到子 URL。

### 设置权重

权重影响检索排序。建议：

- 权重 3：核心原典、核心论文、最可靠材料。
- 权重 2：重要二级文献、较可信资料。
- 权重 1：普通材料、背景材料、网页资料。

权重不会让某个文件“必然被引用”，但会在检索和 rerank 中提高优先级。

### 重建索引

当出现以下情况时，点击“重建索引”：

- 旧库没有 `chunk_id`。
- 切换过 embedding 模型。
- Chroma 与 `chunks.pkl` 不一致。
- 查询时报 `Error finding id`。
- 报 embedding 维度不一致。
- 法文、英文材料检索效果明显异常。
- 删除或追加后怀疑索引状态不稳定。

重建索引会：

- 读取现有 `chunks.pkl`。
- 补齐 `doc_id` 和 `chunk_id`。
- 使用当前 embedding 模型重新向量化。
- 重写 Chroma。
- 更新 `library_meta.json`。

它不会重新读取原始 PDF 文件，而是基于已有 chunk 重建。

### 选择主库和次库

主库是当前问题最核心的材料库。次库是辅助材料库。

建议用法：

- 主库放原典。
- 次库 1 放研究论文。
- 次库 2 放外文资料或背景资料。

如果你只想严格依据一个库，就只选主库。

如果你想让系统更广泛地参考材料，就加入次库。

### 回答导出

点击保存回答后，会弹出保存位置选择。默认文件名是：

```text
rag.txt
```

保存完成后，回答框自动清空。这样下一轮生成不会和上一轮回答混在一起。

## 常见问题

### 启动时模型一直下载怎么办？

通常是模型缓存不完整。系统会检查本地模型是否完整。如果发现缺文件，会重新下载。

建议：

- 保持网络稳定。
- 等待下载完成。
- 不要在下载中途强行关闭程序。
- 如果多次失败，删除不完整模型缓存后重启。

### 出现 `no file named model.safetensors` 怎么办？

这是模型缓存不完整的典型表现。重新启动程序，让系统重新检查并下载。如果仍失败，检查网络或手动清理 `model_cache` 中对应模型目录。

### 出现 `Collection expecting embedding with dimension ... got ...` 怎么办？

这是知识库向量维度和当前 embedding 模型不一致。常见原因是旧库使用了旧模型，新程序切换了新模型。

解决方式：

```text
文件管理页 -> 选择对应库 -> 重建索引
```

或者清空库后重新导入。

### 出现 `Error finding id` 怎么办？

这通常说明 Chroma 索引和 `chunks.pkl` 不一致。

解决方式：

```text
文件管理页 -> 重建索引
```

### PDF 导入时提示读取失败怎么办？

当前项目不内置 OCR。请检查 PDF 是否可以复制文字。

如果不能复制文字，说明它很可能是扫描件。请先用外部 OCR 工具识别成带文本层的 PDF，再导入。

### 终端出现 `invalid pdf header` 或 `incorrect startxref pointer` 是不是严重错误？

这通常是 PDF 文件结构不标准或轻微损坏。系统会尽量读取可用文本，并把无法读取的文件记录为失败。只要导入结束后大部分文件成功，就不一定需要处理。

如果某个关键 PDF 无法读取，请外部修复或重新下载该 PDF。

### 为什么不再内置 OCR？

之前尝试过 PaddleOCR，但在 Windows、GPU、Paddle 版本、CUDA DLL、依赖冲突方面非常容易出问题。为了保证主流程稳定，当前版本移除了 OCR。

当前策略是：系统负责文本层 PDF 的高质量导入；扫描件交给外部 OCR 工具处理。

### URL 递归抓取为什么不完整？

可能原因包括：

- 网站需要登录。
- 页面由 JavaScript 动态渲染。
- 网站反爬。
- 链接结构复杂。
- 超出递归限制。

建议先导入单页 URL，确认有效后再尝试递归。

### 为什么回答没有引用某篇我想要的文献？

可能原因包括：

- 该文献权重较低。
- 问题和文献内容相似度不足。
- 文献切块中相关段落没有被召回。
- 次库配额较少。
- 文献本身导入失败或文本质量差。

可以尝试：

- 提高该文件权重。
- 把它放到主库。
- 问题中明确提到作者、概念或关键词。
- 重建索引。
- 检查文件是否成功导入。

### 为什么参考文献格式仍需要人工检查？

系统会尽量按 GB/T 7714-2015 输出，但原始 metadata 可能缺少作者、出版社、年份、页码等信息。模型不能凭空知道缺失字段。因此正式提交前仍建议人工校对。

### 为什么有时输出仍像分点？

Prompt 已经要求尽量生成完整文段，不要碎片化分点。但如果用户问题本身要求列表，或模型认为审稿意见需要条目化，它仍可能适度分点。可以在问题中明确写：

```text
请不要分点，写成完整连续的论文段落。
```

### `gpt-5.5` 报 `must be called via /v1/responses` 怎么办？

当前代码已经专门适配 `gpt-5.5`。配置时确保：

```text
provider = OpenAI
model = gpt-5.5
base_url = https://api.openai.com/v1
```

不要把 base URL 写成 `/chat/completions` 或 `/responses`。


### 出现 `RequestsDependencyWarning` 怎么办？

如果看到类似：

```text
urllib3 or chardet/charset_normalizer doesn't match a supported version
```

这通常是 Python 环境中的 `requests`、`urllib3`、`charset_normalizer` 版本组合不完全匹配。多数情况下不影响程序运行。

如果想清理环境，可以尝试：

```bash
pip install -U requests urllib3 charset_normalizer
```

### API key 放在 `config.json` 安全吗？

它只保存在你的本地电脑上，但不适合上传到公开仓库。建议：

- 不要把真实 key 发给别人。
- 不要把 `config.json` 上传 GitHub。
- 如果 key 已经泄露，立即去服务商后台重置。

### 为什么问答前要“确认并加载知识库”？

这样可以让系统提前检查知识库是否存在、是否为空、embedding 模型是否匹配、`chunks.pkl` 是否可读。提前加载能减少问答时突然失败。

### 为什么引文补注需要 Word？

因为补注和审稿都需要“目标原文”。如果把几万字原文直接放进问题框，界面和模型上下文都会很难处理。选择 `.docx` 后，系统可以自动读取和分段检索。

### `.doc` 文件可以用吗？

目标文章模块当前只支持 `.docx`。如果你手里是 `.doc`，请先用 Word 或 WPS 另存为 `.docx`。

### 旧库还能用吗？

能，但如果旧库缺少 `chunk_id`，或使用旧 embedding 模型，建议先重建索引。

### 需要联网吗？

需要分情况：

- 首次下载 embedding 模型需要联网。
- 调用远程大模型 API 需要联网。
- 导入网页 URL 需要联网。
- 已下载模型且只管理本地库时，部分操作不需要联网。

## 附加说明

### 项目优点

- 资料导入。
- 知识库管理。
- 权重控制。
- 多库检索。
- 证据排序。
- 学术 prompt 约束。
- 目标文本补注。
- 审稿式批评。
- GB/T 7714-2015 参考文献。
- 导出成可继续编辑的文本。

相比直接把问题丢给大模型，它更适合需要证据、出处、材料覆盖和长期项目管理的研究工作。

### 当前限制

- 引用位置仍由模型判断，需要人工核查。
- 参考文献格式依赖原始材料 metadata，缺字段时无法自动补全。
- PDF 不内置 OCR，扫描件需要外部处理。
- URL 抓取无法保证覆盖所有动态网页。
- 长文补注会消耗较多 token。
- HyDE 可能带来检索偏置。
- 本地库目录结构不能随意移动或删除。

### 数据兼容性

当前系统会尽量兼容旧库。如果旧库缺字段，系统会在加载或重建时补齐：

- `doc_id`
- `chunk_id`
- `chunk_index`
- `chunk_total`
- `page_label`
- `citation_label`

如果旧库使用了不同 embedding 模型，系统会提示重建索引，避免向量维度不一致。

### 开发原则

后续修改代码时，建议遵循：

- 不把检索业务写进 `ui_layout.py`。
- 不把界面状态和核心算法混在一起。
- 不破坏 `chunks.pkl`、Chroma、`file_weights.json` 的对应关系。
- 修改 metadata 时同步更新导入、检索、删除和重建逻辑。
- 修改 prompt 时同步检查输出后处理。
- 修改 embedding 模型时提醒用户重建索引。

### 后续改进方向

可以继续增强：

- 结构化参考文献面板。
- BibTeX、RIS、Word 引文导出。
- 更强的 reranker。
- 长上下文分层摘要。
- 引用置信度提示。
- 文献元数据自动抽取。
- URL 子页面与母页面引用映射。
- 更细粒度的目标文本逐段补注。
- 更完整的论文写作项目面板。

### 免责声明

Academic RAG 是研究辅助工具，不是自动论文生产机。所有生成内容都应人工核查，尤其是：

- 引用是否真的支持正文判断。
- 页码是否准确。
- 参考文献字段是否完整。
- 外文材料是否被正确理解。
- 模型是否过度推断。
- 输出是否符合你的学校、期刊或出版社格式要求。

如果用于正式论文、投稿、出版或学位材料，请务必进行人工复核。
