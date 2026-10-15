# LLM Data Analysis Copilot

面向 CSV 数据的智能分析助手学习项目。最终目标是：用户上传数据、用自然语言提出问题，LLM 选择 Python/Pandas 工具执行真实计算，再根据结果生成可读的分析报告。

当前已完成基础统计模块、单次 LLM 流式调用、system 提示词、JSON 解析校验、Pydantic 回答模型，以及 SDK `parse()` 的接口级结构化输出实验。学习任务 01～06C 的代码已实现并通过离线审查。

**当前尚未将 CSV 统计结果交给模型解释**，这将是学习任务 07。Function Calling、Agent、FastAPI、Streamlit、评估系统与 RAG 均未实现。阶段状态和待确认事项见 [TASKS.md](TASKS.md)。

LLM 核心代码由学习者独立实现，Codex 负责基础工程、答疑、审查和测试；只有明确授权时才修改学习代码。

## 技术栈和结构

Python 3.11+，统一使用 conda 的 `LLM` 环境。Pandas/NumPy 负责真实统计计算；OpenAI Python SDK 调用 DashScope 兼容接口；python-dotenv 读取配置；Pydantic v2 负责本地结构校验。pytest、Ruff、mypy/pandas-stubs 分别用于离线测试、代码规范和类型检查。使用 OpenAI SDK 不代表请求发往 OpenAI，实际服务由 `DASHSCOPE_BASE_URL` 决定。

```text
llm-data-analysis-copilot/
├── src/data_copilot/
│   ├── __init__.py
│   ├── data_loader.py       # CSV 校验、读取、概览
│   ├── analysis_tools.py    # 四类统计工具
│   ├── llm_client.py        # 流式 JSON Schema 结构化调用与 SYSTEM_PROMPT
│   ├── response_parser.py   # 手写 JSON 字段校验
│   └── schemas.py           # Pydantic AnalysisAnswer
├── examples/
│   ├── analyze_sales.py     # 纯统计，无 API 请求
│   ├── hello_llm.py         # 流式调用后手写解析
│   ├── pydantic_validation.py # 纯本地校验实验
│   └── structured_output.py # SDK parse() 非流式实验
├── tests/
│   ├── test_data_loader.py
│   ├── test_analysis_tools.py
│   ├── test_llm_client.py
│   ├── test_response_parser.py
│   ├── test_schemas.py
│   └── test_structured_output.py
├── data/samples/sales.csv
├── frontend/.gitkeep
├── docs/HANDOFF.md
├── .gitignore
├── AGENTS.md
├── pyproject.toml
├── README.md
└── TASKS.md
```

## 安装与验证

在项目目录下使用已有的 `LLM` 环境（Windows PowerShell 同样适用）：

```powershell
conda activate LLM
python -m pip install -e ".[dev,llm]"
python -m pytest -q
python -m ruff check .
python -m mypy .
python -m ruff format --check .
python examples/analyze_sales.py
```

包以 editable 方式安装；无需手工调整 PYTHONPATH。`pyproject.toml` 是运行和开发依赖的唯一声明来源，conda 管理 Python 环境，环境内的 pip 安装项目包。pip 会按照 pyproject 中的版本约束解析依赖，因此可能调整已有包的版本；项目安装命令不切换 Python 解释器。完整测试需要同时安装 dev 和 llm extras，即使不发送真实 API 请求也需要 SDK 与 Pydantic。

新机器尚无 `LLM` 环境时，执行 `conda create -n LLM python=3.12 pip`，再激活环境并按上面的命令安装项目。依赖以 pyproject.toml 为准，约束范围内的安装版本可能不同。

若 PowerShell 未初始化 conda，可先执行 `conda init powershell` 并重新打开终端；自动化中直接使用命名环境：

```powershell
conda run --no-capture-output -n LLM python -m pytest -q
conda run --no-capture-output -n LLM python examples/analyze_sales.py
```

pytest 临时文件固定放在项目内的 `.pytest_cache/tmp`（每次测试会重建这个专用目录），避免系统临时目录权限问题。VS Code 请通过“Python: Select Interpreter”选择 conda 的 LLM 环境；当前工作区的已有配置保留。

## LLM 配置与示例

沿用现有环境变量，也可在项目目录的本地 `.env` 中配置（该文件已被 Git 忽略）：

| 变量 | 含义 |
| --- | --- |
| DASHSCOPE_API_KEY | 自己账号的 API Key，不写进代码或提交 Git |
| DASHSCOPE_BASE_URL | 当前账号/地域的 OpenAI 兼容接口地址 |
| DASHSCOPE_MODEL | 所用模型 ID，结构化实验需支持 JSON Schema |

三个变量都必填，空白值会被拒绝。当前程序设置客户端 timeout=60.0；未额外修改 SDK 的默认重试设置。结构化模式、思考模式与模型支持范围以 [DashScope 官方文档](https://help.aliyun.com/zh/model-studio/qwen-structured-output) 为准。

以下命令在项目目录、LLM 环境中执行：

```powershell
# 离线：不需要密钥，不发送请求
python examples/analyze_sales.py
python examples/pydantic_validation.py

# 真实请求：先配置环境变量
python examples/hello_llm.py
python examples/structured_output.py
```

hello_llm.py 发送一次请求；structured_output.py 的 main 按 QUESTIONS 发送三次独立请求。后者没有对话历史，仍是单轮实验。运行这些示例会产生实际 API 调用及相应 token 消耗。

## 三种结构化处理方式

| 当前实现 | 处理流程 | 返回结果 |
| --- | --- | --- |
| ask_llm + parse_analysis_answer | 流式 create(response_format=json_schema) → 拼接正文 → Pydantic 校验 → 手写解析 | ask_llm 返回已校验的 JSON 正文及调用元数据；解析函数返回回答字典 |
| AnalysisAnswer 本地实验 | model_validate / model_validate_json → 严格类型与摘要校验 | Pydantic 模型，可导出字典/JSON/Schema |
| ask_structured 独立实验 | SDK parse(response_format=AnalysisAnswer) → message.parsed → model_dump | 三字段回答字典，不包含调用元数据 |

三字段为 `summary: str`、`explanation: str`、`missing_information: list[str]`。全部必填，禁止额外字段，summary 不得纯空白；explanation 可为空字符串，missing_information 可为空列表。手写解析与 Pydantic 模型目前并存，尚未替换或合并。

`ask_llm` 现已发送由 `AnalysisAnswer.model_json_schema()` 生成的严格接口级 Schema，保留流式响应和原有返回格式，并显式设置 `enable_thinking=False`。需要配置支持 JSON Schema 的模型；不支持或校验失败时直接报错，不回退到仅提示词模式。`hello_llm.py` 无需修改即可使用该路径。

Pydantic 会生成 Schema，SDK 将它作为接口级约束发送；自定义摘要校验仍在本地执行，不会自动成为同等的远端 Schema 约束。格式通过不代表统计内容正确，截断、拒绝回答和解析失败也仍需处理。完整接口与异常差异见 [HANDOFF](docs/HANDOFF.md)。

## 当前验证范围

2026-10-14 在现有 LLM 环境验收：**205 项离线测试通过**，Ruff 通过，mypy 检查 16 个源文件无错误。结构化测试既有模拟 SDK 返回，也有真实 SDK 配合 httpx.MockTransport 的内存 HTTP 响应；没有调用外部服务，不读取真实 .env。

学习者已提供流式回答及一份三字段 JSON 的实际输出。原生 Schema/Pydantic parse 的三题真实运行记录尚未在仓库归档；离线测试不能代替模型行为评估。

实际验收环境的 OpenAI SDK 为 3.14.1，而 pyproject 的 llm extra 声明 `openai>=1.60,<3`。两者尚未统一；这里如实记录，本次文档更新不调整环境或依赖。执行安装命令会按声明选择版本，不能把现有环境通过测试当成整个版本范围均已验证。详细版本见 HANDOFF。

## 示例数据和工具调用

`data/samples/sales.csv` 是 **40 条人工构造的模拟订单**，无真实个人信息，不代表真实业务表现。字段包括 order_id、region、category、sales、advertising_spend、quantity；设计了少量销售额、广告投入和地区缺失值，用于练习缺失处理。数据无需 API Key。

激活 `LLM` 后，在项目目录用 `python` 开启 Python，然后执行：

```python
import json
from data_copilot.data_loader import load_csv, dataset_overview
from data_copilot.analysis_tools import (
    dataset_info,
    describe_numeric,
    group_aggregate,
    correlation,
)

df = load_csv("data/samples/sales.csv")
results = {
    "overview": dataset_overview(df, preview_rows=3),
    "info": dataset_info(df),
    "description": describe_numeric(df, ["sales", "advertising_spend"]),
    "by_region": group_aggregate(df, "region", "sales", "mean"),
    "correlation": correlation(df, "sales", "advertising_spend"),
}
print(json.dumps(results, ensure_ascii=False, allow_nan=False, indent=2))
```

打印仅发生在调用端，核心模块不输出。相关关系不代表因果关系。

## 数据与统计约定

- 仅接受逗号分隔 CSV（扩展名大小写均可），默认 `utf-8-sig`，可显式传入 `encoding="gbk"` 等编码；不自动猜测编码。
- 严格拒绝空文件、空白列名、完全相同的重复列名、非空记录列数不一致和无效引号。允许只有表头的零行数据、引号内逗号与换行、空行。列名保留原样，不去除空格或自动重命名。
- Pandas 自动推断类型，使用其默认缺失标记（如空单元格、NA、NULL）；这可能影响字符串型 ID、前导零或 NA 字符串。没有额外清洗、填补或类型强制转换。数据语义需要调用者确认。
- 描述统计跳过缺失值；标准差为样本标准差 `ddof=1`；分位数遵循 Pandas 默认线性插值。全缺失和单个有效值等导致不可计算时返回 `null`，列出 `unavailable_statistics`。
- 分组保留缺失键并用 `null` 表示，按首次出现顺序返回，仅包含实际观测的类别；count 只统计目标列非缺失值。sum 对全缺失组返回 `null`，真实和为零仍返回 0。
- Pearson 使用两个字段均非缺失的成对样本；少于 2 对、任一变量零方差返回 `null` 及明确状态。其他浮点溢出返回 `nonfinite_result`。
- 数值工具拒绝字符串、布尔、复数以及正负无穷输入，不悄悄删除无穷值或把字符串转成数字。有限输入仍可能发生浮点溢出；结果非有限时为 `null` 并有状态/不可用字段说明。
- NumPy 标量转 Python 标量；缺失值及概览中的 NaN/Infinity 转 `null`；日期时间转 ISO 字符串，时长转字符串。不对有限数值舍入。其他不支持的对象抛出 TypeError。
- 直接传入的 DataFrame 需要唯一、非空字符串列名。所有工具不修改输入 DataFrame。
- CSV 流式校验一次后再由 Pandas 完整读取，时间开销约两遍；目前不限制文件大小、不分块处理、也不支持远程 URL。适合本地学习数据；大文件需另行设计内存和上传大小限制。读取期间文件须保持不变。

接口格式、异常和后续接入方式详见 [docs/HANDOFF.md](docs/HANDOFF.md)。

CSV 含 NUL（空字节）时会抛出 CSVValidationError，防止底层读取器截断内容而丢失数据。

工具参数在执行前校验：列名必须为非空字符串，columns 必须是单个列名或字符串序列，aggregation/method 必须是字符串。非法类型抛出 TypeError，合法类型但非法值抛出 ValueError。

类型检查使用 mypy，Pandas 的类型信息由开发依赖 pandas-stubs 提供。在项目目录运行 `python -m mypy .`；只使用类型存根，不需要忽略缺失导入错误。
