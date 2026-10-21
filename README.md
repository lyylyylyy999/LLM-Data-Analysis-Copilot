# LLM Data Analysis Copilot

面向 CSV 数据的智能分析助手学习项目：上传 CSV，用自然语言提问，由模型选择工具，Python/Pandas 执行真实计算，再由模型解释结果。

更新日期：2026-10-21。当前已完成学习任务 01～20，包括有限轮 Function Calling、FastAPI 上传分析接口和 Streamlit 前端。任务 21（第一版模型效果评估集）已布置，尚未实现。完整进度见 [TASKS.md](TASKS.md)，代码交接见 [docs/HANDOFF.md](docs/HANDOFF.md)。

LLM 与应用核心代码由学习者编写；Codex 负责基础工程、答疑、审查和测试，明确授权时协助修正。当前是本地学习原型，尚未完成模型效果评估或生产部署验收。

## 当前能力

- 严格读取和校验 CSV，返回字段、类型、缺失数量和预览。
- 描述统计、分组聚合和 Pearson 相关分析，返回严格 JSON 兼容结果。
- 模型提出工具名称及参数，Python 通过白名单校验后执行；DataFrame 由 Python 持有。
- 同轮多个工具顺序执行；支持根据前轮结果继续请求工具，默认最多 3 批工具调用。
- FastAPI 提供健康检查、固定示例分析、临时上传分析及请求/响应校验。
- Streamlit 提供上传、提问、结果和工具详情展示，以及输入和通信错误提示。
- 保留流式调用、JSON Schema、Pydantic、SDK parse 等早期独立学习示例。

尚未实现评估集及评估报告、跨请求聊天历史、数据持久化、用户认证、部署、RAG 或向量数据库；没有引入 LangChain/LangGraph。

## 技术栈与目录

Python 3.11+；统一使用 conda `LLM` 环境。Pandas/NumPy 负责统计，OpenAI Python SDK 调用 DashScope 兼容服务，Pydantic v2 校验结构，FastAPI/Uvicorn 提供后端，Streamlit/httpx 提供前端。pytest、Ruff 和 mypy 用于离线验证。

```text
llm-data-analysis-copilot/
├── src/data_copilot/
│   ├── data_loader.py       # CSV 读取、校验和概览
│   ├── analysis_tools.py    # 纯统计计算
│   ├── tool_registry.py     # 四种工具 Schema、参数校验和白名单分发
│   ├── analysis_service.py  # 有上限的工具调用循环，返回结果而不打印
│   ├── api.py               # HTTP 路由、临时上传和异常处理
│   ├── api_schemas.py       # HTTP 请求与响应模型
│   ├── llm_client.py        # 早期流式结构化调用实验
│   ├── response_parser.py   # 手写 JSON 回答校验
│   ├── schemas.py           # 早期三字段 AnalysisAnswer
│   └── __init__.py
├── examples/               # 统计、结构化输出及工具调用命令行示例
├── frontend/app.py         # 仅通过 HTTP 调用后端
├── tests/                  # 统计、SDK 模拟、HTTP 与页面逻辑离线测试
├── data/samples/sales.csv   # 40 条模拟订单
├── docs/HANDOFF.md
├── AGENTS.md
├── pyproject.toml
├── README.md
└── TASKS.md
```

`evals/` 和评估脚本属于任务 21 的计划，不是当前已有文件。

## 环境安装

以下命令均在包含 `pyproject.toml` 的项目目录执行。已有环境时直接激活；新机器可先创建：

```powershell
conda create -n LLM python=3.12 pip
conda activate LLM
python -m pip install -e ".[dev,llm,api,frontend]"
```

`pyproject.toml` 是依赖声明来源：`dev` 提供测试和检查工具，`llm` 提供模型 SDK，`api` 提供后端及 multipart 解析，`frontend` 提供页面和 HTTP 客户端。完整验证使用全部依赖组。项目采用 src-layout，editable 安装后无需设置 PYTHONPATH。

**依赖一致性待处理**：当前验收环境的 OpenAI SDK 为 **3.14.1**，声明仍是 `openai>=1.60,<3`。执行安装会按声明重新解析，可能更改现有版本；当前 3.14.1 下通过的测试不能证明 `<3` 范围已验证。本次只更新文档，不调整依赖或环境。

自动化不依赖终端激活状态，例如：

```powershell
conda run --no-capture-output -n LLM python -m pytest -q
```

## 配置

后端模型调用期间读取下列变量，也支持项目目录内、已被 Git 忽略的本地 `.env`。不要提交真实凭据。

| 变量 | 用途 |
| --- | --- |
| `DASHSCOPE_API_KEY` | 模型服务凭据 |
| `DASHSCOPE_BASE_URL` | DashScope 的 OpenAI 兼容接口地址，按账号及地域配置 |
| `DASHSCOPE_MODEL` | 请求使用的模型名，工具分析需支持 Function Calling |
| `DATA_COPILOT_API_URL` | 前端访问后端的根地址，默认 `http://127.0.0.1:8000` |

前三项均必填。前端不读取模型凭据；自定义后端地址通过启动 Streamlit 的进程环境设置：

```powershell
$env:DATA_COPILOT_API_URL = "http://127.0.0.1:8000"
```

返回的 `model` 取生成最终回答的实际响应，不直接使用配置中的模型别名。历史真实验收使用过 `qwen3.8-flash`，模型名称及能力需以自己的服务配置为准。

## 启动应用

打开两个终端，均激活 `LLM` 并进入项目目录。

终端一启动后端：

```powershell
python -m uvicorn data_copilot.api:app --reload
```

终端二启动前端：

```powershell
python -m streamlit run frontend/app.py
```

后端文档位于 <http://127.0.0.1:8000/docs>，前端使用 Streamlit 启动时显示的地址，默认通常为 <http://localhost:8501>。上传 UTF-8 CSV，输入问题，点击“开始分析”。提交会产生真实模型请求和相应费用；修改页面输入本身不会自动请求分析。

可使用 `sales.csv` 提问：

- 这份数据有多少行、多少列？
- 每个地区的平均销售额是多少？
- 请先检查哪些数值字段存在缺失，再只对这些字段做描述统计。
- sales 与 advertising_spend 的 Pearson 相关系数及有效样本量是多少？能证明因果吗？

## HTTP 接口

| 接口 | 请求 | 行为 |
| --- | --- | --- |
| `GET /health` | 无 | 返回 `{"status":"ok"}`；不读取 CSV、不请求模型 |
| `POST /analyze` | JSON：`{"question":"这份数据有多少行、多少列？"}` | 使用固定的 `sales.csv` |
| `POST /analyze-file` | multipart：文件字段 `file`、文本字段 `question` | 临时读取上传 CSV 后分析 |

两个分析接口成功返回 `answer`、`model`、`finish_reason`、`tool_results`。`/docs` 底部 Schemas 展示 `AnalyzeRequest`、`AnalyzeResponse` 和 `ToolExecutionResult`。请求/响应模型只校验结构和字段约束，不保证回答内容正确，也不等于约束模型生成的 Structured Outputs。

上传最大 5 MiB（5 × 1024 × 1024 字节），恰好等于上限允许；扩展名大小写不敏感。文件内容按 UTF-8（兼容 BOM）读取，不自动猜测编码。服务端生成临时路径，不使用用户文件名保存；成功或失败均清理临时文件。此应用读取限制不等于服务器层的完整请求体限制。

| 状态码 | 当前含义 |
| --- | --- |
| 200 | 接口成功返回，仍需检查统计解释质量 |
| 400 | 上传文件名、扩展名、空内容或 CSV 格式/编码错误 |
| 413 | 上传文件内容超过 5 MiB |
| 422 | 缺少必填字段、空白问题或其他请求校验失败 |
| 500 | 配置、工具轮数、响应校验或其他内部错误 |
| 502 | 后端访问模型时连接失败或收到上游 API 错误 |
| 504 | 后端模型请求超时 |

服务端异常以固定 `detail` 返回，日志保留堆栈。422 的字段校验错误可能是列表格式。不要把分析阶段的 `ValueError` 都当作 CSV 输入错误。健康检查正常仅说明应用可响应，不证明模型服务可用。

## 命令行与 Python 使用

```powershell
# 纯本地计算/校验，不发送模型请求
python -m examples.analyze_sales
python -m examples.pydantic_validation

# 复用与 HTTP 相同的分析服务，发送真实模型请求
python -m examples.tool_execution --csv data/samples/sales.csv --question "每个地区的平均销售额是多少？"

# 早期独立学习实验，也发送真实请求
python -m examples.hello_llm
python -m examples.structured_output
```

`examples/structured_output.py` 当前 main 只询问一次“什么是过拟合？”，并非自动运行整个 QUESTIONS 列表。它和流式实验使用三字段 `AnalysisAnswer`；当前工具分析服务返回普通解释正文，HTTP 外层使用 `AnalyzeResponse`，两者不应混淆。

```python
import json
from data_copilot.data_loader import load_csv
from data_copilot.analysis_tools import describe_numeric, group_aggregate, correlation

df = load_csv("data/samples/sales.csv")
results = {
    "description": describe_numeric(df, ["sales"]),
    "by_region": group_aggregate(df, "region", "sales", "mean"),
    "correlation": correlation(df, "sales", "advertising_spend"),
}
print(json.dumps(results, ensure_ascii=False, allow_nan=False, indent=2))
```

这段纯统计示例不调用模型。需要模型解释时，调用 `analysis_service.analyze_question(df, question, max_tool_rounds=3)`；该函数不打印、不读取 CSV。

## 统计语义与示例数据

`sales.csv` 是人工构造的模拟数据，无真实个人信息，不代表真实业务表现。40 行、6 列；sales 和 advertising_spend 各缺失 2 条，region 缺失 1 条。

- CSV 严格拒绝空/重复列名、字段数不匹配、无效引号和 NUL 字节；只有表头的 CSV 可返回零行数据。
- Pandas 推断类型及缺失标记；不自动填补、清洗或转换字符串数值。直接调用读取函数可以指定编码，上传接口则固定 UTF-8。
- 描述统计跳过缺失值；样本标准差 `ddof=1`。不可计算的统计量返回 `null` 并记录原因字段。
- 分组保留缺失键为 `null`。count 统计目标列非缺失值，全缺失组的 sum 为 `null`，真实和为零仍为 0。
- Pearson 只使用两列同时非缺失的样本。样本不足或零方差返回 `null` 和状态，不能解释成相关系数为零。
- 数值工具拒绝布尔、复数、字符串列和无穷输入，不修改原 DataFrame；有限统计值不为展示而提前舍入。
- 模拟数据 sales 有效数量 38、均值约 694.6974、样本标准差约 203.1944；与广告投入相关系数约 0.9409743，完整样本 36 对。相关不等于因果。

## 验证与限制

```powershell
python -m pytest -q
python -m ruff check .
python -m mypy .
python -m ruff format --check .
```

最近代码验收快照（2026-10-21）：**675 项测试通过、0 失败**，Ruff 通过，mypy 检查 30 个源文件通过，格式检查 34 个文件通过。测试隔离真实模型请求，使用模拟 SDK、HTTP 传输和页面组件；pytest 临时目录为项目内 `.pytest_cache/tmp`。该数字不是模型准确率，也不是完整浏览器自动化通过率。

用户已人工验收上传分析、零方差、空文件、错误提示及页面连接失败。曾出现非空但无意义的 `"(!256)"` 回复，重试后正常，根因未确认；结构校验无法排除此类问题，任务 21 将区分工具结果检查和人工回答检查。

大文件仍会完整加载到 DataFrame；每次请求重新读取数据，无跨请求会话、存储或用户隔离。有限轮工具循环不等于跨请求聊天。客户端 60 秒超时和 SDK 默认重试不构成整体分析时长保证；前端读取超时为 300 秒。

## Clash / 代理排查

前端访问 `127.0.0.1`、`localhost` 或 `::1` 时显式设置 `trust_env=False`，避免环境代理介入本机通信；远程后端仍沿用环境代理。该修改不改变后端访问模型的代理行为，也不保证绕过系统 TUN 路由。

Clash 开启、FastAPI 停止时，已人工确认修正后显示“无法连接后端”。未知来源的 502/504 仅提示后端或代理网关错误；只有符合后端已知错误 JSON 的响应才显示模型服务失败/超时，不能仅凭状态码判定来源。
