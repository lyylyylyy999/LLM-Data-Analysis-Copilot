# LLM Data Analysis Copilot

面向 CSV 数据的智能分析助手学习项目：上传 CSV，用自然语言提问，由模型选择工具，Python/Pandas 执行真实计算，再由模型解释结果。

更新日期：2026-10-27。学习任务 01～24 已验收，覆盖基础工程、模型调用、有限轮 Function Calling、FastAPI 上传分析、Streamlit、七案例评估及人工审核报告。任务 25 已完成：文档与工程检查通过，用户已确认最终审核和演示无问题，人工结论已记录到独立快照。进度见 [TASKS.md](TASKS.md)，代码交接见 [docs/HANDOFF.md](docs/HANDOFF.md)，演示见 [docs/DEMO.md](docs/DEMO.md)，项目介绍见 [docs/PROJECT_SUMMARY.md](docs/PROJECT_SUMMARY.md)。

LLM 与应用核心代码由学习者编写；Codex 负责基础工程、答疑、审查和测试，明确授权时协助修正。当前定位为完成核心功能和基础评估的本地学习原型；七个固定案例不等于全面质量保证，尚未完成生产部署验收。

## 当前能力

- 严格读取和校验 CSV，返回字段、类型、缺失数量和预览。
- 描述统计、分组聚合和 Pearson 相关分析，返回严格 JSON 兼容结果。
- 模型提出工具名称及参数，Python 通过白名单校验后执行；DataFrame 由 Python 持有。
- 同轮多个工具顺序执行；支持根据前轮结果继续请求工具，默认最多 3 批工具调用。
- FastAPI 提供健康检查、固定示例分析、临时上传分析及请求/响应校验。
- Streamlit 提供上传、提问、结果和工具详情展示，以及输入和通信错误提示。
- 保留流式调用、JSON Schema、Pydantic、SDK parse 等早期独立学习示例。
- 七个评估案例自动检查工具名称、语义参数、统计值与分组键；人工审核脚本生成独立快照，保留原报告。

尚未实现跨请求聊天历史、数据持久化、用户认证、生产部署、RAG 或向量数据库；没有引入 LangChain/LangGraph。成本、时延和大规模模型稳定性尚未系统评估。

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
├── evals/                  # 案例、模拟数据、说明；本地报告目录被忽略
├── docs/                   # HANDOFF.md、DEMO.md、PROJECT_SUMMARY.md
├── AGENTS.md
├── pyproject.toml
├── README.md
└── TASKS.md
```

`examples/run_evaluation.py` 运行评估，`examples/review_evaluation.py` 离线记录人工审核。用例和模拟 CSV 纳入版本管理，`evals/reports/` 下的原始及审核报告默认不提交。

## 环境安装

以下命令均在包含 `pyproject.toml` 的项目目录执行。已有环境时直接激活；新机器可先创建：

```powershell
conda create -n LLM python=3.12 pip
conda activate LLM
python -m pip install -e ".[dev,llm,api,frontend]"
```

`pyproject.toml` 是依赖声明来源：`dev` 提供测试和检查工具，`llm` 提供模型 SDK，`api` 提供后端及 multipart 解析，`frontend` 提供页面和 HTTP 客户端。完整验证使用全部依赖组。项目采用 src-layout，editable 安装后无需设置 PYTHONPATH。

**SDK 版本已统一**：`pyproject.toml` 已固定 `openai==3.14.1`，与 conda `LLM` 环境一致。2026-10-27 学习者提供的验证结果：`pip check` 无依赖冲突，755 项测试通过，Ruff 通过，mypy 检查 34 个源文件通过，39 个文件格式检查通过。固定 SDK 版本不等于锁定整个环境；其他依赖和 Python 版本仍可能影响复现，离线检查也不保证真实模型请求一定成功。

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

最终工程检查（2026-10-27）：**755 项测试通过、0 失败**，`pip check` 无冲突，Ruff 通过，mypy 检查 34 个源文件通过，格式检查 39 个文件通过。测试隔离真实模型请求，使用模拟 SDK、HTTP 传输和页面组件；pytest 临时目录为项目内 `.pytest_cache/tmp`。该数字不是模型准确率，也不是完整浏览器自动化通过率。

用户已人工验收上传分析、零方差、空文件、错误提示及页面连接失败。曾出现非空但无意义的 `"(!256)"` 回复，重试后正常，根因未确认；结构校验无法排除此类问题；现有评估已将工具结果检查和人工回答审核分开，不能因为工具正确就自动认定回答合格。

大文件仍会完整加载到 DataFrame；每次请求重新读取数据，无跨请求会话、存储或用户隔离。有限轮工具循环不等于跨请求聊天。客户端 60 秒超时和 SDK 默认重试不构成整体分析时长保证；前端读取超时为 300 秒。

## 评估与人工审核

```powershell
# 会产生真实模型请求；可使用已有报告完成审核，无需重新运行
python -m examples.run_evaluation

# 不调用模型；请替换为真实文件名，审核结论由审核者自行决定
python -m examples.review_evaluation --report evals/reports/你的报告.json --case dataset_size --status passed --note "已核对回答与真实工具结果"
```

七个案例覆盖数据规模、描述统计、正常相关、零方差、地区均值、类别总和及分组 sum/count 缺失语义。分组按 group 键比较，Pearson 两列允许互换，浮点值按案例容差核对。合理的额外调用或顺序变化不直接判失败。

自动检查只验证工具和指定统计字段；回答是否准确、是否混淆缺失/零或相关/因果仍由人工判断。完整协议、参数及示例见 [evals/README.md](evals/README.md)。每次审核生成新文件，下一次以上一步新文件为输入；同一案例再次审核不会重复计数。

最近七案例真实运行（2026-10-26，qwen3.8-flash）：自动检查 **7 通过、0 失败**。最终人工快照（2026-10-27，`review_20261027T104107648520Z_f182a8e9842b47f88d87562cee447ceb.json`）记录 **通过 7、失败 0、待审核 0**。此前两项由用户命令行记录，其余五项根据用户最终验收确认，通过既有审核脚本写入新快照，原报告不覆盖。最终演示也已获用户确认。两个指标不可混用，也不能将这次小样本全通过表述为系统总体准确率 100%。

## Clash / 代理排查

前端访问 `127.0.0.1`、`localhost` 或 `::1` 时显式设置 `trust_env=False`，避免环境代理介入本机通信；远程后端仍沿用环境代理。该修改不改变后端访问模型的代理行为，也不保证绕过系统 TUN 路由。

Clash 开启、FastAPI 停止时，已人工确认修正后显示“无法连接后端”。未知来源的 502/504 仅提示后端或代理网关错误；只有符合后端已知错误 JSON 的响应才显示模型服务失败/超时，不能仅凭状态码判定来源。
