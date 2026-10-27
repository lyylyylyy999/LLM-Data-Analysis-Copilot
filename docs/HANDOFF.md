# 代码与学习交接

更新日期：2026-10-27。任务 01～24 已完成对应学习验收；任务 25 已完成文档、工程检查及用户最终确认。当前是具有完整分析链路和基础评估的本地学习原型，不代表生产部署或全面模型质量验收。安装与启动见 [README](../README.md)，演示见 [DEMO](DEMO.md)。

## 阅读顺序与职责

| 建议顺序 | 文件 | 重点 |
| --- | --- | --- |
| 1 | `src/data_copilot/data_loader.py` | CSV 输入错误、类型推断与概览 |
| 2 | `src/data_copilot/analysis_tools.py` | 可手工核对的真实计算及统计语义 |
| 3 | `src/data_copilot/tool_registry.py` | Schema 与本地白名单、参数校验各自的作用 |
| 4 | `src/data_copilot/analysis_service.py` | 消息历史、工具结果匹配、轮数上限和返回值 |
| 5 | `examples/tool_execution.py` | 读取 CSV、调用服务与打印的分工 |
| 6 | `src/data_copilot/api_schemas.py`、`api.py` | 请求/响应校验、上传临时文件及错误码 |
| 7 | `frontend/app.py` | HTTP 调用、提交边界和错误展示 |
| 8 | `examples/run_evaluation.py`、`evals/cases.json` | 工具/参数/数值检查与人工回答质量的边界 |
| 9 | `examples/review_evaluation.py` | 独立审核快照、排他写入和人工汇总 |
| 10 | `tests/` | 如何通过模拟请求验证各层，而不调用真实模型 |

早期学习代码仍保留：`llm_client.py`（流式 JSON Schema 调用）、`response_parser.py`（手写 JSON 校验）、`schemas.py`（三字段 AnalysisAnswer），以及 `hello_llm`、`pydantic_validation`、`structured_output` 示例。`analyze_with_llm` 演示固定统计结果交给模型解释；`tool_calling` 仅观察工具请求。它们不是当前 HTTP 服务的业务入口。

## 当前数据流

```text
Streamlit 上传文件和问题
  → POST /analyze-file
  → 文件名/大小校验、服务端临时 CSV
  → load_csv → DataFrame
  → analyze_question(df, question)
      → 模型接收问题、字段名称/类型和工具定义
      → Python 校验模型提出的工具名称及 JSON 参数
      → 绑定当前 df，执行真实统计函数
      → assistant(tool_calls) + tool(tool_call_id, JSON结果) 追加历史
      → 继续请求工具或返回最终正文；最多允许指定批次
  → 临时文件清理、AnalyzeResponse 响应校验
  → Streamlit 展示正文与工具详情
```

命令行复用相同服务但自行读取 CSV。`POST /analyze` 使用固定样例。DataFrame 不由模型提供，也不放入模型参数；模型仍会看到统计结果及分组值。核心统计和服务不打印，展示由 CLI/前端负责。

## 统计接口

以下统计函数返回 JSON 兼容字典，不修改输入 DataFrame。四种工具均可从 `data_copilot.analysis_tools` 导入。

| 函数 | 参数 | 返回字段 |
| --- | --- | --- |
| `dataset_info(dataframe)` | DataFrame | `row_count`、`column_count`、`columns`、`dtypes`、`missing_values` |
| `describe_numeric(dataframe, columns)` | 单个列名或列名序列；仅实数数值列 | `ddof=1`、`statistics`，按列返回 count/mean/std/min/25%/median/75%/max/unavailable_statistics |
| `group_aggregate(dataframe, group_column, target_column, aggregation="mean")` | 分组列、数值目标列、mean/sum/count/min/max | 原参数及 `groups: [{group, value, status}]` |
| `correlation(dataframe, x_column, y_column, method="pearson")` | 两个数值列及 pearson | 原参数及 `coefficient`、`sample_size`、`status` |

`load_csv(file_path, *, encoding="utf-8-sig")` 返回 DataFrame；`dataset_overview(dataframe, *, preview_rows=5)` 返回基本信息和按行字典组织的 `preview`。

描述统计忽略缺失值，标准差自由度为 1，分位数采用 Pandas 默认线性插值。分组 count 仅计目标非缺失值，保留缺失分组键；sum 使用至少一个有效值的约定，全缺失为 null。分组按首次出现顺序，仅输出实际观测类别。

相关分析成对删除缺失值。`sample_size` 不等于任一单列的 count；少于 2 对为 `insufficient_samples`，任一列恒定为 `zero_variance`，浮点结果不可表示为 `nonfinite_result`，这些状态下 coefficient 为 null。零相关只表示无线性相关，不能据此否定非线性关系；工具不提供 p 值、置信区间或因果识别。

NumPy 标量转换为 Python 标量，日期时间为 ISO 字符串。概览中的缺失/非有限值为 null；统计工具拒绝无穷输入，输出非有限值会附状态或不可用字段。有限数值不舍入，调用方使用 `json.dumps(..., allow_nan=False)` 验证。

## 工具注册与服务接口

```text
execute_tool(dataframe, tool_name: str, arguments_json: str) -> dict[str, Any]
analyze_question(dataframe, question: str, *, max_tool_rounds: int = 3) -> dict[str, Any]
```

`execute_tool` 只允许 dataset_info、describe_numeric、group_aggregate、correlation，不使用 eval/exec 或任意动态函数查找。模型侧参数约定如下，所有字段必填，禁止额外字段：

| 工具 | 模型参数 |
| --- | --- |
| dataset_info | `{}` |
| describe_numeric | `{"columns":["sales"]}`；非空字符串数组 |
| group_aggregate | `{"group_column":"region","target_column":"sales","aggregation":"mean"}` |
| correlation | `{"x_column":"sales","y_column":"advertising_spend","method":"pearson"}` |

Schema 帮助模型生成请求，不替代 Python 的类型、列名和允许值校验。统计函数支持单列字符串，不代表模型侧 columns 参数也允许字符串。

`analyze_question` 调用时才读取配置并创建客户端；导入不读 CSV、不读 .env、不请求模型。问题和轮数提前校验；轮数必须为正整数，bool 不允许。

返回结构：

```json
{
  "answer": "模型最终解释正文",
  "model": "最终响应实际返回的模型名",
  "finish_reason": "stop",
  "tool_results": [
    {"tool_call_id":"call_1","name":"dataset_info","arguments":{},
     "result":{"row_count":40,"column_count":6}}
  ]
}
```

上述 result 仅节选格式；真实 dataset_info 还包含列名、类型和缺失数。普通工具服务正文不使用早期三字段 AnalysisAnswer。model 来自最终响应，不是假定配置中的模型名；当前服务不汇总 usage。

每轮先校验整批调用的类型、非空 ID 及批内 ID 唯一，再按原顺序执行。相同工具可多次调用，每条结果通过自己的 tool_call_id 匹配，不能按工具名去重。单轮任一校验、计算或序列化失败即停止，不提交不完整批次，也不伪造成功结果。

工具轮数按批次计数：max_tool_rounds=2 最多执行两批工具，并允许第三次模型请求生成正文；若第三次仍请求工具，抛出 RuntimeError，不执行超限批次。默认最多 3 批、4 次服务层 create 调用；同批多个调用是普通 for 顺序执行，parallel_tool_calls=True 不让 Python 自动并发。SDK 的底层重试可能增加网络尝试次数。

每次模型请求携带完整 system/user/assistant/tool 历史。该历史只存在本次函数调用，不保存到下一次用户请求。已有足够信息时可以首轮直接回答，tool_results=[]，不会额外请求。

## HTTP、上传和异常边界

请求模型 AnalyzeRequest 仅接收非空白字符串 question；AnalyzeResponse 和嵌套 ToolExecutionResult 采用严格类型并禁止额外字段。answer/model/tool_call_id/name 不得全空白，finish_reason 仅允许 stop。统计参数和结果仍保留通用字典，尚未逐种定义 Pydantic 类型。

- 请求校验失败：422，路由未执行，不读取 CSV 或请求模型。
- 响应校验失败：500，服务已经返回了无效数据，属于服务端问题。
- 模型生成的工具参数校验：执行前发生，不能用响应模型代替。
- JSON Schema/Pydantic/HTTP 200 均不保证统计解释正确，非空无意义文本仍可能通过。

上传字段为 file、question；最大 5 MiB，应用最多读上限加 1 字节，超过返回 413。缺失文件字段为 422；缺失文件名、非法后缀、空内容、CSV 格式和解码错误为 400。默认 UTF-8-sig，不推断编码。只有表头可以产生零行 DataFrame。

文件使用 TemporaryDirectory 和服务端固定文件名 uploaded.csv。用户文件名仅校验后缀，避免路径遍历或同名覆盖。读取、分析成功或失败都清理临时目录；不持久化上传，不生成数据集 ID。该 5 MiB 检查发生在框架解析上传后，并不替代网关/服务器层的请求体限制。

HTTP 层先识别 APITimeoutError（APIConnectionError 子类）为 504，再将 APIConnectionError/APIStatusError 映射为 502，其他未处理异常为 500。上游 401 是后端调用模型的认证问题，不代表本接口用户未登录。固定服务端错误 detail 不暴露原始异常，logger.error 使用 exc_info 保留堆栈；日志仍需按服务端敏感信息处理。

统计与服务保持真实异常传播，CLI 不依赖 HTTPException。只在上传的 load_csv 边界将 UnicodeError/ValueError 转为 400，后续分析 ValueError 仍为 500。

| 底层异常 | 常见触发条件 |
| --- | --- |
| FileNotFoundError / OSError / PermissionError | 路径缺失、读取或临时文件权限问题 |
| CSVValidationError（ValueError 子类） | 空文件、重复/空列名、字段数不匹配、NUL 或无法解析 |
| UnicodeDecodeError / LookupError | 文件编码不符、无效编码名 |
| TypeError / ValueError | 非法 DataFrame、列、列类型、聚合方法或参数 |
| json.JSONDecodeError | 工具 arguments 或早期回答解析的 JSON 不合法 |
| RuntimeError | 缺配置、超轮数、空/截断/拒绝/不一致的模型响应 |
| Pydantic / 响应校验异常 | 字段、类型、额外内容或文本约束不符 |
| SDK 连接、状态、超时异常 | 模型调用失败，服务层保留原始异常 |

## 前端与代理修正

frontend/app.py 不导入服务函数，不持有模型密钥。submit_analysis 校验输入后用 httpx.post 发送 multipart；validate_analysis_response 验证最小返回协议；main 负责表单和展示。Streamlit rerun 未提交时不会请求；不能把函数放在提交条件之外。

前端连接/读取/写入/连接池超时分别为 10/300/30/10 秒。连接失败和超时有不同提示，HTTP 错误不作为成功分析展示；正文使用默认安全 Markdown，不开启 unsafe_allow_html。

访问 127.0.0.1、localhost、::1 时使用 trust_env=False，仅绕过前端该次请求的环境代理；远程后端仍允许环境代理，后端 SDK 和系统 TUN 设置未更改。曾在 Clash 开启、FastAPI 停止时出现代理网关 502 被误标为模型错误；修正后用户已确认提示正确。未知 502/504 显示后端或代理网关错误；已知后端 JSON detail 才映射为模型服务错误说明，该识别不是来源认证机制。

## 旧结构化学习接口

- ask_llm(question)：流式 create + AnalysisAnswer JSON Schema，拼接最终正文并本地校验；返回 answer 原始 JSON 字符串、实际 model、finish_reason 和 usage，usage 缺失为 None，不打印推理内容。
- parse_analysis_answer(answer)：手写 json.loads 和三字段校验，不修复非法 JSON，不删除围栏。
- AnalysisAnswer：summary/explanation/missing_information 全必填，严格类型、禁止额外字段；summary 非空白，explanation 可空。
- examples.structured_output.ask_structured：SDK parse(response_format=AnalysisAnswer)，返回 model_dump 的三字段字典。当前 main 只发一个问题，不会遍历 QUESTIONS。

这些是并列学习实验，当前业务服务不通过 ask_llm 或 ask_structured 间接执行工具。自定义本地校验不一定等价于远端 Schema 约束。

## 评估与审核交接

七个案例及两份模拟 CSV 已存在，说明见 [evals/README.md](../evals/README.md)。run_evaluation 逐案例复用 analyze_question；单例失败记录异常类型后继续，不将原始上游错误保存到报告。check_tools 根据名称及参数匹配同一次调用，普通字段和 expected_groups 必须同时通过；按分组键查找，缺组/重复组/null 与 0 混淆均失败。整数统计精确比较，浮点检查显式使用容差；分组聚合允许等价的 0.0 表示数值 0。

自动 passed/failed 仅表示工具及预期字段检查；answer 原样保存，human_review 初始 pending。额外工具可以为解释提供依据，不能要求模型从 sum/count 猜出原始个体数值。分组聚合也不能代替组内相关系数计算。

review_evaluation 只使用标准库，不导入分析服务或模型 SDK，不读凭据、不请求模型。主要入口：

```text
review_report(report_path, case_id, status, note, *, output_dir=DEFAULT_REPORT_DIR)
    -> (新文件路径, 人工审核汇总)
```

case_id 必须恰好匹配一次；status 仅 passed/failed，note 不得全空白。保留 criteria、回答、模型名、工具结果与自动指标，记录 UTC reviewed_at。文件以 x 模式排他创建，原报告和已有文件不覆盖。下一次以新快照为输入才能延续此前审核；重复审核同一案例只更新状态，不增加案例数。

自动与人工可以同时出现“自动通过、人工失败”：工具结果正确不能排除错误因果结论或无意义正文。summary 的 passed/failed 保持原值，人工独立汇总 human_review_passed/human_review_failed/human_review_pending。

## 最终工程验证与证据边界

2026-10-27 在现有 conda LLM 环境重新执行：755 项离线测试通过，0 失败；pip check 无依赖冲突；Ruff 通过；mypy 检查 34 个源文件无错误；39 个文件格式检查通过。本次没有安装/升级包，也没有调用真实模型。代码 import、样例及文档检查与此记录分别保留。

测试覆盖统计手算样例、SDK/HTTP 模拟、有限轮控制、上传清理、响应校验、模拟页面组件、评估器及审核快照。页面组件模拟不是完整浏览器自动化。新建环境全量安装和 Python 3.11 解释器未单独实测；实际使用 Python 3.12.13，SDK 已固定 openai==3.14.1，其他依赖并未全部锁定。

历史人工验收：正常上传/零方差/空 CSV 分别为 200/200/400；Clash 开启且 FastAPI 停止的提示已复测。2026-10-26 七案例真实评估使用 qwen3.8-flash，自动 7/7 通过。四个早期案例和三个新增分组案例已分别在聊天中审阅核心结论，不等于同一份七案例报告的全部状态均已写入。

最终审核文件为 `review_20261027T104107648520Z_f182a8e9842b47f88d87562cee447ceb.json`：自动 7 通过、0 失败；人工 7 通过、0 失败、0 待审核。region_mean 与 category_sum 由用户先前命令行记录；剩余五项按用户“都没问题”的最终人工确认，通过既有脚本写入独立新快照，备注说明结论来源，原始报告及先前快照未覆盖。本次未重新运行模型。文件位于忽略的 evals/reports，本地保留、不提交。

## 收尾分工与限制

Codex 已完成文档、演示脚本和离线工程检查；学习者已确认最终人工审核与 [DEMO.md](DEMO.md) 演示无问题，审核结论已落盘。任务 25 按本地学习原型范围完成，不表示生产部署或全面质量评估完成。项目介绍和个人贡献边界见 [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)。

保留以下限制：

1. 曾出现无意义正文 "(!256)"，重试正常但根因未确认；已有测试确认结构合法不等于语义正确。缺失处理措辞及额外分析建议仍需人工核对。
2. 七个固定案例没有覆盖广泛意图、不同数据分布、提示注入或模型波动，也没有系统测量成本、时延、负载或可靠性。
3. CSV 完整加载，Pandas 类型/缺失推断可能改变前导零及 NA 字符串语义，不自动清洗；上传大小限制发生于框架解析之后。
4. 无跨请求聊天、持久化、认证、生产部署、RAG 或向量数据库。日志保留异常堆栈，不能据客户端固定错误文案推断日志已全面脱敏。

不在收尾阶段扩展上述业务能力，不自行 commit、push、创建远程仓库或发布。早期 uv 初始化仅属历史，当前统一 conda LLM。
