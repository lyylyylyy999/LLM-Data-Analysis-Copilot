# Phase 0 代码交接

## 模块职责和数据流

`data_loader.py` 负责本地文件检查、CSV 结构校验、Pandas 读取、数据集信息和预览。`analysis_tools.py` 负责纯统计计算和参数校验；`dataset_info` 从读取模块复用并公开导出。`__init__.py` 只有包说明。测试均不访问网络。

调用流程：`load_csv(path)` → DataFrame → `dataset_overview(df)` 检查字段及缺失 → 选择统计函数并传入同一 df → 结构化字典 → 调用端 `json.dumps(..., allow_nan=False)`。核心函数无 print、全局 DataFrame、网络请求或 LLM 逻辑。

## 公开接口

以下签名省略完整模块前缀；返回值均为 `dict[str, Any]`，load_csv 除外。可从 `analysis_tools` 导入四个统计函数。

### 文件读取与预览

- `load_csv(file_path: str | Path, *, encoding: str = "utf-8-sig") -> pd.DataFrame`：只读本地 CSV。
- `dataset_overview(dataframe: pd.DataFrame, *, preview_rows: int = 5)`：基本信息加 `preview`，预览为行字典列表，行数必须为非负整数。

### A：dataset_info(dataframe: pd.DataFrame)

```json
{"row_count": 2, "column_count": 1, "columns": ["sales"],
 "dtypes": {"sales": "float64"}, "missing_values": {"sales": 1}}
```

dtype 值是 Pandas dtype 字符串，可能随推断结果和版本变化，不建议作为 LLM 的固定枚举。

### B：describe_numeric(dataframe: pd.DataFrame, columns: str | Sequence[str])

单列名或非空、不重复的列名序列；只接受实数数值列。

```json
{"ddof": 1, "statistics": {"sales": {
 "count": 3, "mean": 3.0, "std": 2.0, "min": 1.0,
 "25%": 2.0, "median": 3.0, "75%": 4.0, "max": 5.0,
 "unavailable_statistics": []}}}
```

缺失值不参与计算；std 使用 `ddof=1`，无法计算的量为 null。列表记录不可用的量，可能因为没有有效数据、样本不足或浮点溢出，不应把 null 解释为零。

### C：group_aggregate(dataframe: pd.DataFrame, group_column: str, target_column: str, aggregation: str = "mean")

aggregation 为 mean/sum/count/min/max。目标列须为实数数值列。分组键须是 Pandas 支持的可分组标量。

```json
{"group_column": "region", "target_column": "sales", "aggregation": "mean",
 "groups": [{"group": "East", "value": 100.0, "status": "ok"},
            {"group": null, "value": null, "status": "undefined_or_nonfinite"}]}
```

固定 group/value 字段避免目标列与分组列同名导致冲突。缺失键保留，类别列仅输出观测到的类别；首次出现排序。count 忽略目标缺失值，sum 在全缺失组返回 null。空 DataFrame 返回空 groups。返回 status 区分正常和无法表示的值。

### D：correlation(dataframe: pd.DataFrame, x_column: str, y_column: str, method: str = "pearson")

```json
{"x_column": "sales", "y_column": "advertising_spend", "method": "pearson",
 "coefficient": 1.0, "sample_size": 3, "status": "ok"}
```

成对删除缺失值。`sample_size` 为真实有效样本对数。只支持 pearson，同一数值列可与自身计算。状态为 ok、insufficient_samples（少于两对）、zero_variance（任一列恒定）、nonfinite_result（数值计算产生非有限值）；后三者 coefficient 均为 null。结果不说明因果。

## 异常与数据约定

| 异常 | 触发条件 |
| --- | --- |
| FileNotFoundError | 文件不存在 |
| ValueError | 目录路径、非 CSV 扩展名、不存在的字段、非实数数值列、无穷输入、不支持的操作、空或重复列选择、无效 DataFrame 列名、负预览行数 |
| CSVValidationError（ValueError 子类） | 空文件、空列名、重复列名、列数不匹配、无法解析的 CSV |
| UnicodeDecodeError | 指定编码无法解码文件，保留原始异常 |
| LookupError | 编码名称无效 |
| OSError / PermissionError | 文件系统或权限问题，保留原始异常 |
| TypeError | 非 DataFrame 输入、预览行数不是整数、无法序列化的对象等 |

CSV parser 错误使用 `raise ... from exc` 保留原因。默认编码 utf-8-sig 兼容 BOM，可指定 gbk；不推断编码、不改列名。Pandas 的类型和 NA 推断是读取语义的一部分，包括 NA/NULL 字符串会视为缺失，数字 ID 可能失去前导零。若有业务语义要求，下一步需设计显式 dtype/NA 参数，而不是任意清洗。

JSON 转换由私有 `_json_value` 统一处理：NumPy 标量转原生类型，NaN/Infinity/NA/NaT 为 null，日期时间 ISO 字符串，时长为字符串。概览不额外记录非有限值原因，查看原 DataFrame 可区分其来源；统计输入无穷值直接报错，输出计算溢出由状态或不可用量列表标识。有限数值不舍入；仍受浮点精度和极端数值溢出影响。所有函数只读，不填补缺失或改变输入。

## 后续如何复用

Phase 1 可独立增加 `src/data_copilot/llm_client.py`，只练习 API 基础调用；本阶段没有创建该文件或安装 LLM SDK。先理解正常响应、超时、认证错误和环境变量，再考虑 Prompt 与结构化结果。

Phase 3 可复用所有统计函数：模型提出函数名称和列名参数，Python 执行层做白名单校验，绑定当前数据集的 DataFrame，再调用对应函数，将结构化结果发送给模型。DataFrame 应由程序持有，不放入模型生成的 JSON 参数。工具定义只描述 columns/group_column/target_column/aggregation/x_column/y_column 等可序列化参数；不要让模型执行任意 Python 或任意读取路径。错误转换应保留真实原因，不伪造结果。以上为接入建议，尚无工具注册、模型调用或 Agent 实现。

## 当前限制和阅读顺序

CSV 校验一遍、读取一遍，无大小上限且完整加载内存，读取时文件须稳定；只支持本地逗号 CSV，无远程 URL/Excel/自动编码推断。数值工具不自动转换对象列，时间数据和复数不作为数值分析对象。输入列名必须唯一非空字符串；组键需可哈希和可序列化。没有持久化、上传接口、认证、缓存、LLM 或前端业务。

建议先读 `data_loader.py`，理解错误边界与 DataFrame；再读 `analysis_tools.py`，理解工具签名、缺失值和计算约定；随后读两份测试，用手算核对结果。最后读 pyproject 和 README，理解 conda 环境、editable 包安装、依赖声明和运行方式。Phase 1 自行按供应商定义环境变量并添加有实际内容的环境模板；真实 `.env` 已忽略。

CSV 含 NUL（空字节）时会抛出 CSVValidationError，防止底层读取器截断内容而丢失数据。

## 初次 Phase 0 验收（2026-10-13，历史记录）

环境：Windows、Python 3.13.9、uv 0.13.0、Pandas 3.0.6、NumPy 2.5.4。

- `python -m uv sync --locked`：依赖安装成功，uv.lock 已保留。
- `python -m uv run pytest -q`：46 passed，0 failed。
- `python -m uv run ruff check .` 与 `python -m uv run ruff format --check .`：通过。
- 包从 editable src 路径成功导入；示例 CSV 为 40 行、6 列。
- 缺失数量：region 1、sales 2、advertising_spend 2，其余 0。
- sales 均值 694.6973684210526；sales 与 advertising_spend 的 Pearson 系数 0.9409742630262548，有效样本 36。
- 示例的均值、样本标准差、各地区均值和相关性使用标准库 csv/statistics 独立核对；所有工具结果通过 `json.dumps(..., allow_nan=False)`。
- Git 已初始化，文件尚未提交，未创建远程仓库、commit 或 push。真实环境文件、虚拟环境、缓存和非示例数据均被忽略。

首次 pytest 因系统临时目录权限出现 setup errors，现通过工作区内的专用缓存目录解决；换行测试显式禁用 Windows 自动转换，未削弱断言。最终无未解决的 Phase 0 问题。最低 Python 3.11 的兼容性由语法/依赖约束声明，本次没有单独运行 3.11 解释器。

## 当前开发环境约定

后续统一使用 conda 的 `LLM` 环境，本机 Python 3.12.13。先在项目目录执行 `conda run -n LLM python -m pip install -e ".[dev]"`；测试用 `conda run -n LLM python -m pytest -q`，代码检查用同一环境的 `python -m ruff`。完整命令见 README。

开发依赖改为 pyproject 的 dev extra，pip 可以直接安装，无需 uv。新机器使用 README 中的 conda create 命令建立环境，项目依赖只在 pyproject.toml 中声明。旧 uv.lock 移除，旧验收记录保留用于追溯。

`examples/analyze_sales.py` 是纯 Pandas 示例，默认从脚本所在仓库定位模拟数据，也支持显式 CSV 路径。它只负责调用和 JSON 输出，没有模型或 API 请求。

工具参数类型不正确时抛出 TypeError：包括非字符串列名、不是字符串序列的 columns 和非字符串操作名；空字符串、空列集合、重复字段、未知操作等抛出 ValueError。

## conda 迁移与仓库优化验收（2026-10-13）

- 解释器已确认：`D:\Anaconda_envs\envs\LLM\python.exe`，Python 3.12.13；Pandas 3.0.6、NumPy 2.5.3。
- `conda run -n LLM python -m pip install -e ".[dev]"` 成功；复用环境中现有依赖，没有升级 Python 或已有分析依赖。
- `conda run -n LLM python -m pytest -q`：59 passed，0 failed。
- 同环境 Ruff 静态检查与格式检查通过，`python -m pip check` 返回 No broken requirements found。
- 示例从仓库外工作目录运行成功；40 行、6 列，相关性有效样本 36，JSON 严格序列化通过。
- 新增参数类型回归及示例数据回归测试。
- 保留现有目录和用户的工作区编辑器配置；未创建 commit、远程仓库或 push。新机器的 conda 环境创建命令已提供，但未另建环境执行验证。

## 仓库精简

移除可选的编辑器与 Git 换行配置、空环境模板及单独的 conda 环境描述文件。保留 pyproject.toml 管理依赖、Ruff 和 pytest 配置；README 记录环境安装步骤；AGENTS.md 保存 conda LLM 和开发边界约定；TASKS.md 保留阶段规划。
