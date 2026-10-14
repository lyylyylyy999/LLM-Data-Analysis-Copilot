# LLM Data Analysis Copilot

面向 CSV 数据的智能分析助手学习项目。最终目标是：用户上传数据、用自然语言提出问题，LLM 选择 Python/Pandas 工具执行真实计算，再根据结果生成可读的分析报告。

当前仅完成 **Phase 0**：工程配置、CSV 读取与校验、数据概览、描述统计、分组聚合、Pearson 相关性和离线测试。LLM API、Prompt、Structured Outputs、Function Calling、Agent、RAG、FastAPI 与 Streamlit 均未实现。后续任务见 [TASKS.md](TASKS.md)。

## 技术栈和结构

Python 3.11+、conda 的 `LLM` 环境、Pandas；pytest 用于测试，Ruff 用于格式与静态规范检查。运行依赖为 Pandas 和其数值基础 NumPy；NumPy 同时用于数值类型与有限值校验。

```text
llm-data-analysis-copilot/
├── src/data_copilot/
│   ├── __init__.py
│   ├── data_loader.py       # CSV 校验、读取、JSON 数据概览
│   └── analysis_tools.py    # 四类统计工具
├── tests/
│   ├── test_data_loader.py
│   └── test_analysis_tools.py
├── data/samples/sales.csv
├── frontend/.gitkeep
├── docs/HANDOFF.md
├── .gitignore
├── pyproject.toml
├── AGENTS.md               # 后续开发约定
├── examples/analyze_sales.py # 可运行示例
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
python -m ruff format --check .
python examples/analyze_sales.py
```

包以 editable 方式安装；无需手工调整 PYTHONPATH。`pyproject.toml` 是运行和开发依赖的唯一声明来源，conda 管理 Python 环境，环境内的 pip 安装项目包。已有环境按上面的安装命令补齐依赖，不升级其 Python 或导入其他项目的全部依赖。

新机器尚无 `LLM` 环境时，执行 `conda create -n LLM python=3.12 pip`，再激活环境并按上面的命令安装项目。依赖以 pyproject.toml 为准，约束范围内的安装版本可能不同。

若 PowerShell 未初始化 conda，可先执行 `conda init powershell` 并重新打开终端；自动化中直接使用命名环境：

```powershell
conda run --no-capture-output -n LLM python -m pytest -q
conda run --no-capture-output -n LLM python examples/analyze_sales.py
```

pytest 临时文件固定放在项目内的 `.pytest_cache/tmp`（每次测试会重建这个专用目录），避免系统临时目录权限问题。VS Code 请通过“Python: Select Interpreter”选择 conda 的 LLM 环境；当前工作区的已有配置保留。

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
