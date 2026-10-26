# 模型效果评估（学习任务 21–22）

本目录目前包含 **7 个小规模评估案例**：任务 21 的数据规模、描述统计、Pearson 相关、零方差，以及任务 22 的地区均值、类别总和、分组 `sum`/`count` 缺失值语义。评估脚本仍直接调用 `data_copilot.analysis_service.analyze_question()`；**不复制分析流程、不修改统计工具、不增加模型调用方式**。

## 文件与数据来源

| 路径 | 用途 | 纳入版本管理 |
| --- | --- | --- |
| `evals/cases.json` | 7 个案例的 ID、问题、CSV、工具名与参数、预期统计结果、人工审核标准 | 是 |
| `data/samples/sales.csv` | 数据规模、描述统计、相关、地区均值、类别总和的共同数据源 | 是（确认允许公开后） |
| `evals/fixtures/zero_variance.csv` | `x` 恒为 10 的零方差模拟数据 | 是 |
| `evals/fixtures/group_missing.csv` | 测试全缺失总和与真实零的模拟数据 | 是 |
| `examples/run_evaluation.py` | 逐个运行案例、自动检查工具结果、生成 JSON 报告 | 是 |
| `evals/reports/` | 本地生成的报告（包含模型回答与工具结果） | 否，已加入 `.gitignore` |

数据基准由仓库中的 CSV 与现有 `load_csv()`、`dataset_info()`、`describe_numeric()`、`correlation()`、`group_aggregate()` 核对。`data/mock_sales.csv` 的列名、数值不同，不能替代 `data/samples/sales.csv`。

### `sales.csv` 的数据结构和缺失值

- **40 行、6 列**：`order_id`、`region`、`category`、`sales`、`advertising_spend`、`quantity`。
- 缺失值：`region` 1 个、`sales` 2 个、`advertising_spend` 2 个，其余列 0 个。
- `sales` 与 `advertising_spend` 的缺失位置不同，两列同时非缺失的有效观测共 **36 对**。

## 七个案例的基准结果

| 案例 ID | 工具调用要求 | 已核对基准 |
| --- | --- | --- |
| `dataset_size` | `dataset_info()` | `row_count=40`，`column_count=6` |
| `sales_descriptive` | `describe_numeric(columns=["sales"])` | 非缺失 `count=38`，均值 `694.6973684210526`，样本标准差 `203.19443954783284`，`ddof=1` |
| `pearson_correlation` | `correlation(sales, advertising_spend, method="pearson")` | `coefficient=0.9409742630262548`，`sample_size=36`，`status="ok"` |
| `zero_variance` | `correlation(x, y, method="pearson")` | `coefficient=null`，`sample_size=3`，`status="zero_variance"` |
| `region_mean` | `group_aggregate(group_column="region", target_column="sales", aggregation="mean")` | East、West、North、South、`null` 五组均值（见下表） |
| `category_sum` | `group_aggregate(group_column="category", target_column="sales", aggregation="sum")` | Books、Electronics、Home 三组总和（见下表） |
| `group_missing_sum_count` | 同一 `group_aggregate` 分别请求 `aggregation="sum"` 和 `aggregation="count"` | East、West、North、`null` 四组，每种聚合独立检查（见下表） |

`region_mean` 的五个分组：

| region | `sales` mean |
| --- | ---: |
| East | 680.375 |
| West | 707.5 |
| North | 686.5 |
| South | 699.3333333333334 |
| `null`（实际缺失分组键） | 721.5 |

`category_sum` 的三个分组：

| category | `sales` sum |
| --- | ---: |
| Books | 9085.5 |
| Electronics | 8935.0 |
| Home | 8378.0 |

`evals/fixtures/group_missing.csv` 的内容：

```csv
region,sales
East,10
East,
West,
West,
North,5
North,-5
,20
```

该模拟数据的分组预期为：

| region | `sum` | `count`（`sales` 非缺失数量） |
| --- | ---: | ---: |
| East | 10 | 1 |
| West | `null` | 0 |
| North | 0 | 2 |
| `null`（缺失的地区键） | 20 | 1 |

**关键语义**：West 组目标列全部缺失，`sum(min_count=1)` 返回 **`null`**；North 组有两条有效数值 5、−5，和为真实 **0**。`count` 计算的是 `sales` **非缺失值数量**，不是每组的总行数。`region` 为缺失值时仍保留单独一组，报告 JSON 中的分组键为 `null`，不是字符串 `"null"`。

## 自动检查的规则

每项 `expected_tools` 要求根据**工具名称和参数**匹配一次或多次真实工具调用；相同工具名但 `aggregation="sum"` 与 `aggregation="count"` 是**两个独立要求**，彼此不能代替。`describe_numeric.columns` 可以包含额外列，Pearson 两列顺序可互换。模型可以额外调用合理工具，也可以改变调用顺序，不会仅因此判失败。

每项工具要求可以同时包含：

- `checks`：对工具结果的普通字段执行路径检查，例如 `aggregation`、`statistics.sales.mean`、`sample_size`。
- `expected_groups`（可选）：对 `group_aggregate` 的 `result.groups` **按 `group` 键查找**，不按列表位置查找。例如：

```json
{
  "name": "group_aggregate",
  "arguments": {"group_column": "region", "target_column": "sales", "aggregation": "sum"},
  "checks": {"aggregation": "sum"},
  "expected_groups": [
    {"group": "West", "value": null},
    {"group": "North", "value": 0},
    {"group": null, "value": 20}
  ]
}
```

每个预期键必须在实际分组结果中**恰好出现一次**；未出现或重复出现都判失败。每组的值还必须符合预期：`null` 只匹配真正的 `null`，不等于数值 0；整数预期允许聚合工具返回等价的浮点表达形式（例如 `0.0`），不允许用 `true` / `false` 冒充数值。浮点数可用 `{ "value": 699.3333333333334, "abs_tol": 1e-6 }` 显式设置容差；其他原有字段比较规则不变。

**普通字段检查与所有分组检查必须同时通过**，该次候选工具调用才通过。若有多次同名且参数匹配的调用，只要其中一次满足该工具要求即可。整例所有工具要求都通过时，`automatic_check.passed` 才为 `true`。不会因为工具结果正确而自动判断最终回答正确。

报告的 `automatic_check.requirements[*].group_checks` 会逐组记录：`group`、`expected`、`actual`、`match_count`、`value_found`、`passed`。其中 `match_count` 能区分“实际值是 `null`”与“根本没有匹配组”；重复组会在 `actual` 中列出所有匹配到的值。原有 `checks` 字段仍保留。

## 运行与报告

安装项目及模型依赖、配置好模型服务凭证后，在仓库根目录执行：

```powershell
python -m examples.run_evaluation
```

默认报告写入 `evals/reports/evaluation_*.json`；可通过 `--cases` 选择案例配置，或通过 `--output` 指定文件。**显式运行会调用真实模型 API，可能产生费用；仅导入脚本不会发请求。**

报告保留案例 ID、模型名、原始回答、工具结果、自动检查结果与错误类型，以及自动通过数／失败数。单个案例发生异常时会记录异常**类型**，不写入原始上游错误消息或 API Key，并继续后续案例。报告可能包含输入数据的统计内容或模型生成的敏感信息，默认不应公开上传。

## 人工审核范围

即使 `automatic_check.passed=true`，`human_review.status` 仍为 `"pending"`。审核者需根据各案例 `human_review.criteria` 检查最终 `answer`：

- 是否正确区分总行数和非缺失数量；`count` 是否被误写为组内总行数。
- 样本标准差是否说明 `ddof=1`；相关系数是否正确说明使用 36 对完整观测、相关不证明因果。
- 零方差案例是否把系数 `null` 解释为**无法计算**，而不是 0。
- 地区均值是否保留缺失分组；类别总和是否准确。
- 分组缺失案例是否正确区分 West 的**全缺失总和 `null`**、North 的**真实总和 0**，以及缺失地区键对应的分组；是否正确解释 `count`。
- 是否将描述性组间差异不恰当地解释为因果关系，或编造 p 值、置信区间，或出现 `"(!256)"` 一类无意义内容。

当前没有模型裁判、并发评估或自动重试；人工审核状态需由审查者另行记录，脚本不会自动判定回答通过。

## 局限性

**7 个小规模案例不足以证明系统整体可靠。** 它们仅覆盖少数特定数据、提问方式、统计工具与缺失值边界，不涵盖大量复杂查询、数据集分布变化、对抗性输入、模型采样波动或更多异常情形。应结合多次运行、人工审核与后续扩展评估进行质量判断。
