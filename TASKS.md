# 项目阶段

| 阶段 | 负责人/方向 | 状态 |
| --- | --- | --- |
| Phase 0 | Codex：基础工程、CSV 读取与统计工具、测试、示例、交接 | 已完成 |
| Phase 1 | 学生独立实现：LLM API 基础调用 | 未开始 |
| Phase 2 | Prompt Engineering 与结构化输出 | 未开始 |
| Phase 3 | Function Calling 与统计工具集成 | 未开始 |
| Phase 4 | FastAPI 后端服务 | 未开始 |
| Phase 5 | Streamlit 前端 | 未开始 |
| Phase 6 | 系统测试、评估与最终交付 | 未开始 |
| Phase 7 | RAG 学习与下一阶段项目 | 未开始 |

## Phase 0 验收

- [x] src-layout、Python 3.11+、conda LLM、pyproject、Git 忽略配置
- [x] CSV 读取、编码选项、校验与 JSON 概览
- [x] 数据集信息、描述统计、分组聚合、Pearson 相关性
- [x] 人工模拟数据、独立离线测试
- [x] 运行测试、格式检查、示例计算和严格 JSON 验证
- [x] README 与 HANDOFF

## Phase 1 建议

亲自新增 `src/data_copilot/llm_client.py`，完成一次最小 LLM API 请求，理解请求参数、响应结构、超时和 API 错误；通过环境变量读取密钥。先使用固定文本验证调用，不急于调用统计工具。根据选用的 API 再添加相应 SDK 依赖，不提前安装框架。后续用可替代的请求层进行离线测试，真实调用作为可选集成检查。

## 仓库优化

- [x] 统一 conda LLM 环境和 pip editable 安装
- [x] 开发依赖 extras 与 AGENTS 开发约定
- [x] 可独立运行的模拟数据分析示例
- [x] 工具参数类型校验及回归测试

- [x] 精简可选配置和空环境模板，环境创建命令统一记录在 README
