# 项目开发约定

- 所有 Python 安装、运行、测试与检查统一使用 conda 的 `LLM` 环境。
- 自动化命令使用 `conda run --no-capture-output -n LLM python ...`，不依赖终端当前激活状态。
- 依赖以 `pyproject.toml` 为准，安装：`conda run -n LLM python -m pip install -e ".[dev,llm,api,frontend]"`。
- 验证：`conda run -n LLM python -m pytest -q`、`conda run -n LLM python -m ruff check .`、`conda run -n LLM python -m mypy .`、`conda run -n LLM python -m ruff format --check .`。
- 保持 src-layout、显式参数、无全局 DataFrame；统计结果须通过严格 JSON 序列化。
- 不替用户提前实现 LLM API、Prompt、Function Calling、Agent、RAG、FastAPI 或 Streamlit 业务；这些阶段由用户独立学习，除非用户另行授权。
- 不自行 commit、push、创建远程仓库或覆盖用户已有内容。

- 学习任务要求是审查标准，不是代写授权。学习者写 LLM 核心代码，Codex 写测试；修改核心实现需用户明确要求。
