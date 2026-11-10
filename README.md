# PowerMind Agent

PowerMind 是面向电力交易场景的多智能体协作演示项目。当前可运行版本以山东省为例，用本地 SQLite 演示数据完成任务识别、固定工作流执行、数据检索、量化分析、策略生成、合规检查、反思与证据校验，并通过 Chainlit 展示结果和执行步骤。

> **演示边界**：仓库中的市场数据、政策条款和案例均为模拟数据。结果仅用于产品与技术演示，不是实时行情、真实电价预测或交易指令；系统不会自动申报、成交或锁价。

## 功能概览

- **五类任务**：市场日报、日前申报建议、实时交易窗口分析、政策查询和中长期锁价分析。
- **固定工作流**：用 LangGraph 编排任务路由、计划校验、数据装配、Quant / Expert / Compliance / Strategy、Reflection、证据汇总和 Verifier。
- **可追溯输出**：从 SQLite 演示库读取结构化行情、天气、案例和政策记录，在页面中展示执行步骤及证据定位。
- **两种运行模式**：默认使用本地 SQLite 数据离线演示；配置 `DEEPSEEK_API_KEY` 后，可让 DeepSeek 参与分析表达。LLM 故障时保留演示数据驱动的降级路径。
- **可扩展模块**：包含检索、提示词注册、量化插件、数据导入、租户隔离和审计等代码。部分模块为扩展或实验实现，不代表已接入生产数据源。

## 技术栈

Python、Chainlit、LangGraph、LangChain、SQLite、ChromaDB、Pydantic、PyTorch。依赖清单见 [`requirements.txt`](requirements.txt)。

## 快速开始

建议使用 Python 3.11 或 3.12，在项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_demo.py
```

`run_demo.py` 会初始化或更新本地演示数据库，启动 Chainlit，并在服务就绪后尝试打开 <http://127.0.0.1:8000>。也可以手动运行：

```powershell
python -m scripts.seed_demo_v1
chainlit run app_chainlit.py --port 8000
```

macOS / Linux 可把虚拟环境激活命令换成 `source .venv/bin/activate`。首次安装 PyTorch 和 ChromaDB 可能需要较长时间。

启动后可试问：

```text
生成山东今天的市场日报
山东明天日前申报怎么做？
山东偏差考核标准是什么？
山东中长期应该锁多少电量？
```

### 可选：启用 DeepSeek

复制 `config/.env.example` 为 `config/.env`，填入自己的 `DEEPSEEK_API_KEY`，再启动应用。`config/.env` 已被 Git 忽略；不要把真实密钥提交到仓库。未配置密钥时，五类工作流仍可使用本地演示模式。

## 代码结构

| 路径 | 用途 |
| --- | --- |
| `app_chainlit.py`、`run_demo.py` | Web 入口与一键演示入口 |
| `agents/`、`workflows/` | 智能体实现与固定工作流 |
| `contracts/`、`middleware/` | 状态和结果契约、审计、降级等中间件 |
| `retrieval/`、`rag/`、`generation/` | 检索、RAG 和带证据的内容生成 |
| `quant/`、`tools/` | 量化插件和业务工具 |
| `services/`、`infrastructure/`、`storage/` | 服务接口、数据库与文件路径、数据存储 |
| `intents/`、`prompts/`、`config/` | 任务意图、运行时提示词与配置 |
| `scripts/seed_demo_v1.py` | 生成可重建的 SQLite 演示数据 |
| `tests/` | 自动化测试 |

`prompts/templates/` 下的 Markdown 文件是程序读取的**运行时输出模板**，不是产品文档。仓库不包含 PRD、知识库建设方案、原始知识文档、已有数据库、模型权重或运行日志。演示数据库由种子脚本生成；需要独立模型权重或外部知识源的实验模块，须由使用者自行提供相应资源。

## 测试

```powershell
python -m pytest tests/test_demo_closed_loop.py -q
```

该测试覆盖演示数据库、检索、提示词注册以及五类任务的主要闭环。运行全部测试可执行 `python -m pytest tests -q`。

## 项目状态

本仓库展示的是原型与离线演示代码，重点在多智能体协作、证据链和合规门禁的实现方式。接入真实市场数据、生产预测模型、企业级权限和交易系统前，需要针对目标环境进行数据校验、安全评审和集成测试。
