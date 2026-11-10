from datetime import date, timedelta
from types import SimpleNamespace

from agents.demo_agents import DeepSeekAgentRunner, run_demo_agent
from agents.react_graph_agent import PowerMindAgentV3
from app_chainlit import _render_business_result, _split_public_analysis
from middleware.dynamic_prompt import PromptRegistry
from retrieval.demo_rag import DemoRAGRetriever
from scripts.seed_demo_v1 import seed_demo_database
from services.repository import DemoSQLiteMarketDataRepository


def test_offline_demo_database_rag_and_five_workflows(tmp_path, monkeypatch):
    db_path = tmp_path / "powermind_demo.db"
    today_date = date.today()
    demo_end = max(today_date + timedelta(days=14), date(today_date.year, 9, 30))
    horizon_days = (demo_end - today_date).days + 1
    counts = seed_demo_database(db_path, base_date=today_date)
    counts_again = seed_demo_database(db_path, base_date=today_date)
    monkeypatch.setenv("POWERMIND_DEMO_DB", str(db_path))
    assert counts_again == counts

    assert counts["price_data"] >= 24 * 120
    assert counts["forecasts"] == 24 * horizon_days
    assert counts["weather_data"] == 24 * horizon_days
    assert counts["bidding_records"] == 45 + horizon_days
    assert counts["knowledge_documents"] >= 8
    assert counts["demo_scenarios"] == 5 * horizon_days
    registry = PromptRegistry()
    assert registry.validate()["valid"] is True
    assert registry.get_asset("strategy_system")["version"] == "2.2.0"
    assert "实时最新数据" in registry.get_asset("strategy_system")["content"]
    assert "SQLite 存储快照" in registry.get_asset("real_time_alert_scenario")["content"]
    repository = DemoSQLiteMarketDataRepository(str(db_path))
    assert len(repository.get_forecast("山东", demo_end.isoformat())) == 24
    assert repository.get_scenario("bidding_advice", "山东", demo_end.isoformat())
    bidding_history = repository.get_bidding_history(
        "山东", limit=7, before_date=demo_end.isoformat()
    )
    assert len(bidding_history) == 7
    assert bidding_history[0]["trade_date"] == (demo_end - timedelta(days=1)).isoformat()

    rag_hits = DemoRAGRetriever(db_path).search(
        "山东偏差考核标准", province="山东", doc_type="policy"
    )
    assert rag_hits[0]["doc_id"] == "POL-SD-DEV-002"

    agent = PowerMindAgentV3()
    today = date.today().isoformat()
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    cases = [
        ("生成山东今天的市场日报", "daily_pulse", today),
        ("山东明天日前申报怎么做？", "bidding_advice", tomorrow),
        ("山东当前实时交易怎么调整？", "real_time_alert", today),
        ("山东偏差考核标准是什么？", "policy_search", today),
        ("山东中长期应该锁多少电量？", "long_term_locking", today),
    ]
    full_flow = {
        "task_init", "supervisor", "workflow_selector", "planner", "plan_validator",
        "context_assembly", "quant", "expert", "compliance", "compliance_gate",
        "strategy", "reflection", "evidence_pipeline", "verifier", "render_result",
    }

    for query, task_type, target_date in cases:
        state = agent.execute_state(
            query, province="山东", task_type=task_type, target_date=target_date
        )
        assert state["task_status"] == "completed"
        assert state["verification_result"]["passed"] is True
        assert state["evidence_records"]
        assert "纯数据库演示" in state["final_response"]
        assert "输出模板" in state["final_response"]
        concise = _render_business_result(state)
        assert "全链路执行轨迹" not in concise
        assert "故障处置记录" not in concise
        if task_type != "policy_search":
            assert full_flow.issubset(set(state["completed_steps"]))
            assert state["reflection_result"]["passed"] is True
            assert state["strategy_result"]["runtime_mode"] == "sqlite_demo_no_api"
            if task_type == "bidding_advice":
                scenario = state["strategy_result"]["scenario"]
                weighted = round(
                    sum(item["volume_mwh"] * item["bid_price"] for item in scenario["segments"])
                    / scenario["total_volume_mwh"],
                    1,
                )
                assert weighted == scenario["recommended_bid_avg"]
            if task_type == "real_time_alert":
                assert "不是实时行情" in state["strategy_result"]["summary"]
                assert state["strategy_result"]["scenario"]["spread"] == 15.0
            if task_type == "long_term_locking":
                strategy = state["strategy_result"]
                assert len(strategy["analysis_data"]["forecast_horizon_10d"]) == 10
                assert strategy["scenario"]["locked_volume_mwh"] / strategy["scenario"]["forecast_demand_mwh"] == 0.375
                evidence_ids = {item["evidence_id"] for item in strategy["evidence"]}
                assert f"contracts:山东:{today}" in evidence_ids
        else:
            assert state["compliance_result"]["retrieval"]["method"] == "sqlite_keyword_rag"


def test_strategy_llm_receives_active_scenario_prompt(tmp_path):
    db_path = tmp_path / "powermind_demo.db"
    seed_demo_database(db_path, base_date=date.today())
    repository = DemoSQLiteMarketDataRepository(str(db_path))

    class CapturingLLM:
        def invoke(self, messages):
            self.messages = messages
            self.calls = getattr(self, "calls", 0) + 1
            return SimpleNamespace(
                content=(
                    "输出不完整"
                    if self.calls == 1
                    else "数据覆盖。当前持仓。未来10日。分批方案。总结。"
                    "提高成交概率。"
                    "<!-- PUBLIC_ANALYSIS_START -->公开分析依据"
                    "<!-- PUBLIC_ANALYSIS_END -->"
                    "必须由交易员人工复核并在正式系统中手工操作。"
                ),
                usage_metadata={},
            )

    runner = DeepSeekAgentRunner.__new__(DeepSeekAgentRunner)
    runner.repository = repository
    runner.retriever = DemoRAGRetriever(db_path)
    runner.registry = PromptRegistry()
    runner.model = "test-model"
    runner.llm = CapturingLLM()

    result = runner.run(
        {
            "user_query": "山东中长期应该锁多少电量？",
            "province": "山东",
            "target_date": date.today().isoformat(),
            "task_type": "long_term_locking",
        },
        "strategy",
    )

    system_prompt = runner.llm.messages[0].content
    user_prompt = runner.llm.messages[1].content
    assert result["llm_used"] is True
    assert result["llm_retry_mode"] == "compact_retry"
    assert "输出契约缺失" in result["llm_first_error"]
    assert "提高报价意愿" in result["summary"]
    assert "提高成交概率" not in result["summary"]
    assert "公开分析路径" in system_prompt
    assert "未来10日逐日策略" in system_prompt
    assert "{province}" not in system_prompt
    assert "forecast_horizon_10d" in user_prompt
    assert f"contracts:山东:{date.today().isoformat()}" in user_prompt


def test_deepseek_runner_keeps_quant_deterministic(tmp_path):
    db_path = tmp_path / "powermind_demo.db"
    seed_demo_database(db_path, base_date=date.today())
    runner = DeepSeekAgentRunner.__new__(DeepSeekAgentRunner)
    runner.repository = DemoSQLiteMarketDataRepository(str(db_path))
    runner.retriever = DemoRAGRetriever(db_path)
    runner.registry = PromptRegistry()
    runner.model = "test-model"

    class MustNotRun:
        def invoke(self, _messages):
            raise AssertionError("Quant不应调用LLM")

    runner.llm = MustNotRun()
    result = runner.run(
        {
            "user_query": "山东明天日前申报怎么做？",
            "province": "山东",
            "target_date": date.today().isoformat(),
            "task_type": "bidding_advice",
        },
        "quant",
    )
    assert result["status"] == "success"
    assert result["runtime_mode"] == "sqlite_demo_no_api"
    assert "llm_used" not in result


def test_reflection_blocks_inconsistent_bid_math(tmp_path):
    db_path = tmp_path / "powermind_demo.db"
    seed_demo_database(db_path, base_date=date.today())
    repository = DemoSQLiteMarketDataRepository(str(db_path))
    retriever = DemoRAGRetriever(db_path)
    strategy = run_demo_agent(
        {
            "province": "山东",
            "target_date": date.today().isoformat(),
            "task_type": "bidding_advice",
        },
        "strategy",
        repository=repository,
        retriever=retriever,
    )
    valid_summary = strategy["summary"]
    strategy["scenario"]["recommended_bid_avg"] -= 30
    strategy["summary"] += "\n预计购电成本为 10 万元，成交概率较高。"
    result = run_demo_agent(
        {"task_type": "bidding_advice", "strategy_result": strategy},
        "reflection",
        repository=repository,
        retriever=retriever,
    )
    assert result["passed"] is False
    assert "weighted_bid_avg_matches" in result["issues"]
    assert "no_bid_as_expected_cost" in result["issues"]
    assert "no_unsupported_probability" in result["issues"]

    strategy["scenario"]["recommended_bid_avg"] += 30
    strategy["summary"] = valid_summary + "\n未计算预计购电成本；没有模型，不能声称成交概率较高或较低。"
    allowed_disclaimer = run_demo_agent(
        {"task_type": "bidding_advice", "strategy_result": strategy},
        "reflection",
        repository=repository,
        retriever=retriever,
    )
    assert allowed_disclaimer["passed"] is True


def test_public_analysis_is_hidden_from_business_result():
    visible, analysis = _split_public_analysis(
        "### 策略正文\n执行建议\n"
        "<!-- PUBLIC_ANALYSIS_START -->\n"
        "### 公开分析依据\n证据映射\n"
        "<!-- PUBLIC_ANALYSIS_END -->\n"
        "### 总结\n人工确认"
    )
    assert "证据映射" not in visible
    assert "人工确认" in visible
    assert "证据映射" in analysis
