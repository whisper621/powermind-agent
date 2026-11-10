#!/usr/bin/env python3
"""
PowerMind Agent — 工具集入口。

25 个 Tool，覆盖 5 大业务领域 + 售电交易决策:
  - 预测: 电价预测、光伏出力预测
  - 分析: 因子计算、多角色分析、场景推演、风控校验
  - 策略: 策略顾问、客户报告、发电报价、储能调度
  - 售电决策: 申报决策表、实时交易信号、中长期策略 (NEW)
  - 运营: 空调节能、需量预警、功率因数、充电桩定价
  - 市场: 省间套利、结算复盘、省份评估、客户组合、DR/VPP
  - 知识库: 文档导入器 (NEW)
"""

from tools.predict_price import predict_electricity_price
from tools.search_policy import search_policy
from tools.factor_engine import (
    compute_market_factors,
    run_multi_role_analysis,
    run_scenario_simulation,
    run_risk_validation,
    run_strategy_advisor,
)
from tools.generate_client_report import (
    generate_client_strategy_report,
    get_all_tools,
)
from tools.bidding_sheet import (
    generate_bidding_sheet,
    generate_realtime_signal,
    generate_long_term_strategy,
)
from tools.weather_adapter import (
    fetch_weather_forecast,
    fetch_weather_current,
)
from tools.energy_tools import (
    pv_forecast,
    storage_optimizer,
    pv_storage_joint_optimize,
    demand_forecast_alert,
    hvac_optimizer,
    power_factor_monitor,
)
from tools.market_tools import (
    cross_province_arbitrage,
    settlement_reconciliation,
    generator_offer_strategy,
    dr_vpp_aggregator,
    real_time_market_adapter,
)
from tools.client_tools import (
    province_entry_assessment,
    client_portfolio_analyzer,
    dynamic_pricing_ev,
)
from tools.prediction_archiver import (
    archive_forecast,
    archive_to_sqlite,
    archive_to_chromadb,
    query_forecasts,
    get_latest_forecast,
)

__all__ = [
    # 预测
    "predict_electricity_price",
    "pv_forecast",
    # 分析
    "search_policy",
    "compute_market_factors",
    "run_multi_role_analysis",
    "run_scenario_simulation",
    "run_risk_validation",
    # 策略
    "run_strategy_advisor",
    "generate_client_strategy_report",
    "generator_offer_strategy",
    "storage_optimizer",
    # 售电交易决策 (NEW)
    "generate_bidding_sheet",
    "generate_realtime_signal",
    "generate_long_term_strategy",
    # 运营
    "pv_storage_joint_optimize",
    "demand_forecast_alert",
    "hvac_optimizer",
    "power_factor_monitor",
    "dynamic_pricing_ev",
    # 市场
    "cross_province_arbitrage",
    "settlement_reconciliation",
    "province_entry_assessment",
    "client_portfolio_analyzer",
    "dr_vpp_aggregator",
    "real_time_market_adapter",
    # 预测归档
    "archive_forecast",
    "archive_to_sqlite",
    "archive_to_chromadb",
    "query_forecasts",
    "get_latest_forecast",
    # 工具列表
    "get_all_tools",
]
