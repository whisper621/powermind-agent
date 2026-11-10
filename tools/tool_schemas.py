#!/usr/bin/env python3
"""
Pydantic 结构化输出模型 —— 为所有 Tool 定义严格的 Schema 约束。

每个 Tool 的返回值都有对应的 Pydantic BaseModel，
确保 LLM 能可靠解析，终结字符串格式的"文本幻觉"。
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional


# ═══════════════════════════════════════════════════════════════════
# 预测工具
# ═══════════════════════════════════════════════════════════════════

class PricePoint(BaseModel):
    hour: int = Field(ge=0, le=23, description="小时 (0-23)")
    price: float = Field(ge=0, le=1500, description="预测电价 (元/MWh)")


class PriceForecastResult(BaseModel):
    """predict_electricity_price 的结构化输出"""
    province: str = Field(description="省份")
    target_date: str = Field(description="预测日期 YYYY-MM-DD")
    model_name: str = Field(description="预测模型名称")
    model_version: str = Field(description="模型版本")
    avg_price: float = Field(description="日均价 (元/MWh)")
    peak_avg: float = Field(description="峰时均价 18-21时 (元/MWh)")
    valley_avg: float = Field(description="谷时均价 0-6时 (元/MWh)")
    confidence: float = Field(ge=0, le=1, description="模型置信度")
    hourly_predictions: list[PricePoint] = Field(description="24点逐时预测")
    p10: Optional[float] = Field(default=None, description="P10置信下界")
    p90: Optional[float] = Field(default=None, description="P90置信上界")


# ═══════════════════════════════════════════════════════════════════
# 因子分析工具
# ═══════════════════════════════════════════════════════════════════

class FactorSignal(BaseModel):
    name: str = Field(description="因子名称")
    direction: str = Field(description="方向: up/down/stable/unknown")
    strength: int = Field(ge=1, le=5, description="强度 1-5")
    confidence: float = Field(ge=0, le=1, description="置信度")


class FactorSnapshotResult(BaseModel):
    """compute_market_factors 的结构化输出"""
    province: str
    date: str
    overall_direction: str = Field(description="bullish/bearish/neutral")
    overall_risk: str = Field(description="low/medium/high/critical")
    total_signals: int = Field(ge=0)
    policy: list[FactorSignal] = Field(default_factory=list)
    supply_demand: list[FactorSignal] = Field(default_factory=list)
    weather: list[FactorSignal] = Field(default_factory=list)
    renewable: list[FactorSignal] = Field(default_factory=list)
    fuel_cost: list[FactorSignal] = Field(default_factory=list)
    price: list[FactorSignal] = Field(default_factory=list)
    customer_risk: list[FactorSignal] = Field(default_factory=list)
    sentiment: list[FactorSignal] = Field(default_factory=list)
    external_maintenance: list[FactorSignal] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
# 策略与风控
# ═══════════════════════════════════════════════════════════════════

class StrategyItem(BaseModel):
    category: str = Field(description="策略类别")
    priority: str = Field(description="high/medium/low")
    advice: list[str] = Field(description="建议列表")


class StrategyResult(BaseModel):
    """run_strategy_advisor 的结构化输出"""
    province: str
    date: str
    market_direction: str
    risk_level: str
    confidence: str
    strategies: list[StrategyItem] = Field(default_factory=list)
    disclaimer: str = ""


class RiskValidationResult(BaseModel):
    """run_risk_validation 的结构化输出"""
    province: str
    date: str
    status: str = Field(description="PASS/WARN/FAIL")
    confidence_score: float = Field(ge=0, le=1)
    confidence_level: str = Field(description="low/medium/high")
    total_issues: int
    high_risk_issues: int
    recommendations: list[str] = Field(default_factory=list)


class ScenarioResult(BaseModel):
    """run_scenario_simulation 的结构化输出"""
    province: str
    date: str
    baseline_direction: str
    baseline_risk: str
    scenarios: list = Field(default_factory=list)


class MultiRoleResult(BaseModel):
    """run_multi_role_analysis 的结构化输出"""
    province: str
    date: str
    overall_direction: str
    overall_risk: str
    signal_count: int
    analyses: list = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
# 客户报告
# ═══════════════════════════════════════════════════════════════════

class ClientReportScenario(BaseModel):
    name: str = Field(description="场景名称")
    adjustable_ratio: str = Field(description="可调负荷比例")
    storage_config: Optional[str] = Field(default=None, description="储能配置")
    predicted_cost: float = Field(description="预测总电费")
    savings: float = Field(description="节省金额")
    savings_pct: float = Field(description="降本率 %")


class ClientReportResult(BaseModel):
    """generate_client_strategy_report 的结构化输出"""
    client_id: str
    province: str
    period_start: str
    period_end: str
    total_load_mwh: float
    avg_price: float
    total_cost: float
    risk_day: Optional[str] = None
    peak_load_pct: float
    valley_load_pct: float
    scenarios: list[ClientReportScenario] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
# PowerAgent OS — 新增领域工具 Schemas
# ═══════════════════════════════════════════════════════════════════

# --- 光伏预测 ---

class PVForecastPoint(BaseModel):
    hour: int = Field(ge=0, le=23)
    solar_radiation: float = Field(ge=0, description="太阳辐射 W/m²")
    pv_power_kw: float = Field(ge=0, description="光伏出力 kW")


class PVForecastResult(BaseModel):
    province: str
    date: str
    capacity_kw: float = Field(ge=0, description="装机容量 kW")
    total_generation_kwh: float = Field(ge=0, description="日总发电量 kWh")
    peak_power_kw: float = Field(ge=0, description="峰值功率 kW")
    peak_hour: int = Field(ge=0, le=23, description="峰值时段")
    hourly_forecast: list[PVForecastPoint] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1, default=0.75)


# --- 储能优化 ---

class StorageAction(BaseModel):
    hour: int = Field(ge=0, le=23)
    action: str = Field(description="charge/discharge/idle")
    power_mw: float = Field(ge=0, description="充/放电功率 MW")
    soc_after: float = Field(ge=0, le=100, description="动作后 SOC %")
    price_at_action: float = Field(description="该时段电价 元/MWh")
    revenue: float = Field(description="该时段收益/成本 元")


class StorageScheduleResult(BaseModel):
    province: str
    date: str
    battery_capacity_mwh: float = Field(ge=0)
    battery_power_mw: float = Field(ge=0)
    schedule: list[StorageAction] = Field(default_factory=list)
    total_revenue: float = Field(description="日总套利收益 元")
    cycle_count: float = Field(ge=0, description="等效循环次数")
    avg_soc: float = Field(ge=0, le=100)


# --- 光储联合优化 ---

class PVStorageJointResult(BaseModel):
    province: str
    date: str
    pv_forecast: PVForecastResult
    storage_schedule: StorageScheduleResult
    self_consumption_rate: float = Field(ge=0, le=100, description="光伏自用率 %")
    grid_export_kwh: float = Field(ge=0, description="余电上网量 kWh")
    total_benefit: float = Field(description="总收益 元 (自用节省+套利+上网)")
    hourly_schedule: list[dict] = Field(default_factory=list)


# --- 需量预测与预警 ---

class DemandForecastResult(BaseModel):
    province: str
    client_id: str
    date: str
    contract_capacity_kw: float = Field(ge=0, description="合同需量 kW")
    forecast_peak_kw: float = Field(ge=0, description="预测峰值 kW")
    alert_level: str = Field(description="normal/warning/critical")
    margin_kw: float = Field(description="剩余裕度 kW")
    risk_hours: list[int] = Field(default_factory=list, description="风险时段")
    pre_discharge_recommendation: str = ""


# --- 空调节能 ---

class HVACSetpoint(BaseModel):
    hour: int = Field(ge=0, le=23)
    temperature_setpoint: float = Field(ge=16, le=30, description="设定温度 °C")
    power_estimate_kw: float = Field(ge=0, description="预计功率 kW")
    electricity_cost: float = Field(ge=0, description="该时段电费 元")


class HVACOptimizeResult(BaseModel):
    province: str
    date: str
    building_type: str = ""
    baseline_cost: float = Field(ge=0, description="基准方案日电费 元")
    optimized_cost: float = Field(ge=0, description="优化方案日电费 元")
    savings: float = Field(description="日节省 元")
    savings_pct: float = Field(description="节能率 %")
    comfort_violation_hours: int = Field(ge=0, description="舒适度违规小时数")
    hourly_schedule: list[HVACSetpoint] = Field(default_factory=list)


# --- 功率因数监测 ---

class PowerFactorResult(BaseModel):
    client_id: str
    province: str
    timestamp: str
    current_pf: float = Field(ge=0, le=1, description="当前功率因数")
    target_pf: float = Field(ge=0.9, le=1, default=0.95)
    status: str = Field(description="ok/warning/penalty_risk")
    required_capacitance_kvar: float = Field(ge=0, description="需补偿容性无功 kVar")
    estimated_penalty: float = Field(ge=0, description="预估力调电费罚款 元")
    recommendation: str = ""


# --- 省间套利 ---

class CrossProvinceArbitrageResult(BaseModel):
    source_province: str
    target_province: str
    date: str
    source_avg_price: float
    target_avg_price: float
    price_spread: float = Field(description="价差 元/MWh")
    transmission_cost: float = Field(ge=0, description="输电费 元/MWh")
    loss_rate: float = Field(ge=0, le=1, default=0.03, description="线损率")
    net_spread: float = Field(description="净价差 元/MWh")
    arbitrage_viable: bool = False
    estimated_daily_profit: float = 0.0
    corridor_info: str = ""


# --- 结算复盘 ---

class SettlementReconciliationResult(BaseModel):
    province: str
    client_id: str
    period_start: str
    period_end: str
    total_energy_mwh: float = Field(ge=0)
    predicted_avg_price: float
    actual_avg_price: float
    price_mape: float = Field(description="价格预测 MAPE %")
    deviation_fee: float = Field(ge=0, description="偏差考核费用 元")
    total_settlement: float = Field(description="总结算费用 元")
    strategy_effectiveness: str = Field(description="策略有效性评级")
    lessons: list[str] = Field(default_factory=list)


# --- 省份入市评估 ---

class ProvinceAssessmentResult(BaseModel):
    province: str
    market_stage: str
    market_maturity_score: float = Field(ge=0, le=10, description="市场成熟度 1-10")
    entry_difficulty: str = Field(description="low/medium/high")
    spot_price_volatility: str = Field(description="low/medium/high")
    key_regulations: list[str] = Field(default_factory=list)
    recommended_strategy: str = ""
    risk_items: list[str] = Field(default_factory=list)
    overall_rating: str = Field(description="A/B/C/D 综合评级")


# --- 客户组合分析 ---

class ClientPortfolioItem(BaseModel):
    client_id: str
    monthly_consumption_mwh: float
    avg_price: float
    cost_saving_potential: float = Field(ge=0, description="降本潜力 元/月")
    contract_expiry: str = ""
    recommended_action: str = ""


class ClientPortfolioResult(BaseModel):
    total_clients: int
    total_monthly_consumption_mwh: float
    total_saving_potential: float
    expiring_soon: int = Field(ge=0, description="即将到期客户数")
    clients: list[ClientPortfolioItem] = Field(default_factory=list)


# --- 动态定价 (充电桩) ---

class DynamicPricePoint(BaseModel):
    hour: int = Field(ge=0, le=23)
    tou_segment: str
    purchase_cost: float = Field(description="购电成本 元/kWh")
    service_fee: float = Field(description="服务费 元/kWh")
    user_price: float = Field(description="用户充电价 元/kWh")
    expected_load_kwh: float = Field(ge=0, description="预期充电量 kWh")
    expected_revenue: float = Field(description="预期收入 元")


class DynamicPricingResult(BaseModel):
    province: str
    date: str
    station_id: str = ""
    total_revenue: float = Field(description="日总收入 元")
    total_profit: float = Field(description="日总利润 元")
    profit_margin_pct: float = Field(description="利润率 %")
    load_shift_pct: float = Field(description="峰→谷负荷转移率 %")
    hourly_pricing: list[DynamicPricePoint] = Field(default_factory=list)


# --- 发电侧申报 ---

class GeneratorOfferResult(BaseModel):
    province: str
    date: str
    unit_capacity_mw: float = Field(ge=0, description="机组容量 MW")
    marginal_cost: float = Field(ge=0, description="边际成本 元/MWh")
    forecast_mcp: float = Field(description="预测出清价 元/MWh")
    recommended_offer_price: float = Field(ge=0, description="建议申报价 元/MWh")
    recommended_offer_volume_mw: float = Field(ge=0, description="建议申报量 MW")
    expected_revenue: float = Field(ge=0, description="预期收入 元")
    risk_of_curtailment: str = Field(description="弃电风险 low/medium/high")
    strategy_note: str = ""


# --- 需求响应/VPP ---

class DRVPPResult(BaseModel):
    province: str
    event_date: str
    total_aggregated_load_mw: float = Field(ge=0, description="聚合可调负荷 MW")
    dr_baseline_mw: float = Field(ge=0, description="基线负荷 MW")
    committed_reduction_mw: float = Field(ge=0, description="承诺削减 MW")
    expected_compensation: float = Field(ge=0, description="预期补偿 元")
    participation_feasible: bool = True
    asset_decomposition: list[dict] = Field(default_factory=list, description="各资产调度分解")
    risk_note: str = ""
