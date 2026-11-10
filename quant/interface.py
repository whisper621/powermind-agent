#!/usr/bin/env python3
"""
PowerMind Quant — 可插拔预测引擎抽象接口。

设计原则:
  1. 所有预测算法必须实现 BaseForecaster 接口
  2. 统一的 ForecastInput / ForecastOutput 数据类保证工具层零修改
  3. 支持运行时热插拔: registry.set_active("新模型名") 即可切换

热插拔示例:
  >>> from quant.registry import ForecasterRegistry
  >>> from quant.plugins.lstm_plugin import LSTMPlugin
  >>> from quant.plugins.future_xgboost_plugin import XGBoostPlugin  # 未来扩展
  >>>
  >>> registry = ForecasterRegistry.get_instance()
  >>> registry.register("lstm_v2", LSTMPlugin(model_path="models/lstm_multivar.pth"))
  >>> registry.register("xgboost_v1", XGBoostPlugin(model_path="models/xgb_shandong.json"))
  >>>
  >>> registry.set_active("lstm_v2")   # 使用 LSTM
  >>> result = registry.get_active().predict(input_data)
  >>>
  >>> registry.set_active("xgboost_v1")  # 一键切换到 XGBoost
  >>> result = registry.get_active().predict(input_data)  # 接口完全不变
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ═══════════════════════════════════════════════════════════════════
# 标准数据结构
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ForecastInput:
    """
    统一预测输入 —— 所有预测插件接收相同的输入结构。

    Attributes:
        province: 省份名称，如 "山东"
        target_date: 目标预测日期，格式 YYYY-MM-DD
        node_code: 可选，节点/分区代码（如 "豫中东"），默认 None = 全省均价
        features: 可选，预构建的特征字典。若为空，由 FeatureBuilder 自动构建。
        seq_length: 输入序列长度（小时数），默认 168 (=7天×24h)
        forecast_horizon: 预测前瞻（小时数），默认 24 (=次日24点)
    """
    province: str
    target_date: str
    node_code: Optional[str] = None
    features: Optional[dict] = None
    seq_length: int = 168
    forecast_horizon: int = 24

    def validate(self) -> bool:
        """基础校验：省份和日期不能为空。"""
        return bool(self.province and self.target_date)


@dataclass
class ForecastOutput:
    """
    统一预测输出 —— 所有预测插件必须返回此结构。

    无论底层是 LSTM、XGBoost、Transformer 还是商业闭源模型，
    Tool 层只依赖这个数据类，保证热插拔时上层代码零修改。

    Attributes:
        province: 省份
        target_date: 预测日期
        model_name: 模型名称（用于审计和日志）
        model_version: 模型版本号
        hourly_predictions: 24 点逐时预测电价 (元/MWh)，长度必须 = forecast_horizon
        avg_price: 日均价 (元/MWh)
        peak_avg: 峰时均价 18:00-21:00 (元/MWh)
        valley_avg: 谷时均价 00:00-06:00 (元/MWh)
        confidence: 模型自评置信度 0.0-1.0
        p10: 可选，P10 置信下界 (24 点)
        p50: 可选，P50 中位预测 (24 点)
        p90: 可选，P90 置信上界 (24 点)
        feature_importance: 可选，特征重要性字典
        metadata: 额外元数据
        created_at: 预测时间戳
    """
    province: str
    target_date: str
    model_name: str
    model_version: str
    hourly_predictions: list[float]          # 长度 = forecast_horizon 的逐时预测
    avg_price: float
    peak_avg: float                          # 晚峰 18:00-21:00 均价
    valley_avg: float                        # 深谷 00:00-06:00 均价
    confidence: float                        # 0.0 - 1.0
    p10: Optional[list[float]] = None        # P10 置信下界
    p50: Optional[list[float]] = None        # P50 中位
    p90: Optional[list[float]] = None        # P90 置信上界
    feature_importance: Optional[dict] = None
    metadata: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_markdown_table(self) -> str:
        """生成 24 点分时电价表（Markdown 格式）。"""
        rows = [
            f"| 时段 | 电价 (元/MWh) | 时段 | 电价 (元/MWh) |",
            f"|------|:-----------:|------|:-----------:|",
        ]
        for i in range(12):
            left_price = self.hourly_predictions[i]
            right_price = self.hourly_predictions[i + 12]
            rows.append(
                f"| {i:02d}:00 | {left_price:7.2f} | {i+12:02d}:00 | {right_price:7.2f} |"
            )
        return "\n".join(rows)


# ═══════════════════════════════════════════════════════════════════
# 抽象基类
# ═══════════════════════════════════════════════════════════════════

class BaseForecaster(ABC):
    """
    预测模型抽象基类。

    所有预测算法（LSTM、XGBoost、Transformer、商业闭源模型）必须实现此接口。

    子类需要实现的三个方法:
      - predict(): 核心预测逻辑
      - get_model_info(): 返回模型元信息（名称/版本/特征列表/训练日期）
      - validate_input(): 输入数据校验

    子类可选实现:
      - preprocess(): 预处理钩子
      - postprocess(): 后处理钩子
    """

    @abstractmethod
    def predict(self, input_data: ForecastInput) -> ForecastOutput:
        """
        执行电价预测。

        Args:
            input_data: 标准预测输入（省份+日期+可选特征）

        Returns:
            ForecastOutput: 标准预测输出（24点逐时电价+统计量+置信度）

        Raises:
            ValueError: 输入数据校验失败
            RuntimeError: 模型推理异常
        """
        ...

    @abstractmethod
    def get_model_info(self) -> dict:
        """
        返回模型元信息。

        Returns:
            {
                "name": str,           # 模型名称 (如 "LSTM")
                "version": str,        # 版本号 (如 "2.0-multivar")
                "type": str,           # 算法类型 (如 "PyTorch LSTM")
                "input_features": list[str],  # 输入特征列表
                "output_dim": int,     # 输出维度 (24)
                "training_date": str,  # 最后训练日期
                "training_data_size": int,    # 训练样本数
                "framework": str,      # 框架 (PyTorch/XGBoost/ONNX)
            }
        """
        ...

    @abstractmethod
    def validate_input(self, input_data: ForecastInput) -> bool:
        """校验输入数据是否满足模型的最小要求。"""
        ...

    # ── 可选钩子方法 ──────────────────────────────────────────

    def preprocess(self, input_data: ForecastInput) -> ForecastInput:
        """预处理钩子。子类可覆写以实现自定义预处理逻辑。"""
        return input_data

    def postprocess(self, output: ForecastOutput) -> ForecastOutput:
        """后处理钩子。子类可覆写以实现自定义后处理逻辑。"""
        return output
