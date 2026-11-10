#!/usr/bin/env python3
"""
PowerMind Quant — 可插拔预测引擎。

用法:
  from quant.registry import ForecasterRegistry
  from quant.plugins.lstm_plugin import LSTMPlugin

  # 注册
  registry = ForecasterRegistry.get_instance()
  registry.register("lstm_default", LSTMPlugin())
  registry.set_active("lstm_default")

  # 预测
  from quant.interface import ForecastInput
  result = registry.predict(ForecastInput(province="山东", target_date="2026-05-21"))

  # 一键切换
  # registry.register("new_model", NewPlugin())
  # registry.set_active("new_model")  # 热插拔完成
"""

from quant.interface import BaseForecaster, ForecastInput, ForecastOutput
from quant.registry import ForecasterRegistry, get_active_forecaster, predict
from quant.feature_builder import FeatureBuilder, FEATURE_COLS, FEATURE_COUNT

__all__ = [
    "BaseForecaster",
    "ForecastInput",
    "ForecastOutput",
    "ForecasterRegistry",
    "FeatureBuilder",
    "FEATURE_COLS",
    "FEATURE_COUNT",
    "get_active_forecaster",
    "predict",
]
