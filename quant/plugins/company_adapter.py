"""CompanyForecasterAdapter — placeholder for the company production algorithm.

V1 task: implement fit / predict / load / save / healthcheck.
The adapter wraps your existing forecasting code and exposes it through
the ForecasterRegistry interface so tools/predict_price.py works
without any code changes.
"""

from __future__ import annotations

from typing import Optional
from quant.interface import BaseForecaster, ForecastInput, ForecastOutput


class CompanyForecasterAdapter(BaseForecaster):
    """Placeholder adapter for the company's production forecasting algorithm.

    V1 implementation checklist:
      1. Implement fit(train_data) — train on historical price data.
      2. Implement predict(input) → ForecastOutput — return 24-point hourly forecast.
      3. Implement load(path) — load serialised model weights.
      4. Implement save(path) — serialise model weights.
      5. Implement healthcheck() → bool — return True if model is loaded and ready.
    """

    def __init__(self):
        self._loaded = False

    def fit(self, train_data) -> None:
        raise NotImplementedError(
            "CompanyForecasterAdapter.fit() not yet implemented — V1 task"
        )

    def predict(self, input: ForecastInput) -> ForecastOutput:
        raise NotImplementedError(
            "CompanyForecasterAdapter.predict() not yet implemented — V1 task. "
            "Use DemoLSTMForecaster via the registry for development."
        )

    def load(self, path: str) -> None:
        self._loaded = False
        raise NotImplementedError(
            "CompanyForecasterAdapter.load() not yet implemented — V1 task"
        )

    def save(self, path: str) -> None:
        raise NotImplementedError(
            "CompanyForecasterAdapter.save() not yet implemented — V1 task"
        )

    def healthcheck(self) -> bool:
        return self._loaded
