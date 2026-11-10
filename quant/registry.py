#!/usr/bin/env python3
"""
PowerMind Quant — 预测模型注册中心 (ForecasterRegistry)。

核心功能:
  1. 注册多个预测模型插件
  2. 运行时热切换激活模型（无需重启、无需改任何 Tool 代码）
  3. 全局单例模式，所有 Tool 共享同一个注册中心

热插拔完整示例:

  # ── 启动时注册 ──
  registry = ForecasterRegistry.get_instance()
  registry.register("lstm_default", LSTMPlugin(model_path="models/lstm_epf.pth"))
  registry.register("lstm_multivar", LSTMPlugin(model_path="models/lstm_multivar.pth"))
  registry.set_active("lstm_default")  # 初始默认

  # ── 运行时切换 ──
  # 当新模型训练完成:
  registry.register("xgboost_v2", XGBoostPlugin(model_path="models/xgb_v2.json"))
  registry.set_active("xgboost_v2")  # 一行代码完成切换

  # ── Tool 层完全无感知 ──
  forecaster = registry.get_active()   # 总是拿到当前激活的模型
  result = forecaster.predict(input_data)  # 接口永远不变
"""

from __future__ import annotations

from typing import Optional
from quant.interface import BaseForecaster, ForecastInput, ForecastOutput


class ForecasterRegistry:
    """
    预测模型注册中心。

    模式: 全局单例
    用法:
      registry = ForecasterRegistry.get_instance()
      registry.register("name", model_instance)
      registry.set_active("name")
      forecaster = registry.get_active()
      result = forecaster.predict(input_data)
    """

    _instance: Optional["ForecasterRegistry"] = None

    def __init__(self) -> None:
        self._models: dict[str, BaseForecaster] = {}
        self._active: Optional[str] = None

    @classmethod
    def get_instance(cls) -> "ForecasterRegistry":
        """获取全局单例。"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """重置单例（仅用于测试）。"""
        cls._instance = None

    # ── 注册与切换 ──────────────────────────────────────────

    def register(self, name: str, forecaster: BaseForecaster) -> None:
        """
        注册一个预测模型插件。

        Args:
            name: 模型唯一名称（如 "lstm_v1", "xgboost_shandong"）
            forecaster: 实现了 BaseForecaster 的模型实例

        Raises:
            TypeError: forecaster 未实现 BaseForecaster 接口

        示例:
            registry.register("lstm_multivar", LSTMPlugin(model_path="models/lstm.pth"))
        """
        if not isinstance(forecaster, BaseForecaster):
            raise TypeError(
                f"模型 '{name}' 未实现 BaseForecaster 接口。"
                f" 实际类型: {type(forecaster).__name__}"
            )
        self._models[name] = forecaster
        if self._active is None:
            self._active = name
            print(f"[ForecasterRegistry] 注册 '{name}' 并自动设为默认激活模型。")
        else:
            print(f"[ForecasterRegistry] 注册 '{name}'。当前激活仍为 '{self._active}'。")

    def unregister(self, name: str) -> None:
        """
        注销一个模型。
        如果注销的是当前激活模型，自动切到第一个可用模型。
        """
        if name not in self._models:
            raise ValueError(f"模型 '{name}' 未注册。可用: {list(self._models.keys())}")
        del self._models[name]
        if self._active == name:
            self._active = next(iter(self._models.keys()), None)
            print(f"[ForecasterRegistry] '{name}' 已注销，激活切换为 '{self._active}'。")

    def set_active(self, name: str) -> BaseForecaster:
        """
        热切换到指定模型 —— 这是热插拔的核心入口。

        调用后，所有后续的 registry.get_active().predict() 都会使用新模型。

        Args:
            name: 要激活的模型名

        Returns:
            新激活的模型实例

        Raises:
            ValueError: 模型名未注册

        示例:
            # 训练好新模型后，一行代码切换到新模型
            registry.set_active("xgboost_v3")
            # 所有 Tool 的后续调用自动使用 xgboost_v3
        """
        if name not in self._models:
            raise ValueError(
                f"模型 '{name}' 未注册。"
                f" 可用模型: {list(self._models.keys())}"
                f" 当前激活: {self._active}"
            )
        old = self._active
        self._active = name
        print(f"[ForecasterRegistry] 🔄 热切换: '{old}' → '{name}'")
        return self._models[name]

    # ── 查询与预测 ──────────────────────────────────────────

    def get_active(self) -> BaseForecaster:
        """
        获取当前激活的预测模型。

        Tool 层调用此方法，始终拿到正确的模型实例。
        无论底层怎么切换，这个方法返回的接口永远不变。

        Returns:
            当前激活的 BaseForecaster 实例

        Raises:
            RuntimeError: 没有注册任何模型
        """
        if self._active is None:
            raise RuntimeError(
                "没有激活的预测模型。请先注册并激活至少一个模型:\n"
                "  registry.register('name', model_instance)\n"
                "  registry.set_active('name')"
            )
        if self._active not in self._models:
            raise RuntimeError(
                f"激活模型 '{self._active}' 已丢失。可用: {list(self._models.keys())}"
            )
        return self._models[self._active]

    def get(self, name: str) -> BaseForecaster:
        """获取指定名称的模型（不改变激活状态）。"""
        if name not in self._models:
            raise ValueError(f"模型 '{name}' 未注册。可用: {list(self._models.keys())}")
        return self._models[name]

    def predict(self, input_data: ForecastInput) -> ForecastOutput:
        """
        快捷方法: 直接用当前激活模型执行预测。

        Tool 层只需:
          result = ForecasterRegistry.get_instance().predict(input_data)
        """
        forecaster = self.get_active()
        return forecaster.predict(input_data)

    # ── 管理 ────────────────────────────────────────────────

    @property
    def active_name(self) -> Optional[str]:
        """当前激活的模型名。"""
        return self._active

    @property
    def model_count(self) -> int:
        """已注册的模型数量。"""
        return len(self._models)

    def list_models(self) -> list[dict]:
        """列出所有已注册模型及其详细信息。"""
        result = []
        for name, model in self._models.items():
            info = model.get_model_info()
            result.append({
                "name": name,
                "active": name == self._active,
                **info,
            })
        return result

    def print_status(self) -> None:
        """打印注册中心当前状态。"""
        print("=" * 60)
        print("  ForecasterRegistry 状态")
        print("=" * 60)
        print(f"  已注册模型: {self.model_count}")
        print(f"  当前激活: {self._active}")
        print(f"  全部模型:")
        for m in self.list_models():
            marker = "★" if m["active"] else " "
            print(f"    [{marker}] {m['name']} — {m.get('type', '?')} v{m.get('version', '?')}")
        print("=" * 60)


# ═══════════════════════════════════════════════════════════════════
# 模块级便捷函数
# ═══════════════════════════════════════════════════════════════════

def get_active_forecaster() -> BaseForecaster:
    """便捷函数: 获取当前激活的预测器。"""
    return ForecasterRegistry.get_instance().get_active()


def predict(input_data: ForecastInput) -> ForecastOutput:
    """便捷函数: 用当前激活模型执行预测。"""
    return ForecasterRegistry.get_instance().predict(input_data)
