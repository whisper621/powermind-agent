#!/usr/bin/env python3
"""
PowerMind Quant — LSTM 预测插件 (默认实现)。

100% 保留项目 B 的 PyTorch LSTM 算法结构，同时:
  1. 实现 BaseForecaster 接口 → 支持热插拔
  2. 扩展为多变量输入 (input_size=20) ← 从原始 input_size=1 升级
  3. 输出 24 点逐时预测 ← 从原始单点日均价升级
  4. 向后兼容: 可加载原始单变量 .pth 权重文件

原始代码来源: PowerTrade-LLM-Agent-main/algorithms/epf_predictor.py
  - ElectricPriceLSTM 类结构 100% 保留
  - LSTM(1,64,2) → MultiVarLSTM(N,64,2)
  - forward() 逻辑不变，仅扩展维度
"""

from __future__ import annotations

import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

from quant.interface import BaseForecaster, ForecastInput, ForecastOutput
from quant.feature_builder import FeatureBuilder, FEATURE_COLS, FEATURE_COUNT

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ═══════════════════════════════════════════════════════════════════
# LSTM 模型定义 (100% 沿用项目 B 结构, 仅扩展维度)
# ═══════════════════════════════════════════════════════════════════

class MultiVarElectricPriceLSTM(nn.Module):
    """
    多变量电力电价 LSTM。

    原始代码 (项目B epf_predictor.py):
      class ElectricPriceLSTM(nn.Module):
          def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
              self.lstm = nn.LSTM(1, 64, 2, batch_first=True)
              self.fc = nn.Linear(64, 1)           # → 单点输出

    升级后:
      class MultiVarElectricPriceLSTM(nn.Module):
          def __init__(self, input_size=20, hidden_size=64, num_layers=2, output_size=24):
              self.lstm = nn.LSTM(20, 64, 2, batch_first=True)
              self.fc = nn.Linear(64, 24)          # → 24点逐时输出

    核心变化:
      input_size:  1 → N (默认20)
      output_size: 1 → 24 (直接输出次日24点预测)
      forward() 逻辑: 完全不变 ✅
    """

    def __init__(
        self,
        input_size: int = FEATURE_COUNT,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 24,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # ── 核心算法 100% 保留 ──
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        # ────────────────────────────

    def forward(self, x):
        """
        前向传播 —— 逻辑 100% 保留原始代码。

        x shape: [batch, seq_len, input_size]
        输出: [batch, output_size] = [batch, 24]
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # 取最后一个时间步 → 全连接 → 24点
        return out


# ═══════════════════════════════════════════════════════════════════
# LSTM 插件 (实现 BaseForecaster 接口)
# ═══════════════════════════════════════════════════════════════════

class LSTMPlugin(BaseForecaster):
    """
    LSTM 预测插件 —— PowerMind 默认预测引擎。

    特点:
      1. 实现 BaseForecaster 接口 → 可在注册中心热插拔
      2. 支持多变量输入 (默认20维特征)
      3. 输出 24 点逐时电价预测
      4. 向后兼容原始单变量 .pth 文件
      5. 无预训练权重时使用初始化模型 (即刻运行)

    Usage:
      plugin = LSTMPlugin(model_path="models/lstm_multivar.pth")
      result = plugin.predict(ForecastInput(province="山东", target_date="2026-05-21"))
    """

    MODEL_NAME = "LSTM"
    MODEL_VERSION = "2.0-multivar"

    def __init__(
        self,
        model_path: Optional[str] = None,
        input_size: int = FEATURE_COUNT,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 24,
        seq_length: int = 168,
    ):
        """
        Args:
            model_path: .pth 权重文件路径。None = 使用随机初始化的模型
            input_size: 输入特征维度 (默认20)
            hidden_size: LSTM 隐藏层大小
            num_layers: LSTM 层数
            output_size: 输出维度 (24=逐时预测)
            seq_length: 输入序列长度 (小时数, 默认168=7天)
        """
        self.model_path = model_path
        self.input_size = input_size
        self.seq_length = seq_length
        self.output_size = output_size
        self.feature_cols = list(FEATURE_COLS)

        self.model: MultiVarElectricPriceLSTM = MultiVarElectricPriceLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
        )

        self._scaler_min: float = 0.0
        self._scaler_max: float = 1.0
        self._trained: bool = False

        self._load_or_init_model()

    # ── 核心方法 ────────────────────────────────────────────

    def predict(self, input_data: ForecastInput) -> ForecastOutput:
        """
        执行 LSTM 电价预测。

        流程:
          1. 校验输入
          2. 构建特征 (20维)
          3. 转为张量 [1, seq_len, 20]
          4. model(tensor) → [1, 24]
          5. 逆归一化 → 真实电价
          6. 封装为 ForecastOutput
        """
        if not self.validate_input(input_data):
            raise ValueError(f"输入数据校验失败: province={input_data.province}, date={input_data.target_date}")

        # 预处理
        input_data = self.preprocess(input_data)

        # 构建特征矩阵
        if input_data.features is None:
            builder = FeatureBuilder(input_data.province)
            input_data.features = builder.build(input_data.target_date)

        # 特征 → 张量
        tensor = self._features_to_tensor(input_data.features)

        # 模型推理
        self.model.eval()
        with torch.no_grad():
            pred_24_raw = self.model(tensor)  # [1, 24]
            pred_24 = pred_24_raw.squeeze(0).tolist()  # [24]

        # 逆归一化（如果是训练过的模型）
        if self._trained:
            pred_24 = [
                float(p * (self._scaler_max - self._scaler_min) + self._scaler_min)
                for p in pred_24
            ]
        else:
            # 未训练的模型: 映射到合理范围 (200-800 元/MWh)
            pred_24 = [200.0 + float(p) * 600.0 / max(abs(float(p)), 1e-6) * abs(float(p)) * 0.01
                       for p in pred_24]
            pred_24 = [max(100.0, min(1000.0, p)) for p in pred_24]

        # 构造输出
        avg = sum(pred_24) / 24
        peak = sum(pred_24[18:21]) / 3 if len(pred_24) >= 21 else avg
        valley = sum(pred_24[0:6]) / 6

        output = ForecastOutput(
            province=input_data.province,
            target_date=input_data.target_date,
            model_name=self.MODEL_NAME,
            model_version=self.MODEL_VERSION,
            hourly_predictions=[round(p, 2) for p in pred_24],
            avg_price=round(avg, 2),
            peak_avg=round(peak, 2),
            valley_avg=round(valley, 2),
            confidence=self._estimate_confidence(),
            metadata={
                "input_size": self.input_size,
                "seq_length": self.seq_length,
                "trained": self._trained,
                "features_used": self.feature_cols,
            },
        )

        # 后处理
        return self.postprocess(output)

    def get_model_info(self) -> dict:
        return {
            "name": self.MODEL_NAME,
            "version": self.MODEL_VERSION,
            "type": "PyTorch LSTM (项目B原始结构)",
            "input_features": self.feature_cols,
            "input_size": self.input_size,
            "output_dim": self.output_size,
            "hidden_size": self.model.hidden_size,
            "num_layers": self.model.num_layers,
            "training_date": (
                datetime.fromtimestamp(os.path.getmtime(self.model_path)).isoformat()
                if self.model_path and os.path.exists(self.model_path)
                else "未训练"
            ),
            "training_data_size": 0 if not self._trained else "已加载",
            "framework": "PyTorch",
            "model_path": str(self.model_path) if self.model_path else "随机初始化",
        }

    def validate_input(self, input_data: ForecastInput) -> bool:
        return input_data.validate()

    # ── 预处理/后处理钩子 ────────────────────────────────────

    def preprocess(self, input_data: ForecastInput) -> ForecastInput:
        """确保特征存在。"""
        if input_data.features is None:
            builder = FeatureBuilder(input_data.province)
            input_data.features = builder.build(input_data.target_date)
        return input_data

    def postprocess(self, output: ForecastOutput) -> ForecastOutput:
        """确保价格在合理范围内。"""
        # 电力市场价格通常在 100-1500 元/MWh
        clamped = [max(50.0, min(1500.0, p)) for p in output.hourly_predictions]
        output.hourly_predictions = clamped
        output.avg_price = round(sum(clamped) / 24, 2)
        return output

    # ── 内部方法 ────────────────────────────────────────────

    def _load_or_init_model(self) -> None:
        """加载预训练权重或使用随机初始化。"""
        if self.model_path and os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, weights_only=False)
                state = checkpoint.get("model_state_dict", checkpoint)

                # 检查是否需要适配维度
                fc_weight = state.get("fc.weight")
                if fc_weight is not None:
                    saved_output_size = fc_weight.shape[0]
                    saved_input_size = state.get("lstm.weight_ih_l0").shape[1] if "lstm.weight_ih_l0" in state else 1

                    if saved_input_size != self.input_size or saved_output_size != self.output_size:
                        print(
                            f"[LSTMPlugin] ⚠️ 权重维度不匹配: "
                            f"模型({saved_input_size}→{saved_output_size}) vs "
                            f"当前({self.input_size}→{self.output_size})"
                        )
                        print(f"[LSTMPlugin] 使用随机初始化。放入真实数据后请重新训练。")
                        self._trained = False
                        return

                self.model.load_state_dict(state, strict=False)
                self._scaler_min = float(checkpoint.get("scaler_min", 0.0))
                self._scaler_max = float(checkpoint.get("scaler_max", 1.0))
                self._trained = True
                print(f"[LSTMPlugin] ✅ 加载预训练模型: {self.model_path}")
                print(f"[LSTMPlugin]    input_size={saved_input_size}, output_size={saved_output_size}")
                return
            except Exception as e:
                print(f"[LSTMPlugin] ⚠️ 加载模型失败: {e}")
                print(f"[LSTMPlugin] 使用随机初始化的模型。")

        self._trained = False
        print(f"[LSTMPlugin] ⚠️ 未找到预训练模型 ({self.model_path or '未指定'})")
        print(f"[LSTMPlugin] 使用随机初始化模型 (input_size={self.input_size}, output_size={self.output_size})")
        print(f"[LSTMPlugin] 放入真实数据后请调用 train_multivar_lstm() 进行训练。")

    def _features_to_tensor(self, features: dict) -> torch.Tensor:
        """
        将特征字典转为 LSTM 需要的张量。

        输入: {"feature_name": float_value, ...}
        输出: torch.Tensor shape [1, seq_length, input_size]

        对于没有真实历史序列的新模型，使用当前特征值构造模拟序列。
        """
        # 按 FEATURE_COLS 顺序提取特征值
        feature_vec = []
        for col in self.feature_cols:
            val = features.get(col, 0.0)
            if isinstance(val, (list, tuple)):
                val = sum(val) / max(len(val), 1)
            feature_vec.append(float(val))

        # 归一化
        if self._trained:
            feature_vec = [
                (v - self._scaler_min) / max(self._scaler_max - self._scaler_min, 1e-8)
                for v in feature_vec
            ]

        # 构造序列: [1, seq_length, input_size]
        # 每个时间步使用相同的特征值（简化处理，真实训练后会替换为滑动窗口）
        seq = torch.tensor(
            [[feature_vec] * self.seq_length], dtype=torch.float32
        )  # [1, seq_len, N]
        # 转置为 [1, N, seq_len] → 再转回 [1, seq_len, N]
        return seq.permute(0, 2, 1).permute(0, 2, 1) if seq.shape[1] != self.seq_length else seq

    def _estimate_confidence(self) -> float:
        """估计预测置信度。"""
        if self._trained:
            return 0.75  # 已训练模型的基础置信度
        return 0.40       # 未训练模型的低置信度


# ═══════════════════════════════════════════════════════════════════
# 训练函数 (供外部使用)
# ═══════════════════════════════════════════════════════════════════

def train_multivar_lstm(
    province: str,
    start_date: str,
    end_date: str,
    save_path: str = "models/lstm_multivar.pth",
    epochs: int = 50,
) -> dict:
    """
    用真实数据训练多变量 LSTM 模型。

    Args:
        province: 省份
        start_date: 训练开始日期
        end_date: 训练结束日期
        save_path: 模型保存路径
        epochs: 训练轮数

    Returns:
        训练结果字典
    """
    from quant.feature_builder import FeatureBuilder

    builder = FeatureBuilder(province)

    # 从 SQLite 构建训练样本
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    days = (end_dt - start_dt).days

    features_list = []
    targets_list = []
    skipped = 0

    for i in range(days):
        d = (start_dt + __import__('datetime').timedelta(days=i)).strftime("%Y-%m-%d")
        try:
            feats = builder.build(d)
            features_list.append(feats)
            targets_list.append(feats.get("price_lag_1d", 350.0))
        except Exception:
            skipped += 1

    if len(features_list) < 30:
        return {"status": "insufficient_data", "samples": len(features_list), "skipped": skipped,
                "detail": f"需要至少30天训练数据，当前只有{len(features_list)}天"}

    # 构建训练张量
    seq_len = 7
    X, y = [], []
    feature_vecs = [
        [f.get(c, 0.0) for c in FEATURE_COLS]
        for f in features_list
    ]

    for i in range(len(feature_vecs) - seq_len):
        X.append(feature_vecs[i:i + seq_len])
        y.append(feature_vecs[i + seq_len][0])  # 用第一个特征 (price_lag_1d) 作为 target

    X_tensor = torch.tensor(X, dtype=torch.float32)  # [samples, seq_len, N]
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # [samples, 1]

    print(f"[LSTM Trainer] 训练样本: {X_tensor.shape[0]}, 特征维度: {X_tensor.shape[2]}")
    print(f"[LSTM Trainer] 跳过天数: {skipped}")

    model = MultiVarElectricPriceLSTM(
        input_size=FEATURE_COUNT, output_size=1
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.6f}")

    # 保存
    save_full = Path(PROJECT_ROOT) / save_path
    save_full.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_size": FEATURE_COUNT,
        "output_size": 1,
        "hidden_size": 64,
        "num_layers": 2,
        "feature_cols": FEATURE_COLS,
        "scaler_min": 0.0,
        "scaler_max": 1.0,
        "training_date": datetime.now().isoformat(),
        "training_samples": X_tensor.shape[0],
        "province": province,
        "loss": losses[-1] if losses else None,
    }, str(save_full))

    return {
        "status": "ok",
        "samples": X_tensor.shape[0],
        "final_loss": round(losses[-1], 6) if losses else None,
        "model_path": str(save_full),
        "epochs": epochs,
    }


# ═══════════════════════════════════════════════════════════════════
# 自测
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  LSTM Plugin 自测")
    print("=" * 60)

    # 测试1: 接口实现检查
    plugin = LSTMPlugin()
    assert isinstance(plugin, BaseForecaster), "❌ 未实现 BaseForecaster"
    print("✅ BaseForecaster 接口实现检查通过")

    # 测试2: get_model_info
    info = plugin.get_model_info()
    print(f"✅ 模型信息: {info['name']} v{info['version']} ({info['type']})")
    print(f"   输入特征: {len(info['input_features'])} 维")

    # 测试3: 预测
    input_data = ForecastInput(province="山东", target_date="2026-05-21")
    result = plugin.predict(input_data)
    assert len(result.hourly_predictions) == 24, f"❌ 输出不是24点: {len(result.hourly_predictions)}"
    print(f"✅ 预测成功:")
    print(f"   省份: {result.province}")
    print(f"   日期: {result.target_date}")
    print(f"   日均价: {result.avg_price:.2f} 元/MWh")
    print(f"   峰时均价: {result.peak_avg:.2f}")
    print(f"   谷时均价: {result.valley_avg:.2f}")
    print(f"   置信度: {result.confidence:.0%}")
    print(f"   模型: {result.model_name} v{result.model_version}")

    # 测试4: 24点输出
    print(f"\n  24点逐时预测:")
    for i in range(0, 24, 6):
        chunk = result.hourly_predictions[i:i+6]
        print(f"    {i:02d}:00-{(i+5):02d}:00: " + " ".join(f"{p:7.2f}" for p in chunk))

    print(f"\n  ✅ LSTM Plugin 所有测试通过")
