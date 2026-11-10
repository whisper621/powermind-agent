#!/usr/bin/env python3
"""
LSTM 数据适配层 —— 桥接项目 A 的 SQLite 结构化数据与项目 B 的 PyTorch LSTM 模型。

职责:
  1. 从 SQLite 提取过去 N 天的电价序列
  2. 转换为 LSTM 需要的 [seq_len, input_size=1] 张量
  3. 无 SQLite 数据时回退到项目 B 的 CSV 数据
"""

from __future__ import annotations

import os
import sqlite3
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# ── LSTM 模型定义（100% 沿用项目 B 的结构）──────────────────────


class ElectricPriceLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(ElectricPriceLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# ── 路径常量 ─────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "lstm_epf.pth"
DB_PATH = (
    Path(r"D:\GitHub_work\china-power-market-agent-main\power_market_agent")
    / "knowledge_base"
    / "knowledge_base.db"
)
FALLBACK_CSV = (
    Path(r"D:\GitHub_work\PowerTrade-LLM-Agent-main")
    / "algorithms"
    / "clean_electric_data.csv"
)

SEQ_LENGTH = 7  # LSTM 输入序列长度（过去 7 天）


# ── 数据适配器 ───────────────────────────────────────────────


class LSTMDataAdapter:
    """从多种数据源提取 LSTM 需要的价格序列，并执行推理。"""

    def __init__(self) -> None:
        self._model: Optional[ElectricPriceLSTM] = None
        self._scaler_min: float = 0.0
        self._scaler_max: float = 1.0
        self._load_model()

    def _load_model(self) -> None:
        """加载训练好的 LSTM 模型和归一化参数。"""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"LSTM 模型权重未找到: {MODEL_PATH}\n"
                "请先运行训练脚本生成 lstm_epf.pth"
            )
        checkpoint = torch.load(str(MODEL_PATH), weights_only=False)
        self._scaler_min = checkpoint["scaler_min"]
        self._scaler_max = checkpoint["scaler_max"]
        self._model = ElectricPriceLSTM()
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.eval()

    def get_price_sequence(
        self, province: str, target_date: str, seq_len: int = SEQ_LENGTH
    ) -> np.ndarray:
        """
        获取过去 seq_len 天的电价序列。
        优先从 SQLite 读取，无数据时回退到 CSV。

        Returns: shape [seq_len, 1] 的 numpy 数组
        """
        # 尝试 SQLite (项目 A 数据)
        sequence = self._from_sqlite(province, target_date, seq_len)
        if sequence is not None:
            return sequence

        # 回退到 CSV (项目 B 数据)
        sequence = self._from_csv(seq_len)
        if sequence is not None:
            return sequence

        raise RuntimeError(
            f"无法获取电价序列: SQLite 无 {province} 数据, CSV 回退也失败"
        )

    def _from_sqlite(
        self, province: str, target_date: str, seq_len: int
    ) -> Optional[np.ndarray]:
        """从项目 A 的 SQLite 提取电价历史序列。"""
        if not DB_PATH.exists():
            return None

        try:
            conn = sqlite3.connect(str(DB_PATH))
            conn.row_factory = sqlite3.Row

            # 先获取省份 ID
            pid_row = conn.execute(
                "SELECT id FROM provinces WHERE name=?", (province,)
            ).fetchone()
            if not pid_row:
                conn.close()
                return None

            # 查询 target_date 之前的电价数据（从 processed_items 或 price_data）
            rows = conn.execute(
                """
                SELECT date, price_direction, summary
                FROM processed_items
                WHERE province_id=? AND date < ?
                ORDER BY date DESC
                LIMIT ?
                """,
                (pid_row["id"], target_date, seq_len * 3),
            ).fetchall()

            conn.close()

            if len(rows) < seq_len:
                return None

            # 从 summary 中尝试提取数值价格（简单启发式）
            prices = []
            for r in reversed(rows):
                price = self._extract_price_from_text(r["summary"] or "")
                if price is not None:
                    prices.append(price)

            if len(prices) >= seq_len:
                return np.array(prices[-seq_len:], dtype=np.float32).reshape(-1, 1)

            return None
        except Exception:
            return None

    def _from_csv(self, seq_len: int) -> Optional[np.ndarray]:
        """从项目 B 的 CSV 获取最后 seq_len 天价格。"""
        if not FALLBACK_CSV.exists():
            return None

        try:
            df = pd.read_csv(str(FALLBACK_CSV))
            df["Delivery start date"] = pd.to_datetime(
                df["Delivery start date"], format="mixed"
            )
            df = df.sort_values(by="Delivery start date")
            prices = df["Wtd avg price $/MWh"].tail(seq_len).values
            return prices.reshape(-1, 1).astype(np.float32)
        except Exception:
            return None

    @staticmethod
    def _extract_price_from_text(text: str) -> Optional[float]:
        """从文本中尝试提取价格数值。"""
        import re

        if not text:
            return None
        # 匹配 "XX元/MWh" 或 "均价XX" 等模式
        patterns = [
            r"(\d+\.?\d*)\s*元/MWh",
            r"均价\s*(\d+\.?\d*)",
            r"(\d+\.?\d*)\s*元",
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                return float(m.group(1))
        return None

    def predict(self, province: str, target_date: str) -> dict:
        """
        执行 LSTM 推理，返回预测结果字典。

        Args:
            province: 省份名称（如 "山东"）
            target_date: 目标日期（如 "2026-01-10"）

        Returns:
            {"target_date": str, "predicted_price": float, "data_source": str, "seq_length": int}
        """
        # 获取价格序列
        price_seq = self.get_price_sequence(province, target_date, SEQ_LENGTH)
        source = "sqlite" if DB_PATH.exists() else "csv_fallback"

        # 归一化
        scaled_input = (price_seq - self._scaler_min) / (
            self._scaler_max - self._scaler_min
        )
        tensor_input = torch.tensor(
            scaled_input, dtype=torch.float32
        ).unsqueeze(0)  # [1, 7, 1]

        # 推理
        with torch.no_grad():
            pred_scaled = self._model(tensor_input).item()

        # 逆归一化
        real_price = pred_scaled * (self._scaler_max - self._scaler_min) + self._scaler_min

        return {
            "target_date": target_date,
            "province": province,
            "predicted_price": round(real_price, 2),
            "data_source": source,
            "seq_length": SEQ_LENGTH,
            "last_known_prices": [round(float(p[0]), 2) for p in price_seq[-3:]],
        }


# ── 全局单例 ─────────────────────────────────────────────────

_adapter: Optional[LSTMDataAdapter] = None


def get_adapter() -> LSTMDataAdapter:
    global _adapter
    if _adapter is None:
        _adapter = LSTMDataAdapter()
    return _adapter


# ── CLI 测试 ─────────────────────────────────────────────────

if __name__ == "__main__":
    adapter = LSTMDataAdapter()
    print("LSTM 模型加载成功")
    print(f"  归一化参数: min={adapter._scaler_min:.4f}, max={adapter._scaler_max:.4f}")

    # 测试推理
    result = adapter.predict("山东", "2026-01-10")
    print(f"\n预测结果:")
    for k, v in result.items():
        print(f"  {k}: {v}")
