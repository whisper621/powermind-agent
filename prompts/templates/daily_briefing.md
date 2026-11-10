# 📊 电力市场日报 · {province} · {date}

> 生成时间: {timestamp}
> 数据来源: PowerMind MAS (QUANT + EXPERT + COMPLIANCE)

---

## 一、昨日回顾

| 指标 | 昨日实际 | 前日 | 环比 |
|------|:------:|:---:|:---:|
| 日前出清均价 (元/MWh) | {yesterday_avg} | {prev_day_avg} | {change} |
| 峰谷价差 (元/MWh) | {spread} | {prev_spread} | {spread_change} |
| 统调负荷峰值 (MW) | {peak_load} | {prev_peak_load} | {load_change} |
| 新能源出力占比 | {re_ratio}% | {prev_re_ratio}% | {re_change} |

## 二、今日预测

### 电价预测
- 预测引擎: {model_name}
- 日均价预测: {avg_price} 元/MWh (P50)
- 价格区间: {p10} ~ {p90} 元/MWh
- 峰时均价 (18-21时): {peak_avg} 元/MWh
- 谷时均价 (00-06时): {valley_avg} 元/MWh
- 置信度: {confidence}

### 24点分时走势

| 时段 | 电价预测 |
|------|:------:|
| 00:00-04:00 | {h0_4} |
| 04:00-08:00 | {h4_8} |
| 08:00-12:00 | {h8_12} |
| 12:00-16:00 | {h12_16} |
| 16:00-20:00 | {h16_20} |
| 20:00-24:00 | {h20_24} |

### 关键因子
- 综合方向: {direction}
- 综合风险: {risk_level}
- 最大影响因素: {top_factors}

## 三、风险提示

{risk_items}

## 四、今日关注

- ✅ {action_item_1}
- ⚠️ {action_item_2}
- 📋 {action_item_3}

---

*本报告由 PowerMind AI Agent 自动生成 · 预测结果仅供参考*
