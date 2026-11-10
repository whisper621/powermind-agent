# 📝 日前申报决策表 · {province} · {target_date}

> 生成时间: {timestamp} | 风险偏好: {risk_preference}
> ⚠️ **本决策表为AI辅助建议，请交易员审核确认后在交易中心系统手动申报。**

---

## 一、决策总览

| 项目 | 建议值 | 置信度 | 关键依据 |
|------|--------|:---:|------|
| **申报均价** | **{bid_avg} 元/MWh** | {confidence} | 预测均价 {forecast_avg}，{adjustment_reason} |
| **峰段策略** | {peak_strategy} | {peak_confidence} | {peak_reason} |
| **谷段策略** | {valley_strategy} | {valley_confidence} | {valley_reason} |
| **中长期参考** | {long_term_ref} | {lt_confidence} | {lt_reason} |

## 二、24点分时申报建议

| 时段 | 小时 | 日前出清预测 (元/MWh) | 建议申报价 (元/MWh) | 策略 | 理由 |
|------|------|:------:|:------:|:----:|------|
| 深谷 | 00:00-06:00 | {v_price_00_06} | {v_bid_00_06} | {v_strategy_00_06} | {v_reason_00_06} |
| 平段 | 06:00-10:00 | {f_price_06_10} | {f_bid_06_10} | {f_strategy_06_10} | {f_reason_06_10} |
| 峰段 | 10:00-15:00 | {p_price_10_15} | {p_bid_10_15} | {p_strategy_10_15} | {p_reason_10_15} |
| 平段 | 15:00-18:00 | {f_price_15_18} | {f_bid_15_18} | {f_strategy_15_18} | {f_reason_15_18} |
| 晚峰 | 18:00-21:00 | {p_price_18_21} | {p_bid_18_21} | {p_strategy_18_21} | {p_reason_18_21} |
| 平段 | 21:00-24:00 | {f_price_21_24} | {f_bid_21_24} | {f_strategy_21_24} | {f_reason_21_24} |

## 三、多角色分析摘要

| 角色 | 判断 | 置信度 | 关键依据 |
|------|:---:|:---:|------|
| 政策分析师 | {policy_dir} | {policy_conf} | {policy_basis} |
| 供需分析师 | {sd_dir} | {sd_conf} | {sd_basis} |
| 新能源分析师 | {re_dir} | {re_conf} | {re_basis} |
| 成本分析师 | {cost_dir} | {cost_conf} | {cost_basis} |
| 价格分析师 | {price_dir} | {price_conf} | {price_basis} |
| 风险分析师 | {risk_dir} | {risk_conf} | {risk_basis} |

## 四、场景推演

| 场景 | 概率 | 电价区间 (元/MWh) | 风险敞口 | 应对策略 |
|------|:---:|:------:|------|---------|
| 🟢 基准 | 50% | {base_low} ~ {base_high} | 基准风险 | {base_action} |
| 🟡 偏紧 | 25% | {tight_low} ~ {tight_high} | 正偏差 | {tight_action} |
| 🟠 宽松 | 20% | {loose_low} ~ {loose_high} | 负偏差 | {loose_action} |
| 🔴 极端 | 5% | {extreme_low} ~ {extreme_high} | 极端风险 | {extreme_action} |

## 五、风险矩阵

| 风险项 | 概率 | 影响程度 | 影响金额估算 | 应对措施 |
|--------|:---:|:---:|------|------|
| {risk_1_name} | {risk_1_prob} | {risk_1_impact} | {risk_1_cost} | {risk_1_mitigation} |
| {risk_2_name} | {risk_2_prob} | {risk_2_impact} | {risk_2_cost} | {risk_2_mitigation} |
| {risk_3_name} | {risk_3_prob} | {risk_3_impact} | {risk_3_cost} | {risk_3_mitigation} |

## 六、历史相似交易日参考

| 日期 | 相似度 | 申报均价 | 出清均价 | 偏差 | 经验教训 |
|------|:---:|:------:|:------:|:---:|------|
| {case_1_date} | {case_1_sim} | {case_1_bid} | {case_1_clear} | {case_1_dev} | {case_1_lesson} |
| {case_2_date} | {case_2_sim} | {case_2_bid} | {case_2_clear} | {case_2_dev} | {case_2_lesson} |

## 七、操作清单

- [ ] 审核预测均价是否合理
- [ ] 确认分时申报策略与风险偏好一致
- [ ] 检查合规边界 (偏差免考核范围)
- [ ] 在交易中心系统中录入申报价格
- [ ] 记录实际申报价与建议价的差异 (用于复盘)

---

> 📌 **数据来源**: QUANT Agent (电价预测) | EXPERT Agent (经验匹配) | COMPLIANCE Agent (合规风控)
> ⚠️ **免责声明**: 本决策表仅供辅助决策参考，不构成交易指令。实际申报由交易员负责。
