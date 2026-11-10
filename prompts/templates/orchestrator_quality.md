# ⚡ PowerAgent OS 统一LangGraph结果 · {task_type} · {province} · {target_date}

> 请求ID: {request_id} | 执行耗时: {duration_ms}ms | 意图: {intent}
> 管道: 意图分类 → 记忆增强 → MAS多智能体(Q→E→C→S) → 证据核查 → 验证 → 记忆记录

---

## 📊 执行概览

| 阶段 | 状态 | 详情 |
|------|:---:|------|
| 意图分类 | ✅ | {intent} (置信度: {intent_confidence}) |
| 量化预测 (QUANT) | {quant_status} | {quant_summary} |
| 经验匹配 (EXPERT) | {expert_status} | {expert_summary} |
| 合规风控 (COMPLIANCE) | {compliance_status} | {compliance_summary} |
| 策略生成 (STRATEGY) | {strategy_status} | {strategy_summary} |
| 证据核查 | {evidence_status} | {evidence_count} 条证据卡片 |
| 事实验证 | {verify_status} | {verified_claims}/{total_claims} 声明已验证 |

{degraded_warning}

---

## 📋 证据卡片

{evidence_cards}

---

## 📝 策略报告

{strategy_output}

---

## 🔍 验证报告

| 指标 | 数值 |
|------|:---:|
| 验证状态 | {verify_passed} |
| 已验证声明 | {verified_claims}/{total_claims} |
| 总体置信度 | {overall_confidence} |

{verify_issues}
{verify_suggestions}

---

## ⚠️ 注意事项

{warnings}

---

> 📌 **运行说明**: 本报告由统一LangGraph生成。完整运行与受控故障处置共用同一任务状态、审计和结果结构。
> 所有数据标注了来源Agent，可追溯到具体工具调用。
> ⚠️ **免责声明**: 本报告仅供辅助决策参考，不构成交易指令。实际申报由交易员负责。
