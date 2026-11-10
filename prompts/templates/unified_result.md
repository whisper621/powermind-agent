# ⚡ PowerMind · {task_type} · {province} · {target_date}

> 请求ID：{request_id} | 执行耗时：{duration_ms}ms | 意图：{intent}
> 运行方式：统一LangGraph | 当前状态：{execution_profile} / {outcome}

---

## 执行概览

| 环节 | 状态 | 说明 |
|---|:---:|---|
| 意图识别 | ✅ | {intent}（置信度：{intent_confidence}） |
| Quant Agent | {quant_status} | {quant_summary} |
| Expert Agent | {expert_status} | {expert_summary} |
| Compliance Agent | {compliance_status} | {compliance_summary} |
| Strategy Agent | {strategy_status} | {strategy_summary} |
| 证据绑定 | {evidence_status} | {evidence_count}条证据 |
| 结果验证 | {verify_status} | {verified_claims}/{total_claims}条声明已验证 |

{degraded_warning}

## 故障处置记录

{degradation_events}

---

## 证据

{evidence_cards}

---

## 业务结果

{strategy_output}

---

## 验证结果

| 指标 | 结果 |
|---|---|
| 验证状态 | {verify_passed} |
| 已验证声明 | {verified_claims}/{total_claims} |
| 数据库证据覆盖率 | {overall_confidence} |

{verify_issues}
{verify_suggestions}

---

## 使用限制

{warnings}

> 本结果由同一套LangGraph任务流程生成。发生故障时，系统只会进入已批准的处理分支，不会切换到另一套兼容架构。
