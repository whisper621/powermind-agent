#!/usr/bin/env python3
"""Architecture probes — verify the 4 acceptance criteria from the repair prompt."""

from agents.react_graph_agent import build_graph, FakeAgentRunner
import sys

def _make_graph(runner):
    return build_graph(agent_runner=runner)

INIT_STATE = {
    "task_mode": "formal", "tenant_context": {}, "degradation_events": [],
    "retry_counters": {}, "execution_profile": "full", "outcome": "completed",
    "warnings": [], "evidence_records": [], "completed_steps": [],
    "task_status": "init", "result_limitations": [], "plan_index": 0,
}

failed = 0

# ==== PROBE 1: policy_search no Strategy ====
print("=" * 60)
print("PROBE 1: policy_search must NOT call Strategy")
runner1 = FakeAgentRunner(responses={
    "quant": {"status": "success", "summary": "quant ok", "agent": "quant"},
    "compliance": {"status": "success", "risk_status": "PASS", "summary": "ok", "agent": "compliance"},
    "strategy": {"status": "success", "summary": "SHOULD_NOT_RUN", "agent": "strategy"},
})
g1 = _make_graph(runner1)
state1 = {**INIT_STATE, "user_query": "policy query", "province": "Shandong",
          "task_type": "policy_search", "target_date": "2026-06-24",
          "request_id": "p1", "thread_id": "p1"}
out1 = g1.invoke(state1, config={"configurable": {"thread_id": "p1"}, "recursion_limit": 80})
print(f"  calls={runner1.calls}")
print(f"  completed={out1.get('completed_steps')}")
print(f"  outcome={out1.get('outcome')}")
print(f"  final_response={bool(out1.get('final_response'))}")

try:
    assert "strategy" not in runner1.calls, "policy_search should not call strategy"
    assert out1.get("final_response") or out1.get("pending_human_action") or out1.get("__interrupt__"), "must have terminal state"
    print("  >> PROBE 1 PASSED")
except AssertionError as e:
    print(f"  >> PROBE 1 FAILED: {e}")
    failed += 1

# ==== PROBE 2: bidding_advice CAN enter Strategy ====
print("=" * 60)
print("PROBE 2: bidding_advice CAN enter Strategy when compliance allows")
runner2 = FakeAgentRunner(responses={
    "quant": {"status": "success", "summary": "quant ok", "agent": "quant"},
    "expert": {"status": "success", "summary": "expert ok", "agent": "expert"},
    "compliance": {"status": "success", "risk_status": "PASS", "summary": "ok", "agent": "compliance"},
    "strategy": {"status": "success", "summary": "strategy ok", "agent": "strategy"},
    "reflection": {"status": "success", "summary": "reflection ok", "agent": "reflection"},
})
g2 = _make_graph(runner2)
state2 = {**INIT_STATE, "user_query": "give me bidding strategy", "province": "Shandong",
          "task_type": "bidding_advice", "target_date": "2026-06-24",
          "request_id": "p2", "thread_id": "p2"}
out2 = g2.invoke(state2, config={"configurable": {"thread_id": "p2"}, "recursion_limit": 80})
print(f"  calls={runner2.calls}")
print(f"  completed={out2.get('completed_steps')}")
print(f"  verification={out2.get('verification_result')}")
print(f"  outcome={out2.get('outcome')}")
print(f"  final_response={bool(out2.get('final_response'))}")

try:
    assert "strategy" in runner2.calls, "bidding_advice should call strategy when compliance allows"
    assert out2.get("final_response") or out2.get("pending_human_action") or out2.get("__interrupt__"), "must have terminal state"
    print("  >> PROBE 2 PASSED")
except AssertionError as e:
    print(f"  >> PROBE 2 FAILED: {e}")
    failed += 1

# ==== PROBE 3: Compliance PROHIBITED blocks Strategy ====
print("=" * 60)
print("PROBE 3: Compliance PROHIBITED must block Strategy")
runner3 = FakeAgentRunner(responses={
    "quant": {"status": "success", "summary": "quant ok", "agent": "quant"},
    "expert": {"status": "success", "summary": "expert ok", "agent": "expert"},
    "compliance": {"status": "success", "risk_status": "FAIL", "summary": "blocked", "agent": "compliance"},
    "strategy": {"status": "success", "summary": "SHOULD_NOT_RUN", "agent": "strategy"},
    "reflection": {"status": "success", "summary": "reflection ok", "agent": "reflection"},
})
g3 = _make_graph(runner3)
state3 = {**INIT_STATE, "user_query": "give me bidding strategy", "province": "Shandong",
          "task_type": "bidding_advice", "target_date": "2026-06-24",
          "request_id": "p3", "thread_id": "p3"}
out3 = g3.invoke(state3, config={"configurable": {"thread_id": "p3"}, "recursion_limit": 80})
print(f"  calls={runner3.calls}")
print(f"  completed={out3.get('completed_steps')}")
print(f"  outcome={out3.get('outcome')}")
print(f"  final_response={bool(out3.get('final_response'))}")

try:
    assert "strategy" not in runner3.calls, "compliance blocked must prevent strategy"
    assert out3.get("final_response"), "compliance block must have final_response"
    assert out3.get("outcome") in ("blocked", "terminated"), f"outcome must be blocked/terminated, got: {out3.get('outcome')}"
    print("  >> PROBE 3 PASSED")
except AssertionError as e:
    print(f"  >> PROBE 3 FAILED: {e}")
    failed += 1

# ==== PROBE 4: Quant failure -> human_wait, not empty ====
print("=" * 60)
print("PROBE 4: Quant failure must not leave empty state")
runner4 = FakeAgentRunner(responses={
    "quant": {"status": "failed", "error": "quant timeout", "agent": "quant",
              "failure_type": "transient", "retryable": True},
    "expert": {"status": "success", "summary": "expert ok", "agent": "expert"},
    "compliance": {"status": "success", "risk_status": "PASS", "summary": "ok", "agent": "compliance"},
    "strategy": {"status": "success", "summary": "strategy ok", "agent": "strategy"},
    "reflection": {"status": "success", "summary": "reflection ok", "agent": "reflection"},
})
g4 = _make_graph(runner4)
state4 = {**INIT_STATE, "user_query": "give me bidding strategy", "province": "Shandong",
          "task_type": "bidding_advice", "target_date": "2026-06-24",
          "request_id": "p4", "thread_id": "p4"}
out4 = g4.invoke(state4, config={"configurable": {"thread_id": "p4"}, "recursion_limit": 80})
print(f"  calls={runner4.calls}")
print(f"  degradation_events={out4.get('degradation_events')}")
print(f"  outcome={out4.get('outcome')}")
print(f"  pending_human_action={bool(out4.get('pending_human_action'))}")
print(f"  final_response={bool(out4.get('final_response'))}")

try:
    assert "expert" not in runner4.calls, "quant failure should not continue to expert"
    assert out4.get("final_response") or out4.get("pending_human_action") or out4.get("__interrupt__"), "fault-to-human cannot be empty return"
    print("  >> PROBE 4 PASSED")
except AssertionError as e:
    print(f"  >> PROBE 4 FAILED: {e}")
    failed += 1

print("=" * 60)
if failed:
    print(f"RESULT: {failed}/4 probes FAILED")
    sys.exit(1)
else:
    print("RESULT: All 4 probes PASSED")
    sys.exit(0)
