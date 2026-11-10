#!/usr/bin/env python3
"""Final round probes — verify human_wait state normalisation."""

import sys
sys.path.insert(0, ".")

from agents.react_graph_agent import PowerMindAgentV3, FakeAgentRunner

failed = 0

# ==== PROBE 1: Missing province -> human_wait with full state ====
print("=" * 60)
print("PROBE 1: Missing province -> human_wait, not empty")
agent = PowerMindAgentV3(agent_runner=FakeAgentRunner())
state = agent.execute_state("analysis", province="", task_type="general", target_date="2026-06-24")

print("keys=", sorted(state.keys()))
print("task_status=", state.get("task_status"))
print("outcome=", state.get("outcome"))
print("pending_human_action=", bool(state.get("pending_human_action")))
print("final_response=", bool(state.get("final_response")))

try:
    assert state.get("task_status") == "waiting_human"
    assert state.get("outcome") == "human_review"
    assert state.get("pending_human_action")
    assert state.get("final_response")
    print(">> PROBE 1 PASSED")
except AssertionError as e:
    print(f">> PROBE 1 FAILED: {e}")
    failed += 1

# ==== PROBE 2: High-risk verifier fail -> human_review state ====
print("=" * 60)
print("PROBE 2: High-risk verifier fail -> human_review, not empty")
runner = FakeAgentRunner(responses={
    "quant": {"status": "success", "summary": "quant ok", "agent": "quant"},
    "expert": {"status": "success", "summary": "expert ok", "agent": "expert"},
    "compliance": {"status": "success", "risk_status": "PASS", "summary": "ok", "agent": "compliance"},
    "strategy": {"status": "success", "summary": "strategy ok", "agent": "strategy"},
    "reflection": {"status": "success", "summary": "reflection ok", "agent": "reflection"},
})
agent2 = PowerMindAgentV3(agent_runner=runner)
state2 = agent2.execute_state("give me bidding strategy", province="Shandong", task_type="bidding_advice", target_date="2026-06-24")

print("calls=", runner.calls)
print("task_status=", state2.get("task_status"))
print("outcome=", state2.get("outcome"))
print("pending_human_action=", bool(state2.get("pending_human_action")))
print("verification.passed=", state2.get("verification_result", {}).get("passed"))
print("final_response=", bool(state2.get("final_response")))

try:
    assert "strategy" in runner.calls
    assert state2.get("verification_result", {}).get("passed") is False
    assert state2.get("task_status") == "waiting_human"
    assert state2.get("outcome") == "human_review"
    assert state2.get("pending_human_action")
    assert state2.get("final_response")
    print(">> PROBE 2 PASSED")
except AssertionError as e:
    print(f">> PROBE 2 FAILED: {e}")
    failed += 1

# ==== PROBE 3: execute() does NOT return fallback 废话 ====
print("=" * 60)
print("PROBE 3: execute() does NOT return fallback nonsense")
text = agent2.execute("give me bidding strategy", province="Shandong", task_type="bidding_advice", target_date="2026-06-24")

print("execute() output (first 300 chars, ascii-safe):")
safe_text = text.encode('ascii', errors='replace').decode('ascii')
print(safe_text[:300])
print()

try:
    assert text
    assert "任务没有返回可用结果" not in text
    has_keyword = any(kw in text for kw in ["人工", "复核", "暂停", "等待"])
    assert has_keyword
    print(">> PROBE 3 PASSED")
except AssertionError as e:
    print(f">> PROBE 3 FAILED: {e}")
    failed += 1

print("=" * 60)
if failed:
    print(f"RESULT: {failed}/3 probes FAILED")
    sys.exit(1)
else:
    print("RESULT: All 3 probes PASSED")
    sys.exit(0)
