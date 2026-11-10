"""Plan Validator — deterministic check, not LLM self-approval.

The Planner produces a plan.  The PlanValidator checks it deterministically.
An invalid plan goes to the fault policy, never silently "fixed".
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PlanValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    can_switch_to_fixed_workflow: bool = False
    suggested_fixed_workflow: str = ""


class PlanValidator:
    """Deterministic plan checks — no LLM involved."""

    MAX_STEPS = 12
    VALID_STEP_TYPES = {
        "retrieve", "analyse", "generate", "verify",
        "compliance_check", "ask_user",
    }
    AGENT_STEP_MAP = {
        "quant": "analyse",
        "expert": "analyse",
        "compliance": "compliance_check",
        "strategy": "generate",
        "verifier": "verify",
    }

    def validate(
        self, plan: dict, task_type: str = "general", is_high_risk: bool = False
    ) -> PlanValidationResult:
        errors: list[str] = []
        warnings: list[str] = []

        steps = plan.get("steps", [])
        if not steps:
            errors.append("计划没有包含任何步骤")
            return PlanValidationResult(valid=False, errors=errors, warnings=warnings)

        step_ids = set()
        step_deps: dict[str, list[str]] = {}

        for step in steps:
            sid = step.get("step_id", "")
            if not sid:
                errors.append("存在没有 step_id 的步骤")
                continue
            if sid in step_ids:
                errors.append(f"重复的 step_id: {sid}")
            step_ids.add(sid)

            step_type = step.get("step_type", "")
            if step_type not in self.VALID_STEP_TYPES:
                errors.append(f"步骤 {sid}: 不合法的步骤类型 '{step_type}'")

            deps = step.get("depends_on", [])
            if isinstance(deps, list):
                step_deps[sid] = [d for d in deps if d]  # filter empty strings

            agent = step.get("agent", "")
            if agent and agent not in self.AGENT_STEP_MAP and agent not in {
                "context_assembly", "compliance_gate", "evidence_pipeline", "reflection",
            }:
                errors.append(f"步骤 {sid}: 不支持的 Agent '{agent}'")

        # Check dependencies exist
        for sid, deps in step_deps.items():
            for dep in deps:
                if dep not in step_ids:
                    errors.append(f"步骤 {sid}: 依赖 '{dep}' 不在计划中")

        # Cycle detection via DFS
        cycle_nodes = self._detect_cycle(step_deps, step_ids)
        if cycle_nodes:
            errors.append(f"DAG存在循环依赖: {' → '.join(cycle_nodes)}")

        # Check for orphan steps
        if len(steps) > 1:
            referenced = {dep for deps in step_deps.values() for dep in deps}
            for sid in step_ids:
                if sid not in step_deps or not step_deps[sid]:
                    if sid not in referenced and sid != steps[0].get("step_id"):
                        warnings.append(f"步骤 {sid}: 可能是孤立步骤（无依赖也不被依赖）")

        # High-risk tasks MUST include compliance and verification
        if is_high_risk:
            has_compliance = any(s.get("step_type") == "compliance_check" or s.get("agent") == "compliance" for s in steps)
            has_verifier = any(s.get("step_type") == "verify" or s.get("agent") == "verifier" for s in steps)
            if not has_compliance:
                errors.append("高风险任务必须包含合规检查步骤")
            if not has_verifier:
                errors.append("高风险任务必须包含验证步骤")

        # Strategy must not appear before compliance
        strategy_idx = next((i for i, s in enumerate(steps) if s.get("agent") == "strategy"), None)
        compliance_idx = next((i for i, s in enumerate(steps) if s.get("agent") == "compliance"), None)
        if strategy_idx is not None and compliance_idx is not None and strategy_idx < compliance_idx:
            errors.append("Strategy 步骤不能排在 Compliance 步骤之前")

        # ASK_USER must route to human_wait
        for s in steps:
            if s.get("step_type") == "ask_user":
                errors.append(f"步骤 {s.get('step_id')}: ASK_USER 类型必须进入人工等待节点")

        # Max steps
        if len(steps) > self.MAX_STEPS:
            errors.append(f"步骤数 {len(steps)} 超过最大限制 {self.MAX_STEPS}")

        return PlanValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

    @staticmethod
    def _detect_cycle(deps: dict[str, list[str]], node_ids: set[str]) -> list[str] | None:
        """Return a cycle path if DAG has one, else None."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {n: WHITE for n in node_ids}
        parent: dict[str, str | None] = {n: None for n in node_ids}

        def dfs(u: str) -> list[str] | None:
            color[u] = GRAY
            for v in deps.get(u, []):
                if v not in color:
                    continue
                if color[v] == GRAY:
                    # Found back-edge: reconstruct cycle
                    cycle = [v, u]
                    cur = u
                    while parent.get(cur) and parent[cur] != v:
                        cur = parent[cur]
                        cycle.append(cur)
                    cycle.append(v)
                    return list(reversed(cycle))
                if color[v] == WHITE:
                    parent[v] = u
                    result = dfs(v)
                    if result:
                        return result
            color[u] = BLACK
            return None

        for n in node_ids:
            if color[n] == WHITE:
                result = dfs(n)
                if result:
                    return result
        return None
