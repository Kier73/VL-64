"""Select an admitted byte operation by desired outcome, not operator name.

The observer requests a result kind. The host advertises independently
installed, side-effect-free relations realizing that kind. Planning is a
bounded typed search; success is scoped to performed local operations.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

from vld import domain_digest
from .ground import QualifiedByteExecutor


@dataclass(frozen=True)
class GoalRelation:
    """Host-declared realization edge: captured bytes -> one result type."""

    name: str
    produces: str
    operator: str
    parameters: tuple = ()  # (operator parameter, requested goal parameter)
    requires_view: str = "any"
    estimated_work: int = 1


BUILTINS = (
    GoalRelation("digest-from-bytes", "digest.sha256", "sha256", (),
                 "any", 1),
    GoalRelation("extract-bytes", "bytes.interval", "slice",
                 (("start", "start"), ("length", "length")), "any", 1),
    GoalRelation("json-field", "value.json", "json.pointer",
                 (("pointer", "path"),), "json", 2),
    GoalRelation("document-links", "links.html", "html.links", (), "html", 2),
    GoalRelation("byte-statistics", "stats.bytes", "bytes.stats", (), "any", 3),
)


def validate_goal(goal):
    if not isinstance(goal, dict) or set(goal) - {
            "kind", "parameters", "max_work", "expected_result"}:
        raise ValueError("Goal must be a bounded typed object")
    kind = goal.get("kind")
    if (not isinstance(kind, str) or not
            re.fullmatch(r"[a-z][a-z0-9_.-]{0,63}", kind)):
        raise ValueError("Goal must name a valid result type")
    params = goal.get("parameters", {})
    if not isinstance(params, dict) or any(
            not isinstance(k, str) for k in params):
        raise ValueError("Goal parameters must be an object")
    cap = goal.get("max_work", 100)
    if type(cap) is not int or not 1 <= cap <= 10000:
        raise ValueError("Goal work budget is invalid")
    return {"kind": kind, "parameters": params,
            "max_work": cap,
            "has_expected_result": "expected_result" in goal}


class GoalNavigator:
    """Deterministic selection over locally authorized output relationships."""

    def __init__(self, *, executor=None, relations=None):
        self.executor = executor if executor is not None else QualifiedByteExecutor()
        self.relations = {}
        for relation in BUILTINS if relations is None else relations:
            self.register(relation)

    def register(self, relation):
        if (not isinstance(relation, GoalRelation) or
                not re.fullmatch(r"[a-z][a-z0-9_.-]{0,63}", relation.name) or
                not re.fullmatch(r"[a-z][a-z0-9_.-]{0,63}", relation.produces) or
                not re.fullmatch(r"[a-z][a-z0-9_.-]{0,63}", relation.operator) or
                relation.name in self.relations or
                type(relation.estimated_work) is not int or
                not 1 <= relation.estimated_work <= 10000 or
                not isinstance(relation.parameters, tuple) or
                any(not isinstance(item, tuple) or len(item) != 2 or
                    any(not isinstance(x, str) or not x for x in item)
                    for item in relation.parameters) or
                not isinstance(relation.requires_view, str) or
                relation.operator not in self.executor.registry):
            raise ValueError("Goal relation not qualified by installed executor")
        capability, _ = self.executor.registry[relation.operator]
        if capability.policy != "pure-local":
            raise ValueError("Goal navigation admits only pure local capabilities")
        self.relations[relation.name] = relation

    def advertised(self):
        return [{
            "relation": r.name, "output_kind": r.produces,
            "operator": r.operator, "requires_view": r.requires_view,
            "estimated_work_units": r.estimated_work,
            "parameters": list(r.parameters),
        } for r in sorted(self.relations.values(), key=lambda x:x.name)]

    def choose(self, goal, ground, capture):
        spec = validate_goal(goal)
        data = capture["data"]
        kinds = {ground["view"]["kind"]}
        kinds.update(view["decoder"] for view in ground.get("extended_views", []))
        considered, qualified = [], []
        for relation in sorted(self.relations.values(),
                               key=lambda x:(x.estimated_work,x.name)):
            if relation.produces != spec["kind"]:
                continue
            capability, _ = self.executor.registry[relation.operator]
            reason = None
            if relation.estimated_work > spec["max_work"]:
                reason = "COST_CEILING"
            elif relation.requires_view != "any" and relation.requires_view not in kinds:
                reason = "REQUIRED_VIEW_UNOBSERVED"
            elif capability.input_kind != "any" and capability.input_kind not in kinds:
                reason = "CAPABILITY_INPUT_TYPE_MISMATCH"
            elif capability.requires_complete and not capture["complete"]:
                reason = "COMPLETE_SOURCE_REQUIRED"
            elif len(data) > capability.max_input:
                reason = "INPUT_LIMIT"
            operator_params = {}
            if reason is None:
                for operator_key, goal_key in relation.parameters:
                    if goal_key not in spec["parameters"]:
                        reason = "GOAL_PARAMETER_MISSING"
                        break
                    operator_params[operator_key] = spec["parameters"][goal_key]
            record = {
                "relation": relation.name, "operator": relation.operator,
                "result_kind": relation.produces,
                "cost_estimate_units": relation.estimated_work,
                "parameters": operator_params,
                "status": "QUALIFIED" if reason is None else "UNRESOLVED",
                "reason": reason,
            }
            considered.append(record)
            if reason is None:
                qualified.append(record)
        selection = {
            "schema": "vl64.goal-navigation-plan.v1",
            "goal": goal,
            "ground_digest": ground["ground_digest"],
            "capture_complete": capture["complete"],
            "candidate_relations": considered,
            "qualified_relations": [x["relation"] for x in qualified],
            "status": "QUALIFIED_ROUTE_AVAILABLE" if qualified
                      else "UNRESOLVED",
            "scope": "local_static_estimates_not_measured_latency_or_provider_authority",
        }
        selection["plan_digest"] = domain_digest(
            "vl64.goal-navigation-plan.v1", selection)
        return selection, qualified

    def realize(self, goal, ground, capture):
        plan, qualified = self.choose(goal, ground, capture)
        attempts = []
        selected = None
        receipt = None
        for candidate in qualified:
            attempt = self.executor.execute(capture, ground, candidate["operator"],
                                            candidate["parameters"])
            attempts.append({"relation": candidate["relation"],
                             "status": attempt["status"],
                             "reason": attempt.get("reason")})
            if attempt["status"] == "EXECUTED_LOCAL":
                if "expected_result" in goal and attempt["result"] != goal["expected_result"]:
                    attempts[-1]["status"] = "REJECTED_EXPECTED_RESULT"
                    continue
                selected, receipt = candidate, attempt
                break
        result = {
            "schema": "vl64.goal-realization.v1",
            "status": "EXECUTED_LOCAL" if receipt is not None else "UNRESOLVED",
            "goal": goal,
            "navigation": plan,
            "attempted_relations": attempts,
            "selected": selected,
            "execution": receipt,
            "verification_scope": "replayable-local-computation-not-independent-remote-attestation",
        }
        result["realization_digest"] = domain_digest("vl64.goal-realization.v1", result)
        return result
