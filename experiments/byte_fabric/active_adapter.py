"""Read-only external observer interface for planned action experiments.

A persisted Ground or catalogue is untrusted data, never an executable action
plugin. The observer can request a prediction or the *next recommended probe*.
Actually running a probe requires a separate host-installed QualifiedResetProbe.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from vld import ObserverIntention, domain_digest
from .active_ground import (
    ActiveGroundSession,ActionGroundNavigator,
)
from .sources import acquire,DEFAULT_HOSTS


def arbitrate_active(request,*,root=None,approved_hosts=None):
    if not isinstance(request,ObserverIntention) or request.spec["operation"] not in (
            "ground.active.plan","ground.action.navigate"):
        raise TypeError("Expected typed action Ground intention")
    req=request.to_dict()
    op=req["operation"]
    plan={
        "status":"ADMITTED_DESCRIPTION_ONLY",
        "intention_digest":request.digest,
        "steps":["acquire_bounded_description","verify_ground_or_catalogue",
                 "plan_qualified_action_experiment" if op=="ground.active.plan"
                 else "navigate_declared_action_model"],
        "execution_authority":"no_host_actions_performed_by_this_observer_route",
    }
    plan["plan_digest"]=domain_digest("vl64.action-intention-plan.v1",plan)
    base={
        "schema":"vl64.action-ground-observer-receipt.v1",
        "status":"UNRESOLVED",
        "intention":req,
        "intention_digest":request.digest,
        "plan":plan,
        "verification_scope":"description_only_not_an_executed_remote_probe",
    }
    opts=req["options"]
    try:
        capture=acquire(
            req["target"],
            root=Path(root) if root is not None else Path.cwd(),
            approved_hosts=approved_hosts if approved_hosts is not None else DEFAULT_HOSTS,
            max_bytes=65536,
        )
        if not capture["complete"]:
            return {**base,"reason":"COMPLETE_DESCRIPTION_REQUIRED"}
        data=json.loads(capture["data"].decode("utf-8"))
        if op=="ground.active.plan":
            if (not isinstance(data,dict) or set(data)!={
                    "schema","candidates","costs"} or
                    data["schema"]!="vl64.action-candidate-catalogue.v1"):
                return {**base,"reason":"UNSUPPORTED_CANDIDATE_CATALOGUE"}
            session=ActiveGroundSession(data["candidates"],costs=data["costs"])
            selected=session.plan(
                max_depth=opts.get("max_depth",4),
                max_words=opts.get("max_words",512),
                max_cost=opts.get("max_cost",12))
            status=selected["status"]
        else:
            if (set(opts)-{"action","steps"} or
                    not isinstance(opts.get("action"),str) or
                    type(opts.get("steps")) is not int):
                return {**base,"status":"REFUSED",
                        "reason":"INVALID_MODELED_ACTION_REQUEST"}
            selected=ActionGroundNavigator(data).repeat(
                opts["action"],opts["steps"])
            status=selected["status"]
        return {**base,"status":status,"result":selected,
                "description_sha256":hashlib.sha256(capture["data"]).hexdigest(),
                "description_source":capture["source"]}
    except (TypeError,KeyError,ValueError,UnicodeError,OSError) as exc:
        return {**base,"status":"UNRESOLVED",
                "reason":type(exc).__name__,"detail":str(exc)[:160]}


def render_active_receipt(receipt,output):
    if output=="json":
        return json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+"\n"
    if output=="jsonl":
        return json.dumps(receipt,separators=(",",":"),sort_keys=True,
                          allow_nan=False)+"\n"
    if output=="text":
        value=receipt.get("result") or {}
        return "\n".join([
            "VL-64 action Ground",
            "Status: "+receipt["status"],
            "Intention: "+receipt["intention_digest"],
            "Recommended actions: "+str(value.get("word",[])),
            "Modeled terminal state: "+str(value.get("terminal_state")),
            "Evidence: "+receipt["verification_scope"],
        ])+"\n"
    raise ValueError("Unqualified output format")
