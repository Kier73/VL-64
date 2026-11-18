"""Expose a validated compressive Ground through the existing observer protocol."""
from __future__ import annotations

import json
import hashlib
import binascii
from pathlib import Path

from vld import ObserverIntention, domain_digest
from .ground_families import GroundFamilies
from .sources import DEFAULT_HOSTS, acquire


def arbitrate_orbit(intention, *, root=None, approved_hosts=None):
    if not isinstance(intention, ObserverIntention) or (
            intention.spec["operation"] != "ground.navigate"):
        raise TypeError("Expected typed ground.navigate observer intention")
    spec=intention.to_dict()
    plan={
        "status":"ADMITTED_SUBJECT_TO_GROUND_VALIDATION",
        "steps":["acquire_ground_description","verify_ground_digest",
                 "discover_typed_path","navigate_compressed_relation",
                 "compose_stage_receipts"],
        "intention_digest":intention.digest,
        "authority":"read-only-descriptor-and-local-pure-navigation",
    }
    plan["plan_digest"]=domain_digest("vl64.compressed-ground-plan.v1",plan)
    base={
        "schema":"vl64.compressed-ground-observer-receipt.v1",
        "intention":spec,
        "intention_digest":intention.digest,
        "plan":plan,
        "verification_scope":"model-derived-state-with-heldout-observations-not-global-proof",
    }
    opts=spec["options"]
    if not isinstance(opts.get("goal"),dict) or set(opts["goal"]) - {
            "kind","steps","max_work","max_hops","include_data"}:
        return {**base,"status":"REFUSED","reason":"INVALID_GROUND_NAVIGATION_GOAL"}
    goal=opts["goal"]
    if (type(goal.get("steps")) is not int or
            not isinstance(goal.get("kind"),str) or
            type(goal.get("include_data",False)) is not bool):
        return {**base,"status":"REFUSED","reason":"INVALID_GROUND_GOAL_PARAMETERS"}
    try:
        capture=acquire(
            spec["target"],
            root=Path(root) if root is not None else Path.cwd(),
            approved_hosts=approved_hosts if approved_hosts is not None else DEFAULT_HOSTS,
            max_bytes=65536,
        )
        if not capture["complete"]:
            return {**base,"status":"UNRESOLVED",
                    "reason":"COMPLETE_GROUND_DESCRIPTION_REQUIRED"}
        descriptor=json.loads(capture["data"].decode("utf-8"))
        navigator=GroundFamilies().open(descriptor)
        result=navigator.realize(
            goal["steps"],goal["kind"],
            max_work=goal.get("max_work",10),
            max_hops=goal.get("max_hops",8),
            include_data=goal.get("include_data",False))
        return {**base,"status":result["status"],"navigation":result,
                "source_ground_digest":descriptor["ground_digest"],
                "source_description_sha256":hashlib.sha256(capture["data"]).hexdigest(),
                "source_scope":capture["source"]}
    except (TypeError,ValueError,KeyError,UnicodeError,OSError,
            binascii.Error) as exc:
        return {**base,"status":"UNRESOLVED",
                "reason":type(exc).__name__,"detail":str(exc)[:160]}


def render_orbit_receipt(receipt,output):
    if output=="json":
        return json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False)+"\n"
    if output=="jsonl":
        return json.dumps(receipt,sort_keys=True,separators=(",",":"),
                          allow_nan=False)+"\n"
    if output=="text":
        nav=receipt.get("navigation") or {}
        outcome=nav.get("result") or {}
        return "\n".join([
            "VL-64 compressed Ground navigation",
            "Status: "+receipt["status"],
            "Intention: "+receipt["intention_digest"],
            "Ground: "+str(receipt.get("source_ground_digest")),
            "Horizon: "+str(nav.get("horizon")),
            "Path: "+str([x["name"] for x in
                          (nav.get("route") or {}).get("stages",[])]),
            "Result: "+str(outcome),
            "Evidence scope: "+receipt["verification_scope"],
        ])+"\n"
    raise ValueError("Unqualified observer output format")
