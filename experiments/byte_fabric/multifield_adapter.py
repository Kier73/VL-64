"""External observer requests for partial multi-field Grounds.

Only an admitted description is read. A partial observation sends *only known
byte intervals*, not an entire full-frame payload. No host action is executed.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from vld import ObserverIntention,domain_digest

from .multifield_ground import MultiFieldNavigator
from .sources import acquire,DEFAULT_HOSTS
from .ground import _json_no_duplicates


def arbitrate_multifield(request,*,root=None,approved_hosts=None):
    if (not isinstance(request,ObserverIntention) or
            request.spec["operation"] not in (
                "ground.multifield.navigate","ground.multifield.observe")):
        raise TypeError("Expected typed multi-field Ground intention")
    spec=request.to_dict()
    op=spec["operation"]
    plan={
        "status":"ADMITTED_READ_ONLY_GROUND",
        "intention_digest":request.digest,
        "steps":["acquire_exact_ground","verify_ground_content",
                 "navigate_known_action_edges" if op.endswith("navigate")
                 else "interpret_only_supplied_byte_intervals",
                 "return_scoped_evidence_receipt"],
        "authority":"no_external_actions_or_unobserved_byte_claims",
    }
    plan["plan_digest"]=domain_digest("vl64.multifield-observer-plan.v1",plan)
    response={
        "schema":"vl64.multifield-observer-receipt.v1",
        "status":"UNRESOLVED",
        "intention":spec,
        "intention_digest":request.digest,
        "plan":plan,
        "scope":"finite_evidenced_byte_ground_not_remote_state_equivalence",
    }
    goal=spec["options"].get("goal")
    if not isinstance(goal,dict):
        return {**response,"status":"REFUSED","reason":"INVALID_GROUND_GOAL"}
    if op.endswith("navigate"):
        if (set(goal)!={"initial_state_hex","action","steps","kind"} or
                type(goal["steps"]) is not int or
                not isinstance(goal["initial_state_hex"],str) or
                not isinstance(goal["action"],str) or
                not isinstance(goal["kind"],str)):
            return {**response,"status":"REFUSED","reason":"INVALID_NAVIGATION_GOAL"}
    else:
        if (set(goal)!={"known_intervals","segments_b64"} or
                not isinstance(goal["known_intervals"],list) or
                not isinstance(goal["segments_b64"],list) or
                len(goal["known_intervals"])!=len(goal["segments_b64"])):
            return {**response,"status":"REFUSED","reason":"INVALID_PARTIAL_BYTE_GOAL"}
    try:
        observed=acquire(
            spec["target"],root=Path(root) if root is not None else Path.cwd(),
            approved_hosts=approved_hosts if approved_hosts is not None else DEFAULT_HOSTS,
            max_bytes=65536)
        if not observed["complete"]:
            return {**response,"reason":"COMPLETE_GROUND_DESCRIPTION_REQUIRED"}
        parsed=_json_no_duplicates(observed["data"].decode("utf-8"))
        ground=MultiFieldNavigator(parsed)
        if op.endswith("navigate"):
            result=ground.navigate(
                goal["initial_state_hex"],goal["action"],goal["steps"],
                result_type=goal["kind"])
        else:
            if not 1<=len(goal["known_intervals"])<=16:
                return {**response,"status":"REFUSED",
                        "reason":"INVALID_PARTIAL_BYTE_INTERVALS"}
            intervals=ground._intervals(goal["known_intervals"])
            data=bytearray(len(ground.template))
            for (first,stop),segment in zip(intervals,goal["segments_b64"]):
                if not isinstance(segment,str) or len(segment)>131072:
                    raise ValueError("Malformed partial-byte segment")
                exact=base64.b64decode(segment,validate=True)
                if len(exact)!=stop-first:
                    raise ValueError("Provided segment length disagrees with interval")
                data[first:stop]=exact
            result=ground.observe_partial(bytes(data),intervals)
        return {
            **response,
            "status":result["status"],
            "result":result,
            "ground_digest":parsed["ground_digest"],
            "ground_source_sha256":hashlib.sha256(observed["data"]).hexdigest(),
            "source":observed["source"],
        }
    except (ValueError,TypeError,KeyError,UnicodeError,OSError) as exc:
        return {**response,"status":"UNRESOLVED","reason":type(exc).__name__,
                "detail":str(exc)[:160]}


def render_multifield(receipt,output):
    if output=="json":
        return json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False)+"\n"
    if output=="jsonl":
        return json.dumps(receipt,sort_keys=True,separators=(",",":"),
                          allow_nan=False)+"\n"
    if output=="text":
        result=receipt.get("result") or {}
        return "\n".join((
            "VL-64 multi-field virtual Ground",
            "Status: "+receipt["status"],
            "Ground: "+str(receipt.get("ground_digest")),
            "Result: "+str(result.get("result",result.get("compatible_model_states"))),
            "Next observation: "+str(result.get("next_probe")),
            "Scope: "+receipt["scope"],
        ))+"\n"
    raise ValueError("Unqualified observer output")
