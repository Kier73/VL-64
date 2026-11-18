"""Observer intent routed to generic byte Ground or qualified pure realization.

This path deliberately avoids architecture- and provider-specific branching:
sources provide bytes; optional decoders propose descriptions; realizer
capabilities decide whether a requested operation is qualified.
"""
from __future__ import annotations

import json
import base64
from pathlib import Path

from vld import ObserverIntention, domain_digest

from .ground import QualifiedByteExecutor, ground_for_capture
from .sources import DEFAULT_HOSTS, acquire


SUPPORTED = frozenset({"bytes.observe", "bytes.execute", "bytes.capabilities"})


def compile_byte_plan(request, *, sources=None, executor=None):
    """Compile intention without granting source/host/operation authority."""
    if not isinstance(request, ObserverIntention):
        raise TypeError("Expected ObserverIntention")
    spec = request.to_dict()
    if spec["operation"] not in SUPPORTED:
        return {"status": "UNRESOLVED", "reason": "UNKNOWN_BYTE_INTENTION",
                "intention_digest": request.digest}
    if spec["operation"] == "bytes.capabilities":
        plan = {"status": "ADMITTED_CAPABILITY_QUERY",
                "kind": "CAPABILITY_INSPECTION",
                "intention_digest": request.digest,
                "steps": ["inspect_installed_capabilities"],
                "output": spec["output"]}
        plan["plan_digest"] = domain_digest("vl64.byte-intention-plan.v1", plan)
        return plan
    target = spec["target"]
    if not (target.startswith(("workspace:", "inline:b64,", "https://")) or
            (sources is not None and any(
                target.startswith(prefix) for prefix in sources.adapters))):
        return {"status": "REFUSED", "reason": "NO_SOURCE_ADAPTER",
                "intention_digest": request.digest}
    if spec["operation"] == "bytes.execute":
        operator = spec["options"].get("operator")
        if operator not in (executor or QualifiedByteExecutor()).registry:
            return {"status": "REFUSED", "reason": "UNREGISTERED_REALIZER",
                    "intention_digest": request.digest}
    plan = {
        "status": "ADMITTED_SUBJECT_TO_SOURCE_AND_CAPABILITY",
        "kind": "OBSERVATION" if spec["operation"] == "bytes.observe" else "PURE_EXECUTION",
        "intention_digest": request.digest,
        "steps": ["qualify_source", "capture_bounded_bytes",
                  "construct_exact_byte_ground", "discover_supported_views"] + (
                  ["qualify_local_realizer", "execute_pure_operator", "bind_receipt"]
                  if spec["operation"] == "bytes.execute" else []),
        "output": spec["output"],
        "execution_authority": "pure-local-allowlisted-operation-only",
        "source_authority": "requires_authorized-acquisition-context",
    }
    plan["plan_digest"] = domain_digest("vl64.byte-intention-plan.v1", plan)
    return plan


def byte_arbitrate(request, *, root=None, approved_hosts=None, executor=None,
                   sources=None, decoders=None):
    """Observe arbitrary authorized byte input; execute only registered pure ops."""
    executor = executor or QualifiedByteExecutor()
    plan = compile_byte_plan(request, sources=sources, executor=executor)
    spec = request.to_dict()
    base = {
        "schema": "vl64.byte-observer-receipt.v1",
        "intention": spec,
        "intention_digest": request.digest,
        "plan": plan,
        "verification_scope": "bytes_and_selected_syntax_not_unobserved_machine_semantics",
    }
    if plan["status"] == "ADMITTED_CAPABILITY_QUERY":
        capabilities = {
            "source_adapters": (["workspace:", "inline:b64,", "https://"] +
                sorted(sources.adapters) if sources is not None else
                ["workspace:", "inline:b64,", "https://"]),
            "registered_decoders": (
                sorted(decoders.decoders) if decoders is not None else []),
            "builtin_recognizers": [
                "bytes", "utf8", "json", "html", "png", "gif",
                "jpeg", "pdf", "zip", "elf", "wasm", "dos-pe-candidate",
                "safetensors-header"],
            "execution_capabilities": {
                name: {"input_kind": cap.input_kind,
                       "requires_complete": cap.requires_complete,
                       "max_input_bytes": cap.max_input, "policy": cap.policy}
                for name,(cap,_) in sorted(executor.registry.items())},
            "allowed_https_hosts": sorted(
                approved_hosts if approved_hosts is not None else DEFAULT_HOSTS),
        }
        ground = {"schema": "vl64.byte-capability-ground.v1",
                  "catalogue": capabilities}
        ground["ground_digest"] = domain_digest(
            "vl64.byte-capability-ground.v1", ground)
        return {**base, "status": "OBSERVED_CAPABILITY_CATALOGUE",
                "ground": ground, "execution": None}
    if plan["status"] != "ADMITTED_SUBJECT_TO_SOURCE_AND_CAPABILITY":
        return {**base, "status": plan["status"], "reason": plan.get("reason")}
    opts = spec["options"]
    try:
        use_registered_source = (sources is not None and any(
            spec["target"].startswith(prefix) for prefix in sources.adapters))
        capture = (sources.acquire(
            spec["target"], max_bytes=opts.get("max_bytes", 65536))
            if use_registered_source else
            acquire(
                spec["target"], root=Path(root) if root is not None else Path.cwd(),
                approved_hosts=approved_hosts if approved_hosts is not None else DEFAULT_HOSTS,
                max_bytes=opts.get("max_bytes", 65536),
                byte_range=opts.get("byte_range", False),
            ))
        hint = opts.get("format_hint", spec["target"])
        if not isinstance(hint, str) or len(hint) > 300:
            raise ValueError("Invalid representation hint")
        ground = ground_for_capture(capture, hint=hint, decoders=decoders)
        result = None
        if spec["operation"] == "bytes.execute":
            params = opts.get("parameters", {})
            if not isinstance(params, dict):
                raise ValueError("Operator parameters must be an object")
            result = executor.execute(capture, ground, opts["operator"], params)
            status = result["status"]
        else:
            status = "OBSERVED_BYTE_GROUND" if capture["complete"] else "PARTIAL_BYTE_GROUND"
        return {**base, "status": status, "ground": ground,
                "execution": result,
                "observed_data": ({
                    "encoding": "base64",
                    "data": base64.b64encode(capture["data"][:4096]).decode("ascii"),
                    "returned_bytes": min(len(capture["data"]), 4096),
                    "captured_bytes": len(capture["data"]),
                    "scope": ("full-resource" if capture["complete"] and
                              len(capture["data"]) <= 4096 else "captured-prefix"),
                } if opts.get("include_data") is True else None),
                "acquisition_scope": "authorized-source-bytes-or-captured-prefix"}
    except (ValueError, KeyError, TypeError, OSError) as exc:
        return {**base, "status": "UNRESOLVED", "reason": type(exc).__name__,
                "detail": str(exc)[:180]}


def render_byte_receipt(receipt, output):
    if output == "json":
        return json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output == "jsonl":
        return json.dumps(receipt, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n"
    if output == "text":
        ground = receipt.get("ground", {})
        view = ground.get("view", {})
        evidence = ground.get("observation", {})
        execution = receipt.get("execution")
        lines = [
            "VL-64 Byte Ground",
            "Status: " + receipt["status"],
            "Intention: " + receipt["intention_digest"],
            "Source: " + str(evidence.get("source", receipt["intention"]["target"])),
            "Observed type: " + str(view.get("kind", "unknown")),
            "Captured bytes: " + str(evidence.get("captured_bytes", "?")),
            "Completeness: " + str(evidence.get("capture_status", "unknown")),
            "Byte digest: " + str(evidence.get("content_sha256", "unobserved")),
            "Ground digest: " + str(ground.get("ground_digest", "unavailable")),
        ]
        if ground.get("schema") == "vl64.byte-capability-ground.v1":
            lines.append("Available capabilities: " +
                         json.dumps(ground["catalogue"], sort_keys=True))
        if receipt.get("observed_data") is not None:
            lines.append("Observed data (base64): " + receipt["observed_data"]["data"])
            lines.append("Observed data scope: " + receipt["observed_data"]["scope"])
        if execution is not None:
            lines.append("Execution: " + str(execution))
        if receipt.get("reason"):
            lines.append("Reason: " + receipt["reason"])
        return "\n".join(lines) + "\n"
    raise ValueError("Unqualified output format")


def replay_byte_receipt(receipt, *, capture, decoders=None, executor=None):
    """Re-evaluate Ground evidence against supplied bytes; no remote attestations."""
    try:
        ground = receipt["ground"]
        from .ground import sha256
        if not isinstance(capture["data"], bytes):
            return {"status": "REJECTED", "reason": "INPUT_NOT_BYTES"}
        if ground["observation"]["content_sha256"] != sha256(capture["data"]):
            return {"status": "REJECTED", "reason": "CONTENT_MISMATCH"}
        if ground["observation"]["capture_status"] != (
                "COMPLETE" if capture["complete"] else "PREFIX"):
            return {"status": "REJECTED", "reason": "COMPLETENESS_MISMATCH"}
        verified = ground_for_capture(capture, hint=receipt["intention"]["options"].get(
            "format_hint", receipt["intention"]["target"]), decoders=decoders)
        if verified["ground_digest"] != ground["ground_digest"]:
            return {"status": "REJECTED", "reason": "GROUND_MISMATCH"}
        if verified["observation"]["observation_digest"] != ground["observation"]["observation_digest"]:
            return {"status": "REJECTED", "reason": "OBSERVATION_MISMATCH"}
        operation = receipt.get("execution")
        if operation is not None:
            options = receipt["intention"]["options"]
            computed = (executor or QualifiedByteExecutor()).execute(
                capture, verified, options["operator"], options.get("parameters", {}))
            if computed != operation:
                return {"status": "REJECTED", "reason": "EXECUTION_RESULT_MISMATCH"}
        return {"status": "VERIFIED_BYTE_GROUND", "ground_digest": verified["ground_digest"],
                "observation_digest": verified["observation"]["observation_digest"],
                "operations_replayed": 1 if operation is not None else 0,
                "scope": "same-input-and-local-operation-replay-not-external-truth"}
    except (ValueError, KeyError, TypeError, OSError):
        return {"status": "REJECTED", "reason": "INVALID_EVIDENCE"}
