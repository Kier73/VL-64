"""Observer intention -> qualified metadata observation -> Virtual Ground.

An external adapter, intentionally outside VL-64's 64 KiB source budget.
Never requests model weight payloads, runs arbitrary repository code, or calls
inference providers. Treats Hugging Face metadata as observed, not verified
model behavior.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from vld import ObserverIntention, domain_digest

SCHEMA = "vl64.observer-arbitration.v1"
ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
MAX_MODEL_BYTES = 192 * 1024
MAX_CONFIG_BYTES = 96 * 1024
MAX_LIST_BYTES = 256 * 1024
SAFE_CONFIG_FIELDS = (
    "architectures", "model_type", "hidden_size", "n_embd", "num_hidden_layers",
    "n_layer", "num_attention_heads", "n_head", "num_key_value_heads",
    "intermediate_size", "vocab_size", "max_position_embeddings",
    "rope_theta", "tie_word_embeddings", "torch_dtype", "dtype",
    "sliding_window", "head_dim", "num_experts", "num_local_experts",
    "num_experts_per_tok", "transformers_version", "vision_config",
    "text_config", "language_config",
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class ConfigOnlyRedirect(urllib.request.HTTPRedirectHandler):
    """Follow only same-host config.json cache redirection, never weight URLs."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        old, new = (urllib.parse.urlsplit(v) for v in (req.full_url, newurl))
        if (old.path.endswith("/config.json")
                and new.scheme == "https" and new.hostname == "huggingface.co"
                and new.username is None and new.password is None
                and new.port in (None, 443)
                and new.path.endswith("/config.json")
                and (new.path.startswith("/api/resolve-cache/models/")
                     or "/resolve/" in new.path)):
            return super().redirect_request(req, fp, code, msg, headers, newurl)
        return None


def metadata_get(url, max_bytes):
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname != "huggingface.co"
            or parsed.port not in (None, 443) or parsed.username
            or parsed.password):
        raise ValueError("Metadata host not permitted")
    if not (parsed.path == "/api/models" or
            parsed.path.startswith("/api/models/") or
            (parsed.path.endswith("/config.json") and "/resolve/" in parsed.path)):
        raise ValueError("Only model listings, model metadata and config.json permitted")
    # Never fetch model weights or remote code.
    if any(t in parsed.path.lower() for t in (
            ".safetensors", ".bin", ".gguf", ".pt", ".py", ".pth",
            "/tokenizer.json", "/vocab.json")):
        raise ValueError("Weight, executable or tokenizer payload prohibited")
    begin = time.perf_counter()
    req = urllib.request.Request(url, headers={
        "User-Agent": "VL64-Observer-Research/0.1",
        "Accept": "application/json",
    }, method="GET")
    try:
        try:
            resp = urllib.request.build_opener(ConfigOnlyRedirect).open(req, timeout=12)
        except urllib.error.HTTPError as err:
            resp = err
        with resp:
            raw = resp.read(max_bytes + 1)
            result = {
                "url": url,
                "http_status": resp.status if hasattr(resp, "status") else resp.code,
                "body_sha256": sha(raw[:max_bytes]),
                "observed_bytes": min(len(raw), max_bytes),
                "truncated": len(raw) > max_bytes,
                "elapsed_ms": round(1000 * (time.perf_counter() - begin), 3),
                "content_type": (resp.headers.get("Content-Type") or "")[:80],
                "redirect_host": urllib.parse.urlsplit(
                    resp.headers.get("Location") or "").hostname,
            }
            if result["http_status"] != 200:
                return result, None
            if result["truncated"]:
                return result, None
            return result, json.loads(raw)
    except (OSError, TimeoutError, UnicodeError, ValueError) as exc:
        return {"url": url, "status": "UNRESOLVED",
                "reason": type(exc).__name__,
                "detail": str(exc)[:120]}, None


def model_identifier(value):
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value) or ".." in value:
        raise ValueError("Unsafe model identifier")
    return value


def pinned_revision(value):
    if not isinstance(value, str) or not HEX40.fullmatch(value):
        raise ValueError("Expected pinned 40-character revision")
    return value


def read_model_listing(limit=12):
    if type(limit) is not int or not 1 <= limit <= 20:
        raise ValueError("Catalogue limit must be 1..20")
    query = urllib.parse.urlencode({
        "pipeline_tag": "text-generation",
        "sort": "lastModified",
        "direction": "-1",
        "limit": str(limit),
    })
    return metadata_get("https://huggingface.co/api/models?" + query,
                        MAX_LIST_BYTES)


def choose_model(intention, evidence):
    target = intention["target"]
    if target != "latest:text-generation":
        return model_identifier(target), {
            "selector": "explicit-repository-id",
            "selection_scope": "user_supplied_model_identifier",
        }
    limit = intention["options"].get("catalog_limit", 12)
    record, listing = read_model_listing(limit)
    evidence.append(record)
    if not isinstance(listing, list):
        raise ValueError("Model catalogue not available as complete JSON list")
    # "Latest" means most recently modified among a bounded, untrusted
    # catalogue response; it does not mean newest published base model.
    candidates = []
    for item in listing:
        if not isinstance(item, dict):
            continue
        candidate = item.get("id") or item.get("modelId")
        if not isinstance(candidate, str) or not ID_PATTERN.fullmatch(candidate):
            continue
        if item.get("gated") or item.get("private"):
            continue
        candidates.append({
            "id": candidate, "last_modified": item.get("lastModified"),
            "pipeline_tag": item.get("pipeline_tag"),
        })
    if not candidates:
        raise ValueError("No accessible text-generation candidates in bounded catalogue")
    return candidates[0]["id"], {
        "selector": "bounded-most-recently-modified",
        "selection_scope": "first-public-eligible-of-retrieved-candidates",
        "catalogue_count": len(listing),
        "eligible_count": len(candidates),
        "candidate_rank": 1,
        "ranking_field": "lastModified",
        "candidates_sample": candidates[:12],
    }


def _typed_config(config):
    if not isinstance(config, dict):
        return {}
    return {field: config[field] for field in SAFE_CONFIG_FIELDS
            if field in config and isinstance(
                config[field], (int, float, bool, str, list, dict, type(None)))}


def make_hf_structure(intention):
    if intention["options"].get("metadata_only", True) is not True:
        return {"status": "REFUSED",
                "reason": "Only metadata-only observations are admitted"}
    if (intention["target"] == "latest:text-generation" and
            intention["options"].get("require_architecture") is True):
        max_candidates = intention["options"].get("max_candidates", 6)
        if type(max_candidates) is not int or not 1 <= max_candidates <= 10:
            raise ValueError("max_candidates must be an integer 1..10")
        observed = []
        _, selection = choose_model(intention, observed)
        attempts = []
        for rank, candidate in enumerate(
                selection["candidates_sample"][:max_candidates], 1):
            probe = dict(intention, target=candidate["id"],
                         options={"metadata_only": True})
            candidate_result = make_hf_structure(probe)
            observed.extend(candidate_result.get("evidence", []))
            attempts.append({
                "rank": rank, "model_id": candidate["id"],
                "status": candidate_result["status"],
                "configuration_status": candidate_result.get(
                    "ground", {}).get("configuration_status"),
            })
            if candidate_result["status"] == "OBSERVED_MODEL_STRUCTURE":
                candidate_result["selection"] = {
                    **selection, "candidate_rank": rank,
                    "selection_scope": "first_config_inspectable_in_bounded_catalogue",
                    "require_architecture": True, "attempts": attempts}
                candidate_result["evidence"] = observed
                candidate_result["total_metadata_bytes_read"] = sum(
                    item.get("observed_bytes", 0) for item in observed)
                return candidate_result
        return {
            "status": "UNRESOLVED",
            "reason": "NO_CANDIDATE_MET_ARCHITECTURE_REQUIREMENT",
            "selection": {**selection, "require_architecture": True,
                          "attempts": attempts},
            "evidence": observed, "weight_payload_bytes_read": 0}
    evidence = []
    model_id, selection = choose_model(intention, evidence)
    name = model_identifier(model_id)
    info_url = "https://huggingface.co/api/models/" + name
    info_rec, info = metadata_get(info_url, MAX_MODEL_BYTES)
    evidence.append(info_rec)
    if not isinstance(info, dict) or info.get("id") not in (name, None):
        return {"status": "UNRESOLVED", "reason": "Model metadata unavailable",
                "model_id": name, "evidence": evidence, "selection": selection}
    rev = pinned_revision(info.get("sha"))
    siblings = info.get("siblings") or []
    if not isinstance(siblings, list):
        siblings = []
    files = [s.get("rfilename") for s in siblings if isinstance(s, dict)
             and isinstance(s.get("rfilename"), str)]
    weight_files = [x for x in files if
                    x.endswith((".safetensors", ".gguf", ".bin", ".pt"))]
    config = {}
    config_status = "NOT_LISTED"
    if "config.json" in files:
        config_url = (f"https://huggingface.co/{name}/resolve/{rev}/config.json")
        config_rec, config_obj = metadata_get(config_url, MAX_CONFIG_BYTES)
        evidence.append(config_rec)
        if isinstance(config_obj, dict):
            config = _typed_config(config_obj)
            config_status = "OBSERVED"
        else:
            config = _typed_config(info.get("config"))
            config_status = ("MODEL_INFO_FALLBACK" if config else
                             "CONFIG_HTTP_" + str(config_rec.get("http_status", "UNRESOLVED")))
    else:
        config = _typed_config(info.get("config"))
        if config:
            config_status = "MODEL_INFO_METADATA_ONLY"
    safetensors = info.get("safetensors")
    safe = {}
    if isinstance(safetensors, dict):
        safe = {k: v for k, v in safetensors.items()
                if k in ("total", "parameters") and isinstance(v, (dict, int, float))}
    nodes = [
        {"id": "model", "kind": "model_metadata"},
        {"id": "revision", "kind": "git_commit"},
        {"id": "config", "kind": "declared_model_architecture"},
        {"id": "weight_manifest", "kind": "filenames_not_weight_payload"},
    ]
    edges = [
        {"from": "model", "to": "revision", "relation": "pinned_revision"},
        {"from": "revision", "to": "config", "relation": "config_at_revision"},
        {"from": "revision", "to": "weight_manifest",
         "relation": "file_listing_at_revision"},
    ]
    ground = {
        "schema": "vl64.hf-model-structure-ground.v1",
        "model_id": name, "revision": rev,
        "pipeline_tag": info.get("pipeline_tag"),
        "last_modified": info.get("lastModified"),
        "gated": info.get("gated", False),
        "configuration_status": config_status,
        "architecture": config,
        "weight_file_count": len(weight_files),
        "weight_filename_sample": weight_files[:32],
        "total_listed_files": len(files),
        "safetensors_summary": safe,
        "graph": {"nodes": nodes, "edges": edges},
        "claims": {
            "source": "public_huggingface_metadata_and_bounded_config_json",
            "model_weights_downloaded": False,
            "model_weight_payload_bytes_read": 0,
            "inference_executed": False,
            "model_behavior_predicted": False,
            "execution_capability": "UNQUALIFIED_NO_WEIGHTS_OR_REMOTE_INFERENCE_RECEIPT",
            "tensor_shape_complete": False,
        },
    }
    ground["ground_digest"] = domain_digest("vl64.hf-model-ground.v1", ground)
    return {
        "status": "OBSERVED_MODEL_STRUCTURE" if config else "PARTIAL_MODEL_STRUCTURE",
        "selection": selection, "ground": ground,
        "evidence": evidence,
        "total_metadata_bytes_read": sum(
            x.get("observed_bytes", 0) for x in evidence),
    }


def run_github_document(intention):
    if intention["target"] != "Kier73/VL-64:docs/RESEARCH_GOALS.md":
        return {"status": "REFUSED", "reason": "Unsupported GitHub document scope"}
    from experiments.internet_lab.virtual_net import (
        InternetGround, github_readme_to_goals)
    graph = InternetGround()
    evidence, route, _ = github_readme_to_goals(graph)
    return {
        "status": route.get("status", "UNRESOLVED"),
        "selection": {"selector": "pinned-commit-document-navigation"},
        "ground": {"graph_digest": graph.fingerprint(),
                   "nodes": graph.nodes, "edges": graph.edges,
                   "route": route},
        "evidence": evidence,
    }


ROUTES = {
    "hf.model.describe": make_hf_structure,
    "github.document.describe": run_github_document,
}


def compile_intention(request):
    if not isinstance(request, ObserverIntention):
        raise TypeError("Expected ObserverIntention")
    spec = request.to_dict()
    operation = spec["operation"]
    if operation == "hf.model.describe":
        if spec["target"] != "latest:text-generation":
            model_identifier(spec["target"])
        steps = (["query_catalogue_latest_modified"] if
                 spec["target"] == "latest:text-generation" else [])
        steps += ["read_model_info", "read_pinned_config_if_listed",
                  "form_metadata_ground", "format_observation_receipt"]
    elif operation == "github.document.describe":
        steps = ["resolve_git_revision", "fetch_pinned_readme",
                 "follow_declared_link", "verify_content_digests"]
    else:
        return {"status": "UNRESOLVED", "reason": "NO_REGISTERED_OPERATOR",
                "intention_digest": request.digest}
    plan = {"status": "ADMITTED_READ_ONLY_PLAN",
            "intention_digest": request.digest,
            "operator": operation,
            "steps": steps,
            "side_effect_policy": "public_read_only_no_weights",
            "format": spec["output"]}
    plan["plan_digest"] = domain_digest("vl64.observer-plan.v1", plan)
    return plan


def arbitrate(request):
    """Resolve observer intent through a registered network metadata operator."""
    plan = compile_intention(request)
    if plan["status"] != "ADMITTED_READ_ONLY_PLAN":
        return {"schema": SCHEMA, "status": "UNRESOLVED",
                "intention": request.to_dict(), "intention_digest": request.digest,
                "plan": plan, "outcome": None}
    try:
        outcome = ROUTES[plan["operator"]](request.to_dict())
    except (ValueError, TypeError, KeyError) as exc:
        outcome = {"status": "UNRESOLVED",
                   "reason": type(exc).__name__, "detail": str(exc)[:160]}
    return {
        "schema": SCHEMA,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "status": outcome["status"],
        "intention": request.to_dict(),
        "intention_digest": request.digest,
        "plan": plan,
        "outcome": outcome,
        "observer_scope": "github_actions_runner_or_local_process",
        "verification_scope": "metadata_integrity_and_scope_not_model_execution",
    }


def render_result(receipt, output):
    if output == "json":
        return json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if output == "jsonl":
        return (json.dumps(receipt, sort_keys=True, separators=(",", ":"),
                           allow_nan=False) + "\n")
    if output == "text":
        outcome = receipt.get("outcome") or {}
        ground = outcome.get("ground") or {}
        if ground.get("schema") == "vl64.hf-model-structure-ground.v1":
            info = [
                "VL-64 observer intention: " + receipt["intention_digest"],
                "Status: " + receipt["status"],
                "Model: " + ground["model_id"],
                "Revision: " + ground["revision"],
                "Architecture: " + str(ground["architecture"]),
                "Weight files: " + str(ground["weight_file_count"]),
                "Model weights downloaded: NO",
                "Model inference executed: NO",
                "Ground digest: " + ground["ground_digest"],
            ]
        else:
            info = ["VL-64 observer intention: " + receipt["intention_digest"],
                    "Status: " + receipt["status"],
                    "Observed Ground: " + str(ground)]
        return "\n".join(info) + "\n"
    raise ValueError("Unsupported output")


def main():
    parser = argparse.ArgumentParser(description="VL-64 structured observer arbitration")
    parser.add_argument("--intent", required=True, help="JSON observer intention")
    parser.add_argument("--out", required=True, help="Requested-format result path")
    args = parser.parse_args()
    request = ObserverIntention.from_dict(
        json.loads(Path(args.intent).read_text(encoding="utf-8")))
    receipt = arbitrate(request)
    output = render_result(receipt, request.spec["output"])
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(output, encoding="utf-8")
    summary = {
        "intention_digest": receipt["intention_digest"],
        "plan_status": receipt["plan"]["status"],
        "status": receipt["status"],
        "format": request.spec["output"],
        "output_sha256": sha(output.encode()),
        "output_bytes": len(output.encode()),
        "model_id": (receipt.get("outcome") or {}).get("ground", {}).get("model_id"),
        "revision": (receipt.get("outcome") or {}).get("ground", {}).get("revision"),
        "weight_payload_bytes_read": (receipt.get("outcome") or {})
            .get("ground", {}).get("claims", {}).get("model_weight_payload_bytes_read"),
        "reason": (receipt.get("outcome") or {}).get("reason"),
        "evidence_statuses": [
            {"http_status": x.get("http_status"), "operation_url": x.get("url"),
             "redirect_host": x.get("redirect_host"), "truncated": x.get("truncated")}
            for x in (receipt.get("outcome") or {}).get("evidence", [])
        ],
    }
    print("VL64_OBSERVER_RESULT " + json.dumps(summary, sort_keys=True))
    if receipt["status"] not in (
            "OBSERVED_MODEL_STRUCTURE", "PARTIAL_MODEL_STRUCTURE",
            "OBSERVED_DESCRIPTION_PATH"):
        raise SystemExit("Observer intention could not be realized")


if __name__ == "__main__":
    main()
