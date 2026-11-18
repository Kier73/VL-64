"""Observer intention transport and read-only model structure extraction."""
import json
from pathlib import Path

import pytest
from vld import ObserverIntention
from experiments.observer_intent import arbitrate as layer


def intent(**overrides):
    base = {"operation": "hf.model.describe",
            "target": "latest:text-generation",
            "output": "json",
            "options": {"metadata_only": True, "catalog_limit": 3}}
    base.update(overrides)
    return ObserverIntention.from_dict(base)


def test_core_intention_is_content_addressed_and_detached():
    options = {"metadata_only": True, "catalog_limit": 4}
    req = ObserverIntention("hf.model.describe", "latest:text-generation",
                            "json", options)
    before = req.digest
    options["catalog_limit"] = 20
    assert req.digest == before
    assert req.to_dict()["options"]["catalog_limit"] == 4
    cloned = ObserverIntention.from_dict(req.to_dict())
    assert cloned.digest == req.digest
    assert cloned.to_dict() == req.to_dict()


@pytest.mark.parametrize("invalid", [
    {"operation": "hf.model.describe", "target": ""},
    {"operation": "../danger", "target": "latest:text-generation"},
    {"operation": "hf.model.describe", "target": "x", "output": "python"},
    {"operation": "hf.model.describe", "target": "x", "options": [1, 2]},
    {"operation": "hf.model.describe", "target": "x", "options": {"a": float("nan")}},
    {"operation": "hf.model.describe", "target": "x", "unexpected": 3},
])
def test_core_refuses_invalid_intentions(invalid):
    with pytest.raises((ValueError, TypeError)):
        ObserverIntention.from_dict(invalid)


def test_unknown_operation_never_dispatches():
    req = ObserverIntention("hf.model.generate", "openai-community/gpt2")
    result = layer.arbitrate(req)
    assert result["status"] == "UNRESOLVED"
    assert result["plan"]["status"] == "UNRESOLVED"
    assert result["outcome"] is None


def test_explicit_model_id_is_not_a_url_injection():
    for target in ("../../private", "https://evil.example/anything",
                   "user/repo/extra", "user/..", "x y/z"):
        with pytest.raises(ValueError):
            layer.compile_intention(intent(target=target))


def test_model_ground_from_mocked_metadata(monkeypatch):
    captures = []
    repo = "openai-community/gpt2"
    sha = "a" * 40

    def controlled_get(url, max_bytes):
        captures.append(url)
        if url == "https://huggingface.co/api/models?pipeline_tag=text-generation&sort=lastModified&direction=-1&limit=3":
            payload = [
                {"id": repo, "gated": False,
                 "pipeline_tag": "text-generation",
                 "lastModified": "2026-11-18T19:00:00Z"}
            ]
        elif url == "https://huggingface.co/api/models/" + repo:
            payload = {
                "id": repo, "sha": sha, "lastModified": "2026-11-18T19:00:00Z",
                "pipeline_tag": "text-generation",
                "siblings": [
                    {"rfilename": "config.json"},
                    {"rfilename": "model.safetensors"},
                    {"rfilename": "pytorch_model.bin"},
                ],
                "safetensors": {"total": 124000000, "parameters": {"F32": 31000000}},
            }
        elif url == "https://huggingface.co/" + repo + "/resolve/" + sha + "/config.json":
            payload = {"model_type": "gpt2", "n_embd": 768,
                       "n_layer": 12, "n_head": 12, "vocab_size": 50257,
                       "untrusted_extra": "ignored"}
        else:
            raise AssertionError("Unexpected request")
        content = json.dumps(payload).encode()
        return {"url": url, "http_status": 200,
                "body_sha256": layer.sha(content),
                "observed_bytes": len(content), "truncated": False}, payload

    monkeypatch.setattr(layer, "metadata_get", controlled_get)
    req = intent()
    plan = layer.compile_intention(req)
    assert plan["status"] == "ADMITTED_READ_ONLY_PLAN"
    assert plan["steps"][0] == "query_catalogue_latest_modified"
    report = layer.arbitrate(req)
    assert report["intention_digest"] == req.digest
    assert report["status"] == "OBSERVED_MODEL_STRUCTURE"
    ground = report["outcome"]["ground"]
    assert ground["architecture"]["model_type"] == "gpt2"
    assert ground["architecture"]["n_layer"] == 12
    assert "untrusted_extra" not in ground["architecture"]
    assert ground["weight_file_count"] == 2
    assert any(node["id"] == "config_field:n_layer" for node in ground["graph"]["nodes"])
    assert any(edge["relation"] == "declares_field" for edge in ground["graph"]["edges"])
    assert ground["family_operator_hints"]["decoder_blocks"] == 12
    assert ground["family_operator_hints"]["scope"] == "unverified_family_template_not_actual_inference"
    assert ground["claims"]["model_weight_payload_bytes_read"] == 0
    assert not ground["claims"]["inference_executed"]
    assert ground["revision"] == sha
    assert len(captures) == 3
    assert all(".safetensors" not in url and ".bin" not in url for url in captures)
    assert report["outcome"]["selection"]["selector"] == "bounded-most-recently-modified"
    assert "Model inference executed: NO" in layer.render_result(report, "text")
    assert json.loads(layer.render_result(report, "jsonl"))["status"] == "OBSERVED_MODEL_STRUCTURE"
    assert json.loads(layer.render_result(report, "json"))["status"] == "OBSERVED_MODEL_STRUCTURE"


def test_partial_ground_preserves_not_observed(monkeypatch):
    def fake_get(url, max_bytes):
        return {"http_status": 200, "observed_bytes": 100, "truncated": False}, {
            "id": "org/model", "sha": "f"*40,
            "siblings": [{"rfilename": "model.safetensors"}],
        }
    monkeypatch.setattr(layer, "metadata_get", fake_get)
    report = layer.arbitrate(intent(target="org/model"))
    assert report["status"] == "PARTIAL_MODEL_STRUCTURE"
    assert report["outcome"]["ground"]["configuration_status"] == "NOT_LISTED"
    assert report["outcome"]["ground"]["claims"]["model_weight_payload_bytes_read"] == 0


def test_no_weight_download_or_redirect_acceptance():
    with pytest.raises(ValueError):
        layer.metadata_get("https://huggingface.co/org/repo/resolve/main/model.safetensors", 32)
    with pytest.raises(ValueError):
        layer.metadata_get("https://huggingface.co/org/repo/resolve/main/model.py", 32)
    with pytest.raises(ValueError):
        layer.metadata_get("https://evil.example/api/models", 32)
    with pytest.raises(ValueError):
        layer.metadata_get("http://huggingface.co/api/models", 32)


def test_refuse_request_to_download_weights(monkeypatch):
    def forbidden(_url, _limit):
        raise AssertionError("Must refuse before network")
    monkeypatch.setattr(layer, "metadata_get", forbidden)
    report = layer.arbitrate(intent(target="org/model",
                                    options={"metadata_only": False}))
    assert report["status"] == "REFUSED"


def test_missing_explicit_model_is_unresolved(monkeypatch):
    def fake_get(_url, _max_bytes):
        return {"status": "UNRESOLVED"}, None
    monkeypatch.setattr(layer, "metadata_get", fake_get)
    report = layer.arbitrate(intent(target="org/model"))
    assert report["status"] == "UNRESOLVED"


def test_github_document_intent_is_supported_without_hf_calls(monkeypatch):
    def fake_fn(intention):
        return {"status": "OBSERVED_DESCRIPTION_PATH",
                "ground": {"route": {"hops": 3}},
                "evidence": [{"status": "OBSERVED"}]}
    monkeypatch.setitem(layer.ROUTES, "github.document.describe", fake_fn)
    request = ObserverIntention(
        "github.document.describe", "Kier73/VL-64:docs/RESEARCH_GOALS.md",
        output="jsonl")
    result = layer.arbitrate(request)
    assert result["status"] == "OBSERVED_DESCRIPTION_PATH"
    assert len(layer.render_result(result, "jsonl").splitlines()) == 1


def test_compact_vl64_source_budget():
    import vld
    root = Path(vld.__file__).parent
    assert sum(p.stat().st_size for p in root.glob("*.py")) <= 65536


def test_config_redirect_policy_only_allows_same_host_config_cache():
    from urllib.request import Request
    handler = layer.ConfigOnlyRedirect()
    original = Request(
        "https://huggingface.co/openai-community/gpt2/resolve/"
        + "a"*40 + "/config.json"
    )
    safe_target = (
        "https://huggingface.co/api/resolve-cache/models/openai-community/"
        "gpt2/" + "a"*40 + "/config.json"
    )
    assert handler.redirect_request(original, None, 302, "found", {},
                                    safe_target) is not None
    assert handler.redirect_request(original, None, 302, "found", {},
                                    "https://other.example/config.json") is None
    assert handler.redirect_request(original, None, 302, "found", {},
                                    "https://huggingface.co/org/repo/weights.safetensors") is None
    from_listing = Request("https://huggingface.co/api/models")
    assert handler.redirect_request(from_listing, None, 302, "found", {},
                                    safe_target) is None


def test_explicit_unknown_operator_can_never_execute_remote_code(monkeypatch):
    monkeypatch.setitem(layer.ROUTES, "hf.model.generate",
                        lambda _item: pytest.fail("No execution admission"))
    req = ObserverIntention("hf.model.generate", "org/model")
    assert layer.compile_intention(req)["status"] == "UNRESOLVED"
    # Dispatch uses the compiled local allowlist, not dynamically added handlers.
    assert layer.arbitrate(req)["status"] == "UNRESOLVED"


def test_observer_requires_architecture_and_selects_next_qualified_model(monkeypatch):
    first, second = "example/train-state", "example/configured-model"
    catalog = [
        {"id": first, "last_modified": "2026-11-18T19:00:00Z"},
        {"id": second, "last_modified": "2026-11-18T18:59:00Z"},
    ]

    def fake_list(limit):
        return {"status": "OBSERVED", "observed_bytes": 200}, catalog

    def fake_get(url, max_bytes):
        if url.endswith("/api/models/" + first):
            value = {"id": first, "sha": "a"*40,
                     "siblings": [{"rfilename": "weights.safetensors"}]}
        elif url.endswith("/api/models/" + second):
            value = {"id": second, "sha": "b"*40,
                     "siblings": [{"rfilename": "config.json"}]}
        elif url.endswith("/resolve/" + "b"*40 + "/config.json"):
            value = {"model_type": "gpt2", "n_layer": 3, "n_embd": 256,
                     "n_head": 4}
        else:
            raise AssertionError("Unexpected URL: " + url)
        return {"http_status": 200, "observed_bytes": 120,
                "body_sha256": layer.sha(json.dumps(value).encode())}, value

    monkeypatch.setattr(layer, "read_model_listing", fake_list)
    monkeypatch.setattr(layer, "metadata_get", fake_get)
    request = intent(options={"metadata_only": True,
                              "require_architecture": True,
                              "catalog_limit": 2,
                              "max_candidates": 2})
    receipt = layer.arbitrate(request)
    assert receipt["status"] == "OBSERVED_MODEL_STRUCTURE"
    chosen = receipt["outcome"]
    assert chosen["ground"]["model_id"] == second
    assert chosen["selection"]["candidate_rank"] == 2
    assert chosen["selection"]["attempts"][0]["status"] == "PARTIAL_MODEL_STRUCTURE"
    assert chosen["ground"]["architecture"]["n_layer"] == 3
    assert chosen["ground"]["claims"]["model_weight_payload_bytes_read"] == 0


def test_architecture_requirement_refuses_inadequate_catalogue(monkeypatch):
    def fake_list(limit):
        return {"status": "OBSERVED", "observed_bytes": 100}, [
            {"id": "example/checkpoint", "last_modified": "2026-11-18T19:00:00Z"}]

    def fake_get(url, max_bytes):
        return {"http_status": 200, "observed_bytes": 150}, {
            "id": "example/checkpoint", "sha": "a"*40,
            "siblings": [{"rfilename": "checkpoint.pt"}]}

    monkeypatch.setattr(layer, "read_model_listing", fake_list)
    monkeypatch.setattr(layer, "metadata_get", fake_get)
    req = intent(options={"require_architecture": True, "max_candidates": 1})
    outcome = layer.arbitrate(req)
    assert outcome["status"] == "UNRESOLVED"
    assert outcome["outcome"]["reason"] == "NO_CANDIDATE_MET_ARCHITECTURE_REQUIREMENT"
    assert outcome["outcome"]["weight_payload_bytes_read"] == 0
    for bad in (0, 11, -1, True, 2.1):
        request = intent(options={"require_architecture": True, "max_candidates": bad})
        assert layer.arbitrate(request)["status"] == "UNRESOLVED"


def test_transformers_library_filter_only_uses_approved_catalogue(monkeypatch):
    observed = []
    def fake_get(url, max_bytes):
        observed.append(url)
        return {"http_status": 200, "observed_bytes": 2}, []
    monkeypatch.setattr(layer, "metadata_get", fake_get)
    rec, content = layer.read_model_listing(5, "transformers")
    assert content == []
    assert observed[0].endswith(
        "pipeline_tag=text-generation&sort=lastModified&direction=-1&limit=5&library=transformers")
    with pytest.raises(ValueError):
        layer.read_model_listing(5, "untrusted-library")
