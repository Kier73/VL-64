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
