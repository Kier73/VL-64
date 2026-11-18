"""General byte-ground observation, source-policy, and qualified realization."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import struct
from pathlib import Path
from unittest.mock import patch

import pytest
from vld import ObserverIntention, domain_digest
from experiments.byte_fabric.ground import (
    describe, ground_for_capture, QualifiedByteExecutor,
    OperationCapability, sha256,
)
from experiments.byte_fabric.sources import (
    acquire, capture_bytes, DEFAULT_HOSTS, read_https, NoRedirect,
)
from experiments.byte_fabric.router import (
    byte_arbitrate, compile_byte_plan, render_byte_receipt, replay_byte_receipt,
)
from experiments.observer_intent.arbitrate import (
    arbitrate, compile_intention, render_result,
)

ROOT = Path(__file__).resolve().parents[1]


def request(operation="bytes.observe",
            target="workspace:experiments/byte_fabric/fixtures/state.json",
            **options):
    return ObserverIntention(operation, target, options=options)


@pytest.mark.parametrize("payload, expected", [
    (b'{"answer":[1,2]}', "json"),
    (b'<!doctype html><html><title>hello</title><a href="/next">x</a></html>', "html"),
    (b"some plain text\n", "utf8-text"),
    (b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" +
     struct.pack(">II", 13, 17) + b"\x08\x02\x00\x00\x00", "png"),
    (b"GIF89a" + struct.pack("<HH", 8, 11), "gif"),
    (b"\x7fELF" + bytes((2, 1)) + bytes(10), "elf"),
    (b"\0asm\x01\0\0\0", "wasm"),
    (b"%PDF-1.7\n...", "pdf"),
    (b"PK\x03\x04random", "zip"),
    (b"\xff\xd8\xff" + bytes(20), "jpeg"),
    (b"MZxxxxxxxxxxxxx", "dos-pe-candidate"),
    (b"\xff\xfe\x00" + bytes(range(5)), "opaque-bytes"),
    (b"", "opaque-bytes"),
])
def test_every_known_and_unknown_format_has_exact_byte_ground(payload, expected):
    view = describe(payload)
    assert view["kind"] == expected
    assert view["captured_bytes"] == len(payload)
    assert view["sha256"] == hashlib.sha256(payload).hexdigest()
    cap = capture_bytes(payload, "memory:test")
    ground = ground_for_capture(cap)
    assert ground["observation"]["capture_status"] == "COMPLETE"
    assert ground["observation"]["content_sha256"] == sha256(payload)
    assert ground["ground_digest"] == domain_digest(
        "vl64.generic-byte-ground.v1",
        {k: v for k,v in ground.items() if k != "ground_digest"})


def test_safetensors_header_can_be_described_without_tensor_values():
    header = json.dumps({
        "__metadata__": {"format": "pt"},
        "weights.layer": {"dtype": "F32", "shape": [2,3],
                          "data_offsets": [0,24]}
    }).encode()
    data = struct.pack("<Q", len(header)) + header
    described = describe(data, hint="model.safetensors")
    assert described["kind"] == "safetensors-header"
    assert described["structure"]["tensor_count"] == 1
    assert described["structure"]["tensor_sample"][0]["shape"] == [2,3]
    assert described["structure"]["weight_payload_interpreted"] is False
    missing = describe(data[:11], hint="model.safetensors")
    assert missing["kind"] == "safetensors-unobserved-header"
    assert missing["structure"]["header_complete"] is False
    malformed = struct.pack("<Q", 3) + b"xxx"
    assert describe(malformed, hint="model.safetensors")["kind"] == "opaque-bytes"


def test_json_pointer_and_encoded_bytes_executable_on_any_source():
    observed = byte_arbitrate(request(), root=ROOT)
    assert observed["status"] == "OBSERVED_BYTE_GROUND"
    assert observed["ground"]["view"]["kind"] == "json"
    target = request("bytes.execute", operator="json.pointer",
                     parameters={"pointer": "/observer/constraints/verified"})
    receipt = arbitrate(target, byte_root=ROOT)
    assert receipt["status"] == "EXECUTED_LOCAL"
    assert receipt["execution"]["result"]["value"] is True
    assert receipt["execution"]["source_ground_digest"] == receipt["ground"]["ground_digest"]
    assert json.loads(render_result(receipt, "json"))["status"] == "EXECUTED_LOCAL"
    assert len(render_result(receipt, "jsonl").splitlines()) == 1
    assert "Status: EXECUTED_LOCAL" in render_result(receipt, "text")
    data = acquire(target.spec["target"], root=ROOT)
    assert replay_byte_receipt(receipt, capture=data)["status"] == "VERIFIED_BYTE_GROUND"
    altered = capture_bytes(data["data"] + b"\n", data["source"])
    assert replay_byte_receipt(receipt, capture=altered)["status"] == "REJECTED"


def test_html_links_are_data_not_followed_as_actions():
    target = "workspace:experiments/byte_fabric/fixtures/page.html"
    rep = arbitrate(request("bytes.execute", target=target,
                            operator="html.links"), byte_root=ROOT)
    assert rep["status"] == "EXECUTED_LOCAL"
    links = rep["execution"]["result"]["links"]
    assert "/docs/USAGE_GUIDE.md" in links
    assert "javascript:alert(1)" in links
    assert rep["execution"]["result"]["scope"] == "untrusted-document-hrefs-not-followed"


def test_any_binary_can_be_sliced_and_hashed():
    data = bytes(range(128)) + b"\xff\x00"
    literal = "inline:b64," + base64.b64encode(data).decode()
    assert len(literal) <= 256
    cap = acquire(literal, root=ROOT)
    assert cap["data"] == data
    observed = byte_arbitrate(request(target=literal))
    assert observed["status"] == "OBSERVED_BYTE_GROUND"
    assert observed["ground"]["view"]["kind"] == "opaque-bytes"
    sliced = byte_arbitrate(request("bytes.execute", literal,
                   operator="slice", parameters={"start": 126, "length": 4}))
    assert sliced["status"] == "EXECUTED_LOCAL"
    assert base64.b64decode(sliced["execution"]["result"]["data"]) == data[126:130]
    hashed = byte_arbitrate(request("bytes.execute", literal, operator="sha256"))
    assert hashed["status"] == "EXECUTED_LOCAL"
    assert hashed["execution"]["result"]["sha256"] == hashlib.sha256(data).hexdigest()


def test_partial_source_refuses_whole_file_ops(tmp_path):
    (tmp_path/"huge.bin").write_bytes(b"\xff" * 300)
    req = request("bytes.execute", "workspace:huge.bin",
                  operator="sha256", max_bytes=64)
    rep = byte_arbitrate(req, root=tmp_path)
    assert rep["status"] == "UNRESOLVED"
    assert rep["ground"]["observation"]["capture_status"] == "PREFIX"
    assert rep["ground"]["observation"]["digest_scope"] == "captured-prefix-only"
    assert rep["execution"]["reason"] == "FULL_RESOURCE_REQUIRED"
    stats = byte_arbitrate(request("bytes.execute", "workspace:huge.bin",
                  operator="bytes.stats", max_bytes=64), root=tmp_path)
    assert stats["status"] == "EXECUTED_LOCAL"
    assert stats["execution"]["input_scope"] == "prefix"
    assert stats["execution"]["result"]["byte_count"] == 64


def test_qualified_registry_rejects_unregistered_and_nonpure_handlers():
    target = request("bytes.execute", operator="python.eval",
                     parameters={"code": "print('no')"})
    assert compile_byte_plan(target)["status"] == "REFUSED"
    refused = arbitrate(target, byte_root=ROOT)
    assert refused["status"] == "REFUSED"
    registry = QualifiedByteExecutor()
    with pytest.raises(ValueError):
        registry.register(OperationCapability("custom", policy="remote-write"), lambda b,o: {})
    registry.register(OperationCapability("prefix", max_input=65536),
                      lambda data, opts: {"prefix_hex": data[:3].hex()})
    data = capture_bytes(b"abcxyz")
    ground = ground_for_capture(data)
    receipt = registry.execute(data, ground, "prefix")
    assert receipt["status"] == "EXECUTED_LOCAL"
    assert receipt["result"]["prefix_hex"] == "616263"
    with pytest.raises(ValueError):
        registry.register(OperationCapability("prefix"), lambda b,o: {})


@pytest.mark.parametrize("data", [
    b'{"x":1,"x":2}', b'{"x":NaN}', b'["under","truncated"'
])
def test_invalid_json_is_not_a_verified_json_ground(data):
    assert describe(data)["kind"] != "json"


@pytest.mark.parametrize("pointer", ["/bad", "/values/99", "/values/-1",
                                      "values", "/bad~2"])
def test_malformed_or_missing_json_path_refuses(pointer):
    rep = byte_arbitrate(request("bytes.execute", operator="json.pointer",
                  parameters={"pointer": pointer}), root=ROOT)
    assert rep["status"] == "UNRESOLVED"


def test_workspace_source_authority_and_budget(tmp_path):
    root = tmp_path/"root"
    root.mkdir()
    (root/"file.txt").write_text("safe")
    (tmp_path/"outside.txt").write_text("private")
    assert acquire("workspace:file.txt", root=root)["data"] == b"safe"
    for target in ("workspace:../outside.txt", "workspace:/etc/passwd",
                   "ftp://example.com/foo", "file:///etc/passwd"):
        with pytest.raises((ValueError, TypeError)):
            acquire(target, root=root)
    with pytest.raises(ValueError):
        acquire("workspace:file.txt", root=root, max_bytes=0)
    with pytest.raises(ValueError):
        acquire("inline:b64,!!!!", root=root)


def test_private_network_targets_are_refused_before_fetch():
    for url in ("http://example.com", "https://127.0.0.1/",
                "https://localhost:443/", "https://example.com:9999/",
                "https://example.com@127.0.0.1/", "https://bad.example/",
                "https://example.com/#bad"):
        with pytest.raises((ValueError, TypeError)):
            acquire(url, root=ROOT)


def test_https_prefix_requires_matched_content_range(monkeypatch):
    import experiments.byte_fabric.sources as source
    monkeypatch.setattr(source, "_public_host", lambda _host: None)
    class Response:
        status = 206
        code = 206
        headers = {"Content-Range": "bytes 0-3/100",
                   "Content-Type": "application/octet-stream"}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, n): return b"\0\xffhi"
    class Opener:
        def open(self, *args, **kwargs): return Response()
    monkeypatch.setattr(source.urllib.request, "build_opener", lambda *a: Opener())
    cap = read_https("https://example.com/resource", approved_hosts=DEFAULT_HOSTS,
                     max_bytes=4, byte_range=True)
    assert cap["complete"] is False
    assert cap["total_bytes"] == 100
    assert cap["transport"]["http_status"] == 206
    assert ground_for_capture(cap)["observation"]["digest_scope"] == "captured-prefix-only"
    Response.headers = {"Content-Range": "bytes 1-4/100"}
    with pytest.raises(ValueError):
        read_https("https://example.com/resource", approved_hosts=DEFAULT_HOSTS,
                   max_bytes=4, byte_range=True)
    Response.headers = {"Content-Encoding": "gzip", "Content-Range": "bytes 0-3/100"}
    with pytest.raises(ValueError):
        read_https("https://example.com/resource", approved_hosts=DEFAULT_HOSTS,
                   max_bytes=4, byte_range=True)


def test_unknown_architecture_remains_describable_and_nonexecuting():
    for raw in (b"\x12\xba\xad\xcc\x10\x00", b"\x00\x00\x01\x02\xaa",
                b"\xff"*200):
        ground = ground_for_capture(capture_bytes(raw, source="opaque"))
        assert ground["view"]["representation"] == "bytes"
        assert ground["view"]["kind"] == "opaque-bytes"
        assert "program_semantics" not in ground


def test_byte_operator_cannot_be_confused_with_hf_inference():
    compiled = compile_intention(request("bytes.execute", operator="sha256"))
    assert compiled["kind"] == "PURE_EXECUTION"
    assert compiled["execution_authority"] == "pure-local-allowlisted-operation-only"
    req = ObserverIntention("hf.model.infer", "openai-community/gpt2")
    assert compile_intention(req)["status"] == "UNRESOLVED"
