"""Offline contract tests for the public Internet observation laboratory."""
import hashlib
import base64
import json
import socket
import struct

import pytest

from experiments.internet_lab import virtual_net as lab
from vld import GitWireV2Parser


def test_graph_navigation_requires_witness_and_existing_nodes():
    g = lab.InternetGround()
    g.node("host", "source")
    g.node("dns", "rrset")
    g.node("document", "data")
    evidence = "a" * 64
    g.link("host", "dns", "resolve", evidence)
    g.link("dns", "document", "read", "b" * 64)
    route = g.navigate("host", "document")
    assert route["hops"] == 2
    assert route["status"] == "OBSERVED_DESCRIPTION_PATH"
    assert route["scope"] == "cached_evidence_only_requires_reobservation_for_fresh_claim"
    assert route["ground_digest"] == g.fingerprint()
    with pytest.raises(ValueError):
        g.navigate("host", "not-observed")
    with pytest.raises(ValueError):
        g.link("host", "document", "fabrication", "invalid")
    with pytest.raises(ValueError):
        g.link("unknown", "document", "fabrication", evidence)
    with pytest.raises(ValueError):
        g.node("host", "different_kind")


def test_graph_fingerprint_attaches_admitted_descriptions():
    g = lab.InternetGround()
    g.node("a", "start")
    initial = g.fingerprint()
    g.node("b", "target")
    g.link("a", "b", "sample", "f" * 64)
    assert initial != g.fingerprint()
    mirror = lab.InternetGround()
    mirror.node("a", "start")
    mirror.node("b", "target")
    mirror.link("a", "b", "sample", "f" * 64)
    assert g.fingerprint() == mirror.fingerprint()


def test_network_target_allowlist_rejects_external_or_unsafe_urls():
    with pytest.raises(ValueError):
        lab.fetch("test", "http://example.com")
    with pytest.raises(ValueError):
        lab.fetch("test", "https://private.example.com")
    with pytest.raises(ValueError):
        lab.fetch("test", "https://user:pass@example.com/")
    with pytest.raises(ValueError):
        lab.fetch("test", "https://example.com:9999/")
    with pytest.raises(ValueError):
        lab.os_dns("localhost")


def test_git_packet_frame_can_be_interpreted_without_network():
    raw = (GitWireV2Parser.encode_pkt_line("version 2\n") +
           GitWireV2Parser.encode_pkt_line("ls-refs\n") +
           GitWireV2Parser.encode_pkt_line(None))
    frames = GitWireV2Parser.parse_pkt_lines(raw)
    assert frames == [b"version 2\n", b"ls-refs\n", b""]
    assert frames[0].startswith(b"version 2")


def test_independent_kernel_tcp_path_and_endpoint_semantics():
    observation = lab.loopback_transform()
    assert observation["status"] == "VERIFIED", observation
    assert observation["endpoint_semantic_check"]
    assert observation["input_bytes"] == 1024
    assert observation["output_bytes"] == 32
    assert observation["request_frame_bytes"] == 1028
    assert observation["input_sha256"] == observation["wire_endpoint_received_digest"]


def test_github_navigation_with_pinned_mocked_observations(monkeypatch):
    revision = "1" * 40
    called = []

    def fake_fetch(operation, url, **options):
        called.append((operation, url))
        if operation == "github_repository_metadata":
            payload = {"full_name": "Kier73/VL-64", "default_branch": "main"}
        elif operation == "github_commit_metadata":
            payload = {"sha": revision}
        elif operation == "github_pinned_readme":
            payload = b"# VL-64\n\n[Research goals](docs/RESEARCH_GOALS.md)\n"
        elif operation == "github_pinned_research_goals":
            payload = b"# VL-64 Research Goals\n"
        else:
            raise AssertionError("Unexpected network request " + operation)
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf8")
        record = {
            "operation": operation, "status": "OBSERVED",
            "http_status": 200, "body_sha256": hashlib.sha256(body).hexdigest(),
        }
        return record, body

    monkeypatch.setattr(lab, "fetch", fake_fetch)
    g = lab.InternetGround()
    receipts, result, _ = lab.github_readme_to_goals(g)
    assert result["status"] == "OBSERVED_DESCRIPTION_PATH"
    assert result["hops"] == 3
    assert result["goal_heading_valid"]
    assert result["commit"] == revision
    assert len(receipts) == 4
    assert called[-2][1].endswith(f"/{revision}/README.md")
    assert called[-1][1].endswith(
        f"/{revision}/docs/RESEARCH_GOALS.md")


def test_repository_navigation_refuses_absent_goal_link(monkeypatch):
    revision = "2" * 40

    def fake_fetch(operation, url, **kwargs):
        values = {
            "github_repository_metadata": {"full_name": "Kier73/VL-64",
                                           "default_branch": "main"},
            "github_commit_metadata": {"sha": revision},
            "github_pinned_readme": b"# No explicit target document\n",
        }
        source = values[operation]
        body = source if isinstance(source, bytes) else json.dumps(source).encode()
        return ({"status": "OBSERVED", "http_status": 200,
                 "body_sha256": lab.digest(body)}, body)

    monkeypatch.setattr(lab, "fetch", fake_fetch)
    g = lab.InternetGround()
    records, result, _ = lab.github_readme_to_goals(g)
    assert result == {"status": "UNRESOLVED", "reason": "requested_link_absent"}
    assert len(records) == 3
    assert not g.edges


def test_dns_wire_parse_and_negative_controls():
    question = b"\x07example\x03com\x00" + struct.pack("!HH", 1, 1)
    head = struct.pack("!HHHHHH", 0, 0x8180, 1, 1, 0, 0)
    answer = b"\xc0\x0c" + struct.pack("!HHIH", 1, 1, 120, 4)
    packet = head + question + answer + b"\x5d\xb8\xd7\x0e"
    output = lab.decode_dns_a(packet)
    assert output["dns_rcode"] == 0
    assert output["answer_count"] == 1
    assert output["ttl_min"] == 120
    assert output["dns_header_answer_count"] == 1
    with pytest.raises(ValueError):
        lab.decode_dns_a(packet[:4])
    with pytest.raises(ValueError):
        lab.decode_dns_a(packet[:-1])
    with pytest.raises(ValueError):
        lab.decode_dns_a(struct.pack("!HHHHHH", 1, 0x8180, 1, 1, 0, 0) +
                         packet[12:])


def test_git_smart_http_advertisement_includes_service_preamble(monkeypatch):
    from vld import GitWireV2Parser as Parser
    payload = (Parser.encode_pkt_line("# service=git-upload-pack\n")
               + Parser.encode_pkt_line(None)
               + Parser.encode_pkt_line("version 2\n")
               + Parser.encode_pkt_line("ls-refs=unborn\n")
               + Parser.encode_pkt_line("fetch=shallow\n")
               + Parser.encode_pkt_line(None))
    def fake_fetch(operation, url, **kwargs):
        return ({"status": "OBSERVED", "http_status": 200,
                 "body_sha256": lab.digest(payload)}, payload)
    monkeypatch.setattr(lab, "fetch", fake_fetch)
    record = lab.git_advertisement()
    assert record["interpretation"] == "git-protocol-v2-capability-advertisement"
    assert record["smart_http_service_preamble"]
    assert "ls-refs" in record["capabilities"]
    assert "fetch" in record["capabilities"]


def test_git_ls_refs_sends_bounded_read_only_request(monkeypatch):
    head = "c" * 40
    raw = (GitWireV2Parser.encode_pkt_line(head + " HEAD symref-target:refs/heads/main\n")
           + GitWireV2Parser.encode_pkt_line(None))
    def fake_fetch(operation, url, **kwargs):
        assert operation == "git_v2_ls_refs"
        assert url.endswith("/Kier73/VL-64/git-upload-pack")
        assert kwargs["method"] == "POST"
        payload = kwargs["payload"]
        assert b"command=ls-refs" in payload
        assert b"ref-prefix HEAD" in payload
        assert b"command=fetch" not in payload
        return ({"status": "OBSERVED", "http_status": 200,
                 "body_sha256": lab.digest(raw)}, raw)
    monkeypatch.setattr(lab, "fetch", fake_fetch)
    rec = lab.git_read_only_ls_refs()
    assert rec["head_commit"] == head
    assert rec["interpretation"] == "git-v2-ls-refs"
    assert rec["request_packet_bytes"] > 0


def test_non_ls_refs_post_is_refused_before_network():
    with pytest.raises(ValueError):
        lab.fetch("arbitrary", "https://github.com/Kier73/VL-64/git-upload-pack",
                  method="POST", payload=b"hello")
    with pytest.raises(ValueError):
        lab.fetch("git_v2_ls_refs", "https://api.github.com/repos/Kier73/VL-64",
                  method="POST", payload=b"hello")


def test_offline_dataset_packet_verification_and_adversarial_edits():
    raw = (struct.pack("!HHHHHH", 0, 0x8180, 1, 1, 0, 0)
           + b"\x07example\x03com\x00"
           + struct.pack("!HH", 1, 1)
           + b"\xc0\x0c"
           + struct.pack("!HHIH", 1, 1, 30, 4)
           + b"\x5d\xb8\xd7\x0e")
    decoded = lab.decode_dns_a(raw)
    ground = lab.InternetGround()
    ground.node("query", "dns_wire_query")
    ground.node("answer", "dns_snapshot")
    ground.link("query", "answer", "doh", lab.digest(raw))
    report = {
        "schema": lab.SCHEMA,
        "observations": [{
            "operation": "cloudflare_doh_wire_A_example.com",
            "raw_packet_b64": base64.b64encode(raw).decode("ascii"),
            "body_sha256": lab.digest(raw),
            "answer_set_sha256": decoded["answer_set_sha256"],
        }],
        "graph": {"nodes": ground.nodes, "edges": ground.edges,
                  "fingerprint": ground.fingerprint()},
        "navigation": {"status": "UNRESOLVED"},
    }
    def resign():
        report["dataset_digest"] = lab.domain_digest(
            "vl64.internet-dataset.v1",
            {k: v for k, v in report.items() if k != "dataset_digest"})
    resign()
    assert lab.verify_dataset(report)["packet_replays"] == 1
    report["observations"][0]["raw_packet_b64"] = base64.b64encode(
        raw[:-1] + b"\x11").decode()
    assert lab.verify_dataset(report)["status"] == "REJECTED"
    resign()  # Even with a newly computed report digest, raw witness is wrong.
    assert lab.verify_dataset(report)["status"] == "REJECTED"
    report["observations"][0]["raw_packet_b64"] = "not:::base64"
    resign()
    assert lab.verify_dataset(report)["status"] == "REJECTED"
