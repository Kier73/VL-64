"""Offline contract tests for the public Internet observation laboratory."""
import hashlib
import json
import socket

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
