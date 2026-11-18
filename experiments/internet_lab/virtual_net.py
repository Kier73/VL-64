"""Internet-as-observed-computation lab for VL-64 (outside the 64 KiB core).

Read-only, bounded public DNS/HTTPS/Git probes plus a controlled TCP
loopback experiment. Records verifiable data, not arbitrary external authority.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import datetime
import hashlib
import ipaddress
import json
import os
import platform
import re
import socket
import ssl
import struct
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from pathlib import Path

from vld import GitWireV2Parser, domain_digest

SCHEMA = "vl64.internet-ground-observation.v1"
MAX_BODY = 65536
HOSTS = frozenset({
    "api.github.com", "raw.githubusercontent.com", "github.com",
    "cloudflare-dns.com", "www.cloudflare.com", "example.com",
})
DOMAINS = ("example.com", "github.com", "cloudflare.com")
HEADERS = {"User-Agent": "VL64-Research-Probe/0.1",
           "Accept": "application/json, */*;q=0.5"}
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
GOAL_LINK = re.compile(r"\[Research goals\]\((docs/RESEARCH_GOALS\.md)\)", re.I)


def digest(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def refusal(label, exception):
    return {"status": "UNRESOLVED", "failure_kind": type(exception).__name__,
            "failure_detail": str(exception)[:150], "operation": label}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """A redirected response is evidence, not permission to visit another host."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def fetch(label: str, url: str, *, headers=None, max_body=MAX_BODY,
          method="GET", payload=None):
    """Bounded read-only HTTPS request, including a pinned Git ls-refs POST."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname not in HOSTS:
        raise ValueError("URL outside explicit read-only HTTPS allowlist")
    if parsed.username or parsed.password or parsed.port not in (None, 443):
        raise ValueError("Unexpected URL credentials or port")
    if method != "GET":
        if not (method == "POST" and label == "git_v2_ls_refs" and
                parsed.hostname == "github.com" and
                parsed.path == "/Kier73/VL-64/git-upload-pack" and
                not parsed.query and isinstance(payload, bytes) and
                len(payload) <= 2048):
            raise ValueError("POST permitted only for bounded read-only Git ls-refs")
    elif payload is not None:
        raise ValueError("GET cannot supply request payload")
    request = urllib.request.Request(
        url, data=payload, headers={**HEADERS, **(headers or {})},
        method=method)
    began = time.perf_counter_ns()
    try:
        response = urllib.request.build_opener(NoRedirect).open(request, timeout=8)
    except urllib.error.HTTPError as exc:
        response = exc
    except (OSError, TimeoutError, ValueError) as exc:
        return refusal(label, exc), b""
    try:
        with response:
            content = response.read(max_body + 1)
            fields = response.headers
            rec = {
                "operation": label, "status": "OBSERVED",
                "url": url, "method": method,
                "http_status": response.status if hasattr(response, "status") else response.code,
                "response_bytes": min(len(content), max_body),
                "body_sha256": digest(content[:max_body]),
                "truncated": len(content) > max_body,
                "elapsed_ms": round((time.perf_counter_ns() - began) / 1e6, 3),
                "content_type": (fields.get("Content-Type") or "")[:100],
                "etag": (fields.get("ETag") or "")[:120],
                "cache_control": (fields.get("Cache-Control") or "")[:120],
                "cf_ray_present": bool(fields.get("CF-Ray")),
                "github_request_id_present": bool(fields.get("X-GitHub-Request-Id")),
            }
            return rec, content[:max_body]
    except (TimeoutError, OSError, ValueError) as exc:
        return refusal(label, exc), b""


def os_dns(domain: str) -> dict:
    if domain not in DOMAINS:
        raise ValueError("Unapproved DNS domain")
    began = time.perf_counter_ns()
    try:
        answers = socket.getaddrinfo(domain, 443, type=socket.SOCK_STREAM)
        addresses = sorted({str(ipaddress.ip_address(a[4][0])) for a in answers})
        return {
            "operation": "os_dns:" + domain, "status": "OBSERVED",
            "hostname": domain, "address_count": len(addresses),
            "address_set_sha256": digest(json.dumps(addresses).encode("utf-8")),
            "families": sorted({a[0] for a in answers}),
            "elapsed_ms": round((time.perf_counter_ns()-began)/1e6, 3),
            "scope": "os_getaddrinfo_snapshot_no_dns_ttl_or_packet_proof",
        }
    except (socket.gaierror, OSError) as exc:
        return refusal("os_dns:" + domain, exc)


def doh_cloudflare():
    url = "https://cloudflare-dns.com/dns-query?name=example.com&type=A"
    rec, body = fetch(
        "cloudflare_doh_A_example.com", url,
        headers={"Accept": "application/dns-json"}, max_body=16384)
    if rec["status"] != "OBSERVED" or rec["http_status"] != 200:
        return rec
    try:
        payload = json.loads(body)
        if type(payload) is not dict or type(payload.get("Status")) is not int:
            raise ValueError("Unrecognized DoH JSON response")
        records = [x for x in payload.get("Answer", []) if
                   isinstance(x, dict) and x.get("type") == 1 and
                   x.get("name", "").rstrip(".").lower() == "example.com"]
        addresses = sorted({x["data"] for x in records
                            if isinstance(x.get("data"), str) and
                            ipaddress.ip_address(x["data"]).version == 4})
        rec.update({
            "dns_rcode": payload["Status"], "answer_count": len(addresses),
            "answer_set_sha256": digest(json.dumps(addresses).encode("utf-8")),
            "min_ttl": min((x.get("TTL", 0) for x in records), default=None),
            "max_ttl": max((x.get("TTL", 0) for x in records), default=None),
            "scope": "dns_https_json_answer_from_public_resolver",
        })
    except (ValueError, KeyError, TypeError) as exc:
        rec["interpretation"] = refusal("doh_json_decode", exc)
    return rec


def _skip_dns_name(raw, offset):
    """Skip a DNS name with bounded labels or a compression pointer."""
    while True:
        if offset >= len(raw):
            raise ValueError("Truncated DNS name")
        n = raw[offset]
        if n & 0xC0 == 0xC0:
            if offset + 1 >= len(raw):
                raise ValueError("Truncated DNS pointer")
            return offset + 2
        if n & 0xC0 or n > 63:
            raise ValueError("Invalid DNS label")
        offset += 1
        if n == 0:
            return offset
        offset += n
        if offset > len(raw):
            raise ValueError("Truncated DNS label")


def decode_dns_a(raw):
    """Decode a bounded DNS message, preserving RCODE and A-answer evidence."""
    if len(raw) < 12:
        raise ValueError("DNS header incomplete")
    ident, flags, qd, an, ns, ar = struct.unpack_from("!HHHHHH", raw)
    if not (flags & 0x8000) or ident != 0 or qd != 1 or an > 128:
        raise ValueError("Unexpected DNS response identity or counts")
    offset = _skip_dns_name(raw, 12)
    if offset + 4 > len(raw):
        raise ValueError("Truncated DNS question")
    qtype, qclass = struct.unpack_from("!HH", raw, offset)
    if (qtype, qclass) != (1, 1):
        raise ValueError("Unexpected DNS query type")
    offset += 4
    addresses, ttls = set(), []
    for _ in range(an):
        offset = _skip_dns_name(raw, offset)
        if offset + 10 > len(raw):
            raise ValueError("Truncated DNS answer")
        kind, cls, ttl, length = struct.unpack_from("!HHIH", raw, offset)
        offset += 10
        if offset + length > len(raw):
            raise ValueError("Truncated DNS answer data")
        if (kind, cls, length) == (1, 1, 4):
            addresses.add(str(ipaddress.IPv4Address(raw[offset:offset+4])))
            ttls.append(ttl)
        offset += length
    return {
        "dns_rcode": flags & 15,
        "answer_count": len(addresses),
        "answer_set_sha256": digest(json.dumps(sorted(addresses)).encode("utf-8")),
        "ttl_min": min(ttls) if ttls else None,
        "ttl_max": max(ttls) if ttls else None,
        "dns_wire_response_bytes": len(raw),
        "dns_header_answer_count": an,
        "dns_extra_sections": ns + ar,
    }


def doh_wire_cloudflare():
    """Encode an actual DNS wire-format A query carried over HTTPS."""
    name = b"\x07example\x03com\x00"
    raw_query = struct.pack("!HHHHHH", 0, 0x0100, 1, 0, 0, 0)
    raw_query += name + struct.pack("!HH", 1, 1)
    encoded = base64.urlsafe_b64encode(raw_query).rstrip(b"=").decode("ascii")
    rec, raw = fetch(
        "cloudflare_doh_wire_A_example.com",
        "https://cloudflare-dns.com/dns-query?dns=" + encoded,
        headers={"Accept": "application/dns-message"}, max_body=16384,
    )
    rec["query_packet_sha256"] = digest(raw_query)
    rec["query_packet_bytes"] = len(raw_query)
    if rec["status"] != "OBSERVED" or rec["http_status"] != 200:
        return rec
    try:
        rec.update(decode_dns_a(raw))
        rec["raw_packet_b64"] = base64.b64encode(raw).decode("ascii")
        rec["interpretation"] = "dns-wire-response-a"
        rec["scope"] = "verified_dns_wire_decode_of_https_response_not_dnssec"
    except (ValueError, struct.error) as exc:
        rec["interpretation"] = refusal("dns_wire_decode", exc)
    return rec


def tls_handshake():
    """A direct TCP + TLS attempt, separate from HTTPS requests using proxies."""
    started = time.perf_counter_ns()
    try:
        with socket.create_connection(("example.com", 443), timeout=5) as sock:
            sock.settimeout(5)
            with ssl.create_default_context().wrap_socket(
                    sock, server_hostname="example.com") as secured:
                cert = secured.getpeercert(binary_form=True)
                return {
                    "operation": "tls_direct:example.com", "status": "OBSERVED",
                    "tls_version": secured.version(),
                    "cipher": secured.cipher()[0],
                    "certificate_sha256": digest(cert),
                    "elapsed_ms": round((time.perf_counter_ns()-started)/1e6, 3),
                    "scope": "direct_socket_tls_handshake_not_route_attestation",
                }
    except (TimeoutError, OSError, ssl.SSLError) as exc:
        return refusal("tls_direct:example.com", exc)


def git_advertisement():
    url = ("https://github.com/Kier73/VL-64/info/refs"
           "?service=git-upload-pack")
    rec, raw = fetch("git_v2_capabilities", url, headers={
        "Git-Protocol": "version=2",
        "Accept": "application/x-git-upload-pack-advertisement",
    }, max_body=32768)
    if rec["status"] != "OBSERVED" or rec["http_status"] != 200:
        return rec
    try:
        frames = GitWireV2Parser.parse_pkt_lines(raw)
        version_at = next((i for i, frame in enumerate(frames)
                           if frame.strip() == b"version 2"), None)
        if version_at is None:
            raise ValueError("No Git protocol v2 advertisement")
        # Smart HTTP may include "# service=git-upload-pack" and flush
        # before the version line; these are protocol framing, not errors.
        caps = [frame.decode("ascii").strip() for frame in frames[version_at + 1:]
                if frame and len(frame) < 256]
        capabilities = sorted({item.split("=")[0] for item in caps})
        rec.update({
            "interpretation": "git-protocol-v2-capability-advertisement",
            "raw_packet_b64": base64.b64encode(raw).decode("ascii"),
            "frames": len(frames),
            "smart_http_service_preamble": any(
                frame.startswith(b"# service=git-upload-pack") for frame in frames),
            "capabilities": capabilities,
            "scope": "remote_protocol_advertisement_not_local_git_execution",
        })
    except (UnicodeError, ValueError) as exc:
        rec["interpretation"] = refusal("git_pktline_decode", exc)
    return rec


def git_read_only_ls_refs():
    """Issue a real Git upload-pack v2 ls-refs command, no fetch or push."""
    body = (GitWireV2Parser.encode_pkt_line("command=ls-refs\n")
            + b"0001"
            + GitWireV2Parser.encode_pkt_line("symrefs\n")
            + GitWireV2Parser.encode_pkt_line("ref-prefix HEAD\n")
            + b"0000")
    rec, raw = fetch(
        "git_v2_ls_refs",
        "https://github.com/Kier73/VL-64/git-upload-pack",
        headers={
            "Git-Protocol": "version=2",
            "Content-Type": "application/x-git-upload-pack-request",
            "Accept": "application/x-git-upload-pack-result",
        },
        method="POST", payload=body, max_body=16384,
    )
    rec["request_packet_bytes"] = len(body)
    rec["request_packet_sha256"] = digest(body)
    if rec["status"] != "OBSERVED" or rec["http_status"] != 200:
        return rec
    try:
        frames = GitWireV2Parser.parse_pkt_lines(raw)
        head = next((line for line in frames
                     if re.match(rb"^[0-9a-f]{40} HEAD(?: |$)", line)), None)
        if head is None:
            raise ValueError("Missing HEAD in Git ls-refs response")
        sha = head[:40].decode("ascii")
        rec.update({
            "interpretation": "git-v2-ls-refs",
            "raw_packet_b64": base64.b64encode(raw).decode("ascii"),
            "head_commit": sha, "response_frames": len(frames),
            "scope": "read_only_git_protocol_response_not_host_authority",
        })
    except (ValueError, UnicodeError) as exc:
        rec["interpretation"] = refusal("git_ls_refs_decode", exc)
    return rec


def _recv_all(sock, n):
    parts = []
    while n:
        block = sock.recv(n)
        if not block:
            raise EOFError("Socket closed during framed receive")
        parts.append(block)
        n -= len(block)
    return b"".join(parts)


def loopback_transform():
    """A real loopback TCP stream carrying framed data to a distinct worker."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(5)
    port = listener.getsockname()[1]
    payload = bytes((i * 17 + 9) % 256 for i in range(1024))
    result = {}
    error = []
    def serve():
        try:
            conn, _ = listener.accept()
            with conn:
                conn.settimeout(5)
                size, = struct.unpack("!I", _recv_all(conn, 4))
                if size > 4096:
                    raise ValueError("Frame too large")
                received = _recv_all(conn, size)
                transformed = hashlib.sha256(b"vl64.endpoint.v1:" + received).digest()
                conn.sendall(struct.pack("!I", len(transformed)) + transformed)
                result["rx_sha256"] = digest(received)
                result["transformed_sha256"] = digest(transformed)
        except (OSError, EOFError, ValueError) as exc:
            error.append(str(exc))
        finally:
            listener.close()
    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    begun = time.perf_counter_ns()
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
            sock.settimeout(5)
            sock.sendall(struct.pack("!I", len(payload)) + payload)
            size, = struct.unpack("!I", _recv_all(sock, 4))
            output = _recv_all(sock, size)
        thread.join(timeout=5)
        expected = hashlib.sha256(b"vl64.endpoint.v1:" + payload).digest()
        accepted = (output == expected and not error and
                    result.get("rx_sha256") == digest(payload))
        return {
            "operation": "tcp_loopback_endpoint_transform",
            "status": "VERIFIED" if accepted else "REJECTED",
            "input_bytes": len(payload), "output_bytes": len(output),
            "request_frame_bytes": len(payload) + 4,
            "response_frame_bytes": len(output) + 4,
            "input_sha256": digest(payload),
            "output_sha256": digest(output),
            "wire_endpoint_received_digest": result.get("rx_sha256"),
            "endpoint_semantic_check": accepted,
            "elapsed_ms": round((time.perf_counter_ns()-begun)/1e6, 3),
            "scope": "actual_local_tcp_endpoint_operation_not_in_flight_transformation",
        }
    except (OSError, EOFError, ValueError) as exc:
        return refusal("tcp_loopback_endpoint_transform", exc)
    finally:
        if thread.is_alive():
            thread.join(timeout=5)


class InternetGround:
    """Observed representation relations, with read-only cached-path navigation.

    Node/edge identities refer to captured evidence. A route through this
    graph is an existing *description*, never a live request or authority.
    """
    def __init__(self):
        self.nodes = {}
        self.edges = []

    def node(self, name: str, kind: str):
        if not name or not kind:
            raise ValueError("Nodes require nonempty names and kinds")
        if name in self.nodes and self.nodes[name] != kind:
            raise ValueError("Node identity reused with different kind")
        self.nodes[name] = kind
        return name

    def link(self, source: str, target: str, relation: str, witness: str):
        if source not in self.nodes or target not in self.nodes:
            raise ValueError("Unregistered relation endpoint")
        if not re.fullmatch(r"[0-9a-f]{64}", witness):
            raise ValueError("Relation witness must be a SHA-256 digest")
        edge = {"from": source, "to": target, "relation": relation,
                "witness_sha256": witness}
        if edge not in self.edges:
            self.edges.append(edge)

    def navigate(self, start: str, destination: str):
        if start not in self.nodes or destination not in self.nodes:
            raise ValueError("Unobserved node")
        queue = deque([(start, [])])
        visited = {start}
        while queue:
            name, path = queue.popleft()
            if name == destination:
                return {
                    "status": "OBSERVED_DESCRIPTION_PATH",
                    "ground_digest": self.fingerprint(),
                    "hops": len(path), "path": path,
                    "scope": "cached_evidence_only_requires_reobservation_for_fresh_claim",
                }
            for edge in self.edges:
                if edge["from"] == name and edge["to"] not in visited:
                    visited.add(edge["to"])
                    queue.append((edge["to"], path + [edge]))
        raise ValueError("No observed description path to requested destination")

    def fingerprint(self):
        return domain_digest("vl64.internet-ground.v1", {
            "nodes": self.nodes, "edges": self.edges,
        })


def github_readme_to_goals(graph: InternetGround):
    """Live navigation: repo metadata -> pinned commit -> README link -> goal."""
    records = []
    repo_url = "https://api.github.com/repos/Kier73/VL-64"
    meta, data = fetch("github_repository_metadata", repo_url)
    records.append(meta)
    if meta["status"] != "OBSERVED" or meta["http_status"] != 200:
        return records, {"status": "UNRESOLVED", "reason": "metadata"}, None
    try:
        parsed = json.loads(data)
        if parsed.get("full_name") != "Kier73/VL-64":
            raise ValueError("Repository identity mismatch")
        branch = parsed["default_branch"]
        if not re.fullmatch(r"[A-Za-z0-9._/-]{1,120}", branch):
            raise ValueError("Unsafe branch name")
    except (ValueError, KeyError, TypeError) as exc:
        return records, refusal("github_repo_parse", exc), None
    commit_url = ("https://api.github.com/repos/Kier73/VL-64/commits/" +
                  urllib.parse.quote(branch, safe=""))
    commit_rec, raw_commit = fetch("github_commit_metadata", commit_url)
    records.append(commit_rec)
    if commit_rec["status"] != "OBSERVED" or commit_rec["http_status"] != 200:
        return records, {"status": "UNRESOLVED", "reason": "commit"}, None
    try:
        rev = json.loads(raw_commit)["sha"]
        if not HEX40.fullmatch(rev):
            raise ValueError("Commit not SHA-1 object ID")
    except (ValueError, KeyError, TypeError) as exc:
        return records, refusal("github_commit_parse", exc), None

    base = "https://raw.githubusercontent.com/Kier73/VL-64/" + rev + "/"
    readme, raw = fetch("github_pinned_readme", base + "README.md")
    records.append(readme)
    if readme["status"] != "OBSERVED" or readme["http_status"] != 200:
        return records, {"status": "UNRESOLVED", "reason": "readme"}, None
    match = GOAL_LINK.search(raw.decode("utf-8", errors="replace"))
    if not match:
        return records, {"status": "UNRESOLVED", "reason": "requested_link_absent"}, None
    goals, content = fetch("github_pinned_research_goals", base + match.group(1))
    records.append(goals)
    if goals["status"] != "OBSERVED" or goals["http_status"] != 200:
        return records, {"status": "UNRESOLVED", "reason": "goals"}, None
    heading_valid = content.startswith(b"# VL-64 Research Goals")
    a = graph.node("repo:Kier73/VL-64", "github_repository")
    b = graph.node("git-commit:" + rev, "git_commit")
    c = graph.node("readme:" + digest(raw), "github_raw_document")
    d = graph.node("research-goals:" + digest(content), "github_raw_document")
    graph.link(a, b, "resolve_default_branch_commit", commit_rec["body_sha256"])
    graph.link(b, c, "read_pinned_readme", readme["body_sha256"])
    graph.link(c, d, "navigate_markdown_link", goals["body_sha256"])
    result = graph.navigate(a, d)
    result.update({
        "commit": rev, "readme_sha256": digest(raw),
        "goals_sha256": digest(content), "goal_heading_valid": heading_valid,
        "branch": branch, "goal_bytes": len(content),
        "claim": "observed_remote_document_navigation_not_packet_native_execution",
    })
    return records, result, (rev, raw, content)


def run():
    observations = []
    graph = InternetGround()
    for domain in DOMAINS:
        dns = os_dns(domain)
        observations.append(dns)
        if dns["status"] == "OBSERVED":
            a = graph.node("dns-name:" + domain, "dns_query")
            b = graph.node("dns-answers:" + dns["address_set_sha256"], "dns_snapshot")
            graph.link(a, b, "os_resolve", dns["address_set_sha256"])
    doh = doh_cloudflare()
    observations.append(doh)
    wire = doh_wire_cloudflare()
    observations.append(wire)
    if doh["status"] == "OBSERVED" and "answer_set_sha256" in doh:
        a = graph.node("doh:example.com:A", "dns_over_https_query")
        b = graph.node("dns-answers:" + doh["answer_set_sha256"], "dns_snapshot")
        graph.link(a, b, "cloudflare_doh_answer", doh["body_sha256"])
    if wire.get("interpretation") == "dns-wire-response-a":
        a = graph.node("doh-wire:example.com:A", "dns_wire_query")
        b = graph.node("dns-answers:" + wire["answer_set_sha256"], "dns_snapshot")
        graph.link(a, b, "cloudflare_doh_dns_packet", wire["body_sha256"])
    for probe in (tls_handshake, git_advertisement, git_read_only_ls_refs,
                  loopback_transform):
        observations.append(probe())
    cf, _ = fetch("cloudflare_homepage", "https://www.cloudflare.com/",
                  max_body=16384)
    observations.append(cf)
    example, _ = fetch("example_com_homepage", "https://example.com/",
                       max_body=16384)
    observations.append(example)
    records, repo_nav, _ = github_readme_to_goals(graph)
    observations.extend(records)
    git_heads = [x["head_commit"] for x in observations
                 if x.get("interpretation") == "git-v2-ls-refs"]
    if repo_nav.get("commit") and git_heads:
        repo_nav["git_wire_rest_head_match"] = repo_nav["commit"] in git_heads

    # A cached graph route is not an authenticated claim of a current result.
    success = [x for x in observations if
               x.get("status") in ("OBSERVED", "VERIFIED")]
    summary = {
        "observations": len(observations),
        "successful_observations": len(success),
        "unresolved_or_rejected": len(observations) - len(success),
        "nodes": len(graph.nodes), "relations": len(graph.edges),
        "github_navigation": repo_nav.get("status"),
        "github_navigation_hops": repo_nav.get("hops"),
        "git_wire_rest_head_match": repo_nav.get("git_wire_rest_head_match"),
        "cloudflare_doh": doh.get("status"),
        "cloudflare_dns_wire": wire.get("interpretation", wire.get("status")),
        "nested_decoding_refusals": sum(
            isinstance(x.get("interpretation"), dict) and
            x["interpretation"].get("status") == "UNRESOLVED"
            for x in observations),
        "loopback": next((x["status"] for x in observations if
                          x["operation"] == "tcp_loopback_endpoint_transform"), "UNRESOLVED"),
    }
    return {
        "schema": SCHEMA,
        "captured_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "runner": {"python": platform.python_version(), "os": platform.system(),
                   "arch": platform.machine(),
                   "git_revision": os.environ.get("GITHUB_SHA", "")[:40]},
        "probe_policy": {
            "bounded_http_body_bytes": MAX_BODY,
            "target_hosts": sorted(HOSTS),
            "dns_names": list(DOMAINS),
            "operations": "public_read_only_plus_local_loopback",
            "max_http_timeout_seconds": 8,
            "no_credential_use": True, "redirect_following": False,
        },
        "observations": observations, "graph": {
            "fingerprint": graph.fingerprint(),
            "nodes": graph.nodes, "edges": graph.edges,
        }, "navigation": repo_nav, "summary": summary,
    }


def verify_dataset(result):
    """Replay retained protocol packets and relation evidence without Internet."""
    try:
        claimed_digest = result["dataset_digest"]
        unsigned = {k: v for k, v in result.items() if k != "dataset_digest"}
        if domain_digest("vl64.internet-dataset.v1", unsigned) != claimed_digest:
            raise ValueError("Dataset content digest mismatch")
        graph = InternetGround()
        for name, kind in result["graph"]["nodes"].items():
            graph.node(name, kind)
        for edge in result["graph"]["edges"]:
            graph.link(edge["from"], edge["to"], edge["relation"],
                       edge["witness_sha256"])
        if graph.fingerprint() != result["graph"]["fingerprint"]:
            raise ValueError("Graph fingerprint mismatch")
        verified_packets = 0
        for item in result["observations"]:
            if "raw_packet_b64" not in item:
                continue
            raw = base64.b64decode(item["raw_packet_b64"], validate=True)
            if digest(raw) != item["body_sha256"]:
                raise ValueError("Raw protocol packet digest mismatch")
            if item["operation"] == "cloudflare_doh_wire_A_example.com":
                if decode_dns_a(raw)["answer_set_sha256"] != item["answer_set_sha256"]:
                    raise ValueError("DNS answer interpretation mismatch")
            elif item["operation"] == "git_v2_capabilities":
                frames = GitWireV2Parser.parse_pkt_lines(raw)
                if not any(line.strip() == b"version 2" for line in frames):
                    raise ValueError("Git v2 capability interpretation mismatch")
            elif item["operation"] == "git_v2_ls_refs":
                frames = GitWireV2Parser.parse_pkt_lines(raw)
                if not any(line.startswith(item["head_commit"].encode() + b" HEAD")
                           for line in frames):
                    raise ValueError("Git head interpretation mismatch")
            else:
                raise ValueError("Unexpected retained packet operation")
            verified_packets += 1
        nav = result.get("navigation", {})
        if nav.get("status") == "OBSERVED_DESCRIPTION_PATH":
            repo = "repo:Kier73/VL-64"
            goal = "research-goals:" + nav["goals_sha256"]
            candidate = graph.navigate(repo, goal)
            if candidate["path"] != nav["path"]:
                raise ValueError("Recorded navigation path mismatch")
        return {"status": "VERIFIED", "packet_replays": verified_packets,
                "graph_relations": len(graph.edges),
                "scope": "dataset_integrity_and_retained_protocol_bytes_only"}
    except (ValueError, KeyError, TypeError, IndexError, UnicodeError,
            binascii.Error) as exc:
        return {"status": "REJECTED", "reason": str(exc)[:160]}


def main():
    parser = argparse.ArgumentParser(description="Observe public internet relations as VL-64 Grounds")
    parser.add_argument("--out", default="internet-ground-dataset.json")
    parser.add_argument("--verify-file", help="Offline replay of a saved dataset")
    args = parser.parse_args()
    if args.verify_file:
        saved = json.loads(Path(args.verify_file).read_text(encoding="utf-8"))
        verdict = verify_dataset(saved)
        print("VL64_INTERNET_OFFLINE_VERIFY " + json.dumps(verdict, sort_keys=True))
        if verdict["status"] != "VERIFIED":
            raise SystemExit(1)
        return
    result = run()
    result["dataset_digest"] = domain_digest("vl64.internet-dataset.v1", result)
    check = verify_dataset(result)
    if check["status"] != "VERIFIED":
        raise SystemExit("Failed to independently replay captured packet evidence")
    serialized = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(serialized, encoding="utf-8")
    print("VL64_INTERNET_SUMMARY " + json.dumps(result["summary"], sort_keys=True))
    print("VL64_INTERNET_NAVIGATION " + json.dumps(result["navigation"], sort_keys=True))
    print("VL64_INTERNET_RECORDS " + json.dumps(result["observations"], sort_keys=True))
    print("VL64_INTERNET_GROUND_SHA256 " + result["graph"]["fingerprint"])
    print("VL64_INTERNET_DATASET_SHA256 " + result["dataset_digest"])
    print("VL64_INTERNET_OFFLINE_VERIFY " + json.dumps(check, sort_keys=True))
    # A successful lab run includes honest unresolved probes. Fail only on loss
    # of all observation capability or failure of the independent TCP reference.
    if not result["summary"]["successful_observations"]:
        raise SystemExit("No observations were acquired")
    if result["summary"]["loopback"] != "VERIFIED":
        raise SystemExit("Independent TCP semantic reference failed")


if __name__ == "__main__":
    main()
