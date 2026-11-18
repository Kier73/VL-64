"""Source adapters for bytes regardless of their meaning.

Input sources do not grant execution authority. Public HTTPS targets require
explicit hostname allowlisting; redirects and unsafe destinations are refused.
"""
from __future__ import annotations

import base64
import ipaddress
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .ground import sha256

MAX_CAPTURE = 65536
DEFAULT_HOSTS = frozenset({
    "example.com", "www.cloudflare.com", "raw.githubusercontent.com",
    "github.com", "api.github.com", "huggingface.co",
})


def capture_bytes(data: bytes, source="memory", *, complete=True,
                  total_bytes=None, transport=None):
    if not isinstance(data, bytes):
        raise TypeError("Source must supply bytes")
    if type(complete) is not bool:
        raise TypeError("Completeness must be explicit")
    if total_bytes is not None and (
            type(total_bytes) is not int or total_bytes < len(data)):
        raise ValueError("Invalid declared total")
    if complete and total_bytes not in (None, len(data)):
        raise ValueError("Complete resource cannot have unmatched total size")
    return {"data": data, "complete": complete, "source": source,
            "total_bytes": len(data) if complete else total_bytes,
            "transport": transport or {"kind": "memory"}}


def read_file(target: str, *, root: Path, max_bytes: int):
    if not target.startswith("workspace:"):
        raise ValueError("Not a workspace target")
    suffix = target[len("workspace:"):]
    if not suffix or suffix.startswith(("/", "\\")) or "\x00" in suffix:
        raise ValueError("Invalid workspace target")
    root = Path(root).resolve()
    path = (root / suffix).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("File outside approved workspace or absent")
    size = path.stat().st_size
    with path.open("rb") as file:
        data = file.read(max_bytes + 1)
    return capture_bytes(
        data[:max_bytes], "workspace:" + str(path.relative_to(root)).replace("\\", "/"),
        complete=(size <= max_bytes and len(data) == size),
        total_bytes=size,
        transport={"kind": "local-file", "observed_size": size,
                   "prefix_sha256": sha256(data[:max_bytes])})


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _public_host(hostname):
    # Validation is a safety check, not a DNS-pinning guarantee. Only
    # explicitly trusted hosts should be added to the network allowlist.
    if not hostname or hostname.endswith(".") or re.search(r"\s", hostname):
        raise ValueError("Invalid URL host")
    try:
        if not ipaddress.ip_address(hostname).is_global:
            raise ValueError("Nonpublic IP destination refused")
    except ValueError as exc:
        if "Nonpublic IP" in str(exc):
            raise
    results = socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)
    if not results:
        raise ValueError("Host has no DNS answer")
    for entry in results:
        ip = ipaddress.ip_address(entry[4][0])
        if not ip.is_global:
            raise ValueError("DNS resolved to nonpublic address")


def read_https(url: str, *, approved_hosts, max_bytes: int,
               byte_range=False):
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname not in approved_hosts or
            parsed.username or parsed.password or
            parsed.port not in (None, 443) or parsed.fragment):
        raise ValueError("HTTPS target not allowed by capture policy")
    _public_host(parsed.hostname)
    headers = {"User-Agent": "VL64-ByteFabric/0.1",
               "Accept": "*/*", "Accept-Encoding": "identity"}
    if byte_range:
        headers["Range"] = f"bytes=0-{max_bytes-1}"
    request = urllib.request.Request(url, method="GET", headers=headers)
    begin = time.perf_counter_ns()
    try:
        try:
            response = urllib.request.build_opener(NoRedirect).open(request, timeout=10)
        except urllib.error.HTTPError as e:
            response = e
        with response:
            status = response.status if hasattr(response, "status") else response.code
            if status not in (200, 206):
                raise ValueError(f"HTTP response not accepted: {status}")
            if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                raise ValueError("Compressed remote representation not accepted")
            data = response.read(max_bytes + 1)
            if status == 206:
                cr = response.headers.get("Content-Range", "")
                match = re.fullmatch(r"bytes 0-(\d+)/(\d+)", cr)
                if not match:
                    raise ValueError("Invalid Content-Range")
                last, total = map(int, match.groups())
                if last >= total or len(data) != last + 1 or len(data) > max_bytes:
                    raise ValueError("Range response does not match observed bytes")
                complete = (len(data) == total)
            else:
                length = response.headers.get("Content-Length")
                total = int(length) if length and length.isdigit() else None
                complete = (len(data) <= max_bytes and
                            (total is None or total == len(data)))
            truncated = len(data) > max_bytes
            data = data[:max_bytes]
            if truncated:
                complete = False
            return capture_bytes(
                data, url, complete=complete, total_bytes=total,
                transport={
                    "kind": "https", "http_status": status,
                    "range_requested": byte_range,
                    "content_range": response.headers.get("Content-Range"),
                    "content_type": (response.headers.get("Content-Type") or "")[:100],
                    "etag": (response.headers.get("ETag") or "")[:120],
                    "elapsed_ms": round((time.perf_counter_ns()-begin)/1e6, 3),
                    "redirect_followed": False,
                    "truncated": truncated,
                })
    except (OSError, urllib.error.URLError) as e:
        raise ValueError("HTTPS acquisition unsuccessful: " + type(e).__name__) from e


def acquire(target: str, *, root: Path, approved_hosts=DEFAULT_HOSTS,
            max_bytes=MAX_CAPTURE, byte_range=False):
    if type(max_bytes) is not int or not 1 <= max_bytes <= 1024*1024:
        raise ValueError("Capture budget must be 1..1048576 bytes")
    if type(byte_range) is not bool:
        raise ValueError("Byte-range request must be boolean")
    if not isinstance(target, str):
        raise TypeError("Target must be a source identifier")
    if target.startswith("workspace:"):
        return read_file(target, root=root, max_bytes=max_bytes)
    if target.startswith("inline:b64,"):
        raw = target[len("inline:b64,"):]
        if len(raw) > 2048:
            raise ValueError("Inline bytes exceed budget")
        data = base64.b64decode(raw, validate=True)
        return capture_bytes(data[:max_bytes], "inline:b64",
                             complete=len(data) <= max_bytes,
                             total_bytes=len(data))
    if target.startswith("https://"):
        return read_https(target, approved_hosts=approved_hosts,
                          max_bytes=max_bytes, byte_range=byte_range)
    raise ValueError("No registered capture adapter for source")
