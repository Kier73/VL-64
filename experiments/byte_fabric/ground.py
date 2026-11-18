"""Substrate-independent byte Grounds and an admitted, typed local realizer.

Every accepted input remains an exact byte observation. Semantic interpreters
are optional and never grant execution authority or prove machine behavior.
"""
from __future__ import annotations

import base64
import hashlib
import html.parser
import json
import math
import re
import struct
from dataclasses import dataclass

from vld import domain_digest

MAX_INLINE = 4096
MAX_FIELDS = 96


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_no_duplicates(text):
    def unique(pairs):
        result = {}
        for key, val in pairs:
            if key in result:
                raise ValueError("Duplicate JSON object key")
            result[key] = val
        return result
    def refuse_constant(value):
        raise ValueError("Invalid JSON numerical constant " + value)
    def finite_float(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("Nonfinite JSON number")
        return parsed
    return json.loads(text, object_pairs_hook=unique,
                      parse_constant=refuse_constant, parse_float=finite_float)


class _HTMLLinks(html.parser.HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links = []
        self.title = []
        self.in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self.in_title = True
        if tag in ("a", "link") and len(self.links) < 64:
            href = dict(attrs).get("href")
            if isinstance(href, str):
                self.links.append(href[:512])

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False

    def handle_data(self, data):
        if self.in_title and sum(len(x) for x in self.title) < 256:
            self.title.append(data[:256])


def _json_structure(value):
    if isinstance(value, dict):
        return {"root_type": "object", "field_count": len(value),
                "field_names": sorted(value)[:MAX_FIELDS]}
    if isinstance(value, list):
        return {"root_type": "array", "item_count": len(value)}
    return {"root_type": type(value).__name__,
            "scalar_digest": domain_digest("vl64.scalar.v1", value)}


def describe(data: bytes, *, hint: str = "") -> dict:
    """Identify safe supported views; bytes remain the universal fallback."""
    if not isinstance(data, bytes):
        raise TypeError("Expected bytes")
    generic = {
        "representation": "bytes",
        "prefix_hex": data[:24].hex(),
        "captured_bytes": len(data),
        "sha256": sha256(data),
    }
    kind, structure = "opaque-bytes", {}
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        w, h = struct.unpack(">II", data[16:24])
        if data[12:16] == b"IHDR" and w and h:
            kind, structure = "png", {
                "width": w, "height": h, "header_only": True,
                "decode_scope": "PNG-IHDR-not-verified-image-decode"}
    elif data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        if len(data) >= 10:
            w, h = struct.unpack("<HH", data[6:10])
            kind, structure = "gif", {"width": w, "height": h,
                                       "header_only": True}
    elif data.startswith(b"\x7fELF") and len(data) >= 16:
        kind, structure = "elf", {
            "elf_class": {1:"32-bit", 2:"64-bit"}.get(data[4], "unknown"),
            "endianness": {1:"little", 2:"big"}.get(data[5], "unknown"),
            "header_only": True}
    elif data.startswith(b"\x00asm") and len(data) >= 8:
        kind, structure = "wasm", {
            "version": int.from_bytes(data[4:8], "little"),
            "header_only": True}
    elif data.startswith(b"PK\x03\x04"):
        kind, structure = "zip", {"header_only": True}
    elif data.startswith(b"%PDF-"):
        kind, structure = "pdf", {"version_hint": data[5:8].decode("ascii", "replace"),
                                   "header_only": True}
    elif data.startswith(b"\xff\xd8\xff"):
        kind, structure = "jpeg", {"header_only": True}
    elif data.startswith(b"MZ") and len(data) >= 2:
        kind, structure = "dos-pe-candidate", {
            "header_only": True, "pe_signature_verified": False}
    elif hint.lower().endswith(".safetensors") and len(data) >= 8:
        length = int.from_bytes(data[:8], "little")
        if 0 < length <= 1024*1024 and len(data) >= 8+length:
            try:
                header = _json_no_duplicates(
                    data[8:8+length].decode("utf-8").strip())
                if not isinstance(header, dict):
                    raise ValueError("Safetensors header not an object")
                items = []
                for name, entry in header.items():
                    if name == "__metadata__":
                        continue
                    if (not isinstance(entry, dict) or
                        not isinstance(entry.get("shape"), list) or
                        not isinstance(entry.get("dtype"), str) or
                        not isinstance(entry.get("data_offsets"), list) or
                        len(entry["data_offsets"]) != 2 or
                        any(type(x) is not int or x < 0 for x in entry["data_offsets"]) or
                        entry["data_offsets"][1] < entry["data_offsets"][0]):
                        raise ValueError("Malformed tensor entry")
                    items.append({
                        "name": name[:200], "dtype": entry["dtype"][:20],
                        "shape": entry["shape"][:24],
                        "data_offsets": entry["data_offsets"]})
                kind, structure = "safetensors-header", {
                    "header_bytes": length, "tensor_count": len(items),
                    "tensor_sample": items[:MAX_FIELDS],
                    "header_only": True,
                    "weight_payload_interpreted": False,
                    "tensor_metadata_complete": len(items) <= MAX_FIELDS}
            except (ValueError, UnicodeDecodeError, TypeError):
                kind, structure = "opaque-bytes", {}
        else:
            kind, structure = "safetensors-unobserved-header", {
                "declared_header_bytes": length,
                "captured_header_bytes": max(0, len(data)-8),
                "header_complete": False}
    else:
        try:
            text = data.decode("utf-8-sig")
            value = _json_no_duplicates(text)
            kind, structure = "json", _json_structure(value)
        except (UnicodeDecodeError, ValueError, RecursionError):
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                text = None
            if text is not None:
                if re.search(r"(?is)<!doctype\s+html|<html(?:\s|>)", text[:4096]):
                    parser = _HTMLLinks()
                    parser.feed(text)
                    kind, structure = "html", {
                        "title": "".join(parser.title)[:256],
                        "link_count_observed": len(parser.links),
                        "link_sample": parser.links[:64],
                        "scope": "syntactic-links-untrusted"}
                elif text and all((ord(x) >= 32 or x in "\n\r\t") for x in text[:1024]):
                    kind, structure = "utf8-text", {
                        "characters": len(text), "line_count": len(text.splitlines()),
                        "prefix": text[:160]}
    generic.update({
        "kind": kind, "structure": structure,
        "view_scope": "syntax-or-header-observation-not-program-semantics",
    })
    return generic


def ground_for_capture(capture, *, hint="", decoders=None):
    """Construct identity of exact/prefix evidence and typed optional views."""
    payload = capture["data"]
    view = describe(payload, hint=hint)
    evidence = {
        "source": capture["source"],
        "capture_status": "COMPLETE" if capture["complete"] else "PREFIX",
        "captured_bytes": len(payload),
        "declared_total_bytes": capture.get("total_bytes"),
        "content_sha256": sha256(payload),
        "digest_scope": "full-resource" if capture["complete"] else "captured-prefix-only",
        "transport": capture.get("transport", {}),
    }
    evidence["observation_digest"] = domain_digest(
        "vl64.byte-capture-evidence.v1", evidence)
    ground = {"schema": "vl64.generic-byte-ground.v1",
              "observation": evidence, "view": view,
              "supported_realizers": list(DEFAULT_OPS)}
    if decoders is not None:
        ground["extended_views"] = decoders.describe(payload, hint=hint)
    # Structural identity is independent of source, clock, network latency
    # and transport. Evidence identity is separately capture-specific.
    structure = {"schema": ground["schema"], "view": view,
                 "capture_scope": evidence["digest_scope"],
                 "extended_views": ground.get("extended_views", [])}
    ground["ground_digest"] = domain_digest(
        "vl64.generic-byte-structure.v1", structure)
    return ground


def _json_pointer(data: bytes, pointer):
    if not isinstance(pointer, str) or len(pointer) > 1024 or (
            pointer and not pointer.startswith("/")):
        raise ValueError("Invalid JSON pointer")
    value = _json_no_duplicates(data.decode("utf-8-sig"))
    if pointer:
        for part in pointer.split("/")[1:]:
            if re.search(r"~(?![01])", part):
                raise ValueError("Invalid JSON-pointer escape")
            key = part.replace("~1", "/").replace("~0", "~")
            if isinstance(value, dict):
                value = value[key]
            elif isinstance(value, list) and re.fullmatch(r"0|[1-9][0-9]*", key):
                value = value[int(key)]
            else:
                raise ValueError("JSON pointer cannot reach target")
    return value


def _encode_slice(data):
    return {"encoding": "base64", "data": base64.b64encode(data).decode("ascii"),
            "length": len(data), "sha256": sha256(data)}


def op_hash(data, options):
    return {"sha256": sha256(data), "length": len(data)}


def op_slice(data, options):
    start, length = options.get("start", 0), options.get("length", 32)
    if (type(start) is not int or type(length) is not int or
            start < 0 or not 0 <= length <= MAX_INLINE or
            start + length > len(data)):
        raise ValueError("Slice outside observed bytes or output budget")
    return _encode_slice(data[start:start+length])


def op_json_pointer(data, options):
    value = _json_pointer(data, options.get("pointer", ""))
    if len(json.dumps(value, ensure_ascii=True).encode()) > MAX_INLINE:
        raise ValueError("Selected JSON value exceeds output budget")
    return {"value": value, "value_type": type(value).__name__}


def op_html_links(data, options):
    text = data.decode("utf-8")
    parser = _HTMLLinks()
    parser.feed(text)
    return {"links": parser.links, "title": "".join(parser.title)[:256],
            "scope": "untrusted-document-hrefs-not-followed"}


def op_byte_count(data, options):
    return {"byte_count": len(data),
            "zero_bytes": data.count(b"\0"), "distinct_byte_values": len(set(data))}


DEFAULT_OPS = {
    "sha256": op_hash,
    "slice": op_slice,
    "json.pointer": op_json_pointer,
    "html.links": op_html_links,
    "bytes.stats": op_byte_count,
}


@dataclass(frozen=True)
class OperationCapability:
    name: str
    requires_complete: bool = True
    input_kind: str = "any"
    max_input: int = 65536
    policy: str = "pure-local"


class QualifiedByteExecutor:
    """Explicit capability registry: a Ground does not authorize execution."""

    def __init__(self):
        self.registry = {}
        for name, fn in DEFAULT_OPS.items():
            self.register(OperationCapability(
                name, requires_complete=name not in ("slice", "bytes.stats"),
                input_kind=("json" if name == "json.pointer" else
                            "html" if name == "html.links" else "any")), fn)

    def register(self, capability, handler):
        if (not isinstance(capability, OperationCapability) or
                capability.name in self.registry or
                not callable(handler) or capability.policy != "pure-local"):
            raise ValueError("Capability not admitted")
        self.registry[capability.name] = (capability, handler)

    def execute(self, capture, ground, operator, options=None):
        if operator not in self.registry:
            return {"status": "REFUSED", "reason": "UNREGISTERED_REALIZER"}
        capability, handler = self.registry[operator]
        data = capture["data"]
        if ground["observation"]["content_sha256"] != sha256(data):
            return {"status": "REFUSED", "reason": "GROUND_INPUT_MISMATCH"}
        if capability.requires_complete and not capture["complete"]:
            return {"status": "UNRESOLVED", "reason": "FULL_RESOURCE_REQUIRED"}
        if len(data) > capability.max_input:
            return {"status": "UNRESOLVED", "reason": "REALIZER_INPUT_LIMIT"}
        if capability.input_kind != "any" and (
                ground["view"]["kind"] != capability.input_kind or
                not capture["complete"]):
            return {"status": "UNRESOLVED", "reason": "REQUIRED_TYPED_VIEW_ABSENT"}
        try:
            result = handler(data, dict(options or {}))
        except (ValueError, TypeError, KeyError, IndexError, UnicodeError,
                json.JSONDecodeError, RecursionError) as exc:
            return {"status": "UNRESOLVED", "reason": "OPERATOR_INPUT_UNSATISFIED",
                    "detail": str(exc)[:160]}
        try:
            size = len(json.dumps(result, allow_nan=False).encode())
        except (ValueError, TypeError, OverflowError):
            return {"status": "UNRESOLVED", "reason": "INVALID_OPERATOR_RESULT"}
        if size > 16384:
            return {"status": "UNRESOLVED", "reason": "RESULT_OUTPUT_LIMIT"}
        receipt = {
            "status": "EXECUTED_LOCAL",
            "operator": operator,
            "source_ground_digest": ground["ground_digest"],
            "source_observation_digest": ground["observation"]["observation_digest"],
            "input_sha256": sha256(data),
            "input_scope": "full" if capture["complete"] else "prefix",
            "result": result,
            "verification_scope": "deterministic-local-result-not-independent-remote-attestation",
        }
        receipt["receipt_digest"] = domain_digest("vl64.byte-execution-receipt.v1", receipt)
        return receipt
