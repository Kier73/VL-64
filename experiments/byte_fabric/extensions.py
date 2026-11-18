"""Host-installed adapters extend sources and typed byte views.

No data-provided field is executable Python. A host must explicitly register
each callback before using it; registration is an authority decision.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from vld import domain_digest

MAX_VIEW_BYTES = 8192


@dataclass(frozen=True)
class SourceCapability:
    prefix: str
    effect: str = "read-only"
    max_bytes: int = 65536


class SourceRegistry:
    """Permit custom machine, process, screen, or stream byte capture sources."""

    def __init__(self):
        self.adapters = {}

    def register(self, capability, callback):
        if (not isinstance(capability, SourceCapability) or
                not re.fullmatch(r"[a-z][a-z0-9+.-]*:", capability.prefix) or
                capability.effect != "read-only" or
                type(capability.max_bytes) is not int or
                not 1 <= capability.max_bytes <= 1048576 or
                not callable(callback) or
                capability.prefix in self.adapters):
            raise ValueError("Source capability not admitted")
        self.adapters[capability.prefix] = capability, callback

    def acquire(self, target, *, max_bytes):
        for prefix, (capability, handler) in self.adapters.items():
            if target.startswith(prefix):
                if type(max_bytes) is not int or not 1 <= max_bytes <= capability.max_bytes:
                    raise ValueError("Requested source exceeds capability")
                capture = handler(target, max_bytes)
                if (not isinstance(capture, dict) or
                        not isinstance(capture.get("data"), bytes) or
                        type(capture.get("complete")) is not bool or
                        not isinstance(capture.get("source"), str) or
                        len(capture["data"]) > max_bytes):
                    raise ValueError("Source adapter produced invalid evidence")
                from .sources import capture_bytes
                return capture_bytes(
                    capture["data"], source=capture["source"],
                    complete=capture["complete"],
                    total_bytes=capture.get("total_bytes"),
                    transport=capture.get("transport"))
        raise ValueError("No registered source for target")


class DecoderRegistry:
    """Extend optional interpretations; never replaces exact byte evidence."""

    def __init__(self):
        self.decoders = {}

    def register(self, name, predicate, decoder):
        if (not isinstance(name, str) or
                not re.fullmatch(r"[a-z][a-z0-9_.-]{0,63}", name) or
                name in self.decoders or
                not callable(predicate) or not callable(decoder)):
            raise ValueError("Decoder capability not admitted")
        self.decoders[name] = predicate, decoder

    def describe(self, data: bytes, hint: str = ""):
        if not isinstance(data, bytes):
            raise TypeError("Expected bytes")
        views = []
        for name, (predicate, decoder) in self.decoders.items():
            if not predicate(data, hint):
                continue
            result = decoder(data, hint)
            if not isinstance(result, dict):
                raise ValueError("Decoder did not return a typed description")
            json_data = json.dumps(result, sort_keys=True, allow_nan=False)
            if len(json_data.encode()) > MAX_VIEW_BYTES:
                raise ValueError("Decoder returned an oversized description")
            record = {
                "decoder": name, "status": "INTERPRETED_LOCAL",
                "semantic_scope": "host-registered-decoder-not-verified-system-behavior",
                "value": result,
            }
            record["view_digest"] = domain_digest(
                "vl64.extended-byte-view.v1", record)
            views.append(record)
        return views
