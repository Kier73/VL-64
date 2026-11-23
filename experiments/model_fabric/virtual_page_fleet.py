"""VL64 MPF-001: many observers and model pages as one virtual computation fabric.

This module retains *descriptions* of remotely hosted sources; no model bytes,
weights, tensor ranges, model loading, or remote code are transferred/executed.
Its input is E-CPU HSS-002's verified source-description Ground; the same
structural laws are reconstructed independently with VL-64's native
ObserverIntention and domain-separated hash-chained receipts.

1 model : N observers = one source identity, N independent causal contexts.
M models : N observers = independently scoped model/source/intention Grounds.
A structural processor is not a running pretrained neural model. Trained
behavior requires a distinct source-local qualified operator contract.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import re
import threading
from typing import Any, Mapping, Sequence

from vld import ObserverIntention, domain_digest

SOURCE_SCHEMA = "mailcpu.external-virtual-structure.v1"
SCHEMA = "vl64.virtual-model-page-fleet.v1"
RECEIPT_DOMAIN = "vl64.virtual-model-page-fleet-receipt.v1"
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_PATH = re.compile(r"[A-Za-z0-9_.+@/-]{1,1024}\Z")
_OPS = frozenset((
    "source.catalogue", "source.file", "source.extent",
    "source.architecture", "source.layer",
))
_MAX_MODELS = 512
_MAX_OBSERVERS = 8192
_MAX_EVENTS = 32768


class ModelPageRefused(ValueError):
    """An external model Ground, observer or qualified operation is ambiguous."""


def _canon(value: Any) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (ValueError, TypeError) as exc:
        raise ModelPageRefused("model description contains noncanonical data") from exc


def _hash(value: Any) -> str:
    return hashlib.sha256(_canon(value)).hexdigest()


def _name(value: Any, label: str) -> str:
    if type(value) is not str or _NAME.fullmatch(value) is None:
        raise ModelPageRefused(label + " is not a receiver-owned identity")
    return value


def _path(value: Any) -> str:
    if (type(value) is not str or _PATH.fullmatch(value) is None
        or value.startswith("/") or "//" in value
        or any(x in (".", "..") for x in value.split("/"))):
        raise ModelPageRefused("source coordinate path is not admitted")
    return value


def _plain(document: Any) -> dict:
    if not isinstance(document, Mapping):
        raise ModelPageRefused("expected independently observed source document")
    try:
        raw = _canon(dict(document))
        if len(raw) > 2 * 1024 * 1024:
            raise ModelPageRefused("source metadata inventory exceeds policy")
        result = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise ModelPageRefused("source metadata inventory is not JSON") from exc
    if type(result) is not dict:
        raise ModelPageRefused("source metadata document is not a mapping")
    return result


class HostedModelSource:
    """A VL64 view of one source without importing the whole E-CPU runtime."""

    def __init__(self, record: Mapping[str, Any], *, expected_head: str):
        doc = _plain(record)
        if (type(expected_head) is not str or _HEX64.fullmatch(expected_head) is None
            or set(doc) != {
                "schema", "repository", "revision", "files", "config",
                "observation_digests", "scope",
            }
            or doc["schema"] != SOURCE_SCHEMA + ".source"
            or doc["scope"] != "external_structure_only_zero_model_payload"
            or _hash(doc) != expected_head
            or type(doc["repository"]) is not str
            or _REPO.fullmatch(doc["repository"]) is None
            or type(doc["revision"]) is not str
            or _HEX40.fullmatch(doc["revision"]) is None
            or type(doc["config"]) is not dict
            or len(_canon(doc["config"])) > 16384
            or type(doc["files"]) is not list
            or len(doc["files"]) > 10000
            or type(doc["observation_digests"]) is not list
            or any(type(h) is not str or _HEX64.fullmatch(h) is None
                   for h in doc["observation_digests"])):
            raise ModelPageRefused("external source structural Ground lacks exact HSS identity")
        index: dict[str, dict] = {}
        for item in doc["files"]:
            if (type(item) is not dict
                or set(item) != {"path", "size_bytes", "source_sha256"}
                or type(item["size_bytes"]) is not int or item["size_bytes"] < 0):
                raise ModelPageRefused("source has an untyped or invalid remote file")
            p = _path(item["path"])
            h = item["source_sha256"]
            if h is not None and (type(h) is not str or _HEX64.fullmatch(h) is None):
                raise ModelPageRefused("source advertises a malformed content witness")
            if p in index:
                raise ModelPageRefused("source file path appears more than once")
            index[p] = item
        if list(index) != sorted(index):
            raise ModelPageRefused("external source inventory order is noncanonical")
        self.record = doc
        self.head = expected_head
        self.repository = doc["repository"]
        self.revision = doc["revision"]
        self.config = doc["config"]
        self._index = index
        self.files = tuple(index)
        self.declared_bytes = sum(f["size_bytes"] for f in index.values())

    def read(self, operation: str, arguments: Mapping[str, Any] | None = None) -> dict:
        """Pure source-Ground interpretation, no external tensor data movement."""
        args = dict(arguments or {})
        if operation not in _OPS or len(_canon(args)) > 4096:
            raise ModelPageRefused("source has no admitted structural interpretation law")
        if operation == "source.catalogue":
            if args:
                raise ModelPageRefused("catalogue accepts no arguments")
            kinds: dict[str, int] = {}
            for name in self.files:
                ext = name.rsplit(".", 1)[-1] if "." in name else "unknown"
                kinds[ext] = kinds.get(ext, 0) + 1
            return {"files": len(self.files),
                    "declared_external_bytes": self.declared_bytes,
                    "file_extensions": dict(sorted(kinds.items()))}
        if operation == "source.architecture":
            if args:
                raise ModelPageRefused("architecture accepts no arguments")
            return {"observed_configuration": self.config,
                    "semantics": "source_declared_only"}
        if operation == "source.layer":
            if set(args) != {"index"} or type(args["index"]) is not int:
                raise ModelPageRefused("layer intention needs exact index")
            n = self.config.get("num_hidden_layers", self.config.get("n_layer"))
            if type(n) is not int or n < 1 or not 0 <= args["index"] < n:
                raise ModelPageRefused("layer absent from source-declared architecture")
            return {"layer_index": args["index"], "declared_layers": n,
                    "model_type": self.config.get("model_type"),
                    "meaning": "abstract_layer_coordinate_not_neural_execution"}
        if set(args) != ({"path"} if operation == "source.file"
                         else {"path", "offset", "length"}):
            raise ModelPageRefused("remote file intention requires exact coordinate fields")
        path = _path(args["path"])
        try:
            f = self._index[path]
        except KeyError as exc:
            raise ModelPageRefused("source file address unknown") from exc
        address = f"hf://{self.repository}@{self.revision}/{path}"
        if operation == "source.file":
            return {"address": address, "declared_size_bytes": f["size_bytes"],
                    "source_sha256": f["source_sha256"]}
        start, width = args["offset"], args["length"]
        if (type(start) is not int or type(width) is not int
            or start < 0 or width <= 0 or start + width > f["size_bytes"]):
            raise ModelPageRefused("virtual extent exceeds declared external source")
        return {"address": address, "offset": start, "length": width,
                "end": start + width, "model_payload_bytes_read": 0,
                "meaning": "virtual_source_coordinate_not_tensor_values"}

    def position(self, operation: str, arguments: Mapping[str, Any] | None = None) -> dict:
        """Independently reproduce E-CPU HSS's three original control frames."""
        args = dict(arguments or {})
        result = self.read(operation, args)
        head = _hash({"schema": SOURCE_SCHEMA + ".genesis", "source_head": self.head})
        frames = []
        for i, (kind, description) in enumerate((
            ("ground.source", {"source_head": self.head}),
            ("ground.intention", {"operation": operation, "arguments": args}),
            ("ground.realization", {"result_digest": _hash(result)}),
        )):
            frame = {"index": i, "kind": kind, "description": description,
                     "predecessor": head}
            head = _hash(frame)
            frames.append({**frame, "head": head})
        position = {
            "schema": SOURCE_SCHEMA + ".position",
            "source_head": self.head,
            "operation": operation,
            "arguments": args, "result": result, "frames": frames,
            "model_payload_reads": 0, "model_payload_bytes": 0,
            "interpretation": "source_structural_intention_only",
        }
        position["position_digest"] = _hash(position)
        return position


def _genesis(observer: str, alias: str, source_head: str) -> str:
    return domain_digest(SCHEMA + ".genesis", {
        "observer": observer, "model": alias, "source_head": source_head,
    })


class VirtualModelPageFleet:
    """Source-sharing fabric for independently recoverable VL64 observers.

    No connection to an external model/data host is made during navigation;
    all observers share its once-qualified structural description. Each
    observer has an independent hash-linked causal stream. Receipts do not
    confer permission to invoke source-local neural operators.
    """

    def __init__(self) -> None:
        self._models: dict[str, HostedModelSource] = {}
        self._sessions: dict[str, dict] = {}
        self._events: list[dict] = []
        self._request_receipts: dict[tuple[str, str], dict] = {}
        self._lock = threading.RLock()
        self._observer_locks: dict[str, threading.RLock] = {}

    @property
    def models(self) -> tuple[str, ...]:
        return tuple(sorted(self._models))

    def register(self, alias: str, source: HostedModelSource) -> str:
        _name(alias, "model alias")
        if not isinstance(source, HostedModelSource):
            raise ModelPageRefused("source must have a source-pinned HSS-002 Ground")
        with self._lock:
            if alias in self._models:
                raise ModelPageRefused("model alias already bound")
            if len(self._models) >= _MAX_MODELS:
                raise ModelPageRefused("model registry capacity exhausted")
            self._models[alias] = source
        return source.head

    def open(self, observer_id: str, alias: str, *, host_id: str = "local") -> str:
        _name(observer_id, "observer")
        _name(host_id, "host")
        with self._lock:
            if alias not in self._models:
                raise ModelPageRefused("unknown model source alias")
            if observer_id in self._sessions:
                raise ModelPageRefused("observer already exists")
            if len(self._sessions) >= _MAX_OBSERVERS:
                raise ModelPageRefused("observer capacity exhausted")
            head = _genesis(observer_id, alias, self._models[alias].head)
            self._sessions[observer_id] = {
                "observer": observer_id, "model": alias, "host": host_id,
                "epoch": 0, "sequence": 0, "tip": head,
            }
            self._observer_locks[observer_id] = threading.RLock()
            return head

    def state(self, observer_id: str) -> dict:
        with self._lock:
            try:
                return dict(self._sessions[observer_id])
            except KeyError as exc:
                raise ModelPageRefused("unknown observer session") from exc

    def _observer_lock(self, observer_id: str) -> threading.RLock:
        with self._lock:
            try:
                return self._observer_locks[observer_id]
            except KeyError as exc:
                raise ModelPageRefused("unknown observer identity") from exc

    def observe(self, observer_id: str, operation: str, arguments: Mapping[str, Any] | None,
                *, request_id: str, expected_tip: str | None = None) -> dict:
        """Record one native VL64 ObserverIntention + E-CPU-equivalent position.

        A repeated request_id returns the identical committed receipt;
        a conflicting reuse or stale expected_tip is rejected.
        """
        _name(request_id, "idempotency request")
        if not isinstance(arguments, Mapping):
            raise ModelPageRefused("observer arguments must be explicit JSON mapping")
        args = json.loads(_canon(dict(arguments)))
        with self._observer_lock(observer_id):
            s = self.state(observer_id)
            source = self._models[s["model"]]
            target = f"hf://{source.repository}@{source.revision}"
            intent = ObserverIntention(
                "model.structure.read", target, options={
                    "operation": operation, "arguments": args,
                },
            )
            prior = self._request_receipts.get((observer_id, request_id))
            if prior is not None:
                if prior["intention_digest"] != intent.digest:
                    raise ModelPageRefused("idempotency key reused for another intention")
                return json.loads(_canon(prior))
            if expected_tip is not None and expected_tip != s["tip"]:
                raise ModelPageRefused("stale observer tip cannot publish a new observation")
            position = source.position(operation, args)
            receipt = {
                "schema": SCHEMA + ".observation",
                "kind": "observe", "observer": observer_id,
                "model": s["model"], "host": s["host"],
                "source_head": source.head,
                "epoch": s["epoch"], "sequence": s["sequence"],
                "request_id": request_id, "intention": intent.to_dict(),
                "intention_digest": intent.digest,
                "position": position, "position_head": position["position_digest"],
                "predecessor": s["tip"], "model_payload_bytes": 0,
            }
            receipt["head"] = domain_digest(RECEIPT_DOMAIN, receipt)
            with self._lock:
                if len(self._events) >= _MAX_EVENTS:
                    raise ModelPageRefused("observer receipt history capacity exhausted")
                s["sequence"] += 1
                s["tip"] = receipt["head"]
                self._sessions[observer_id] = s
                self._events.append(receipt)
                self._request_receipts[(observer_id, request_id)] = receipt
            return json.loads(_canon(receipt))

    def rehome(self, observer_id: str, new_host: str, *, expected_tip: str) -> dict:
        """Change the interpretation host under the existing observer's lineage.

        The coordinator owns the one current tip; a second disconnected
        coordinator would require native E-CPU M3/quorum for global fencing.
        """
        _name(new_host, "replacement host")
        with self._observer_lock(observer_id):
            s = self.state(observer_id)
            if expected_tip != s["tip"] or new_host == s["host"]:
                raise ModelPageRefused("rehome lacks current observer tip or new host")
            receipt = {
                "schema": SCHEMA + ".handoff", "kind": "handoff",
                "observer": observer_id, "model": s["model"],
                "source_head": self._models[s["model"]].head,
                "from_host": s["host"], "to_host": new_host,
                "epoch": s["epoch"] + 1, "sequence": s["sequence"],
                "predecessor": s["tip"], "model_payload_bytes": 0,
            }
            receipt["head"] = domain_digest(RECEIPT_DOMAIN, receipt)
            with self._lock:
                if len(self._events) >= _MAX_EVENTS:
                    raise ModelPageRefused("observer receipt history capacity exhausted")
                s["host"] = new_host
                s["epoch"] += 1
                s["sequence"] += 1
                s["tip"] = receipt["head"]
                self._sessions[observer_id] = s
                self._events.append(receipt)
            return json.loads(_canon(receipt))

    def observe_many(self, tasks: Sequence[tuple[str, str, Mapping[str, Any], str]],
                     *, workers: int = 16) -> tuple[dict, ...]:
        """Run independent observer sessions in a bounded thread pool."""
        if (not isinstance(tasks, (tuple, list)) or not 1 <= len(tasks) <= 8192
            or type(workers) is not int or not 1 <= workers <= 64):
            raise ModelPageRefused("observation batch exceeds receiver bound")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            receipts = executor.map(lambda t: self.observe(
                t[0], t[1], t[2], request_id=t[3]), tasks)
            return tuple(receipts)

    def snapshot(self) -> dict:
        """Only descriptions, intentions and receipts; no model data or handlers."""
        with self._lock:
            result = {
                "schema": SCHEMA + ".checkpoint",
                "models": {key: value.record for key, value in sorted(self._models.items())},
                "source_heads": {key: value.head for key, value in sorted(self._models.items())},
                "sessions": [dict(s) for _, s in sorted(self._sessions.items())],
                "events": self._events,
                "scope": "metadata_only_virtual_model_fleet",
            }
            result = json.loads(_canon(result))
            result["checkpoint_digest"] = domain_digest(SCHEMA + ".checkpoint", result)
            return result

    @classmethod
    def restore(cls, checkpoint: Mapping[str, Any], *, expected_digest: str) -> "VirtualModelPageFleet":
        """Independently reconstruct *all* observer operations, source and handoffs."""
        if (not isinstance(checkpoint, Mapping)
            or type(expected_digest) is not str or _HEX64.fullmatch(expected_digest) is None
            or checkpoint.get("checkpoint_digest") != expected_digest
            or set(checkpoint) != {
                "schema", "models", "source_heads", "sessions", "events",
                "scope", "checkpoint_digest",
            } or checkpoint["schema"] != SCHEMA + ".checkpoint"
            or checkpoint["scope"] != "metadata_only_virtual_model_fleet"):
            raise ModelPageRefused("model fleet checkpoint identity invalid")
        doc = _plain(checkpoint)
        target = doc.pop("checkpoint_digest")
        if domain_digest(SCHEMA + ".checkpoint", doc) != target:
            raise ModelPageRefused("model fleet checkpoint digest changed")
        result = cls()
        for alias, data in doc["models"].items():
            if alias not in doc["source_heads"]:
                raise ModelPageRefused("model source has no independent expected head")
            result.register(alias, HostedModelSource(
                data, expected_head=doc["source_heads"][alias],
            ))
        if set(doc["models"]) != set(doc["source_heads"]):
            raise ModelPageRefused("model source heads disagree")
        # Session creation is distinct from interpretation. Restore genesis
        # sessions before independently reenacting their ordered history.
        if not isinstance(doc["sessions"], list) or not isinstance(doc["events"], list):
            raise ModelPageRefused("session or event ledger is not a sequence")
        for row in doc["sessions"]:
            if (type(row) is not dict or set(row) != {
                "observer", "model", "host", "epoch", "sequence", "tip",
            }):
                raise ModelPageRefused("session checkpoint fields invalid")
            # Original host may be the very first observation receipt.
            initial = next((
                event["host"] for event in doc["events"]
                if event.get("observer") == row["observer"]
                and event.get("kind") == "observe"
                and event.get("sequence") == 0
            ), None)
            if initial is None:
                initial = next((
                    event["from_host"] for event in doc["events"]
                    if event.get("observer") == row["observer"]
                    and event.get("kind") == "handoff"
                    and event.get("sequence") == 0
                ), row["host"])
            result.open(row["observer"], row["model"], host_id=initial)
        for event in doc["events"]:
            if type(event) is not dict:
                raise ModelPageRefused("source observation event has invalid shape")
            if event.get("kind") == "observe":
                intent = ObserverIntention.from_dict(event["intention"])
                args = intent.spec["options"]
                if set(args) != {"operation", "arguments"}:
                    raise ModelPageRefused("source observation has unsupported options")
                recreated = result.observe(
                    event["observer"], args["operation"], args["arguments"],
                    request_id=event["request_id"], expected_tip=event["predecessor"],
                )
            elif event.get("kind") == "handoff":
                recreated = result.rehome(
                    event["observer"], event["to_host"], expected_tip=event["predecessor"],
                )
            else:
                raise ModelPageRefused("unknown model observer event kind")
            if recreated != event:
                raise ModelPageRefused("original model observation or handoff changed")
        if result.snapshot() != checkpoint:
            raise ModelPageRefused("reconstructed fleet checkpoint differs")
        return result
