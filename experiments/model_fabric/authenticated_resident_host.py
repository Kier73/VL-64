"""MPF-003: authenticated HTTPS operator calls to an already-running model host.

Mutually pinned source Ground, operator Ground, HMAC authenticated frames,
bounded freshness/nonces and an exclusive host-side causal execution lease.
TLS provides endpoint authentication and transport security; the HMAC key must
be independently provisioned. Neither endpoint sends checkpoint/weight bytes.
HMAC validates a server holding the key, not the truth of its model weights.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re
import secrets
import ssl
import threading
import time
from typing import Any, Callable
import urllib.error
import urllib.parse
import urllib.request

from .operational_ground import CALL_SCHEMA, canonical, observe_operational_ground

SCHEMA = "vl64.source-pinned-https-operator.v1"
MAX_FRAME = 65536
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_HEX_NONCE = re.compile(r"[0-9a-f]{32}\Z")
_MAX_NONCES = 8192
_MAX_FRESHNESS = 60
_LEASE_TTL = 120


class AuthenticatedHostRefused(ValueError):
    """Invalid host identity, request signature, lease or bounded response."""


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _valid_hex(value: Any) -> bool:
    return type(value) is str and _SHA.fullmatch(value) is not None


def _key(value: bytes) -> bytes:
    if type(value) is not bytes or not 32 <= len(value) <= 128:
        raise AuthenticatedHostRefused("host-shared key must be 32..128 private bytes")
    return value


def _mac(secret: bytes, label: str, doc: dict) -> str:
    return hmac.new(secret, (SCHEMA + "." + label).encode() + b"\x00" +
                    canonical(doc), hashlib.sha256).hexdigest()


def _encode(value: Any) -> bytes:
    raw = canonical(value)
    if not 2 <= len(raw) <= MAX_FRAME:
        raise AuthenticatedHostRefused("operator transport frame exceeds bound")
    return raw


def _decode(raw: bytes) -> dict:
    if not isinstance(raw, bytes) or not 2 <= len(raw) <= MAX_FRAME:
        raise AuthenticatedHostRefused("incomplete or oversized operator transport")
    def pairs(items):
        doc = {}
        for key, value in items:
            if key in doc:
                raise AuthenticatedHostRefused("duplicate JSON operator frame keys")
            doc[key] = value
        return doc
    try:
        doc = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                         parse_constant=lambda _: (_ for _ in ()).throw(
                             ValueError("non-finite JSON")))
    except (ValueError, UnicodeError) as exc:
        raise AuthenticatedHostRefused("malformed operator transport frame") from exc
    if type(doc) is not dict:
        raise AuthenticatedHostRefused("operator frame must be a mapping")
    return doc


_HOST_DESCRIPTION_KEYS = {
    "schema", "model", "revision", "observer_host", "model_artifact_identity",
    "tokenizer_id", "vocab_size", "max_context", "operations", "trust_scope",
}
_HOST_RESULT_KEYS = {
    "encode": {"tokens"},
    "decode": {"text"},
    "open": {"session_id", "prefix_digest", "logits_sha256", "token_count"},
    "logits_tile": {"session_id", "prefix_digest", "start", "count",
                   "f32le_hex", "sha256", "full_logits_sha256"},
    "advance": {"session_id", "prefix_digest", "logits_sha256", "token_count"},
}
_HOST_ENVELOPE_KEYS = {
    "schema", "operation", "ground_digest", "model", "revision",
    "observer_host", "status", "result", "result_digest",
}


def _admit_host_reply(operation: str, reply: dict) -> dict:
    """Forbid arbitrary tensors, checkpoint material or invented result fields."""
    if type(reply) is not dict:
        raise AuthenticatedHostRefused("resident host sent an untyped result")
    if operation == "describe":
        if set(reply) != _HOST_DESCRIPTION_KEYS:
            raise AuthenticatedHostRefused("resident description exposes unadmitted fields")
    else:
        if (operation not in _HOST_RESULT_KEYS
                or set(reply) != _HOST_ENVELOPE_KEYS
                or reply.get("operation") != operation
                or type(reply.get("result")) is not dict
                or set(reply["result"]) != _HOST_RESULT_KEYS[operation]):
            raise AuthenticatedHostRefused("host result exposes unadmitted tensor fields")
    return reply


class SignedResidentHostGateway:
    """Host-side authenticated lease and actual resident numerical executor.

    The gateway is instantiated *at the existing original model host*. The
    supplied callable is an existing OperationalHostBase/transformer adapter.
    No loader, fetch, weight transfer or inferred model internals occur here.
    """

    def __init__(self, host: Callable, *, source_head: str, model: str,
                 revision: str, artifact_identity: str, observer_host: str,
                 secret: bytes, clock: Callable = time.time):
        if not _valid_hex(source_head) or not _valid_hex(artifact_identity):
            raise AuthenticatedHostRefused("original source and artifact pins missing")
        self.secret = _key(secret)
        self.source_head = source_head
        self.clock = clock
        self.host = host
        self.ground = observe_operational_ground(
            host, expected_model=model, expected_revision=revision,
            expected_artifact=artifact_identity,
            approved_hosts={observer_host},
        )
        self._lock = threading.RLock()
        self._recent: dict[str, int] = {}
        self._lease_id: str | None = None
        self._lease_until = 0

    def dispatch(self, raw: bytes) -> bytes:
        request = _decode(raw)
        if (set(request) != {"schema", "source_head", "ground_digest",
                            "nonce", "issued_at", "kind", "lease_id",
                            "payload", "hmac"}
                or request["schema"] != SCHEMA + ".request"
                or request["source_head"] != self.source_head
                or request["ground_digest"] != self.ground["ground_digest"]
                or type(request["nonce"]) is not str
                or _HEX_NONCE.fullmatch(request["nonce"]) is None
                or type(request["issued_at"]) is not int
                or request["kind"] not in ("acquire", "release", "call")
                or not isinstance(request["payload"], dict)
                or not _valid_hex(request["hmac"])):
            raise AuthenticatedHostRefused("source-operator request is not admitted")
        unsigned = {k: v for k, v in request.items() if k != "hmac"}
        if not hmac.compare_digest(request["hmac"],
                                   _mac(self.secret, "request", unsigned)):
            raise AuthenticatedHostRefused("operator call authentication failed")
        now = int(self.clock())
        if not now - _MAX_FRESHNESS <= request["issued_at"] <= now + 5:
            raise AuthenticatedHostRefused("expired or future-dated host request")
        with self._lock:
            self._recent = {nonce: timestamp for nonce, timestamp in
                            self._recent.items() if timestamp >= now-_MAX_FRESHNESS}
            if request["nonce"] in self._recent or len(self._recent) >= _MAX_NONCES:
                raise AuthenticatedHostRefused("replayed or over-capacity operator nonce")
            self._recent[request["nonce"]] = now
            if now >= self._lease_until:
                self._lease_id = None
            kind, lease = request["kind"], request["lease_id"]
            if kind == "acquire":
                if request["payload"] or lease is not None or self._lease_id is not None:
                    raise AuthenticatedHostRefused("numerical host already leased")
                self._lease_id = secrets.token_hex(16)
                self._lease_until = now + _LEASE_TTL
                result = {"lease_id": self._lease_id}
            elif kind == "release":
                if (request["payload"] or lease != self._lease_id
                        or self._lease_id is None):
                    raise AuthenticatedHostRefused("invalid causal-lease release")
                self._lease_id = None
                self._lease_until = 0
                result = {"released": True}
            else:
                payload = request["payload"]
                if payload.get("schema") != CALL_SCHEMA or not isinstance(
                        payload.get("operation"), str):
                    raise AuthenticatedHostRefused("invalid host operation protocol")
                if payload["operation"] == "describe":
                    if lease is not None:
                        raise AuthenticatedHostRefused("description must not hold a lease")
                elif self._lease_id is None or lease != self._lease_id:
                    raise AuthenticatedHostRefused("noncurrent numerical execution lease")
                if payload["operation"] != "describe":
                    self._lease_until = now + _LEASE_TTL
                result = {"host_result": _admit_host_reply(
                    payload["operation"], self.host(payload))}
            response = {
                "schema": SCHEMA + ".response",
                "source_head": self.source_head,
                "ground_digest": self.ground["ground_digest"],
                "nonce": request["nonce"],
                "request_sha256": _sha(unsigned),
                "kind": kind,
                "result": result,
            }
            response["hmac"] = _mac(self.secret, "response", response)
            return _encode(response)


class AuthenticatedResidentHostClient:
    """Signed HTTPS client exposing the original OperationalHost callable API.

    A transport callback may replace HTTPS for deterministic CI tests. Production
    defaults to TLS-verified HTTPS, exactly one configured endpoint, no redirects.
    """
    def __init__(self, *, endpoint: str, secret: bytes, source_head: str,
                 ground_digest: str, transport: Callable | None = None,
                 clock: Callable = time.time, nonce_factory: Callable = lambda: secrets.token_hex(16),
                 timeout: int = 45):
        parsed = urllib.parse.urlsplit(endpoint)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username
                or parsed.password or parsed.query or parsed.fragment
                or parsed.path != "/v1/operational" or not 1 <= timeout <= 300
                or not _valid_hex(source_head) or not _valid_hex(ground_digest)):
            raise AuthenticatedHostRefused("host endpoint must be exact HTTPS source route")
        self.endpoint, self.secret = endpoint, _key(secret)
        self.source_head, self.ground_digest = source_head, ground_digest
        self.clock, self.nonce_factory = clock, nonce_factory
        self.timeout = timeout
        self.transport = transport or self._https
        self._lock = threading.RLock()
        self._lease = None
        self._depth = 0

    def _https(self, raw: bytes) -> bytes:
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None
        opener = urllib.request.build_opener(NoRedirect)
        req = urllib.request.Request(
            self.endpoint, data=raw, method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with opener.open(req, timeout=self.timeout) as result:
                if result.status != 200:
                    raise AuthenticatedHostRefused("resident HTTPS host refused request")
                response = result.read(MAX_FRAME+1)
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            raise AuthenticatedHostRefused("original HTTPS model host unavailable") from exc
        return response

    def _rpc(self, kind: str, payload: dict, lease_id: str | None) -> dict:
        now = int(self.clock())
        nonce = self.nonce_factory()
        if type(nonce) is not str or _HEX_NONCE.fullmatch(nonce) is None:
            raise AuthenticatedHostRefused("cannot create unique host request nonce")
        request = {
            "schema": SCHEMA + ".request", "source_head": self.source_head,
            "ground_digest": self.ground_digest,
            "nonce": nonce, "issued_at": now, "kind": kind,
            "lease_id": lease_id, "payload": payload,
        }
        request["hmac"] = _mac(self.secret, "request", request)
        response = _decode(self.transport(_encode(request)))
        if (set(response) != {"schema", "source_head", "ground_digest", "nonce",
                             "request_sha256", "kind", "result", "hmac"}
                or response["schema"] != SCHEMA + ".response"
                or response["source_head"] != self.source_head
                or response["ground_digest"] != self.ground_digest
                or response["nonce"] != nonce or response["kind"] != kind
                or response["request_sha256"] != _sha(
                    {k:v for k,v in request.items() if k!="hmac"})
                or not isinstance(response["result"], dict)
                or not _valid_hex(response["hmac"])):
            raise AuthenticatedHostRefused("host reply has different causal request identity")
        if not hmac.compare_digest(response["hmac"], _mac(self.secret, "response", {
                k:v for k,v in response.items() if k != "hmac"})):
            raise AuthenticatedHostRefused("host result authentication failed")
        return response["result"]

    def __call__(self, payload: dict) -> dict:
        if not isinstance(payload, dict):
            raise AuthenticatedHostRefused("operational call must be explicit JSON")
        with self._lock:
            if payload.get("operation") != "describe" and self._lease is None:
                raise AuthenticatedHostRefused("host numerical call requires execution lease")
            result = self._rpc("call", payload, None if payload.get("operation") ==
                               "describe" else self._lease)
            if set(result) != {"host_result"} or not isinstance(result["host_result"], dict):
                raise AuthenticatedHostRefused("host result is not a qualified object")
            return result["host_result"]

    @contextmanager
    def execution_lease(self):
        """One complete conversation is atomic across separate receiver processes.

        Host-side TTL fences crashed clients. For live inference, each subcall
        extends its lease. This protocol does not make model execution itself
        exactly-once across a network crash.
        """
        with self._lock:
            if self._depth == 0:
                result = self._rpc("acquire", {}, None)
                token = result.get("lease_id")
                if type(token) is not str or _HEX_NONCE.fullmatch(token) is None:
                    raise AuthenticatedHostRefused("missing authenticated causal lease")
                self._lease = token
            self._depth += 1
            try:
                yield self
            finally:
                self._depth -= 1
                if self._depth == 0:
                    token, self._lease = self._lease, None
                    self._rpc("release", {}, token)


def make_https_handler(gateway: SignedResidentHostGateway):
    """Bridge a caller-provisioned HTTPS server to the resident gateway."""
    if not isinstance(gateway, SignedResidentHostGateway):
        raise AuthenticatedHostRefused("resident host gateway required")

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/v1/operational":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 2 <= length <= MAX_FRAME:
                    raise AuthenticatedHostRefused("invalid request length")
                response = gateway.dispatch(self.rfile.read(length))
            except (ValueError, TypeError):
                self.send_error(400, "Unadmitted source-pinned model operation")
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)
        def log_message(self, fmt, *args):
            return  # do not accidentally log request tokens or host credentials

    return Handler


def serve_https(gateway: SignedResidentHostGateway, *, address: str, port: int,
                tls_context: ssl.SSLContext):
    """Serve an already-running model behind caller-provisioned TLS certificates.

    The server never locates, downloads, initializes, or copies the pretrained
    model. It serves only the already-admitted live resident operator callable.
    """
    if (not isinstance(tls_context, ssl.SSLContext)
            or tls_context.protocol != ssl.PROTOCOL_TLS_SERVER
            or not isinstance(port, int) or not 1 <= port <= 65535):
        raise AuthenticatedHostRefused("TLS server context and explicit port required")
    server = ThreadingHTTPServer((address, port), make_https_handler(gateway))
    server.socket = tls_context.wrap_socket(server.socket, server_side=True)
    server.serve_forever()
