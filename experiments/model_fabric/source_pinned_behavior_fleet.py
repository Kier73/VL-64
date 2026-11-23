"""MPF-002: source-pinned VL64 observers executing qualified resident model behavior.

The source-page Ground is immutable and descriptive.  A separately admitted
host callable performs real numerical inference with its own resident weights.
Observer causal receipts, messages and model checkpoints contain no weights.
This bridge never fetches a checkpoint, tokenizer, tensor extent or model page.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import re
import threading
from typing import Any, Callable, Mapping, Sequence

from vld import ObserverIntention, domain_digest
from .virtual_page_fleet import VirtualModelPageFleet, ModelPageRefused, _name
from .operational_ground import (
    _check_ground, canonical, digest, observe_operational_ground,
    OperationalModelNavigator, prefix_digest,
    CHECKPOINT_SCHEMA as MODEL_CHECKPOINT_SCHEMA,
)
from .intention_fabric import (
    IntentionSession, compile_intention,
    OUTPUT_SCHEMA,
)

SCHEMA = "vl64.source-pinned-behavior-fleet.v1"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_MAX_OBSERVERS = 2048
_MAX_EVENTS = 8192


def _copy(value: Any) -> Any:
    return json.loads(canonical(value))


def _seal(value: dict) -> dict:
    record = _copy(value)
    record["head"] = domain_digest(SCHEMA + ".receipt", record)
    return record


def _genesis(initial: dict, ground_digest: str) -> str:
    return domain_digest(SCHEMA + ".genesis", {
        **initial, "operator_ground": ground_digest,
    })


def _valid_hex(value: Any) -> bool:
    return type(value) is str and _HEX.fullmatch(value) is not None


class SourcePinnedBehaviorFleet:
    """One resident trained-model host, many independently ordered observers.

    Host callbacks are trusted only within explicit receiver admission pins.
    An in-process lock encloses one complete causal inference on a mutable
    host, including its logit tiles, preventing session/KV eviction races.
    Different host objects can execute concurrently.  For a shared host this
    is concurrent admission and queueing, not concurrent numerical kernels.
    """

    def __init__(self, sources: VirtualModelPageFleet):
        if not isinstance(sources, VirtualModelPageFleet):
            raise ModelPageRefused("requires an MPF-001 source-pinned observer fleet")
        self.sources = sources
        self._bindings: dict[str, dict] = {}
        self._host_locks: dict[int, threading.RLock] = {}
        self._initials: dict[str, dict] = {}
        self._sessions: dict[str, dict] = {}
        self._events: list[dict] = []
        self._requests: dict[tuple[str, str], dict] = {}
        self._lock = threading.RLock()
        self._observer_locks: dict[str, threading.RLock] = {}

    def bind(self, alias: str, host: Callable, *, approved_hosts: set[str],
             expected_artifact: str) -> str:
        """Admit the actual operator, matching exact source revision and artifact.

        expected_artifact is an independently supplied host-artifact pin;
        a host's self-description alone is never accepted as this pin.
        """
        if alias not in self.sources._models or not _valid_hex(expected_artifact):
            raise ModelPageRefused("source alias or independently pinned artifact invalid")
        if not isinstance(approved_hosts, set) or not approved_hosts:
            raise ModelPageRefused("no approved numerical execution host")
        src = self.sources._models[alias]
        ground = observe_operational_ground(
            host, expected_model=src.repository, expected_revision=src.revision,
            expected_artifact=expected_artifact, approved_hosts=approved_hosts,
        )
        with self._lock:
            if alias in self._bindings:
                if self._bindings[alias]["ground"] != ground:
                    raise ModelPageRefused("source's admitted numerical operator changed")
            else:
                self._bindings[alias] = {"ground": _copy(ground),
                                         "host": None, "approved_hosts": set(),
                                         "host_lock": threading.RLock()}
            # A restored or replaced resident host must reproduce the exact
            # last causal logits for every affected observer. Mere re-advertising
            # the original artifact/revision string is insufficient.
            seen = set()
            challenges = []
            for ev in reversed(self._events):
                if (ev["kind"] == "turn" and ev["model"] == alias
                        and ev["observer"] not in seen):
                    seen.add(ev["observer"])
                    challenges.append(ev["checkpoint"]["model_checkpoint"])
            host_lock = self._host_locks.setdefault(id(host), threading.RLock())
            with host_lock:
                for checkpoint in challenges:
                    nav = OperationalModelNavigator(ground, host,
                                                    approved_hosts=approved_hosts)
                    try:
                        nav.open_tokens(checkpoint["tokens"],
                                        prompt_length=checkpoint["prompt_length"])
                    except (ValueError, TypeError, KeyError) as exc:
                        raise ModelPageRefused(
                            "replacement trained host refused causal prefix challenge"
                        ) from exc
                    if (nav.logits_digest != checkpoint["logits_sha256"]
                            or prefix_digest(nav.tokens) != checkpoint["prefix_digest"]):
                        raise ModelPageRefused(
                            "replacement trained host failed exact causal-logits challenge")
            self._bindings[alias]["host"] = host
            self._bindings[alias]["host_lock"] = host_lock
            self._bindings[alias]["approved_hosts"] = set(approved_hosts)
        return ground["ground_digest"]

    def open(self, observer_id: str) -> str:
        _name(observer_id, "behavioral observer")
        with self._lock:
            source_state = self.sources.state(observer_id)
            alias = source_state["model"]
            if alias not in self._bindings or observer_id in self._sessions:
                raise ModelPageRefused("observer or its source operator is not admitted")
            if len(self._sessions) >= _MAX_OBSERVERS:
                raise ModelPageRefused("behavioral observer capacity exhausted")
            init = {
                "observer": observer_id, "model": alias,
                "source_head": self.sources._models[alias].head,
                "source_tip": source_state["tip"],
                "epoch": source_state["epoch"],
            }
            self._initials[observer_id] = init
            self._sessions[observer_id] = {
                **init, "sequence": 0, "messages": [],
                "tip": _genesis(init, self._bindings[alias]["ground"]["ground_digest"]),
            }
            self._observer_locks[observer_id] = threading.RLock()
            return self._sessions[observer_id]["tip"]

    def state(self, observer_id: str) -> dict:
        with self._lock:
            if observer_id not in self._sessions:
                raise ModelPageRefused("unknown behavioral observer")
            return _copy(self._sessions[observer_id])

    def _assert_source(self, state: dict) -> None:
        now = self.sources.state(state["observer"])
        if (now["model"] != state["model"] or now["tip"] != state["source_tip"]
                or now["epoch"] != state["epoch"]
                or self.sources._models[state["model"]].head != state["source_head"]):
            raise ModelPageRefused("source-pinned observer moved; explicit reanchor required")

    def _request_intention(self, source, *, kind, text, max_new_tokens,
                           max_output_bytes, tokenizer_mode, stop_tokens):
        if type(text) is not str or not text or len(text.encode("utf-8")) > 2048:
            raise ModelPageRefused("bounded new conversation text required")
        return ObserverIntention(
            "model.behavior.generate",
            f"hf://{source.repository}@{source.revision}",
            options={
                "kind": kind, "text": text, "max_new_tokens": max_new_tokens,
                "max_output_bytes": max_output_bytes,
                "tokenizer_mode": tokenizer_mode,
                "stop_tokens": list(stop_tokens),
            },
        )

    def turn(self, observer_id: str, user_text: str, *, request_id: str,
             kind: str = "paragraph", max_new_tokens: int = 8,
             max_output_bytes: int = 4096, tokenizer_mode: str = "plain_roles",
             stop_tokens: Sequence[int] = (), tile_size: int = 256,
             expected_tip: str | None = None) -> dict:
        """Execute one actual host-resident causal dialogue, then commit receipt.

        A retry of the same request returns the exact committed receipt.
        Errors never publish a completed behavioral observation.
        """
        _name(request_id, "behavioral request")
        if not isinstance(stop_tokens, (tuple, list)):
            raise ModelPageRefused("stop tokens must be an explicit bounded sequence")
        with self._lock:
            lock = self._observer_locks.get(observer_id)
        if lock is None:
            raise ModelPageRefused("unknown behavioral observer")
        with lock, self.sources._observer_lock(observer_id):
            s = self.state(observer_id)
            source = self.sources._models[s["model"]]
            self._assert_source(s)
            intention = self._request_intention(
                source, kind=kind, text=user_text,
                max_new_tokens=max_new_tokens, max_output_bytes=max_output_bytes,
                tokenizer_mode=tokenizer_mode, stop_tokens=stop_tokens,
            )
            prior = self._requests.get((observer_id, request_id))
            if prior is not None:
                if prior["intention_digest"] != intention.digest:
                    raise ModelPageRefused("request ID reused with a different behavior")
                return _copy(prior)
            if expected_tip is not None and expected_tip != s["tip"]:
                raise ModelPageRefused("stale behavioral observer tip")
            binding = self._bindings[s["model"]]
            if binding["host"] is None:
                raise ModelPageRefused("recovered observer requires reattached source host")
            messages = s["messages"] + [{"role": "user", "content": user_text}]
            model_intent = compile_intention(
                binding["ground"], kind=kind, messages=messages,
                max_new_tokens=max_new_tokens, max_output_bytes=max_output_bytes,
                tokenizer_mode=tokenizer_mode, stop_tokens=stop_tokens,
            )
            with binding["host_lock"]:
                run = IntentionSession(model_intent, binding["ground"], binding["host"],
                                       approved_hosts=binding["approved_hosts"])
                observation = run.run(tile_size=tile_size)
                checkpoint = run.checkpoint()
            self._assert_source(s)
            event = _seal({
                "schema": SCHEMA + ".event", "kind": "turn",
                "observer": observer_id, "model": s["model"],
                "source_head": s["source_head"], "source_tip": s["source_tip"],
                "epoch": s["epoch"], "sequence": s["sequence"],
                "request_id": request_id, "predecessor": s["tip"],
                "intention": intention.to_dict(),
                "intention_digest": intention.digest,
                "operator_ground": binding["ground"]["ground_digest"],
                "model_intention": model_intent,
                "observation": observation, "checkpoint": checkpoint,
                "model_payload_bytes": 0,
            })
            with self._lock:
                if len(self._events) >= _MAX_EVENTS or self._sessions[observer_id]["tip"] != s["tip"]:
                    raise ModelPageRefused("behavioral receiver's high-water commit changed")
                updated = _copy(s)
                updated["tip"] = event["head"]
                updated["sequence"] += 1
                updated["messages"] = messages + [
                    {"role": "assistant", "content": observation["observed_text"]}
                ]
                self._sessions[observer_id] = updated
                self._events.append(event)
                self._requests[(observer_id, request_id)] = event
            return _copy(event)

    def turn_many(self, requests: Sequence[Mapping[str, Any]], *, workers: int = 8) -> tuple[dict, ...]:
        if (not isinstance(requests, (list, tuple)) or not 1 <= len(requests) <= 512
                or type(workers) is not int or not 1 <= workers <= 64):
            raise ModelPageRefused("invalid bounded concurrent behavior fleet")
        for request in requests:
            if not isinstance(request, dict) or not {"observer_id", "user_text", "request_id"} <= set(request):
                raise ModelPageRefused("behavior task missing source-scoped observer identity")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return tuple(pool.map(lambda r: self.turn(**r), requests))

    def reanchor(self, observer_id: str, *, expected_tip: str) -> dict:
        """Follow a qualified MPF-001 observer handoff without moving model data."""
        with self._lock:
            lock = self._observer_locks.get(observer_id)
        if lock is None:
            raise ModelPageRefused("unknown behavioral observer")
        with lock, self.sources._observer_lock(observer_id):
            s = self.state(observer_id)
            now = self.sources.state(observer_id)
            if (s["tip"] != expected_tip or now["model"] != s["model"]
                    or now["epoch"] <= s["epoch"] or now["tip"] == s["source_tip"]):
                raise ModelPageRefused("stale or unrelated source observer handoff")
            event = _seal({
                "schema": SCHEMA + ".event", "kind": "reanchor",
                "observer": observer_id, "model": s["model"],
                "source_head": s["source_head"], "source_tip": s["source_tip"],
                "new_source_tip": now["tip"], "epoch": s["epoch"],
                "new_epoch": now["epoch"], "sequence": s["sequence"],
                "predecessor": s["tip"], "model_payload_bytes": 0,
            })
            with self._lock:
                if len(self._events) >= _MAX_EVENTS:
                    raise ModelPageRefused("behavioral history capacity exhausted")
                s["tip"] = event["head"]
                s["sequence"] += 1
                s["source_tip"] = now["tip"]
                s["epoch"] = now["epoch"]
                self._sessions[observer_id] = s
                self._events.append(event)
            return _copy(event)

    def snapshot(self) -> dict:
        """Portable source + behavioral receipts; never serialize host or weights."""
        with self._lock:
            document = {
                "schema": SCHEMA + ".checkpoint",
                "source_fleet": self.sources.snapshot(),
                "operators": {k: _copy(v["ground"]) for k, v in sorted(self._bindings.items())},
                "initials": _copy(self._initials),
                "sessions": _copy(self._sessions),
                "events": _copy(self._events),
                "scope": "qualified_host_claims_and_actual_observed_behaviors_no_weight_data",
            }
            document["checkpoint_digest"] = domain_digest(SCHEMA + ".checkpoint", document)
            return document

    @classmethod
    def restore(cls, snapshot: Mapping[str, Any], *, expected_digest: str):
        """Independently validate every behavioral event without rerunning inference.

        Original numerical correctness is established through the qualified
        host execution, not by receipt replay. Reattach a resident host with
        bind() before executing further turns.
        """
        if (not isinstance(snapshot, Mapping) or not _valid_hex(expected_digest)
                or snapshot.get("checkpoint_digest") != expected_digest):
            raise ModelPageRefused("behavioral checkpoint hash pin invalid")
        doc = _copy(snapshot)
        sha = doc.pop("checkpoint_digest", None)
        if (set(doc) != {"schema", "source_fleet", "operators", "initials",
                         "sessions", "events", "scope"}
                or doc["schema"] != SCHEMA + ".checkpoint"
                or doc["scope"] != "qualified_host_claims_and_actual_observed_behaviors_no_weight_data"
                or domain_digest(SCHEMA + ".checkpoint", doc) != sha):
            raise ModelPageRefused("behavioral checkpoint description changed")
        base = VirtualModelPageFleet.restore(
            doc["source_fleet"], expected_digest=doc["source_fleet"]["checkpoint_digest"],
        )
        result = cls(base)
        for alias, ground in doc["operators"].items():
            src = base._models.get(alias)
            _check_ground(ground)
            if (src is None or ground["model"] != src.repository
                    or ground["revision"] != src.revision):
                raise ModelPageRefused("replayed operator has a different source identity")
            result._bindings[alias] = {
                "ground": _copy(ground), "host": None, "approved_hosts": set(),
                "host_lock": threading.RLock(),
            }
        known_tips: dict[str, set[str]] = {}
        for observer, page in base._sessions.items():
            known_tips[observer] = {page["tip"]}
        for receipt in base._events:
            known_tips.setdefault(receipt["observer"], set()).update((
                receipt["predecessor"], receipt["head"],
            ))
        for observer, init in doc["initials"].items():
            if (not isinstance(init, dict) or init.get("observer") != observer
                    or init.get("model") not in result._bindings
                    or init.get("source_head") != base._models[init["model"]].head
                    or init.get("source_tip") not in known_tips.get(observer, set())
                    or type(init.get("epoch")) is not int):
                raise ModelPageRefused("behavioral genesis not bound to a source observer")
            result._initials[observer] = init
            result._sessions[observer] = {
                **init, "sequence": 0, "messages": [],
                "tip": _genesis(init, result._bindings[init["model"]]["ground"]["ground_digest"]),
            }
            result._observer_locks[observer] = threading.RLock()
        if not isinstance(doc["events"], list) or len(doc["events"]) > _MAX_EVENTS:
            raise ModelPageRefused("invalid bounded behavior history")
        for ev in doc["events"]:
            if (not isinstance(ev, dict) or ev.get("schema") != SCHEMA + ".event"
                    or ev.get("head") != domain_digest(
                        SCHEMA + ".receipt", {k: v for k, v in ev.items() if k != "head"}
                    )):
                raise ModelPageRefused("behavior receipt changed")
            observer = ev.get("observer")
            if observer not in result._sessions:
                raise ModelPageRefused("behavior receipt refers to missing observer")
            s = result._sessions[observer]
            if (ev.get("predecessor") != s["tip"] or ev.get("sequence") != s["sequence"]
                    or ev.get("model") != s["model"]
                    or ev.get("source_head") != s["source_head"]
                    or ev.get("source_tip") != s["source_tip"]
                    or ev.get("epoch") != s["epoch"]
                    or ev.get("model_payload_bytes") != 0):
                raise ModelPageRefused("behavior causal chain disagrees with source observer")
            if ev.get("kind") == "reanchor":
                if (ev.get("new_source_tip") not in known_tips.get(observer, set())
                        or type(ev.get("new_epoch")) is not int
                        or ev["new_epoch"] <= s["epoch"]):
                    raise ModelPageRefused("reanchor has no source observer evidence")
                s["source_tip"], s["epoch"] = ev["new_source_tip"], ev["new_epoch"]
            elif ev.get("kind") == "turn":
                ground = result._bindings[s["model"]]["ground"]
                obj = ObserverIntention.from_dict(ev["intention"])
                options = obj.spec["options"]
                source = base._models[s["model"]]
                if (obj.digest != ev.get("intention_digest")
                        or obj.spec["operation"] != "model.behavior.generate"
                        or obj.spec["target"] != f"hf://{source.repository}@{source.revision}"
                        or ev.get("operator_ground") != ground["ground_digest"]
                        or ev.get("request_id") is None
                        or (observer, ev["request_id"]) in result._requests):
                    raise ModelPageRefused("behavior intent substituted")
                messages = s["messages"] + [{"role": "user", "content": options["text"]}]
                exact = compile_intention(
                    ground, kind=options["kind"], messages=messages,
                    max_new_tokens=options["max_new_tokens"],
                    max_output_bytes=options["max_output_bytes"],
                    tokenizer_mode=options["tokenizer_mode"],
                    stop_tokens=options["stop_tokens"],
                )
                if exact != ev.get("model_intention"):
                    raise ModelPageRefused("reconstructed model intention differs")
                obs = ev["observation"]
                cp = ev["checkpoint"]
                mc = cp["model_checkpoint"]
                if (obs.get("observation_digest") != digest(
                        OUTPUT_SCHEMA, {k: v for k, v in obs.items() if k != "observation_digest"}
                    ) or obs.get("intent_digest") != exact["intent_digest"]
                    or obs.get("model_ground_digest") != ground["ground_digest"]
                    or obs.get("remote_weight_bytes_received") != 0
                    or cp.get("checkpoint_digest") != digest(
                        cp["schema"], {k: v for k, v in cp.items() if k != "checkpoint_digest"}
                    ) or cp.get("model_ground_digest") != ground["ground_digest"]
                    or cp.get("intent_digest") != exact["intent_digest"]
                    or mc.get("checkpoint_digest") != digest(
                        MODEL_CHECKPOINT_SCHEMA, {k: v for k, v in mc.items()
                                                  if k != "checkpoint_digest"}
                    ) or obs.get("checkpoint_digest") != mc["checkpoint_digest"]
                    or mc.get("transcript_head") != obs.get("transcript_head")
                    or mc.get("prompt_length") != obs.get("input_token_count")
                    or mc.get("tokens", [])[mc["prompt_length"]:][:len(obs["generated_tokens"])]
                        != obs["generated_tokens"]):
                    raise ModelPageRefused("host numerical observation evidence differs")
                s["messages"] = messages + [
                    {"role": "assistant", "content": obs["observed_text"]}
                ]
                result._requests[(observer, ev["request_id"])] = ev
            else:
                raise ModelPageRefused("unknown source-pinned behavior event")
            s["sequence"] += 1
            s["tip"] = ev["head"]
            result._events.append(ev)
        for observer, s in result._sessions.items():
            now = base.state(observer)
            if now["tip"] != s["source_tip"] or now["epoch"] != s["epoch"]:
                raise ModelPageRefused("behavioral checkpoint has stale source-page head")
        if result._sessions != doc["sessions"] or result.snapshot() != snapshot:
            raise ModelPageRefused("behavioral checkpoint reconstruction mismatch")
        return result
