"""MPF-003: original-source numerical host authentication and no-copy behavior tests.

The remote arithmetic fixture uses real count-trained weights resident ONLY
inside the serving host process in the cross-process case. Other tests assert
that attaching a preloaded transformer never calls a model loader.
"""
from __future__ import annotations

import contextlib
import copy
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from experiments.model_fabric.attached_resident_pretrained import (
    AttachedResidentPretrainedLanguage,
)
from experiments.model_fabric.authenticated_resident_host import (
    AuthenticatedHostRefused, AuthenticatedResidentHostClient,
    SignedResidentHostGateway, _encode, _decode,
)
from experiments.model_fabric.byte_lm import fit_byte_lm
from experiments.model_fabric.operational_ground import (
    GROUND_SCHEMA, CALL_SCHEMA, digest, observe_operational_ground,
)
from experiments.model_fabric.operational_hosts import HostedTrainedByteLanguage
from experiments.model_fabric.source_pinned_behavior_fleet import SourcePinnedBehaviorFleet
from experiments.model_fabric.virtual_page_fleet import (
    HostedModelSource, VirtualModelPageFleet, _hash, ModelPageRefused,
)

SECRET = bytes(range(32))
TRAIN = (b"Pretrained computation is performed at the original numerical host. "
         b"Separate observers navigate causal streams and trusted receipts. ") * 12
REPO = "lab/source-resident-trained-model"
REV = "a"*40


def source(repository=REPO, revision=REV):
    record = {
        "schema": "mailcpu.external-virtual-structure.v1.source",
        "repository": repository, "revision": revision,
        "files": [{"path": "config.json", "size_bytes": 238, "source_sha256": None}],
        "config": {"model_type": "resident_language", "num_hidden_layers": 1},
        "observation_digests": ["d"*64],
        "scope": "external_structure_only_zero_model_payload",
    }
    return HostedModelSource(record, expected_head=_hash(record))


def host_for(src, *, learned=None):
    learned = learned if learned is not None else fit_byte_lm(TRAIN)
    host = HostedTrainedByteLanguage(learned, observer_host="existing-original-host")
    host.description["model"] = src.repository
    host.description["revision"] = src.revision
    host.bound_ground = digest(GROUND_SCHEMA, host.description)
    return host, learned["ground_digest"]


def attached_transport(src=None, *, secret=SECRET, clock=time.time):
    src = src or source()
    host, artifact = host_for(src)
    gateway = SignedResidentHostGateway(
        host, source_head=src.head, model=src.repository, revision=src.revision,
        artifact_identity=artifact, observer_host="existing-original-host",
        secret=secret, clock=clock,
    )
    client = AuthenticatedResidentHostClient(
        endpoint="https://already-resident.example/v1/operational",
        secret=secret, source_head=src.head,
        ground_digest=gateway.ground["ground_digest"],
        transport=gateway.dispatch, clock=clock,
    )
    return src, host, artifact, gateway, client


def make_bridge(n=4):
    src, host, artifact, gateway, client = attached_transport()
    pages = VirtualModelPageFleet()
    pages.register("resident", src)
    bridge = SourcePinnedBehaviorFleet(pages)
    bridge.bind("resident", client, approved_hosts={"existing-original-host"},
                expected_artifact=artifact)
    for i in range(n):
        pages.open(f"observer-{i}", "resident", host_id=f"receiver-{i%2}")
        bridge.open(f"observer-{i}")
    return bridge, host, artifact, gateway, client


def test_authenticated_one_host_many_trained_causal_observers_no_weights():
    bridge, host, artifact, gateway, client = make_bridge(12)
    jobs = [{"observer_id": f"observer-{i}", "user_text": f"Observed input {i}.",
             "request_id": "step-0", "max_new_tokens": 3,
             "tile_size": 37} for i in range(12)]
    replies = bridge.turn_many(jobs, workers=12)
    assert len(replies) == len({r["head"] for r in replies}) == 12
    assert len({r["source_head"] for r in replies}) == 1
    assert len({r["operator_ground"] for r in replies}) == 1
    assert len({r["checkpoint"]["model_checkpoint"]["transcript_head"]
                for r in replies}) >= 1
    assert all(len(r["observation"]["generated_tokens"]) == 3 for r in replies)
    assert all(r["observation"]["host_calls"] > 3 for r in replies)
    assert all(r["model_payload_bytes"] == 0 and
               r["observation"]["remote_weight_bytes_received"] == 0 for r in replies)
    assert len(host.sessions) <= 8
    snap = bridge.snapshot()
    assert SECRET.hex() not in json.dumps(snap)
    assert "sparse_weights" not in json.dumps(snap)
    assert SourcePinnedBehaviorFleet.restore(
        snap, expected_digest=snap["checkpoint_digest"]).snapshot() == snap
    first = bridge.turn("observer-0", "Continue the first conversation.",
                        request_id="step-1", max_new_tokens=2)
    assert first["sequence"] == 1
    assert first["model_intention"]["messages"][-2]["role"] == "assistant"
    assert first["model_intention"]["messages"][-1]["role"] == "user"


def test_signed_frame_replay_bad_secret_expiry_and_forged_result_refused():
    src, host, artifact, gateway, client = attached_transport()
    captured = []
    def recording(raw):
        captured.append(raw)
        return gateway.dispatch(raw)
    client.transport = recording
    ground = observe_operational_ground(
        client, expected_model=src.repository, expected_revision=src.revision,
        expected_artifact=artifact, approved_hosts={"existing-original-host"})
    assert ground["ground_digest"] == gateway.ground["ground_digest"]
    with pytest.raises(AuthenticatedHostRefused, match="replayed"):
        gateway.dispatch(captured[-1])
    wrong = AuthenticatedResidentHostClient(
        endpoint=client.endpoint, secret=b"x"*32,
        source_head=src.head, ground_digest=gateway.ground["ground_digest"],
        transport=gateway.dispatch)
    with pytest.raises(AuthenticatedHostRefused, match="authentication"):
        wrong({"schema": CALL_SCHEMA, "operation": "describe"})
    expired = AuthenticatedResidentHostClient(
        endpoint=client.endpoint, secret=SECRET,
        source_head=src.head, ground_digest=gateway.ground["ground_digest"],
        transport=gateway.dispatch, clock=lambda: int(time.time())-300)
    with pytest.raises(AuthenticatedHostRefused, match="expired"):
        expired({"schema": CALL_SCHEMA, "operation": "describe"})
    def tamper(raw):
        response = _decode(gateway.dispatch(raw))
        response["result"]["host_result"]["revision"] = "b"*40
        return _encode(response)
    client.transport = tamper
    with pytest.raises(AuthenticatedHostRefused, match="authentication"):
        client({"schema": CALL_SCHEMA, "operation": "describe"})
    with pytest.raises(AuthenticatedHostRefused, match="HTTPS"):
        AuthenticatedResidentHostClient(
            endpoint="http://untrusted.example/v1/operational",
            secret=SECRET, source_head=src.head,
            ground_digest=gateway.ground["ground_digest"],
            transport=gateway.dispatch)


def test_lease_fences_two_remote_clients_and_refuses_unleased_inference():
    src, host, artifact, gateway, client = attached_transport()
    second = AuthenticatedResidentHostClient(
        endpoint=client.endpoint, secret=SECRET, source_head=src.head,
        ground_digest=gateway.ground["ground_digest"], transport=gateway.dispatch)
    g = gateway.ground
    call = {"schema": CALL_SCHEMA, "operation": "encode",
            "model": src.repository, "revision": src.revision,
            "ground_digest": g["ground_digest"],
            "arguments": {"text": "HELLO"}}
    with pytest.raises(AuthenticatedHostRefused, match="lease"):
        client(call)
    with client.execution_lease():
        assert client(call)["result"]["tokens"] == list(b"HELLO")
        with pytest.raises(AuthenticatedHostRefused, match="already leased"):
            with second.execution_lease():
                pass
    with second.execution_lease():
        assert second(call)["result"]["tokens"] == list(b"HELLO")


def test_restore_original_host_reattachment_requires_matching_causal_logits():
    bridge, host, artifact, gateway, client = make_bridge(1)
    before = bridge.turn("observer-0", "Commit a trained numerical prefix.",
                         request_id="before", max_new_tokens=2)
    checkpoint = bridge.snapshot()
    restored = SourcePinnedBehaviorFleet.restore(
        checkpoint, expected_digest=checkpoint["checkpoint_digest"])
    other, same_artifact = host_for(source())
    different, _ = host_for(source())
    different.calculate_logits = lambda tokens: [0.0]*256
    invalid_gateway = SignedResidentHostGateway(
        different, source_head=source().head, model=REPO, revision=REV,
        artifact_identity=artifact, observer_host="existing-original-host",
        secret=SECRET)
    invalid_client = AuthenticatedResidentHostClient(
        endpoint=client.endpoint, secret=SECRET, source_head=source().head,
        ground_digest=invalid_gateway.ground["ground_digest"],
        transport=invalid_gateway.dispatch)
    with pytest.raises(ModelPageRefused, match="causal-logits"):
        restored.bind("resident", invalid_client,
                      approved_hosts={"existing-original-host"},
                      expected_artifact=artifact)
    valid_gateway = SignedResidentHostGateway(
        other, source_head=source().head, model=REPO, revision=REV,
        artifact_identity=artifact, observer_host="existing-original-host",
        secret=SECRET)
    valid_client = AuthenticatedResidentHostClient(
        endpoint=client.endpoint, secret=SECRET, source_head=source().head,
        ground_digest=valid_gateway.ground["ground_digest"],
        transport=valid_gateway.dispatch)
    restored.bind("resident", valid_client,
                  approved_hosts={"existing-original-host"}, expected_artifact=artifact)
    continued = restored.turn("observer-0", "Continue on original resident host.",
                              request_id="next", max_new_tokens=2)
    assert continued["predecessor"] == before["head"]
    assert continued["model_payload_bytes"] == 0


def test_source_revision_artifact_and_unauthorized_model_substitution_refused():
    src, host, artifact, gateway, client = attached_transport()
    pages = VirtualModelPageFleet()
    pages.register("right", src)
    pages.register("wrong", source("lab/different", "b"*40))
    bridge = SourcePinnedBehaviorFleet(pages)
    with pytest.raises(ValueError, match="Host model"):
        bridge.bind("wrong", client, approved_hosts={"existing-original-host"},
                    expected_artifact=artifact)
    with pytest.raises(ValueError, match="artifact"):
        bridge.bind("right", client, approved_hosts={"existing-original-host"},
                    expected_artifact="f"*64)
    bridge.bind("right", client, approved_hosts={"existing-original-host"},
                expected_artifact=artifact)


class FakeTorch:
    long = "long"
    def tensor(self, values, dtype):
        assert dtype == self.long
        return values
    def inference_mode(self):
        return contextlib.nullcontext()


class FakeVector:
    def __init__(self, values): self.values = values
    def float(self): return self
    def cpu(self): return self
    def tolist(self): return list(self.values)


class FakeLogits:
    def __init__(self, values): self.values = values
    def __getitem__(self, index):
        assert index == (0, -1)
        return FakeVector(self.values)


class AlreadyLoadedDecoder:
    def __init__(self):
        self.config = SimpleNamespace(vocab_size=256, max_position_embeddings=512)
        self.calls = []
        self.training = True
    def eval(self):
        self.training = False
        return self
    def __call__(self, *, input_ids, use_cache, past_key_values=None):
        current = tuple(input_ids[0])
        old = past_key_values or ()
        prefix = old + current
        self.calls.append((current, old))
        values = [-1000.0]*256
        values[(sum(prefix)+37) % 256] = 1.0
        return SimpleNamespace(logits=FakeLogits(values), past_key_values=prefix)


class AlreadyLoadedTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode())
    def decode(self, tokens, skip_special_tokens=True):
        return bytes(tokens).decode("utf-8", "replace")
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        return list(json.dumps(messages).encode("utf-8"))


def test_preloaded_transformer_attachment_invokes_existing_host_object_no_loader():
    torch, decoder, tokenizer = FakeTorch(), AlreadyLoadedDecoder(), AlreadyLoadedTokenizer()
    host = AttachedResidentPretrainedLanguage(
        torch=torch, model=decoder, tokenizer=tokenizer, repository=REPO,
        revision=REV, artifact_identity="1"*64, observer_host="resident-transformer",
        max_context=512)
    assert host.model is decoder and host.tokenizer is tokenizer
    assert not decoder.training
    g = observe_operational_ground(
        host, expected_model=REPO, expected_revision=REV,
        expected_artifact="1"*64, approved_hosts={"resident-transformer"})
    from experiments.model_fabric.operational_ground import OperationalModelNavigator
    nav = OperationalModelNavigator(g, host, approved_hosts={"resident-transformer"})
    nav.open_text("Prompt")
    one = nav.advance(tile_size=53)
    two = nav.advance(tile_size=53)
    assert len(decoder.calls) == 3
    assert decoder.calls[-2][0] == (one,)
    assert decoder.calls[-1][0] == (two,)
    assert nav.remote_weight_bytes_received == 0
    assert host.host_kv_reuses == 2
    chat = AttachedResidentPretrainedLanguage(
        torch=torch, model=AlreadyLoadedDecoder(), tokenizer=tokenizer,
        repository=REPO, revision=REV, artifact_identity="1"*64,
        observer_host="resident-chat", native_chat=True)
    assert chat.description["tokenizer_id"] == "hf_native_chat_template_v1"
    with pytest.raises(ValueError, match="Native chat"):
        chat.encode("raw text")


def test_original_trained_host_held_in_separate_process_with_signed_rpc():
    src = source()
    pinned_learned = fit_byte_lm(TRAIN)
    artifact = pinned_learned["ground_digest"]
    script = '''
import sys
from experiments.model_fabric.byte_lm import fit_byte_lm
from experiments.model_fabric.operational_hosts import HostedTrainedByteLanguage
from experiments.model_fabric.operational_ground import GROUND_SCHEMA,digest
from experiments.model_fabric.authenticated_resident_host import SignedResidentHostGateway
TRAIN=(b"Pretrained computation is performed at the original numerical host. "
       b"Separate observers navigate causal streams and trusted receipts. ")*12
host=HostedTrainedByteLanguage(fit_byte_lm(TRAIN),observer_host="existing-original-host")
host.description["model"]="lab/source-resident-trained-model"
host.description["revision"]="a"*40
host.bound_ground=digest(GROUND_SCHEMA,host.description)
gateway=SignedResidentHostGateway(host,source_head=SOURCE_HEAD,
    model="lab/source-resident-trained-model",revision="a"*40,
    artifact_identity=host.description["model_artifact_identity"],
    observer_host="existing-original-host",secret=bytes(range(32)))
print(gateway.ground["ground_digest"],flush=True)
for line in sys.stdin:
    if line.strip()=="EXIT":break
    try:
        print(gateway.dispatch(bytes.fromhex(line.strip())).hex(),flush=True)
    except Exception:
        print("ERR",flush=True)
'''.replace("SOURCE_HEAD",repr(src.head))
    # The model host, its numerical weights and KV/session memory exist only
    # in the child; this parent receives authenticated control frames.
    child = subprocess.Popen([sys.executable, "-u", "-c", script],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             text=True)
    try:
        ground_digest = child.stdout.readline().strip()
        assert len(ground_digest) == 64
        def over_pipe(raw):
            child.stdin.write(raw.hex()+"\n")
            child.stdin.flush()
            answer = child.stdout.readline().strip()
            if answer == "ERR" or not answer:
                raise AuthenticatedHostRefused("separate source host rejected RPC")
            return bytes.fromhex(answer)
        client = AuthenticatedResidentHostClient(
            endpoint="https://original-external-host.example/v1/operational",
            secret=SECRET, source_head=src.head,
            ground_digest=ground_digest, transport=over_pipe)
        pages = VirtualModelPageFleet()
        pages.register("source", src)
        bridge = SourcePinnedBehaviorFleet(pages)
        bridge.bind("source", client, approved_hosts={"existing-original-host"},
                    expected_artifact=artifact)
        for i in range(3):
            pages.open(f"separate-{i}", "source")
            bridge.open(f"separate-{i}")
        results = bridge.turn_many([
            {"observer_id": f"separate-{i}", "user_text": f"Original-host task {i}.",
             "request_id": "first", "max_new_tokens": 2}
            for i in range(3)
        ], workers=3)
        assert len({x["head"] for x in results}) == 3
        assert all(x["model_payload_bytes"] == 0 for x in results)
        checkpoint = bridge.snapshot()
        assert SourcePinnedBehaviorFleet.restore(
            checkpoint, expected_digest=checkpoint["checkpoint_digest"]).snapshot() == checkpoint
    finally:
        child.stdin.write("EXIT\n")
        child.stdin.flush()
        child.communicate(timeout=15)
        assert child.returncode == 0

def test_live_benchmark_entrypoint_on_separate_resident_protocol(monkeypatch):
    """Exercise the actual two-turn measurement path using an authenticated fixture."""
    from experiments.model_fabric import authenticated_pretrained_lab as live
    src, _, artifact, gateway, client = attached_transport()
    monkeypatch.setattr(live, "AuthenticatedResidentHostClient",
                        lambda **kwargs: client)
    report = live.run(
        source_record=src.record, source_head=src.head,
        operator_ground=gateway.ground["ground_digest"],
        artifact_head=artifact,
        endpoint=client.endpoint,
        approved_host="existing-original-host", secret=SECRET,
        observers=3, workers=3, tokens=2, tile_size=37)
    assert [r["computed_tokens"] for r in report["behavioral_rounds"]] == [6, 6]
    assert [r["independent_causal_receipts"] for r in report["behavioral_rounds"]] == [3, 3]
    assert all(r["host_protocol_calls"] >= 6 for r in report["behavioral_rounds"])
    assert report["model_payload_bytes_downloaded"] == 0
    assert report["model_payload_bytes_copied_to_vl64"] == 0
    assert report["final_checkpoint"]["checkpoint_digest"]
    assert report["checkpoint_reconstruction_ns"] >= 0
