"""Host-installed source, decoder and realizer: no architecture hardcoding."""
import struct

import pytest
from vld import ObserverIntention
from experiments.byte_fabric.sources import capture_bytes
from experiments.byte_fabric.extensions import (
    SourceCapability, SourceRegistry, DecoderRegistry,
)
from experiments.byte_fabric.ground import (
    QualifiedByteExecutor, OperationCapability, ground_for_capture,
)
from experiments.byte_fabric.router import (
    byte_arbitrate, replay_byte_receipt, compile_byte_plan,
)


def test_machine_sensor_adapter_and_custom_decoder_require_registration():
    sources = SourceRegistry()
    sensors = {5: struct.pack("<I", 73)}
    def capture(target, limit):
        return capture_bytes(sensors[5], source=target)
    sources.register(SourceCapability("sensor:"), capture)
    decoders = DecoderRegistry()
    decoders.register(
        "u32-register",
        lambda raw, hint: len(raw) == 4 and hint == "register32",
        lambda raw, hint: {
            "architecture": "unknown",
            "field": "little-endian-unsigned-register",
            "observed_value": struct.unpack("<I", raw)[0],
        },
    )
    request = ObserverIntention(
        "bytes.observe", "sensor:register-5",
        options={"format_hint": "register32"})
    assert compile_byte_plan(request)["status"] == "REFUSED"
    assert compile_byte_plan(request, sources=sources)["status"].startswith("ADMITTED")
    receipt = byte_arbitrate(request, sources=sources, decoders=decoders)
    assert receipt["status"] == "OBSERVED_BYTE_GROUND"
    assert receipt["ground"]["view"]["kind"] == "opaque-bytes"
    view = receipt["ground"]["extended_views"][0]
    assert view["decoder"] == "u32-register"
    assert view["value"]["observed_value"] == 73
    assert view["semantic_scope"] == (
        "host-registered-decoder-not-verified-system-behavior")
    captured = sources.acquire("sensor:register-5", max_bytes=8)
    assert replay_byte_receipt(receipt, capture=captured, decoders=decoders)["status"] == "VERIFIED_BYTE_GROUND"
    assert replay_byte_receipt(receipt, capture=captured)["status"] == "REJECTED"


def test_custom_observed_process_operator_is_explicitly_admitted():
    sources = SourceRegistry()
    sources.register(SourceCapability("process:"),
                     lambda target, budget: capture_bytes(
                        b"\x02\x00\x00\x00", target))
    executor = QualifiedByteExecutor()
    executor.register(OperationCapability("decode.u32"), lambda raw,opts: {
        "value": struct.unpack("<I", raw)[0]})
    req = ObserverIntention(
        "bytes.execute", "process:synthetic",
        options={"operator": "decode.u32"})
    assert compile_byte_plan(req)["status"] == "REFUSED"
    assert compile_byte_plan(req, sources=sources, executor=executor)["status"].startswith("ADMITTED")
    result = byte_arbitrate(req, sources=sources, executor=executor)
    assert result["status"] == "EXECUTED_LOCAL"
    assert result["execution"]["result"]["value"] == 2
    assert result["execution"]["verification_scope"] == (
        "deterministic-local-result-not-independent-remote-attestation")


def test_bad_host_plugin_and_oversized_description_refused():
    sources = SourceRegistry()
    with pytest.raises(ValueError):
        sources.register(SourceCapability("file:", effect="write"),
                         lambda a,b: {})
    with pytest.raises(ValueError):
        sources.register(SourceCapability("../../"),
                         lambda a,b: {})
    sources.register(SourceCapability("sensor:"),lambda target,n:{
        "data": "not bytes", "complete": True, "source": target})
    with pytest.raises(ValueError):
        sources.acquire("sensor:one", max_bytes=128)
    decoders = DecoderRegistry()
    decoders.register("bad-decoder",lambda data,hint: True,
                      lambda data,hint: {"oversized": "X"*10000})
    with pytest.raises(ValueError):
        ground_for_capture(capture_bytes(b"ok"), decoders=decoders)
    with pytest.raises(ValueError):
        decoders.register("bad-decoder",lambda a,b:False,lambda a,b:{})
