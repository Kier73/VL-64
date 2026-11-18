"""The external observer can inspect qualified host capabilities first."""
import json

from vld import ObserverIntention
from experiments.byte_fabric.extensions import (
    SourceRegistry, SourceCapability, DecoderRegistry,
)
from experiments.byte_fabric.sources import capture_bytes
from experiments.byte_fabric.ground import (
    QualifiedByteExecutor, OperationCapability,
)
from experiments.byte_fabric.router import (
    byte_arbitrate, compile_byte_plan, render_byte_receipt,
)
from experiments.observer_intent.arbitrate import arbitrate


def test_generic_capability_query_is_not_tied_to_models_or_files():
    req = ObserverIntention("bytes.capabilities", "*")
    plan = compile_byte_plan(req)
    assert plan["status"] == "ADMITTED_CAPABILITY_QUERY"
    receipt = arbitrate(req)
    assert receipt["status"] == "OBSERVED_CAPABILITY_CATALOGUE"
    ground = receipt["ground"]
    assert ground["schema"] == "vl64.byte-capability-ground.v1"
    catalogue = ground["catalogue"]
    assert "workspace:" in catalogue["source_adapters"]
    assert "https://" in catalogue["source_adapters"]
    assert "json" in catalogue["builtin_recognizers"]
    assert "wasm" in catalogue["builtin_recognizers"]
    assert "sha256" in catalogue["execution_capabilities"]
    assert catalogue["execution_capabilities"]["sha256"]["requires_complete"]
    assert "model_type" not in catalogue
    assert json.loads(render_byte_receipt(receipt, "json"))["status"] == "OBSERVED_CAPABILITY_CATALOGUE"


def test_host_registration_changes_what_observer_can_discover():
    sources = SourceRegistry()
    sources.register(
        SourceCapability("screen:"),
        lambda identifier, limit: capture_bytes(b"\xff\x00", identifier))
    decoders = DecoderRegistry()
    decoders.register("screen-pixel-probe",
                      lambda data,hint: len(data) == 2,
                      lambda data,hint: {"bytes":2})
    executor = QualifiedByteExecutor()
    executor.register(OperationCapability("pixel.sample"),
                      lambda data, options: {"first": data[0]})
    req = ObserverIntention("bytes.capabilities", "host:local")
    catalogue = byte_arbitrate(
        req,sources=sources,decoders=decoders,executor=executor,
    )["ground"]["catalogue"]
    assert "screen:" in catalogue["source_adapters"]
    assert "screen-pixel-probe" in catalogue["registered_decoders"]
    assert "pixel.sample" in catalogue["execution_capabilities"]


def test_4k_observer_target_supports_long_source_identifiers():
    long_target = "workspace:" + "component_"*40 + ".bin"
    assert len(long_target) > 256
    request = ObserverIntention("bytes.observe", long_target)
    assert request.to_dict()["target"] == long_target
    assert len(long_target) < 4096
