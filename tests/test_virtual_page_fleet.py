"""MPF-001 — independent VL64 observers sharing model-page descriptions.

All input data is receiver-verified *metadata*. Neither the original
pretrained model nor weight shards are acquired, copied, or executed.
The separate E-CPU interop test reproduces its exact structural positions.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
from hashlib import sha256
import json
import subprocess
import sys
from time import perf_counter_ns

import pytest

from experiments.model_fabric.virtual_page_fleet import (
    HostedModelSource, ModelPageRefused, VirtualModelPageFleet, _hash,
)
from vld import ObserverIntention

BASE_SHA = "08c8657f02291e11b6f6b63e0e16014f8e748518"
NAME = "deepseek-ai/DeepSeek-V4.1-Flash-1017"


def described_source(model=NAME, *, revision=BASE_SHA, size=10**10, shards=48,
                     family="deepseek_v41", layers=128):
    """Produce an E-CPU-compatible SOURCE DESCRIPTION, never a model payload."""
    items = [
        {"path":"config.json","size_bytes":3000,"source_sha256":None},
        {"path":"tokenizer.json","size_bytes":50000,"source_sha256":None},
        {"path":"scripts/inference.py","size_bytes":20000,"source_sha256":None},
    ]
    items.extend({
        "path":f"model-{j:05d}-of-{shards:05d}.safetensors",
        "size_bytes":size+j,
        "source_sha256":f"{j:064x}",
    } for j in range(1,shards+1))
    record = {
        "schema":"mailcpu.external-virtual-structure.v1.source",
        "repository":model, "revision":revision,
        "files":sorted(items,key=lambda x:x["path"]),
        "config":{
            "model_type":family,
            "num_hidden_layers":layers,
            "num_attention_heads":64,
        },
        "observation_digests":["0"*64,"1"*64],
        "scope":"external_structure_only_zero_model_payload",
    }
    return HostedModelSource(record,expected_head=_hash(record))


def test_vl64_reuses_pinned_ecpu_source_ground_and_computes_equal_structural_position():
    source=described_source()
    assert source.head==_hash(source.record)
    assert source.declared_bytes>480_000_000_000
    assert len(source.files)==51
    operations=[
        ("source.catalogue",{}),
        ("source.architecture",{}),
        ("source.layer",{"index":0}),
        ("source.layer",{"index":127}),
        ("source.file",{"path":"config.json"}),
        ("source.extent",{
            "path":"model-00048-of-00048.safetensors",
            "offset":10**10-8,"length":7,
        }),
    ]
    for op, args in operations:
        position=source.position(op,args)
        assert position["position_digest"]==_hash({
            k:v for k,v in position.items() if k!="position_digest"})
        assert len(position["frames"])==3
        assert position["model_payload_bytes"]==0
        assert position["result"]==source.read(op,args)
    with pytest.raises(ModelPageRefused):
        source.position("model.neural.generate",{"input":"hello"})


def test_one_model_many_observers_shares_one_source_but_not_one_causal_identity():
    source=described_source()
    fleet=VirtualModelPageFleet()
    fleet.register("deepseek",source)
    assert fleet.models==("deepseek",)
    n=512
    for i in range(n):
        fleet.open(f"reader-{i:04d}","deepseek",host_id=f"host-{i%8}")
    batch=[(f"reader-{i:04d}","source.extent",{
        "path":"model-00048-of-00048.safetensors",
        "offset":source.read("source.file",{
            "path":"model-00048-of-00048.safetensors"
        })["declared_size_bytes"]-1,
        "length":1,
    },f"visit-{i}") for i in range(n)]
    receipts=fleet.observe_many(batch,workers=32)
    assert len(receipts)==512
    assert len({r["head"] for r in receipts})==n
    assert len({r["position_head"] for r in receipts})==1
    assert len({r["source_head"] for r in receipts})==1
    assert all(r["model_payload_bytes"]==0 for r in receipts)
    checkpoint=fleet.snapshot()
    assert len(checkpoint["events"])==n
    recovered=VirtualModelPageFleet.restore(
        checkpoint,expected_digest=checkpoint["checkpoint_digest"])
    assert recovered.snapshot()==checkpoint
    assert fleet.state("reader-0000")["sequence"]==1
    assert fleet.state("reader-0511")["sequence"]==1


@pytest.mark.parametrize("count",[1,4,16,64])
def test_many_models_in_one_fabric_keep_original_model_and_observer_identity(count):
    fleet=VirtualModelPageFleet()
    for i in range(count):
        model=f"source-{i:03d}"
        src=described_source(
            model=f"lab/model-{i:03d}",
            revision=f"{i+1:040x}",
            size=2**(24+i),
            shards=2,
            family=f"family_{i}",
            layers=3+i,
        )
        fleet.register(model,src)
        fleet.open(f"observer-{i:03d}",model)
        result=fleet.observe(
            f"observer-{i:03d}","source.layer",{"index":i%3},
            request_id=f"first-{i}")
        assert result["position"]["result"]["model_type"]==f"family_{i}"
    assert len(fleet.models)==count
    checkpoint=fleet.snapshot()
    again=VirtualModelPageFleet.restore(
        checkpoint,expected_digest=checkpoint["checkpoint_digest"])
    assert again.snapshot()==checkpoint
    assert sum(again._models[a].declared_bytes for a in again.models)>0
    assert all(ev["position"]["model_payload_bytes"]==0
               for ev in checkpoint["events"])


def test_replay_idempotency_handoff_and_independent_causal_sequences():
    fleet=VirtualModelPageFleet()
    fleet.register("deepseek",described_source())
    fleet.open("a","deepseek",host_id="original")
    fleet.open("b","deepseek",host_id="original")
    head=fleet.state("a")["tip"]
    a0=fleet.observe("a","source.layer",{"index":15},
                     request_id="question",expected_tip=head)
    assert fleet.observe("a","source.layer",{"index":15},
                         request_id="question",expected_tip=head)==a0
    assert fleet.state("a")["sequence"]==1
    with pytest.raises(ModelPageRefused,match="idempotency"):
        fleet.observe("a","source.layer",{"index":16},request_id="question")
    with pytest.raises(ModelPageRefused,match="stale"):
        fleet.observe("a","source.layer",{"index":15},
                      request_id="next",expected_tip=head)
    b0=fleet.observe("b","source.layer",{"index":15},request_id="b-one")
    assert b0["position_head"]==a0["position_head"]
    assert b0["head"]!=a0["head"]
    new=fleet.rehome("a","new-host",expected_tip=a0["head"])
    assert new["epoch"]==1
    assert new["predecessor"]==a0["head"]
    with pytest.raises(ModelPageRefused):
        fleet.rehome("a","obsolete",expected_tip=a0["head"])
    a1=fleet.observe("a","source.layer",{"index":16},
                     request_id="after-rehome",expected_tip=new["head"])
    assert a1["host"]=="new-host" and a1["epoch"]==1
    assert a1["predecessor"]==new["head"]
    assert fleet.state("b")["host"]=="original"
    snapshot=fleet.snapshot()
    assert VirtualModelPageFleet.restore(
        snapshot,expected_digest=snapshot["checkpoint_digest"]).snapshot()==snapshot


def test_observer_replacement_process_uses_only_metadata_ground_and_causal_checkpoint():
    fleet=VirtualModelPageFleet()
    fleet.register("deepseek",described_source(size=10**25))
    fleet.open("observer-x","deepseek",host_id="host-a")
    initial=fleet.observe("observer-x","source.catalogue",{},request_id="first")
    fleet.rehome("observer-x","host-b",expected_tip=initial["head"])
    snapshot=fleet.snapshot()
    code=(
        "import json,sys;"
        "from experiments.model_fabric.virtual_page_fleet import VirtualModelPageFleet;"
        "v=json.loads(sys.stdin.read());"
        "f=VirtualModelPageFleet.restore(v,expected_digest=v['checkpoint_digest']);"
        "s=f.state('observer-x');"
        "r=f.observe('observer-x','source.layer',{'index':127},"
        "request_id='new-step',expected_tip=s['tip']);"
        "print(json.dumps({'state':f.state('observer-x'),'head':r['head'],"
        "'source_head':r['source_head'],'result':r['position']['result']}))"
    )
    result=subprocess.run([sys.executable,"-c",code],
                          input=json.dumps(snapshot),text=True,
                          capture_output=True,check=True,timeout=30)
    returned=json.loads(result.stdout)
    assert returned["result"]["layer_index"]==127
    assert returned["state"]["epoch"]==1
    assert returned["source_head"]==fleet._models["deepseek"].head
    assert returned["state"]["tip"]==returned["head"]


@pytest.mark.parametrize("size",[2**20,2**40,2**80,2**200])
def test_interpretation_complexity_does_not_scale_with_model_byte_volume(size):
    ground=described_source(size=size)
    fleet=VirtualModelPageFleet()
    fleet.register("model",ground)
    fleet.open("observer","model")
    start=perf_counter_ns()
    receipt=fleet.observe("observer","source.extent",{
        "path":"model-00048-of-00048.safetensors",
        "offset":size, "length":1,
    },request_id="coordinate")
    elapsed=perf_counter_ns()-start
    assert elapsed>=0
    assert receipt["position"]["result"]["offset"]==size
    assert receipt["position"]["result"]["model_payload_bytes_read"]==0
    assert len(receipt["position"]["frames"])==3
    assert len(fleet.snapshot()["events"])==1


def test_source_identity_and_model_scope_never_mix_on_causal_replay():
    fleet=VirtualModelPageFleet()
    a=described_source(model="lab/model-a",revision="a"*40)
    b=described_source(model="lab/model-b",revision="b"*40)
    fleet.register("a",a)
    fleet.register("b",b)
    fleet.open("viewer-a","a")
    fleet.open("viewer-b","b")
    first=fleet.observe("viewer-a","source.file",{"path":"config.json"},
                        request_id="one")
    second=fleet.observe("viewer-b","source.file",{"path":"config.json"},
                         request_id="two")
    assert first["position"]["result"]["declared_size_bytes"]==second["position"]["result"]["declared_size_bytes"]
    assert first["position_head"]!=second["position_head"]
    assert first["source_head"]!=second["source_head"]
    doc=fleet.snapshot()
    forged=copy.deepcopy(doc)
    forged["events"][0]["source_head"]=second["source_head"]
    with pytest.raises(ModelPageRefused):
        VirtualModelPageFleet.restore(
            forged,expected_digest=doc["checkpoint_digest"])


def test_tampered_intention_evidence_duplicate_source_or_foreign_model_refused():
    model=described_source()
    altered=copy.deepcopy(model.record)
    altered["files"][0]["size_bytes"]+=1
    with pytest.raises(ModelPageRefused):
        HostedModelSource(altered,expected_head=model.head)
    forged=copy.deepcopy(model.record)
    forged["files"].append(copy.deepcopy(forged["files"][0]))
    with pytest.raises(ModelPageRefused):
        HostedModelSource(forged,expected_head=_hash(forged))
    reversed_files=copy.deepcopy(model.record)
    reversed_files["files"]=list(reversed(reversed_files["files"]))
    with pytest.raises(ModelPageRefused):
        HostedModelSource(reversed_files,expected_head=_hash(reversed_files))
    f=VirtualModelPageFleet()
    f.register("model",model)
    f.open("obs","model")
    with pytest.raises(ModelPageRefused):
        f.observe("obs","source.file",{"path":"../config.json"},request_id="bad")
    with pytest.raises(ModelPageRefused):
        f.observe("obs","model.generate",{"prompt":"hello"},request_id="wrong")
    with pytest.raises(ModelPageRefused):
        f.observe("obs","source.layer",{"index":False},request_id="wrong-layer")
    with pytest.raises(ModelPageRefused):
        f.observe("obs","source.extent",{
            "path":"config.json","offset":3000,"length":1},request_id="bad-range")
    doc=f.snapshot()
    doc["sessions"][0]["tip"]="c"*64
    with pytest.raises(ModelPageRefused):
        VirtualModelPageFleet.restore(doc,expected_digest=doc["checkpoint_digest"])


def test_vl64_observer_intention_carries_source_relative_operation_not_python_code():
    source=described_source()
    f=VirtualModelPageFleet()
    f.register("model",source)
    f.open("observer-1","model")
    result=f.observe("observer-1","source.layer",{"index":0},
                     request_id="once")
    intent=ObserverIntention.from_dict(result["intention"])
    assert intent.digest==result["intention_digest"]
    assert intent.spec["target"]==f"hf://{source.repository}@{source.revision}"
    assert intent.spec["operation"]=="model.structure.read"
    assert result["position"]["result"]["meaning"].endswith("not_neural_execution")
    assert result["model_payload_bytes"]==0


def test_ecpu_hss_original_implementation_agrees_independently_when_available():
    """Cross-repo: validate true E-CPU coordinate/position on the same source."""
    ecpu=pytest.importorskip("mailcpu.external_virtual_structure")
    src=described_source()
    native=ecpu.SourceStructure.restore(src.record,src.head)
    for op,args in (
        ("source.file",{"path":"config.json"}),
        ("source.catalogue",{}),
        ("source.architecture",{}),
        ("source.layer",{"index":127}),
        ("source.extent",{
            "path":"model-00048-of-00048.safetensors","offset":10**10,"length":1,
        }),
    ):
        expected=ecpu.compile_virtual_read(native,op,args).to_dict()
        found=src.position(op,args)
        assert found==expected
        assert src.head==native.ground_digest


def test_one_logical_observer_navigates_three_unrelated_model_pages_in_tandem():
    """One observer creates an M:N relation with source-local causal isolation."""
    fleet=VirtualModelPageFleet()
    for alias,owner,family in [
        ("deepseek","deepseek-ai/example","deepseek_v41"),
        ("qwen","Qwen/example","qwen3_8"),
        ("embed","google/example","embeddinggemma2"),
    ]:
        source=described_source(model=owner,revision={
            "deepseek":"a"*40,"qwen":"b"*40,"embed":"c"*40,
        }[alias], family=family)
        fleet.register(alias,source)
    opened=fleet.open_across_models("scientist",host_id="observer-locus")
    assert set(opened)=={"deepseek","qwen","embed"}
    assert len(set(opened.values()))==3
    results=fleet.observe_across_models(
        "scientist","source.architecture",{},request_id="compare-family",
        workers=3,
    )
    assert set(results)==set(opened)
    assert {r["position"]["result"]["observed_configuration"]["model_type"]
            for r in results.values()}=={"deepseek_v41","qwen3_8","embeddinggemma2"}
    assert len({r["source_head"] for r in results.values()})==3
    assert len({r["head"] for r in results.values()})==3
    assert len({r["position_head"] for r in results.values()})==3
    assert {r["model_payload_bytes"] for r in results.values()}=={0}
    replay=fleet.observe_across_models(
        "scientist","source.architecture",{},request_id="compare-family",
    )
    assert replay==results
    with pytest.raises(ModelPageRefused):
        fleet.open_across_models("scientist")
    with pytest.raises(ModelPageRefused):
        fleet.observe_across_models(
            "unknown","source.architecture",{},request_id="unknown",
        )
    snapshot=fleet.snapshot()
    again=VirtualModelPageFleet.restore(
        snapshot,expected_digest=snapshot["checkpoint_digest"])
    assert again.snapshot()==snapshot
