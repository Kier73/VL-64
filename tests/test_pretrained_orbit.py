"""Independent causal restart, byte-page, lineage and observer rehome tests."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from vld import domain_digest
from experiments.model_fabric.execution_dag import compile_decoder_dag, MAX_RANGE
from experiments.model_fabric.pretrained_chain import execute_pretrained_chain
from experiments.model_fabric.pretrained_orbit import (
    ContentAddressedPages, RecoverableModelOrbit, save_checkpoint, load_checkpoint,
    _get_blob, _put_blob, _get_f64, SCHEMA,
)
from test_execution_dag import qualified_host, scalar_reference, synthetic_pretrained_ground


def fixture(tmp_path, *, dtype="BF16", layers=2):
    observed, raw, weights = synthetic_pretrained_ground(dtype=dtype,layers=layers)
    plan=compile_decoder_dag(observed)
    a=qualified_host(observed,raw,observer="host-a")
    b=qualified_host(observed,raw,observer="host-b")
    store=ContentAddressedPages(tmp_path/"pages")
    policy=dict(approved_producers={"host-a-compute","host-b-compute"})
    def create(host, actor, epoch, owner):
        return RecoverableModelOrbit(plan,{-1:host,**{i:host for i in range(layers)}},
            approved_hosts={i:{"host-a" if host is a else "host-b"}
                            for i in range(-1,layers)},
            compute_actors={i:actor for i in range(layers-1)},
            store=store,owner=owner,epoch=epoch,last_layer=layers-1,**policy)
    def restore(snapshot,head,*,epoch=1,host=b):
        return RecoverableModelOrbit.restore(plan,snapshot,expected_head=head,
            new_owner="receiver-b",new_epoch=epoch,
            range_hosts={i:host for i in range(-1,layers)},
            approved_hosts={i:{"host-b" if host is b else "host-a"}
                            for i in range(-1,layers)},
            compute_actors={i:"host-b-compute" for i in range(layers-1)},
            store=store,**policy)
    return plan,observed,raw,weights,a,b,store,create,restore


def _assert_decoder_logits_agree(actual, weights, architecture, tokens):
    x=actual
    eps=architecture["rms_norm_eps"]
    gamma=weights["model.norm.weight"][1]
    scale=1/math.sqrt(math.fsum(z*z for z in x)/len(x)+eps)
    norm=[z*scale*g for z,g in zip(x,gamma)]
    shape,table=weights["model.embed_tokens.weight"]
    logits=[math.fsum(table[r*shape[1]+c]*norm[c]
                      for c in range(shape[1])) for r in range(shape[0])]
    ref=scalar_reference(tokens,weights,architecture)
    assert max(abs(a-b) for a,b in zip(logits,ref))<2e-6


@pytest.mark.parametrize("dtype",["BF16","F32"])
def test_restored_two_layer_kv_matches_fresh_independent_decoder_and_rehomes(tmp_path,dtype):
    plan,observed,raw,weights,a,b,store,create,restore=fixture(tmp_path,dtype=dtype)
    first=create(a,"host-a-compute",0,"receiver-a")
    first.advance(4,expected_position=0)
    first.advance(3,expected_position=1)
    checkpoint=first.checkpoint()
    head=save_checkpoint(tmp_path/"checkpoint.json",checkpoint)
    persisted=load_checkpoint(tmp_path/"checkpoint.json")
    resumed,last=restore(persisted,head)
    assert resumed.tokens==[4,3]
    assert resumed.previous_checkpoint==head
    assert resumed.epoch==1
    assert resumed.layers[0].step==resumed.layers[1].step==2
    assert last==_get_f64(store, checkpoint["last_hidden"],plan["architecture"]["hidden"])
    # Replay a fresh, uninterrupted reference rather than taking the restart output on faith.
    third=resumed.advance(6,expected_position=2)
    arch=observed["architecture_fields"]
    _assert_decoder_logits_agree(third["state"],weights,arch,[4,3,6])
    assert resumed.tokens==[4,3,6]
    assert checkpoint["weight_witness_chains"]["0"] != resumed.readers[0].receipt_chain
    assert all(len(layer.keys)==3 for layer in resumed.layers)
    assert resumed.checkpoint()["previous_checkpoint"]==head


def test_restarted_checkpoint_is_byte_identical_and_recomputed_head_is_stable(tmp_path):
    _,_,_,_,a,_,store,create,restore=fixture(tmp_path)
    s=create(a,"host-a-compute",0,"local")
    s.advance(4)
    head=save_checkpoint(tmp_path/"state.json",s.checkpoint())
    assert head == s.checkpoint()["checkpoint_digest"]
    r,_=restore(load_checkpoint(tmp_path/"state.json"),head)
    assert r.checkpoint()["transcript_head"] == s.transcript_head
    assert r.tokens == [4]


def test_stale_or_recomputed_checkpoint_refused_against_external_pin(tmp_path):
    _,_,_,_,a,_,store,create,restore=fixture(tmp_path)
    s=create(a,"host-a-compute",0,"origin")
    s.advance(4)
    old=s.checkpoint()
    s.advance(3)
    newest=s.checkpoint()
    with pytest.raises(ValueError,match="externally pinned"):
        restore(old,newest["checkpoint_digest"])
    changed=copy.deepcopy(newest)
    changed["tokens"][-1]=7
    changed["checkpoint_digest"]=domain_digest(SCHEMA,{k:v for k,v in changed.items() if k!="checkpoint_digest"})
    with pytest.raises(ValueError,match="externally pinned"):
        restore(changed,newest["checkpoint_digest"])
    with pytest.raises(ValueError,match="epoch"):
        restore(newest,newest["checkpoint_digest"],epoch=0)


def test_corrupted_missing_or_replaced_kv_page_refused(tmp_path):
    _,_,_,_,a,_,store,create,restore=fixture(tmp_path)
    s=create(a,"host-a-compute",0,"origin")
    s.advance(4)
    checkpoint=s.checkpoint()
    head=checkpoint["checkpoint_digest"]
    page=next(iter(s.kv_refs[0][0]["k"]["pages"]))
    (store.root/page).write_bytes(b"NOT ORIGINAL")
    with pytest.raises(ValueError,match="page|digest"):
        restore(checkpoint,head)
    (store.root/page).unlink()
    with pytest.raises(FileNotFoundError):
        restore(checkpoint,head)


def test_wrong_model_dag_refused_before_acquiring_any_new_weights(tmp_path):
    plan,_,_,_,a,b,store,create,_=fixture(tmp_path)
    s=create(a,"host-a-compute",0,"origin")
    s.advance(4)
    snap=s.checkpoint()
    wrong=dict(plan,model="not-same/model")
    def forbidden(*args,**kwargs):
        raise AssertionError("No weight read allowed before state validation")
    with pytest.raises(ValueError,match="identity"):
        RecoverableModelOrbit.restore(wrong,snap,expected_head=snap["checkpoint_digest"],
            new_owner="b",new_epoch=1,range_hosts={-1:forbidden,0:forbidden,1:forbidden},
            approved_hosts={-1:{"host-b"},0:{"host-b"},1:{"host-b"}},
            compute_actors={0:"host-b-compute"},approved_producers={"host-b-compute"},store=store)


def test_position_failure_does_not_advance_causal_frontier_or_checkpoint(tmp_path):
    _,_,_,_,a,_,store,create,_=fixture(tmp_path)
    session=create(a,"host-a-compute",0,"origin")
    session.advance(4)
    head=session.checkpoint()["checkpoint_digest"]
    steps=[layer.step for layer in session.layers]
    session.transfer=lambda packet: (_ for _ in ()).throw(ValueError("transport lost"))
    with pytest.raises(ValueError,match="transport"):
        session.advance(3,expected_position=1)
    assert session.tokens==[4]
    assert [layer.step for layer in session.layers]==steps
    assert session.checkpoint()["checkpoint_digest"]==head
    session.transfer=None
    session.advance(3,expected_position=1)
    with pytest.raises(ValueError,match="out-of-order"):
        session.advance(2,expected_position=1)


def test_checkpoint_rehome_refuses_unqualified_new_range_host(tmp_path):
    _,_,_,_,a,b,store,create,restore=fixture(tmp_path)
    session=create(a,"host-a-compute",0,"origin")
    session.advance(4)
    snapshot=session.checkpoint()
    head=snapshot["checkpoint_digest"]
    relocated,_=restore(snapshot,head,host=b)
    real=relocated.readers[0].host
    def impostor(request):
        result=real(request)
        result["observer_host"]="unauthorized-source"
        return result
    relocated.readers[-1].host=impostor
    with pytest.raises(ValueError,match="Unqualified"):
        relocated.advance(3)
    assert relocated.tokens==[4]
    assert all(layer.step==1 for layer in relocated.layers)


def test_bounded_pages_preserve_exact_f64_and_reject_nonfinite(tmp_path):
    store=ContentAddressedPages(tmp_path/"pages")
    raw=bytes(range(256))*1000
    manifest=_put_blob(store,raw)
    assert len(manifest["pages"])==math.ceil(len(raw)/MAX_RANGE)
    assert all((store.root/p).stat().st_size<=MAX_RANGE for p in manifest["pages"])
    assert _get_blob(store,manifest,expected_bytes=len(raw))==raw
    with pytest.raises(ValueError,match="finite"):
        _get_f64(store,_put_blob(store,bytes.fromhex("000000000000f87f")),1)


def test_checkpoint_not_required_to_repeat_previous_token_weight_reads(tmp_path):
    _,_,_,_,a,b,store,create,restore=fixture(tmp_path)
    before=create(a,"host-a-compute",0,"origin")
    before.advance(4)
    checkpoint=before.checkpoint()
    saved=[]
    def tracking(request):
        saved.append((request["tensor"],request["byte_begin"]))
        return b(request)
    resumed,_=RecoverableModelOrbit.restore(before.plan,checkpoint,
        expected_head=checkpoint["checkpoint_digest"],new_owner="receiver-b",new_epoch=1,
        range_hosts={-1:tracking,0:tracking,1:tracking},
        approved_hosts={-1:{"host-b"},0:{"host-b"},1:{"host-b"}},
        compute_actors={0:"host-b-compute"},approved_producers={"host-b-compute"},store=store)
    assert not any(x[0]=="model.embed_tokens.weight" for x in saved)
    resumed.advance(3)
    assert len([x for x in saved if x[0]=="model.embed_tokens.weight"])==1


def test_restart_in_fresh_python_process_uses_persisted_pages(tmp_path):
    plan,observed,raw,weights,a,_,store,create,_=fixture(tmp_path)
    session=create(a,"host-a-compute",0,"origin")
    session.advance(4)
    session.advance(3)
    head=save_checkpoint(tmp_path/"state.json",session.checkpoint())
    (tmp_path/"model.json").write_text(json.dumps(observed))
    (tmp_path/"weights.bin").write_bytes(raw)
    script=r'''
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); sys.path.insert(0,"tests")
try: from test_execution_dag import qualified_host
except ImportError:
    from test_execution_dag import qualified_host
from experiments.model_fabric.execution_dag import compile_decoder_dag
from experiments.model_fabric.pretrained_orbit import ContentAddressedPages, RecoverableModelOrbit,load_checkpoint
observed=json.loads((root/"model.json").read_text()); plan=compile_decoder_dag(observed)
provider=qualified_host(observed,(root/"weights.bin").read_bytes(),observer="new-host")
resumed,previous=RecoverableModelOrbit.restore(plan,load_checkpoint(root/"state.json"),
  expected_head=sys.argv[2],new_owner="new-process",new_epoch=1,
  range_hosts={-1:provider,0:provider,1:provider},
  approved_hosts={-1:{"new-host"},0:{"new-host"},1:{"new-host"}},
  compute_actors={0:"new-process"},approved_producers={"new-process"},
  store=ContentAddressedPages(root/"pages"))
result=resumed.advance(6,expected_position=2)
print(json.dumps({"state":result["state"],"tokens":resumed.tokens,"source":resumed.owner}))
'''
    repo_root=Path(__file__).resolve().parents[1]
    subprocess_cwd=(repo_root if (repo_root/"experiments").is_dir()
                    else Path(__file__).resolve().parent)
    proc=subprocess.run([sys.executable,"-c",script,str(tmp_path),head],
        cwd=subprocess_cwd,capture_output=True,text=True,timeout=30)
    assert proc.returncode==0,proc.stderr
    output=json.loads(proc.stdout.strip())
    assert output["source"]=="new-process" and output["tokens"]==[4,3,6]
    _assert_decoder_logits_agree(output["state"],weights,observed["architecture_fields"],[4,3,6])


def test_io_failure_before_publication_leaves_orbit_frontier_unchanged(tmp_path,monkeypatch):
    _,_,_,_,a,_,store,create,_=fixture(tmp_path)
    session=create(a,"host-a-compute",0,"origin")
    session.advance(4)
    head=session.checkpoint()["checkpoint_digest"]
    original=store.put
    counter=[0]
    def fail_once(raw):
        counter[0]+=1
        if counter[0]==2:
            raise OSError("simulated partial paged store failure")
        return original(raw)
    monkeypatch.setattr(store,"put",fail_once)
    with pytest.raises(OSError,match="partial"):
        session.advance(3)
    assert session.tokens==[4]
    assert session.checkpoint()["checkpoint_digest"]==head
    assert all(layer.step==1 for layer in session.layers)
    monkeypatch.setattr(store,"put",original)
    session.advance(3,expected_position=1)
    assert session.tokens==[4,3]


def test_approved_rehome_cannot_substitute_new_weights_with_fresh_self_hashes(tmp_path):
    plan,_,_,_,a,b,store,create,_=fixture(tmp_path)
    first=create(a,"host-a-compute",0,"origin")
    first.advance(4)
    snap=first.checkpoint()
    head=snap["checkpoint_digest"]
    def substituted(request):
        reply=b(request)
        if request["tensor"]=="model.layers.0.mlp.down_proj.weight":
            values=bytearray(reply["payload"])
            values[0] ^= 1
            reply["payload"]=bytes(values)
            reply["sha256"]=hashlib.sha256(reply["payload"]).hexdigest()
        return reply
    with pytest.raises(ValueError,match="different static weight snapshot"):
        RecoverableModelOrbit.restore(plan,snap,expected_head=head,
            new_owner="replacement",new_epoch=1,
            range_hosts={-1:substituted,0:substituted,1:substituted},
            approved_hosts={-1:{"host-b"},0:{"host-b"},1:{"host-b"}},
            compute_actors={0:"host-b-compute"},approved_producers={"host-b-compute"},
            store=store)


def test_weight_identity_survives_rehome_but_host_evidence_changes(tmp_path):
    _,_,_,_,a,b,store,create,restore=fixture(tmp_path)
    session=create(a,"host-a-compute",0,"origin")
    session.advance(4)
    snap=session.checkpoint()
    later,_=restore(snap,snap["checkpoint_digest"])
    assert later.static_weight_identities == snap["static_weight_identities"]
    assert later.readers[0].receipt_chain != snap["weight_witness_chains"]["0"]


def test_v1_checkpoint_cannot_implicitly_claim_v2_weight_continuity(tmp_path):
    _,_,_,_,a,b,store,create,restore=fixture(tmp_path)
    s=create(a,"host-a-compute",0,"origin")
    s.advance(4)
    snap=s.checkpoint()
    downgraded=dict(snap)
    downgraded["schema"]="vl64.model-orbit-checkpoint.v1"
    del downgraded["static_weight_identities"]
    downgraded["checkpoint_digest"]=domain_digest(downgraded["schema"],
        {k:v for k,v in downgraded.items() if k!="checkpoint_digest"})
    with pytest.raises(ValueError,match="identity"):
        restore(downgraded,downgraded["checkpoint_digest"])
