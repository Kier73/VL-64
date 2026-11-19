"""A monolithic self-contained model seed really executes in isolated Python."""
from __future__ import annotations

import io
import json
import subprocess
import sys
import zipfile

import pytest

from experiments.model_fabric.model_seed import (
    ROOT,build_model_seed,model_seed_members,
)
from experiments.model_fabric.slice_ground import replay_slice_ground
from experiments.seed_fabric.build import make_archive,input_sources


def subprocess_seed(binary,tmp_path,*args):
    path=tmp_path/"VL64-ModelSeed.pyz"
    path.write_bytes(binary)
    return subprocess.run(
        [sys.executable,"-I",str(path),*args],cwd=tmp_path,
        text=True,capture_output=True,timeout=30)


def test_monolithic_model_seed_is_reproducible_under_64k_and_contains_actual_weights():
    one,report=build_model_seed(ROOT)
    two,second=build_model_seed(ROOT)
    assert one==two
    assert report==second
    assert len(one)<=65536
    assert report["model_seed_bytes"]==len(one)
    assert report["core_source_bytes"]==65093
    assert report["actual_pretrained_weight_payload_bytes"]==5760
    assert report["complete_pretrained_model_execution"] is False
    with zipfile.ZipFile(io.BytesIO(one)) as archive:
        names=set(archive.namelist())
        assert "model_slice.py" in names
        assert "vl64/pretrained_slice_ground.json" in names
        capsule=json.loads(archive.read("vl64/pretrained_slice_ground.json"))
        assert capsule["ground_digest"]==report["model_slice_ground_digest"]
        assert replay_slice_ground(capsule)[
            "status"]=="VERIFIED_LOCAL_REPLAY_OF_PRETRAINED_TENSOR_SLICE"
        for name,raw in input_sources(ROOT).items():
            assert archive.read(name)==raw


def test_single_zipapp_reproduces_pretrained_operator_without_repo_or_install(tmp_path):
    payload,report=build_model_seed()
    verified=subprocess_seed(payload,tmp_path,"model-verify")
    assert verified.returncode==0,verified.stderr
    parsed=json.loads(verified.stdout)
    assert parsed["status"]=="VERIFIED_PORTABLE_VL64_MODEL_SEED"
    assert parsed["no_network_required"] is True
    assert parsed["seed_bytes"]==len(payload)
    result=subprocess_seed(payload,tmp_path,"model-run")
    assert result.returncode==0,result.stderr
    outcome=json.loads(result.stdout)
    assert outcome["status"]=="VERIFIED_LOCAL_REPLAY_OF_PRETRAINED_TENSOR_SLICE"
    assert outcome["actual_model_weight_bytes"]==5760
    assert outcome["ground_digest"]==report["model_slice_ground_digest"]
    assert len(outcome["output_coordinates"])==4
    ground=subprocess_seed(payload,tmp_path,"model-ground")
    assert ground.returncode==0
    assert json.loads(ground.stdout)["ground_digest"]==report[
        "model_slice_ground_digest"]
    source_ground=subprocess_seed(payload,tmp_path,"seed-ground")
    assert source_ground.returncode==0
    assert json.loads(source_ground.stdout)["ground_digest"]==report[
        "software_ground_digest"]
    assert not (tmp_path/"vld").exists()


def test_original_vl64_commands_still_execute_in_portable_model_seed(tmp_path):
    payload,report=build_model_seed()
    result=subprocess_seed(payload,tmp_path,"verify")
    assert result.returncode==0,result.stderr
    navigation=subprocess_seed(
        payload,tmp_path,"navigate","--a","5","--b","1",
        "--x0","3","--steps","1000000000000","--modulus","256")
    assert navigation.returncode==0,navigation.stderr
    assert "Result after 1000000000000 steps: 3" in navigation.stdout


def test_tampering_weight_or_code_is_rejected_before_inference(tmp_path):
    members,manifest,software,capsule=model_seed_members()
    for filename in ("vl64/pretrained_slice_ground.json","model_slice.py",
                     "vld/micro_operators.py"):
        changed=dict(members)
        changed[filename]=changed[filename]+b"\n# tampered\n"
        binary=make_archive(changed)
        out=subprocess_seed(binary,tmp_path,"model-run")
        assert out.returncode!=0
        assert "VL64_MODEL_SEED_REFUSED" in out.stderr
