"""Reproducible source-derived VL-64 zipapp and structural software Ground."""
from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from experiments.seed_fabric.build import (
    ROOT,SEED_CEILING,build,extract_sources,input_sources,
    make_archive,make_members,synthesize_ground,verify_seed_bytes,
)


def invoke(binary,tmp_path,*args):
    output=tmp_path/"VL64.pyz"
    output.write_bytes(binary)
    return subprocess.run([sys.executable,"-I",str(output),*args],
                          cwd=tmp_path,text=True,capture_output=True,timeout=25)


def test_monolithic_seed_is_below_ceiling_and_reproducible_byte_for_byte():
    first,ground,report=build(ROOT)
    second,ground2,report2=build(ROOT)
    assert first==second
    assert ground==ground2
    assert report==report2
    assert len(first)<SEED_CEILING
    assert report["status"]=="VERIFIED_MONOLITHIC_SEED"
    assert report["source_bytes"]==65093
    assert report["seed_bytes"]==len(first)
    assert report["ground_digest"]==ground["ground_digest"]
    assert report["compressed_vs_source_ratio"]>1
    assert ground["qualified_reconstruction"]["exact_source_bytes_in_seed"]
    assert not ground["qualified_reconstruction"]["arbitrary_host_equivalence_proven"]


def test_monolithic_seed_runs_with_isolated_python_and_no_installed_package(tmp_path):
    binary,ground,report=build(ROOT)
    verified=invoke(binary,tmp_path,"seed-verify")
    assert verified.returncode==0,verified.stderr
    parsed=json.loads(verified.stdout)
    assert parsed["ground_digest"]==ground["ground_digest"]
    assert parsed["source_bytes"]==65093
    actual=invoke(binary,tmp_path,"seed-ground")
    assert actual.returncode==0,actual.stderr
    observed=json.loads(actual.stdout)
    assert observed==ground
    runtime=invoke(binary,tmp_path,"navigate","--a","5","--b","1",
                   "--x0","3","--steps","1000000000000","--modulus","256")
    assert runtime.returncode==0,runtime.stderr
    assert "terminal_state" in runtime.stdout or "result" in runtime.stdout
    selftest=invoke(binary,tmp_path,"verify")
    assert selftest.returncode==0,selftest.stderr


def test_source_ground_declares_actual_modules_imports_and_public_api():
    sources=input_sources(ROOT)
    ground=synthesize_ground(sources)
    assert len(ground["source_files"])==7
    assert ground["core_source_bytes"]==sum(map(len,sources.values()))
    byname={x["module"]:x for x in ground["observed_program_structure"]}
    assert "vld.micro_control" in byname
    assert "vld.micro_translation" in byname
    assert "vld.__init__" in byname
    assert "ObserverIntention" in byname["vld.__init__"]["public_exports"]
    assert "GroundRegistry" in byname["vld.micro_control"]["top_level_classes"]
    assert any(r["module"]=="micro_kernel" for r in
               byname["vld.__init__"]["relative_imports"])
    assert "ObservedGround" in byname["vld.__init__"]["public_exports"]
    assert ground["entrypoints"]["python_module"]=="vld.__main__"


def test_seed_exact_reconstruction_matches_all_original_source_bytes(tmp_path):
    binary,ground,report=build(ROOT)
    out=tmp_path/"reconstruction"
    unpack=extract_sources(binary,out)
    assert unpack==report
    original=input_sources(ROOT)
    for name,raw in original.items():
        assert (out/name).read_bytes()==raw
    assert synthesize_ground({
        name:(out/name).read_bytes() for name in original})==ground
    with pytest.raises(ValueError,match="occupied"):
        extract_sources(binary,out)


def test_mutation_of_source_in_archive_is_detected_before_import(tmp_path):
    binary,ground,report=build(ROOT)
    with zipfile.ZipFile(io.BytesIO(binary),"r") as z:
        members={name:z.read(name) for name in z.namelist()}
    name="vld/micro_translation.py"
    changed=bytearray(members[name])
    changed[25] ^= 1
    members[name]=bytes(changed)
    altered=make_archive(members)
    with pytest.raises(ValueError,match="source module"):
        verify_seed_bytes(altered)
    test=invoke(altered,tmp_path,"seed-verify")
    assert test.returncode!=0
    assert "VL64_SEED_REFUSED" in test.stderr


def test_bootstrap_ground_and_unknown_file_tampering_are_rejected(tmp_path):
    binary,ground,report=build(ROOT)
    with zipfile.ZipFile(io.BytesIO(binary),"r") as z:
        members={name:z.read(name) for name in z.namelist()}
    for name in ("__main__.py","vl64/system_ground.json"):
        corrupted=dict(members)
        corrupted[name]+=b"#altered"
        altered=make_archive(corrupted)
        with pytest.raises(ValueError,match="bootstrap|Ground"):
            verify_seed_bytes(altered)
    corrupted=dict(members)
    corrupted["surprise.py"]=b"print('not admitted')\n"
    with pytest.raises(ValueError,match="Undeclared"):
        verify_seed_bytes(make_archive(corrupted))


def test_mutation_of_checked_source_recomputes_ground_and_seed(tmp_path):
    copy=tmp_path/"repository"
    (copy/"src/vld").mkdir(parents=True)
    (copy/"experiments/seed_fabric").mkdir(parents=True)
    for name,raw in input_sources(ROOT).items():
        (copy/"src"/name).write_bytes(raw)
    (copy/"experiments/seed_fabric/bootstrap.py").write_bytes(
        (ROOT/"experiments/seed_fabric/bootstrap.py").read_bytes())
    before,old_ground,_=build(copy)
    source=copy/"src/vld/micro_translation.py"
    source.write_bytes(source.read_bytes().replace(
        b"ObservedGround",b"ObservedGround",1)+b"\n# reproducibility-test\n")
    after,new_ground,_=build(copy)
    assert before!=after
    assert old_ground["ground_digest"]!=new_ground["ground_digest"]
    assert old_ground["core_source_bytes"]<new_ground["core_source_bytes"]


def test_no_implicit_extraction_or_execution_authority_in_ground(tmp_path):
    binary,ground,report=build(ROOT)
    root=tmp_path/"isolated"
    root.mkdir()
    entry=invoke(binary,root,"seed-ground")
    assert entry.returncode==0
    assert list(root.iterdir())==[root/"VL64.pyz"]
    assert ground["qualified_reconstruction"][
        "remote_actions_authorized_by_digest"] is False
