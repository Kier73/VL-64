"""Procedurally package the 64 KiB VL-64 runtime with a real model slice.

The model seed is one executable Python ZIP containing:
- exact 65,093-byte VL64 source modules, transparently ZIP-compressed,
- a source-generated Ground of module Grounds,
- a bounded BF16 pretrained weight-slice Ground (5,760 raw tensor bytes),
- an audited fixed computation that invokes the compact core operators.

Both SOURCE and ARCHIVE remain <= 65,536 bytes. Requires Python 3.11+.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

from experiments.seed_fabric.build import (
    ROOT,input_sources,synthesize_ground,make_archive,canonical
)
from .slice_ground import replay_slice_ground

CAPSULE=Path("experiments/model_fabric/datasets/2026-11-18-pretrained-bf16-slice-ground.json")
BOOTSTRAP=Path("experiments/model_fabric/model_bootstrap.py")
INTERPRETER=Path("experiments/model_fabric/slice_ground.py")
MAX_BYTES=65536


def sha(value):
    return hashlib.sha256(value).hexdigest()


def model_seed_members(root=ROOT,*,capsule=None):
    root=Path(root)
    sources=input_sources(root)
    software=synthesize_ground(sources)
    if capsule is None:
        capsule=json.loads((root/CAPSULE).read_text())
    verified=replay_slice_ground(capsule)
    if verified["status"]!="VERIFIED_LOCAL_REPLAY_OF_PRETRAINED_TENSOR_SLICE":
        raise ValueError("Unverified actual model slice Ground")
    model_bytes=canonical(capsule)
    bootstrap=(root/BOOTSTRAP).read_bytes()
    operator=(root/INTERPRETER).read_bytes()
    src_files=software["source_files"]
    manifest={
        "schema":"vl64.pretrained-model-seed.v1",
        "python_requires":">=3.11",
        "core_source_bytes":software["core_source_bytes"],
        "source_files":src_files,
        "bootstrap_sha256":sha(bootstrap),
        "interpreter_sha256":sha(operator),
        "software_ground_sha256":sha(canonical(software)),
        "model_slice_ground_sha256":sha(model_bytes),
        "model_slice_ground_digest":capsule["ground_digest"],
        "source_model":capsule["model"],
        "model_revision":capsule["revision"],
        "no_remote_inference_or_network_access_required":True,
        "scope":"source_verified_python_model_slice_replay_not_full_LLM",
    }
    members=dict(sources)
    members.update({
        "__main__.py":bootstrap,
        "model_slice.py":operator,
        "vl64/manifest.json":canonical(manifest),
        "vl64/system_ground.json":canonical(software),
        "vl64/pretrained_slice_ground.json":model_bytes,
    })
    return members,manifest,software,capsule


def build_model_seed(root=ROOT,*,capsule=None):
    members,manifest,software,model=model_seed_members(root,capsule=capsule)
    raw=make_archive(members)
    if len(raw)>MAX_BYTES:
        raise ValueError("Monolithic pretrained model seed exceeds 64 KiB")
    if manifest["core_source_bytes"]>MAX_BYTES:
        raise ValueError("Expanded runtime exceeds 64 KiB")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for name,source in members.items():
            if archive.read(name)!=source:
                raise ValueError("Archive did not reconstruct admitted bytes")
    report={
        "schema":"vl64.pretrained-model-seed-build.v1",
        "status":"REPRODUCIBLE_PORTABLE_MODEL_SEED",
        "model_seed_bytes":len(raw),
        "model_seed_sha256":sha(raw),
        "core_source_bytes":manifest["core_source_bytes"],
        "software_ground_digest":software["ground_digest"],
        "model_slice_ground_digest":model["ground_digest"],
        "actual_pretrained_weight_payload_bytes":5760,
        "model":model["model"],"revision":model["revision"],
        "executable_subgraph":"layer0 first four Q rows after actual RMSNorm",
        "native_python_dependency":">=3.11",
        "complete_pretrained_model_execution":False,
    }
    return raw,report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",default="artifacts/VL64-ModelSeed.pyz")
    args=parser.parse_args()
    path=Path(args.out)
    path.parent.mkdir(parents=True,exist_ok=True)
    archive,report=build_model_seed()
    path.write_bytes(archive)
    (path.with_suffix(".report.json")).write_text(
        json.dumps(report,sort_keys=True,indent=2)+"\n")
    print("VL64_PRETRAINED_MODEL_SEED "+json.dumps(report,sort_keys=True))


if __name__=="__main__":
    main()
