"""Bootstrap for the portable VL-64 pretrained-model-slice seed.

Verifies every expected source and weight file before using the bundled core.
No network, package installation or repository checkout is needed, but Python
3.11+ is still required. No model metadata is executed as arbitrary Python.
"""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

MAX_SEED=65536
MANIFEST="vl64/manifest.json"
GROUND="vl64/system_ground.json"
SLICE="vl64/pretrained_slice_ground.json"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(obj):
    return json.dumps(obj,sort_keys=True,separators=(",",":"),
                      ensure_ascii=True,allow_nan=False).encode()


def audit(path):
    if not path.is_file() or path.stat().st_size>MAX_SEED:
        raise ValueError("Missing or oversized monolithic model seed")
    with zipfile.ZipFile(path) as archive:
        infos=archive.infolist()
        names=[item.filename for item in infos]
        if len(names)!=len(set(names)) or len(names)<10 or len(names)>20:
            raise ValueError("Unqualified model seed inventory")
        if any(item.file_size>MAX_SEED or item.flag_bits&1 or
               item.filename.endswith("/") or
               ".." in item.filename.split("/") for item in infos):
            raise ValueError("Unsafe model seed member")
        manifest=json.loads(archive.read(MANIFEST))
        if manifest.get("schema")!="vl64.pretrained-model-seed.v1" or (
                manifest.get("python_requires")!=">=3.11"):
            raise ValueError("Unknown model seed manifest")
        sources=manifest["source_files"]
        paths=[item["path"] for item in sources]
        if (len(paths)!=7 or paths!=sorted(paths) or
                any(not name.startswith("vld/") or name.count("/")!=1 or
                    not name.endswith(".py") for name in paths)):
            raise ValueError("Unrecognized model seed software modules")
        expected=set(paths)|{MANIFEST,GROUND,SLICE,"__main__.py","model_slice.py"}
        if set(names)!=expected:
            raise ValueError("Undeclared or missing model seed member")
        for item in sources:
            raw=archive.read(item["path"])
            if len(raw)!=item["length"] or sha(raw)!=item["sha256"]:
                raise ValueError("Invalid embedded VL64 source identity")
        if sum(item["length"] for item in sources)!=manifest["core_source_bytes"] or (
                manifest["core_source_bytes"]>MAX_SEED):
            raise ValueError("Compact runtime source ceiling violated")
        for name,key in (
            ("__main__.py","bootstrap_sha256"),
            ("model_slice.py","interpreter_sha256"),
            (GROUND,"software_ground_sha256"),
            (SLICE,"model_slice_ground_sha256"),
        ):
            if sha(archive.read(name))!=manifest[key]:
                raise ValueError("Invalid admitted model seed member digest")
        software=json.loads(archive.read(GROUND))
        capsule=json.loads(archive.read(SLICE))
        if (software.get("source_files")!=sources or
                software.get("core_source_bytes")!=manifest["core_source_bytes"] or
                software.get("schema")!="vl64.software-ground.v1" or
                capsule.get("schema")!="vl64.ranged-pretrained-slice-ground.v1"):
            raise ValueError("Invalid structural model-seed Ground")
        software_bare={k:v for k,v in software.items() if k!="ground_digest"}
        if software.get("ground_digest")!=sha(
                b"vl64.software-ground.v1\x00"+canonical(software_bare)):
            raise ValueError("Software Ground digest invalid")
        if capsule.get("ground_digest")!=manifest["model_slice_ground_digest"]:
            raise ValueError("Pretrained model slice Ground identity mismatch")
        return manifest,software,capsule


def main():
    if sys.version_info<(3,11):
        raise SystemExit("VL64_MODEL_SEED_REFUSED: Python 3.11+ required")
    try:
        manifest,software,model=audit(Path(sys.argv[0]))
    except (OSError,ValueError,KeyError,TypeError,zipfile.BadZipFile) as exc:
        raise SystemExit("VL64_MODEL_SEED_REFUSED: "+str(exc)) from None
    command=sys.argv[1] if len(sys.argv)>1 else "model-verify"
    if command=="seed-ground":
        print(json.dumps(software,sort_keys=True,indent=2))
        return
    if command=="model-ground":
        print(json.dumps(model,sort_keys=True,indent=2))
        return
    # Both verified module sources are now imported through zipimport.
    from model_slice import replay_slice_ground
    verified=replay_slice_ground(model)
    if command=="model-verify":
        print(json.dumps({
            "status":"VERIFIED_PORTABLE_VL64_MODEL_SEED",
            "seed_bytes":Path(sys.argv[0]).stat().st_size,
            "core_source_bytes":manifest["core_source_bytes"],
            "software_ground_digest":software["ground_digest"],
            "pretrained_slice_ground_digest":model["ground_digest"],
            "model_execution_receipt":verified["receipt_digest"],
            "no_network_required":True,
        },sort_keys=True))
        return
    if command=="model-run":
        print(json.dumps(verified,sort_keys=True,indent=2))
        return
    from vld.micro_control import cli_main
    cli_main()


if __name__=="__main__":
    main()
