"""Zip-imported monolithic VL-64 bootstrap, embedded as archive __main__.py.

Only the Python standard library is required. Verify the archive's complete
admitted member inventory and SHA-256 source Ground before importing vld.
This is integrity, not authentication of who supplied the seed.
"""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

MAX_ARCHIVE_BYTES = 65536
MAX_MEMBER_BYTES = 65536
MANIFEST = "vl64/manifest.json"
GROUND = "vl64/system_ground.json"
GROUND_DOMAIN = b"vl64.software-ground.v1\x00"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def verify_seed(path):
    """Audit every member before importing any code from this archive."""
    if not path.is_file() or path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("Missing or oversized monolithic seed")
    with zipfile.ZipFile(path,"r") as archive:
        entries=archive.infolist()
        names=[item.filename for item in entries]
        if (len(names)<4 or len(names)>16 or len(set(names))!=len(names) or
                MANIFEST not in names or GROUND not in names or
                "__main__.py" not in names):
            raise ValueError("Incomplete, duplicate or unexpected seed inventory")
        if any(item.file_size>MAX_MEMBER_BYTES or
               item.flag_bits&1 or item.filename.endswith("/") or
               ".." in item.filename.split("/") for item in entries):
            raise ValueError("Unsafe seed archive member")
        manifest=json.loads(archive.read(MANIFEST))
        ground=json.loads(archive.read(GROUND))
        if (not isinstance(manifest,dict) or
                manifest.get("schema")!="vl64.monolithic-seed.v1" or
                manifest.get("python_requires")!=">=3.11" or
                not isinstance(manifest.get("source_files"),list) or
                not isinstance(ground,dict) or
                ground.get("schema")!="vl64.software-ground.v1"):
            raise ValueError("Unknown monolithic seed contract")
        modules=manifest["source_files"]
        paths=[item["path"] for item in modules]
        if (not 1<=len(modules)<=10 or paths!=sorted(paths) or
                len(set(paths))!=len(paths) or
                not all(p.startswith("vld/") and p.endswith(".py") and
                        p.count("/")==1 for p in paths)):
            raise ValueError("Unknown seed module layout")
        expected=set(paths)|{MANIFEST,GROUND,"__main__.py"}
        if set(names)!=expected:
            raise ValueError("Undeclared archive members")
        for item in modules:
            data=archive.read(item["path"])
            if (len(data)!=item["length"] or _sha(data)!=item["sha256"]):
                raise ValueError("Module content fingerprint disagrees")
        if sum(item["length"] for item in modules)!=manifest["core_source_bytes"]:
            raise ValueError("Declared source budget inconsistent")
        if manifest["core_source_bytes"]>MAX_ARCHIVE_BYTES:
            raise ValueError("Expanded runtime exceeds 64 KiB")
        if _sha(archive.read("__main__.py"))!=manifest["bootstrap_sha256"]:
            raise ValueError("Bootstrap source fingerprint disagrees")
        if _sha(archive.read(GROUND))!=manifest["ground_sha256"]:
            raise ValueError("Software Ground bytes disagree")
        ground_claim={k:v for k,v in ground.items()
                      if k!="ground_digest"}
        if (_sha(GROUND_DOMAIN+canonical(ground_claim))!=ground.get("ground_digest") or
                ground.get("source_files")!=modules or
                ground.get("core_source_bytes")!=manifest["core_source_bytes"] or
                ground.get("python_requires")!=manifest["python_requires"]):
            raise ValueError("Self Ground is not consistent with admitted source")
        if len(ground["ground_digest"])!=64:
            raise ValueError("Invalid Ground fingerprint")
        return {
            "status":"VERIFIED_MONOLITHIC_SEED",
            "source_bytes":manifest["core_source_bytes"],
            "seed_bytes":path.stat().st_size,
            "ground_digest":ground["ground_digest"],
            "module_count":len(modules),
            "manifest":manifest,
            "ground":ground,
        }


def main():
    if sys.version_info < (3,11):
        raise SystemExit("VL-64 requires Python 3.11 or newer")
    try:
        checked=verify_seed(Path(sys.argv[0]))
    except (OSError,ValueError,KeyError,TypeError,zipfile.BadZipFile,
            json.JSONDecodeError) as exc:
        raise SystemExit("VL64_SEED_REFUSED: "+str(exc)) from None
    command=sys.argv[1] if len(sys.argv)>1 else "help"
    if command=="seed-verify":
        print(json.dumps({k:v for k,v in checked.items()
                          if k not in ("manifest","ground")},sort_keys=True))
        return
    if command=="seed-ground":
        print(json.dumps(checked["ground"],sort_keys=True,indent=2))
        return
    # The existing CLI remains the execution authority and supports verify,
    # navigate and established operations. Seed-specific introspection never
    # installs new operators or enables arbitrary side-effecting code.
    from vld.micro_control import cli_main
    cli_main()


if __name__=="__main__":
    main()
