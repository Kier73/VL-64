"""Deterministic, source-derived VL-64 monolithic seed and software Ground.

The seed is a zipped Python application containing the original seven modules,
a self-verifying entrypoint, a source inventory, and a structural software
Ground. The *archive* must fit the same 64 KiB ceiling as the source runtime.
The current project-wide experiments and tests are deliberately NOT bundled.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path("src/vld")
BOOTSTRAP=Path("experiments/seed_fabric/bootstrap.py")
MANIFEST="vl64/manifest.json"
GROUND="vl64/system_ground.json"
SEED_CEILING=65536
MODEL_PROFILE="vl64.standalone-model-readout.v1"
ZIP_DATE=(1980,1,1,0,0,0)
DOMAIN=b"vl64.software-ground.v1\x00"


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),
                      allow_nan=False,ensure_ascii=True).encode("utf-8")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def program_structure(name,data):
    tree=ast.parse(data.decode("utf-8"),filename=name)
    classes=[]
    functions=[]
    relatives=[]
    exported=[]
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
            functions.append(node.name)
        elif isinstance(node,ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node,ast.ImportFrom) and node.level:
            relatives.append({
                "module":node.module or "",
                "depth":node.level,
                "symbols":[item.name for item in node.names],
            })
        elif (isinstance(node,ast.Assign) and
                any(isinstance(t,ast.Name) and t.id=="__all__"
                    for t in node.targets)):
            try:
                extracted=ast.literal_eval(node.value)
            except (ValueError,TypeError,MemoryError,RecursionError):
                extracted=[]
            if isinstance(extracted,(list,tuple)) and all(
                    isinstance(s,str) for s in extracted):
                exported=list(extracted)
    return {
        "module":name.removesuffix(".py").replace("/","."),
        "top_level_classes":classes,
        "top_level_functions":functions,
        "relative_imports":relatives,
        "public_exports":exported,
    }


def input_sources(root):
    source=root/SOURCE
    files=sorted(source.glob("*.py"))
    if (len(files)!=7 or any(not file.is_file() or file.is_symlink()
                             for file in files) or
            not (source/"__init__.py").is_file() or
            not (source/"__main__.py").is_file()):
        raise ValueError("Expected exactly seven admitted VL-64 module files")
    files_bytes={
        "vld/"+file.name:file.read_bytes()
        for file in files
    }
    source_bytes=sum(map(len,files_bytes.values()))
    if not 1<=source_bytes<=SEED_CEILING:
        raise ValueError("VL-64 core source exceeds 65,536 bytes")
    if any(len(value)>SEED_CEILING for value in files_bytes.values()):
        raise ValueError("Oversized seed source member")
    return files_bytes


def synthesize_ground(source_files):
    """Discover the actual source-package declarations from Python AST."""
    if not source_files or not all(isinstance(raw,bytes)
                                  for raw in source_files.values()):
        raise TypeError("Ground generator requires actual byte source files")
    inventory=[
        {"path":path,"length":len(raw),"sha256":sha(raw)}
        for path,raw in sorted(source_files.items())
    ]
    if sum(item["length"] for item in inventory)>SEED_CEILING:
        raise ValueError("Expanded source exceeds 64 KiB")
    structures=[program_structure(path,raw) for path,raw in
                sorted(source_files.items())]
    # Every module has its own content-derived software Ground; the root Ground
    # is the immutable relation/index over those per-module descriptions.
    module_grounds=[]
    for member,structure in zip(inventory,structures):
        atom={
            "schema":"vl64.source-module-ground.v1",
            "module":structure["module"],
            "source_path":member["path"],
            "source_sha256":member["sha256"],
            "structure":structure,
        }
        module_grounds.append({
            "module":atom["module"],
            "source_path":atom["source_path"],
            "source_sha256":atom["source_sha256"],
            "ground_digest":sha(b"vl64.source-module-ground.v1\x00"+
                                canonical(atom)),
        })
    present={item["module"] for item in module_grounds}
    dependencies=[]
    for structure in structures:
        for edge in structure["relative_imports"]:
            if edge["depth"]!=1 or not edge["module"]:
                continue
            target="vld."+edge["module"]
            if target in present:
                dependencies.append({
                    "source_module":structure["module"],
                    "target_module":target,
                    "imported_names":edge["symbols"],
                    "evidence_kind":"parsed_relative_import",
                })
    dependencies.sort(key=lambda row:(
        row["source_module"],row["target_module"],row["imported_names"]))

    ground={
        "schema":"vl64.software-ground.v1",
        "description":"Observed Python-source structure of the exact VL-64 runtime",
        "source_files":inventory,
        "core_source_bytes":sum(item["length"] for item in inventory),
        "python_requires":">=3.11",
        "observed_program_structure":structures,
        "module_grounds":module_grounds,
        "module_relations":dependencies,
        "ground_role":"source-derived-ground-of-module-grounds",
        "entrypoints":{
            "package":"vld",
            "python_module":"vld.__main__",
            "seed":"__main__.py",
        },
        "qualified_reconstruction":{
            "exact_source_bytes_in_seed":True,
            "independent_python_interpreter_required":True,
            "arbitrary_host_equivalence_proven":False,
            "remote_actions_authorized_by_digest":False,
        },
        "representation_scope":
            "exact-source-files-and-syntactic-structural-relations-only",
    }
    ground["ground_digest"]=sha(DOMAIN+canonical(ground))
    return ground


def make_members(root, *, sources=None, bootstrap=None):
    """A seed inventory commits to each pair (length(s_i), SHA256(s_i))."""
    if sources is None:
        sources=input_sources(root)
    ground=synthesize_ground(sources)
    if bootstrap is None:
        bootstrap=(root/BOOTSTRAP).read_bytes()
    if not bootstrap or len(bootstrap)>SEED_CEILING:
        raise ValueError("Invalid bootstrap loader")
    inventory=ground["source_files"]
    manifest={
        "schema":"vl64.monolithic-seed.v1",
        "python_requires":">=3.11",
        "core_source_bytes":ground["core_source_bytes"],
        "source_files":inventory,
        "bootstrap_sha256":sha(bootstrap),
        "ground_sha256":sha(canonical(ground)),
        "scope":"reproducible_software_source_carrier_not_native_machine_code",
    }
    members=dict(sources)
    members["__main__.py"]=bootstrap
    members[MANIFEST]=canonical(manifest)
    members[GROUND]=canonical(ground)
    return members,manifest,ground


def make_archive(members):
    """Canonical member ordering, mode, time, compression and filename."""
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,"w",compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9,strict_timestamps=True) as archive:
        for filename,data in sorted(members.items()):
            if filename.startswith("/") or ".." in filename.split("/"):
                raise ValueError("Unsafe archive path")
            info=zipfile.ZipInfo(filename,date_time=ZIP_DATE)
            info.create_system=3
            info.external_attr=0o644<<16
            info.compress_type=zipfile.ZIP_DEFLATED
            info.extra=b""
            archive.writestr(info,data,compress_type=zipfile.ZIP_DEFLATED,
                             compresslevel=9)
    binary=stream.getvalue()
    if len(binary)>SEED_CEILING:
        raise ValueError("Monolithic executable seed exceeds 65,536 bytes")
    return binary


def verify_model_budget(manifest, expanded):
    """For the model profile, B = sum |p| over every .py, including __main__, and B <= 2**16."""
    declared = manifest.get("expanded_python_bytes")
    if (manifest.get("profile") != MODEL_PROFILE or type(declared) is not int
            or declared != expanded or not 0 < expanded <= SEED_CEILING):
        raise ValueError("Invalid model profile or expanded Python byte budget")


def verify_seed_bytes(binary):
    """Full independent member and self-Ground validation before use."""
    if not isinstance(binary,bytes) or len(binary)>SEED_CEILING:
        raise ValueError("Invalid bounded seed bytes")
    with zipfile.ZipFile(io.BytesIO(binary),"r") as archive:
        entries=archive.infolist()
        names=[item.filename for item in entries]
        if len(names)!=len(set(names)) or not 4<=len(names)<=16:
            raise ValueError("Invalid seed member count")
        if MANIFEST not in names or GROUND not in names or "__main__.py" not in names:
            raise ValueError("Seed has no verified bootstrap")
        if any(item.file_size>SEED_CEILING or item.flag_bits&1 or
               item.filename.endswith("/") or
               ".." in item.filename.split("/") for item in entries):
            raise ValueError("Unsafe seed archive member")
        manifest=json.loads(archive.read(MANIFEST))
        ground=json.loads(archive.read(GROUND))
        if manifest.get("schema")!="vl64.monolithic-seed.v1" or (
                ground.get("schema")!="vl64.software-ground.v1"):
            raise ValueError("Unrecognized seed contract")
        files=manifest["source_files"]
        if (not isinstance(files,list) or
                not 1<=len(files)<=10 or
                any(not isinstance(item,dict) or
                    set(item)!={"path","length","sha256"} or
                    not isinstance(item["path"],str) or
                    not item["path"].startswith("vld/") or
                    item["path"].count("/")!=1 or
                    not item["path"].endswith(".py") or
                    type(item["length"]) is not int or
                    not 0<=item["length"]<=SEED_CEILING or
                    not isinstance(item["sha256"],str) or
                    len(item["sha256"])!=64 for item in files) or
                [item["path"] for item in files]!=sorted(
                    item["path"] for item in files) or
                set(names)!={MANIFEST,GROUND,"__main__.py"}|
                            {item["path"] for item in files}):
            raise ValueError("Undeclared seed archive member")
        module_bytes={}
        for file in files:
            data=archive.read(file["path"])
            if len(data)!=file["length"] or sha(data)!=file["sha256"]:
                raise ValueError("Unverified source module")
            module_bytes[file["path"]]=data
        if sha(archive.read("__main__.py"))!=manifest["bootstrap_sha256"]:
            raise ValueError("Unverified archive bootstrap")
        if sha(archive.read(GROUND))!=manifest["ground_sha256"]:
            raise ValueError("Unverified structural Ground member")
        regenerated=synthesize_ground(module_bytes)
        if regenerated!=ground or regenerated["source_files"]!=files:
            raise ValueError("Self Ground is not reproducible from source")
        if sum(x["length"] for x in files)!=manifest["core_source_bytes"]:
            raise ValueError("Expanded module budget mismatch")
        report = {
            "status":"VERIFIED_MONOLITHIC_SEED",
            "seed_bytes":len(binary),
            "source_bytes":manifest["core_source_bytes"],
            "ground_digest":ground["ground_digest"],
            "modules":len(files),
            "compressed_vs_source_ratio":round(
                manifest["core_source_bytes"]/len(binary),4),
        }
        if "vld/local_seed.py" in names or "profile" in manifest:
            expanded = sum(item.file_size for item in entries if item.filename.endswith(".py"))
            verify_model_budget(manifest, expanded)
            report.update(profile=MODEL_PROFILE, expanded_python_bytes=expanded)
        return report


def build(root=ROOT):
    root=Path(root).resolve()
    members,manifest,ground=make_members(root)
    binary=make_archive(members)
    report=verify_seed_bytes(binary)
    return binary,ground,report


def extract_sources(binary,target):
    """Optional explicit reconstruction; source paths only, never extractall."""
    verified=verify_seed_bytes(binary)
    target=Path(target)
    if target.is_symlink() or (target.exists() and (
            not target.is_dir() or any(target.iterdir()))):
        raise ValueError("Refusing to overwrite an occupied destination")
    with zipfile.ZipFile(io.BytesIO(binary)) as archive:
        manifest=json.loads(archive.read(MANIFEST))
        for member in manifest["source_files"]:
            if not member["path"].startswith("vld/") or (
                    member["path"].count("/")!=1):
                raise ValueError("Invalid reconstruction target")
            final=target/member["path"]
            final.parent.mkdir(parents=True,exist_ok=True)
            final.write_bytes(archive.read(member["path"]))
    return verified


def main():
    parser=argparse.ArgumentParser(description="Procedurally derive the VL-64 monolithic seed and self Ground")
    parser.add_argument("action",choices=("build","inspect","extract"),
                        nargs="?",default="build")
    parser.add_argument("--root",default=str(ROOT))
    parser.add_argument("--out",default="artifacts/VL64.pyz")
    parser.add_argument("--extract-to",default="artifacts/reconstructed")
    args=parser.parse_args()
    if args.action=="build":
        binary,ground,report=build(Path(args.root))
        output=Path(args.out)
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_bytes(binary)
        ground_path=output.with_suffix(".ground.json")
        ground_path.write_text(json.dumps(ground,sort_keys=True,indent=2)+"\n")
    elif args.action=="inspect":
        binary=Path(args.out).read_bytes()
        report=verify_seed_bytes(binary)
    else:
        binary=Path(args.out).read_bytes()
        report=extract_sources(binary,Path(args.extract_to))
    print("VL64_MONOLITHIC_SEED "+json.dumps(report,sort_keys=True))


if __name__=="__main__":
    main()
