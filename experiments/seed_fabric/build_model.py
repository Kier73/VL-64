"""Build a standalone model readout using exact declarations of the VL-64 runtime."""
from __future__ import annotations

import argparse
import ast
import json
import sys
import zipfile
from pathlib import Path

from .build import (
    BOOTSTRAP, MANIFEST, MODEL_PROFILE, ROOT, canonical, make_archive, make_members,
    verify_model_budget, verify_seed_bytes,
)
from .bootstrap import verify_seed


def source_declarations(path, names, *, include_imports=True):
    """Project an ordered syntax tree onto declarations N, preserving their exact source bytes."""
    source = path.read_bytes()
    lines = source.splitlines(keepends=True)
    required = set(names)
    found = set()
    pieces = []
    for node in ast.parse(source, filename=str(path)).body:
        symbol = getattr(node, "name", None)
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            symbol = getattr(node.targets[0], "id", None)
        if symbol in required:
            if symbol in found:
                raise ValueError(f"Ambiguous source declaration {symbol} in {path}")
            found.add(symbol)
        elif not (include_imports and isinstance(node, (ast.Import, ast.ImportFrom))):
            continue
        start = min([node.lineno] + [item.lineno for item in getattr(node, "decorator_list", ())])
        pieces.append(b"".join(lines[start - 1:node.end_lineno]).rstrip(b"\r\n"))
    if found != required:
        raise ValueError(f"Missing source declarations {sorted(required - found)} in {path}")
    return b"\n\n".join(pieces) + b"\n"


def model_entrypoint():
    """Verify H(module_i) and sum_i |python_i| <= 2**16 before dispatching the model CLI."""
    if sys.version_info < (3, 11):
        raise SystemExit("VL-64 requires Python 3.11 or newer")
    try:
        archive = Path(sys.argv[0])
        result = verify_seed(archive)
        with zipfile.ZipFile(archive) as packed:
            expanded = result["source_bytes"] + packed.getinfo("__main__.py").file_size
        verify_model_budget(result["manifest"], expanded)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        raise SystemExit("VL64_SEED_REFUSED: " + str(exc)) from None
    if sys.argv[1:2] == ["seed-verify"]:
        receipt = {k: v for k, v in result.items() if k not in ("manifest", "ground")}
        receipt.update(profile=MODEL_PROFILE, expanded_python_bytes=expanded)
        print(json.dumps(receipt, sort_keys=True))
    elif sys.argv[1:2] == ["seed-ground"]:
        print(json.dumps(result["ground"], indent=2, sort_keys=True))
    else:
        from vld.local_seed import cli_main
        cli_main()


def model_bootstrap(root):
    """Compose the canonical integrity predicate V with profile budget predicate B, accepting iff V and B."""
    verifier = source_declarations(root / BOOTSTRAP, (
        "MAX_ARCHIVE_BYTES", "MAX_MEMBER_BYTES", "MANIFEST", "GROUND", "GROUND_DOMAIN",
        "canonical", "_sha", "verify_seed",
    ))
    budget = source_declarations(root / "experiments/seed_fabric/build.py", (
        "MODEL_PROFILE", "SEED_CEILING", "verify_model_budget",
    ), include_imports=False)
    entry = source_declarations(root / "experiments/seed_fabric/build_model.py",
                                ("model_entrypoint",), include_imports=False)
    return b"\n".join((verifier, budget, entry, b'if __name__ == "__main__":\n    model_entrypoint()\n'))


def model_sources(root):
    """P(S) retains exact imported model definitions and the complete decoder and local reader."""
    core = root / "src/vld"
    model = root / "experiments/model_fabric"
    sources = {
        "vld/__init__.py": (
            b"from .micro_kernel import canonical_bytes, domain_digest\n"
            b"from .micro_operators import StructuralLinearOperator, rms_norm\n"
            b"from .micro_substrates import SafetensorsGlance\n"
        ),
        "vld/__main__.py": b'from .local_seed import cli_main\n\nif __name__ == "__main__":\n    cli_main()\n',
        "vld/execution_dag.py": (model / "execution_dag.py").read_bytes(),
        "vld/local_seed.py": (model / "local_seed.py").read_bytes(),
        "vld/hf_glance.py": source_declarations(
            model / "hf_glance.py", ("TIMEOUT", "_sha", "_read", "_url")),
        "vld/tensor_ground.py": source_declarations(
            model / "tensor_ground.py", ("HEADER_LIMIT", "MAX_TENSORS", "MAX_ELEMENTS",
                                         "DTYPE_BYTES", "_sha", "_distinct_pairs", "observe_safetensors_header")),
    }
    selections = {
        "micro_kernel": ("canonical_bytes", "domain_digest"),
        "micro_operators": ("StructuralLinearOperator", "rms_norm"),
        "micro_substrates": ("SafetensorsGlance",),
    }
    for module, names in selections.items():
        sources[f"vld/{module}.py"] = source_declarations(core / f"{module}.py", names)
    return sources


def build_model(root=ROOT):
    """Enforce both |ZIP(P)| <= 2**16 and sum_{p in P, p ends in .py} |p| <= 2**16."""
    root = Path(root).resolve()
    bootstrap = model_bootstrap(root)
    members, manifest, ground = make_members(root, sources=model_sources(root), bootstrap=bootstrap)
    expanded = sum(len(raw) for path, raw in members.items() if path.endswith(".py"))
    manifest.update(profile=MODEL_PROFILE, expanded_python_bytes=expanded)
    verify_model_budget(manifest, expanded)
    members[MANIFEST] = canonical(manifest)
    binary = make_archive(members)
    return binary, ground, verify_seed_bytes(binary)


def main():
    """Persist the verified archive A, with |A| equal to the byte count in its receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts/VL64-model.pyz"))
    args = parser.parse_args()
    binary, _, report = build_model()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(binary)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
