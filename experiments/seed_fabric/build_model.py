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


def source_declarations(path, names, *, include_imports=True, used_imports_only=False):
    """Project an ordered syntax tree onto declarations N, preserving their exact source bytes."""
    source = path.read_bytes()
    lines = source.splitlines(keepends=True)
    required = set(names)
    found = set()
    pieces = []
    nodes = ast.parse(source, filename=str(path)).body
    def symbol_of(node):
        """π(node) is the bound declaration name for a function, class or single assignment."""
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            return getattr(node.targets[0], "id", None)
        return getattr(node, "name", None)

    referenced = {child.id for node in nodes if symbol_of(node) in required
                  for child in ast.walk(node) if isinstance(child, ast.Name)}
    for node in nodes:
        symbol = symbol_of(node)
        if symbol in required:
            if symbol in found:
                raise ValueError(f"Ambiguous source declaration {symbol} in {path}")
            found.add(symbol)
        elif not (include_imports and isinstance(node, (ast.Import, ast.ImportFrom))):
            continue
        elif used_imports_only and getattr(node, "module", None) != "__future__":
            if not any((item.asname or item.name.split(".")[0]) in referenced for item in node.names):
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
        "vld/execution_dag.py": source_declarations(model / "execution_dag.py", (
            "MAX_RANGE", "DTYPE_SIZE", "_sha", "_check_digest", "compile_decoder_dag",
            "apply_rope", "grouped_query_attention", "swiglu", "QualifiedTensorReader",
            "PAIR_SCHEMA", "_CompiledPairReadout", "_check_pair", "compile_pair_readout",
            "_verify_pair_readout", "_pair_receipt", "execute_tokens",
        ), used_imports_only=True),
        "vld/local_seed.py": (model / "local_seed.py").read_bytes(),
        "vld/page_files.py": (model / "page_files.py").read_bytes(),
        "vld/hf_glance.py": source_declarations(
            model / "hf_glance.py", ("TIMEOUT", "_sha", "_read", "_url", "range_bytes"), used_imports_only=True),
        "vld/tensor_ground.py": source_declarations(
            model / "tensor_ground.py", ("HEADER_LIMIT", "MAX_TENSORS", "MAX_ELEMENTS",
                                         "DTYPE_BYTES", "_sha", "_distinct_pairs", "observe_safetensors_header"),
            used_imports_only=True),
    }
    selections = {
        "micro_kernel": ("canonical_bytes", "domain_digest"),
        "micro_operators": ("StructuralLinearOperator", "rms_norm"),
        "micro_substrates": ("SafetensorsGlance",),
    }
    for module, names in selections.items():
        sources[f"vld/{module}.py"] = source_declarations(core / f"{module}.py", names, used_imports_only=True)
    return sources


def external_sources(root):
    """Project the model seed onto the source adapters and the BERT/GPT-2 operator set."""
    sources = model_sources(root)
    model = root / "experiments/model_fabric"
    del sources["vld/execution_dag.py"], sources["vld/micro_operators.py"]
    sources["vld/__init__.py"] = (b"from .micro_kernel import domain_digest\n"
                                  b"from .micro_substrates import SafetensorsGlance\n")
    sources["vld/local_seed.py"] = source_declarations(model / "local_seed.py", (
        "INDEX","INDEX_LIMIT","SHARD_NAME","LocalFiles","metadata_object","TensorDirectory",
    ),used_imports_only=True) + b'\ndef cli_main():\n    """Evaluate F_W(ids) from external source ranges."""\n    from .external_transformer import main\n    main()\n'
    for module in ("external_transformer","http_directory"):
        sources[f"vld/{module}.py"] = (model / f"{module}.py").read_bytes()
    return sources


def build_model(root=ROOT, *, external=False):
    """Enforce both |ZIP(P)| <= 2**16 and sum_{p in P, p ends in .py} |p| <= 2**16."""
    root = Path(root).resolve()
    bootstrap = model_bootstrap(root)
    sources = external_sources(root) if external else model_sources(root)
    members, manifest, ground = make_members(root, sources=sources, bootstrap=bootstrap)
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
    parser.add_argument("--external", action="store_true", help="Build the streaming GPT-2/BERT interpreter")
    args = parser.parse_args()
    binary, _, report = build_model(external=args.external)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(binary)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
