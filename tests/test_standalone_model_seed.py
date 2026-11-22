"""Exercise the deployed archive with equivalent one- and two-shard model encodings."""
import io
import json
import struct
import subprocess
import sys
import zipfile

import pytest

from experiments.seed_fabric.build import make_archive, verify_model_budget
from experiments.seed_fabric.build_model import build_model
from test_execution_dag import synthetic_pretrained_ground


def snapshot(path, partitions):
    """Split the same tensor bytes W across j mod partitions; per-shard offsets start at zero."""
    ground, payload, _ = synthetic_pretrained_ground(layers=2)
    path.mkdir()
    (path / "config.json").write_text(json.dumps(ground["architecture_fields"]))
    records = ground["tensors"]["tensors"]
    mapping = {}
    for j in range(partitions):
        name = f"part-{j}.safetensors" if partitions > 1 else "model.safetensors"
        data = bytearray()
        header = {}
        for item in records[j::partitions]:
            lo, hi = item["data_offsets"]
            start = len(data)
            data.extend(payload[lo:hi])
            key = item["name"]
            mapping[key] = name
            header[key] = dict(dtype=item["dtype"], shape=item["shape"], data_offsets=[start, len(data)])
        raw = json.dumps(header).encode()
        (path / name).write_bytes(struct.pack("<Q", len(raw)) + raw + data)
    if partitions > 1:
        (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))


def invoke(seed, *args, succeeds=True):
    """Execute P under I∩S (isolated, no site packages) and compare exit(P) with expected membership."""
    run = subprocess.run([sys.executable, "-I", "-S", str(seed), *map(str, args)],
                         cwd=seed.parent, capture_output=True, text=True)
    assert (run.returncode == 0) == succeeds, (run.stdout, run.stderr)
    return json.loads(run.stdout) if succeeds else run.stderr


def test_archive_budget_and_integrity(tmp_path):
    """Verify P₁=P₂, |P|≤2¹⁶ and Σ|expanded Python members|≤2¹⁶, including the bootstrap."""
    blob, _, report = build_model()
    assert blob == build_model()[0]
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    expanded = sum(len(raw) for name, raw in members.items() if name.endswith(".py"))
    assert expanded == report["expanded_python_bytes"] <= 65536
    assert len(blob) == report["seed_bytes"] <= 65536
    with pytest.raises(ValueError):
        verify_model_budget({"profile": report["profile"], "expanded_python_bytes": 65537}, 65537)
    seed = tmp_path / "model.pyz"
    seed.write_bytes(blob)
    assert invoke(seed, "seed-verify")["expanded_python_bytes"] == expanded
    members["vld/local_seed.py"] += b"\n"
    seed.write_bytes(make_archive(members))
    assert "REFUSED" in invoke(seed, "inspect", succeeds=False)


def test_shards_preserve_generation_and_refuse_inconsistent_sources(tmp_path):
    """Changing the address map preserves logits; assigning W_i to the wrong shard breaks the bijection."""
    seed = tmp_path / "model.pyz"
    seed.write_bytes(build_model()[0])
    one, two = tmp_path / "one", tmp_path / "two"
    snapshot(one, 1)
    snapshot(two, 2)
    glance = invoke(seed, "inspect", "--snapshot", two)
    assert glance["executable"] and glance["weight_bytes_at_inspection"] == 0
    assert len(glance["shard_header_sha256"]) == 2
    args = ["--tokens", 1, 2, "--steps", 2, "--include-logits"]
    full = invoke(seed, "run", "--snapshot", one, *args)
    split = invoke(seed, "run", "--snapshot", two, *args)
    assert full["logits"] == split["logits"] and len(split["generated_ids"]) == 2
    pair = invoke(seed, "run", "--snapshot", two, "--tokens", 1, 2, "--pair", 4, 7)
    assert pair["generated_ids"] == [max((4, 7), key=full["logits"][0].__getitem__)]
    config = two / "config.json"
    config.write_text(json.dumps(dict(json.loads(config.read_text()), rope_scaling={"factor": 2})))
    assert invoke(seed, "inspect", "--snapshot", two)["executable"] is False
    assert "Unimplemented" in invoke(seed, "run", "--snapshot", two, "--tokens", 1, succeeds=False)
    index = two / "model.safetensors.index.json"
    doc = json.loads(index.read_text())
    doc["weight_map"][next(iter(doc["weight_map"]))] = "part-1.safetensors"
    index.write_text(json.dumps(doc))
    assert "index differs" in invoke(seed, "inspect", "--snapshot", two, succeeds=False)
