"""Compile layer-zero post-training dynamics of one Qwen3.5-family value head."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .architecture_ops import qwen_gates
from .local_seed import Snapshot
from .page_files import PageFiles
from ..seed_fabric.build import make_archive
from ..seed_fabric.build_model import source_declarations


def lift(page, tokens, head=0, row=0):
    """Specialize W·RMSNorm(E[t]) once per token; leave causal convolution and Δ-state dynamic."""
    alphabet = sorted(set(tokens))
    with PageFiles(page) as files:
        snapshot = Snapshot(files)
        config = snapshot.config.get("text_config", {})
        if (config.get("model_type") != "qwen3_5_text" or config["layer_types"][0] != "linear_attention"
                or config.get("hidden_act") != "silu" or config.get("attention_bias", False)
                or snapshot.config.get("quantization_config") or config.get("quantization_config")):
            raise ValueError("Unsupported first-layer semantics")
        hidden, vocab = config["hidden_size"], config["vocab_size"]
        keys, heads = config["linear_num_key_heads"], config["linear_num_value_heads"]
        k, v, taps = (config[name] for name in ("linear_key_head_dim", "linear_value_head_dim", "linear_conv_kernel_dim"))
        if (any(type(n) is not int or n < 1 for n in (hidden, vocab, keys, heads, k, v, taps))
                or heads % keys or type(head) is not int or not 0 <= head < heads
                or type(row) is not int or not 0 <= row < hidden or not alphabet
                or any(type(t) is not int or not 0 <= t < vocab for t in alphabet)):
            raise ValueError("Invalid model coordinate")
        mixed = 2*keys*k + heads*v
        eps = config["rms_norm_eps"]
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("Invalid RMS epsilon")
        # Input table, static conv kernel, gain and output row must fit even before packaging.
        if 8*((2*k+v)*taps+2*v+len(alphabet)*(2*k+2*v+2)) >= 65536:
            raise ValueError("Token alphabet exceeds compact data budget")
        descriptors = {t["name"]: t for t in snapshot.ground["tensors"]["tensors"]}
        witness = []
        remote = {}

        def read(name, shape, start, count, *, role=None):
            """Read exactly [h+o+bs,h+o+b(s+n)) after verifying dtype, shape and shard size."""
            item = descriptors[name]
            dtype = item["dtype"]
            if item["shape"] != list(shape) or dtype not in ("BF16", "F32", "F16"):
                raise ValueError("Unexpected trained operand: " + name)
            shard, prefix, _, lo, hi, size = snapshot.locations[name]
            width = 4 if dtype == "F32" else 2
            if start < 0 or count < 1 or width*(start+count) > hi-lo:
                raise ValueError("Operand outside its source tensor")
            offset = prefix+lo+width*start
            data, total = files.read_range(shard, offset, width*count)
            if total != size or len(data) != width*count:
                raise ValueError("Incomplete operand")
            witness.append(dict(tensor=name, shard=shard, offset=offset, size=len(data),
                                dtype=dtype, sha256=hashlib.sha256(data).hexdigest()))
            if role is not None:
                remote.setdefault(role, []).append(dict(witness[-1], file_size=size))
            if dtype == "BF16":
                value = (np.frombuffer(data, "<u2").astype("<u4") << 16).view("<f4")
            else:
                value = np.frombuffer(data, "<f4" if width == 4 else "<f2")
            if not np.isfinite(value).all():
                raise ValueError("Nonfinite trained operand")
            return value.astype(np.float64)

        prefix = "model.language_model.layers.0."
        linear = prefix+"linear_attn."
        shared = head // (heads//keys)
        blocks = [(shared*k, k), (keys*k+shared*k, k), (2*keys*k+head*v, v)]
        weights = {}
        weights["embedding"] = np.stack([read("model.language_model.embed_tokens.weight",
            (vocab,hidden), t*hidden, hidden) for t in alphabet])
        embedding = "model.language_model.embed_tokens.weight"
        shard, base, _, lo, _, size = snapshot.locations[embedding]
        remote["embedding"] = dict(shard=shard, offset=base+lo, file_size=size, vocab=vocab,
                                   hidden=hidden, dtype=descriptors[embedding]["dtype"])
        weights["norm"] = read(prefix+"input_layernorm.weight", (hidden,), 0, hidden, role="norm")
        weights["qkv"] = np.concatenate([read(linear+"in_proj_qkv.weight", (mixed,hidden),
            start*hidden, length*hidden, role="qkv").reshape(length,hidden) for start,length in blocks])
        weights["conv"] = np.concatenate([read(linear+"conv1d.weight", (mixed,1,taps),
            start*taps, length*taps).reshape(length,taps) for start,length in blocks])
        weights["z"] = read(linear+"in_proj_z.weight", (heads*v,hidden),head*v*hidden,v*hidden,role="z").reshape(v,hidden)
        for gate in ("a","b"):
            weights[gate] = read(linear+f"in_proj_{gate}.weight", (heads,hidden), head*hidden, hidden, role=gate)
        weights["alog"] = read(linear+"A_log", (heads,), head, 1)
        weights["dt"] = read(linear+"dt_bias", (heads,), head, 1)
        weights["gain"] = read(linear+"norm.weight", (v,), 0, v)
        weights["row"] = read(linear+"out_proj.weight", (hidden,heads*v), (row*heads+head)*v, v)
        source = dict(page=page, visits=files.visits, index_sha256=snapshot.ground["index_sha256"],
                      metadata_bytes=snapshot.metadata_bytes, operands=witness,
                      trained_bytes=sum(r["size"] for r in witness))
        ground = dict(schema="vl64.qwen35-gdn-head.v1", model=files.model, revision=files.revision,
                      config_sha256=snapshot.ground["config_sha256"], structure=snapshot.ground["ground_digest"],
                      layer=0, head=head, output_row=row, key_width=k, value_width=v, conv_width=taps,
                      eps=eps, context=config["max_position_embeddings"], alphabet=alphabet, remote=remote,
                      a_log=float(weights["alog"][0]), dt_bias=float(weights["dt"][0]),
                      operands_sha256=hashlib.sha256(json.dumps(witness,sort_keys=True).encode()).hexdigest())
    parts = [weights["conv"].ravel(), weights["gain"], weights["row"]]
    for embedding in weights["embedding"]:
        x = embedding*(1+weights["norm"])/np.sqrt(np.mean(embedding**2)+eps)
        beta, decay = qwen_gates([float(weights["a"]@x)], [float(weights["b"]@x)],
                                 weights["alog"].tolist(), weights["dt"].tolist())
        parts.extend([weights["qkv"]@x, weights["z"]@x, [beta[0],math.exp(decay[0])]])
    raw = np.concatenate(parts).astype("<f8").tobytes()
    ground["coefficients_sha256"] = hashlib.sha256(raw).hexdigest()
    return ground, raw, source, weights


def package(ground, raw):
    """Emit A only if max(|ZIP(A)|, Σ uncompressed member lengths) ≤ 2¹⁶."""
    directory = Path(__file__).parent
    members = {"__main__.py": (directory/"addressed_delta.py").read_bytes(),
               "architecture_ops.py": source_declarations(directory/"architecture_ops.py",
                   ("_finite", "_sigmoid", "_l2", "_softplus", "qwen_gates", "depthwise_causal_conv_step", "qwen_gated_rms_norm"),
                   used_imports_only=True),
               "hf_glance.py": source_declarations(directory/"hf_glance.py",
                   ("TIMEOUT","_sha","_url","_read","range_bytes"), used_imports_only=True),
               "ground.json": json.dumps(ground,sort_keys=True,separators=(",",":"),allow_nan=False).encode(),
               "coefficients.bin": raw}
    expanded = sum(map(len,members.values()))
    if expanded > 65536:
        raise ValueError("Expanded code and coefficients exceed 64 KiB")
    return make_archive(members), expanded


def main():
    """Map (page,head,row,A) to an executable observation for any prefix in A⁺ within context."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("page")
    parser.add_argument("--tokens", required=True, type=int, nargs="+")
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    ground, raw, _, _ = lift(args.page, args.tokens, args.head, args.row)
    blob, expanded = package(ground, raw)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(blob)
    print(json.dumps(dict(archive_bytes=len(blob), expanded_bytes=expanded, **ground)))


if __name__ == "__main__":
    main()
