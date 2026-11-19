"""Optional actual pinned SmolLM2 two-layer execution, independent scalar check.

Uses only strict HTTP 206 ranges, 64-KiB messages, and a pinned revision.
The reference reconstructs real observed tensor bytes; neither model shape
nor a prior layer receipt is substituted for missing weight evidence.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from .execution_dag import compile_decoder_dag, huggingface_range_host
from .pretrained_chain import execute_pretrained_chain
from .pretrained_layer import LAYER_TENSORS
from .real_layer_lab import _decode

MANIFEST = (Path(__file__).parent / "datasets" /
            "2026-11-18-pinned-smollm2-tensor-ground.json")
TOKENS = (42, 75)


def independent_two_layer_reference(plan, fragments, tokens, *, last_layer=1):
    """Independent dense scalar Llama math, no VL64 numerical operations."""
    a = plan["architecture"]
    h, d, nh, nkv = a["hidden"], a["head_dim"], a["heads"], a["kv_heads"]

    def fetch_tensor(name, *, start=0, count=None):
        tensor = plan["tensors"][name]
        dtype = tensor["dtype"]
        size = 2 if dtype == "BF16" else 4
        if count is None:
            count = math.prod(tensor["shape"])
        first = plan["header_bytes"] + tensor["data_offsets"][0] + start * size
        end = first + count * size
        target, cursor = bytearray(), first
        for offset, raw in sorted(fragments[name]):
            stop = offset + len(raw)
            if stop <= cursor:
                continue
            if offset > cursor:
                break
            take = min(stop, end) - cursor
            if take > 0:
                target.extend(raw[cursor - offset: cursor - offset + take])
                cursor += take
            if cursor == end:
                break
        if cursor != end:
            raise ValueError("Independent layer reference lacks exact weight bytes: " + name)
        return _decode(bytes(target), dtype)

    tables = {}
    for layer in range(last_layer + 1):
        for suffix in LAYER_TENSORS:
            name = f"model.layers.{layer}." + suffix
            tables[name] = fetch_tensor(name)

    def dot(suffix, layer, x):
        name = f"model.layers.{layer}." + suffix
        rows, cols = plan["tensors"][name]["shape"]
        w = tables[name]
        return [math.fsum(w[r * cols + c] * x[c] for c in range(cols))
                for r in range(rows)]

    def norm(x, suffix, layer):
        gamma = tables[f"model.layers.{layer}." + suffix]
        scale = 1 / math.sqrt(math.fsum(z*z for z in x)/len(x) + a["eps"])
        return [z * scale * g for z, g in zip(x, gamma)]

    def rope(values, position):
        half = len(values)//2
        out = values.copy()
        for j in range(half):
            angle = position / (a["theta"] ** (j / half))
            co, si = math.cos(angle), math.sin(angle)
            out[j] = values[j]*co - values[j+half]*si
            out[j+half] = values[j]*si + values[j+half]*co
        return out

    def silu(z):
        return (z/(1 + math.exp(-z)) if z >= 0
                else z*math.exp(z)/(1 + math.exp(z)))

    history = [([], []) for _ in range(last_layer + 1)]
    final = None
    for position, token in enumerate(tokens):
        x = fetch_tensor("model.embed_tokens.weight", start=token*h, count=h)
        for layer in range(last_layer + 1):
            n = norm(x, "input_layernorm.weight", layer)
            q = dot("self_attn.q_proj.weight", layer, n)
            k = dot("self_attn.k_proj.weight", layer, n)
            v = dot("self_attn.v_proj.weight", layer, n)
            qh = [rope(q[j*d:(j+1)*d], position) for j in range(nh)]
            kh = [rope(k[j*d:(j+1)*d], position) for j in range(nkv)]
            vh = [v[j*d:(j+1)*d] for j in range(nkv)]
            keys, values = history[layer]
            keys.append(kh)
            values.append(vh)
            context = []
            for j in range(nh):
                shared = j // (nh//nkv)
                scores = [math.fsum(qh[j][c]*frame[shared][c] for c in range(d)) / math.sqrt(d)
                          for frame in keys]
                peak = max(scores)
                probabilities = [math.exp(s-peak) for s in scores]
                total = math.fsum(probabilities)
                context.extend(math.fsum(p/total * frame[shared][c]
                                         for p, frame in zip(probabilities, values))
                               for c in range(d))
            o = dot("self_attn.o_proj.weight", layer, context)
            residual = [u+v for u,v in zip(x,o)]
            post = norm(residual, "post_attention_layernorm.weight", layer)
            gate = dot("mlp.gate_proj.weight", layer, post)
            up = dot("mlp.up_proj.weight", layer, post)
            x = [s+t for s,t in zip(residual, dot("mlp.down_proj.weight", layer,
                                                 [silu(g)*u for g,u in zip(gate,up)]))]
        final = x
    return final


def run(*, tokens=TOKENS):
    observed = json.loads(MANIFEST.read_text(encoding="utf-8"))
    plan = compile_decoder_dag(observed)
    remote = huggingface_range_host(plan)
    fragments = defaultdict(list)

    def capture(request):
        reply = remote(request)
        fragments[request["tensor"]].append((request["byte_begin"], reply["payload"]))
        return reply

    result = execute_pretrained_chain(
        plan, list(tokens), {-1:capture, 0:capture, 1:capture},
        approved_hosts={-1:{"pinned-huggingface-https-range"},
                        0:{"pinned-huggingface-https-range"},
                        1:{"pinned-huggingface-https-range"}},
        compute_actors={0:"local-layer0-interpreter"},
        approved_producers={"local-layer0-interpreter"})
    independent = independent_two_layer_reference(plan, fragments, tokens)
    error = max(abs(x-y) for x,y in zip(result["output"], independent))
    tol = 2e-5
    if not math.isfinite(error) or error > tol:
        raise ValueError("Two actual pretrained layers disagree with independent scalar reference")
    return {
        "schema": "vl64.real-pretrained-two-layer-check.v1",
        "status": "REAL_BF16_TWO_LAYER_CHAIN_INDEPENDENTLY_VERIFIED",
        "model": plan["model"], "revision":plan["revision"],
        "dag_digest": plan["dag_digest"],
        "layer_chain_receipt_digest": result["receipt_digest"],
        "tokens": list(tokens), "layers": [0,1],
        "last_hidden_first_8": [round(x,8) for x in result["output"][:8]],
        "last_handoff_digest": result["last_handoff_digest"],
        "range_chains": result["range_chains"],
        "independent_reference_max_abs_error": round(error,12),
        "reference_tolerance_abs": tol,
        "weight_payload_bytes_acquired": result["weight_payload_bytes_acquired"],
        "exact_range_count": result["exact_range_count"],
        "scope": "two_pinned_pretrained_layers_not_full_model_text_or_logits",
        "compute_scope": "single_python_interpreter_with_qualified_https_weight_ranges",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/pretrained-two-layer-report.json")
    args = parser.parse_args()
    path = Path(args.out)
    path.parent.mkdir(parents=True,exist_ok=True)
    try:
        report = run()
    except (ValueError, OSError, TimeoutError) as error:
        report = {"status":"UNRESOLVED", "reason":type(error).__name__,
                  "scope":"no_two_layer_execution_claim_without_exact_observed_weights"}
        path.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
        print("VL64_REAL_TWO_LAYER "+json.dumps(report,sort_keys=True))
        raise SystemExit(1) from error
    path.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print("VL64_REAL_TWO_LAYER "+json.dumps(report,sort_keys=True))


if __name__ == "__main__":
    main()
