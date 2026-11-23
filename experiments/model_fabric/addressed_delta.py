"""Evaluate a trained, token-addressed Qwen Gated DeltaNet head from zero history."""
import argparse
import hashlib
import json
import math
import struct
import sys
import zipfile

if __package__:
    from .architecture_ops import _l2, depthwise_causal_conv_step, qwen_gated_rms_norm, qwen_gates
    from .hf_glance import _url, range_bytes
else:
    from architecture_ops import _l2, depthwise_causal_conv_step, qwen_gated_rms_norm, qwen_gates
    from hf_glance import _url, range_bytes


def read_ground(archive):
    """Decode exactly M = c·K + 2v + |alphabet|·(c+v+2) finite binary64 coefficients."""
    with zipfile.ZipFile(archive) as packed:
        if sum(i.file_size for i in packed.infolist()) > 65536:
            raise ValueError("Expanded program and data exceed 64 KiB")
        ground = json.loads(packed.read("ground.json"))
        raw = packed.read("coefficients.bin")
    if ground["schema"] != "vl64.qwen35-gdn-head.v1" or hashlib.sha256(raw).hexdigest() != ground["coefficients_sha256"]:
        raise ValueError("Unknown ground or changed coefficients")
    k, v, taps = (ground[key] for key in ("key_width", "value_width", "conv_width"))
    ids = ground["alphabet"]
    if (any(type(n) is not int or n < 1 for n in (k, v, taps)) or not ids
            or any(type(i) is not int or i < 0 for i in ids) or ids != sorted(set(ids))):
        raise ValueError("Invalid head geometry or alphabet")
    c = 2*k+v
    count = c*taps + 2*v + len(ids)*(c+v+2)
    if len(raw) != 8*count:
        raise ValueError("Coefficient extent differs from declared geometry")
    values = struct.unpack(f"<{count}d", raw)
    if not all(map(math.isfinite, values)):
        raise ValueError("Nonfinite coefficient")
    bank = {}
    offset = c*taps + 2*v
    for token in ids:
        bank[token] = values[offset:offset+c+v+2]
        offset += c+v+2
    return (ground, [values[i:i+taps] for i in range(0, c*taps, taps)],
            values[c*taps:c*taps+v], values[c*taps+v:c*taps+2*v], bank)


def evaluate(program, tokens):
    """Evaluate y=S_t^T q via u←f(u−β(k·u)k), y←y+β(k·u)v in reverse time."""
    ground, kernel, gain, output_row, bank = program
    if (not tokens or len(tokens) > ground["context"] or
            any(type(t) is not int or t not in bank for t in tokens)):
        raise ValueError("Prefix empty, beyond context, or token outside compiled alphabet")
    k, v = ground["key_width"], ground["value_width"]
    c = 2*k+v
    conv_history, events = [], []
    for token in tokens:
        entry = bank[token]
        mixed, conv_history = depthwise_causal_conv_step(entry[:c], kernel, conv_history)
        beta, fade = entry[-2:]
        events.append((_l2(mixed[k:2*k]), mixed[2*k:], beta, fade))
    # At the requested address only the last query and output gate are observable.
    u = [x / math.sqrt(k) for x in _l2(mixed[:k])]
    core = [0.] * v
    for key, value, beta, fade in reversed(events):
        coefficient = beta * math.fsum(a*b for a,b in zip(key, u))
        core = [a + coefficient*b for a,b in zip(core, value)]
        u = [fade*(a - coefficient*b) for a,b in zip(u, key)]
    output = qwen_gated_rms_norm(core, entry[c:c+v], gain, eps=ground["eps"])
    return {"head_output": output, "row_contribution": math.fsum(a*b for a,b in zip(output, output_row))}


def resolve(program, tokens):
    """Extend the token table by Φ(t)=W·RMSNorm(E[t]); existing entries cost zero reads."""
    g, _, _, _, bank = program
    remote = g["remote"]
    emb = remote["embedding"]
    if (not tokens or len(tokens) > g["context"] or
            any(type(t) is not int or not 0 <= t < emb["vocab"] for t in tokens)):
        raise ValueError("Outside model token or context domain")
    missing = sorted(set(tokens) - bank.keys())
    bytes_read = 0
    if not missing:
        return bytes_read

    def fetch(part):
        """Read B=S[a:a+n], then check the pinned file extent and any compiled H(B)."""
        nonlocal bytes_read
        url = _url(g["model"], g["revision"], part["shard"])
        data, total = range_bytes(url, part["offset"], part["size"])
        if total != part["file_size"] or ("sha256" in part and hashlib.sha256(data).hexdigest() != part["sha256"]):
            raise ValueError("Source extent or trained bytes diverged")
        bytes_read += len(data)
        if part["dtype"] == "BF16":
            return [struct.unpack("<f", b"\0\0"+data[i:i+2])[0] for i in range(0,len(data),2)]
        fmt = {"F16":"<e", "F32":"<f"}[part["dtype"]]
        return [x for x, in struct.iter_unpack(fmt, data)]

    h = emb["hidden"]
    gain = fetch(remote["norm"][0])
    xs = {}
    stride = h * (4 if emb["dtype"] == "F32" else 2)
    for token in missing:
        e = fetch(dict(emb, offset=emb["offset"]+token*stride, size=stride))
        scale = 1/math.sqrt(math.fsum(x*x for x in e)/h+g["eps"])
        xs[token] = [x*scale*(1+w) for x,w in zip(e,gain)]
    fields = {t: [] for t in missing}
    for role in ("qkv", "z", "a", "b"):
        for part in remote[role]:
            matrix = fetch(part)
            for start in range(0,len(matrix),h):
                row = matrix[start:start+h]
                for t in missing:
                    fields[t].append(math.fsum(a*b for a,b in zip(row,xs[t])))
    for values in fields.values():
        a, b = values[-2:]
        beta, decay = qwen_gates([a],[b],[g["a_log"]],[g["dt_bias"]])
        values[-2:] = [beta[0], math.exp(decay[0])]
    bank.update(fields)
    return bytes_read


def main():
    """Map (checkpoint, head, row, token prefix) ↦ last gated head vector and row contribution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokens", type=int, nargs="+")
    parser.add_argument("--resolve", action="store_true", help="Read missing token operands at the pinned checkpoint")
    args = parser.parse_args()
    try:
        program = read_ground(sys.argv[0])
        size = resolve(program, args.tokens) if args.resolve else 0
        print(json.dumps(dict(evaluate(program, args.tokens), trained_bytes_read=size), allow_nan=False))
    except ValueError as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
