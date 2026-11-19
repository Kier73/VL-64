"""Manifest-compiled, weight-qualified decoder execution outside the 64-KiB VL64 core.

Currently this adapter accepts a strictly checked Llama/SwiGLU/RoPE/GQA graph.
The manifest is a structural Ground, NOT evidence of available model weights.
Every numerical weight access requires an observer-host's exact ranged receipt.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct

from vld import StructuralLinearOperator, domain_digest, rms_norm

MAX_RANGE = 16384  # bounded payload per observer-host request, not a model cap
DTYPE_SIZE = {"BF16": 2, "F32": 4}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _check_digest(ground, schema):
    if ground.get("schema") != schema or ground.get("ground_digest") != domain_digest(
        schema, {k: v for k, v in ground.items() if k != "ground_digest"}
    ):
        raise ValueError("Unverified model Ground digest: " + schema)


def compile_decoder_dag(observed):
    """Bind all Llama operator types/edges to exact manifest tensor coordinates.

    This produces zero weight requests. It fails closed for unmodeled families.
    Returned graph is JSON serializable, reproducible and topologically ordered.
    """
    _check_digest(observed, "vl64.remote-model-topology-ground.v1")
    if observed.get("tensor_header_status") != "OBSERVED_COMPLETE_TENSOR_HEADER":
        raise ValueError("Tensor header was not observed")
    header = observed.get("tensors")
    if not isinstance(header, dict):
        raise ValueError("Missing tensor Ground")
    _check_digest(header, "vl64.remote-tensor-structure-ground.v1")
    arch = observed["architecture_fields"]
    if (arch.get("model_type") != "llama" or arch.get("hidden_act") != "silu"
            or arch.get("attention_bias") not in (None, False)
            or arch.get("mlp_bias") not in (None, False)):
        raise ValueError("Unimplemented pretrained decoder architecture")
    def integer(field, low, high):
        v = arch.get(field)
        if type(v) is not int or not low <= v <= high:
            raise ValueError("Invalid decoder dimension: " + field)
        return v
    hidden = integer("hidden_size", 2, 32768)
    layers = integer("num_hidden_layers", 1, 512)
    heads = integer("num_attention_heads", 1, 512)
    kv = integer("num_key_value_heads", 1, 512)
    intermediate = integer("intermediate_size", 1, 131072)
    vocab = integer("vocab_size", 2, 1048576)
    context = integer("max_position_embeddings", 1, 1048576)
    eps = arch.get("rms_norm_eps")
    theta = arch.get("rope_theta")
    if (hidden % heads or heads % kv or (hidden // heads) % 2
            or type(eps) not in (int, float) or not 0 < eps <= 0.01
            or type(theta) not in (int, float) or not 1 < theta < 1e12):
        raise ValueError("Invalid RoPE, GQA or normalization contract")
    head_dim = hidden // heads
    kv_dim = kv * head_dim
    manifest = header.get("tensors")
    if not isinstance(manifest, list) or len(manifest) != header.get("tensor_count"):
        raise ValueError("Incomplete tensor manifest")
    tensors = {}
    intervals = []
    for item in manifest:
        name = item.get("name")
        shape = item.get("shape")
        dtype = item.get("dtype")
        offsets = item.get("data_offsets")
        if (not isinstance(name, str) or name in tensors or
                not isinstance(shape, list) or any(type(x) is not int or x <= 0 for x in shape) or
                dtype not in DTYPE_SIZE or not isinstance(offsets, list) or len(offsets) != 2 or
                any(type(x) is not int or x < 0 for x in offsets) or
                offsets[1] - offsets[0] != math.prod(shape)*DTYPE_SIZE[dtype]):
            raise ValueError("Malformed/duplicate tensor extent")
        tensors[name] = item
        intervals.append((offsets[0], offsets[1]))
    intervals.sort()
    if any(b[0] < a[1] for a, b in zip(intervals, intervals[1:])):
        raise ValueError("Overlapping tensor payload extents")
    source_bytes = header.get("source_file_bytes")
    if source_bytes is not None and (type(source_bytes) is not int or
            intervals[-1][1] + header["header_bytes"] > source_bytes):
        raise ValueError("Tensor beyond observed artifact")
    if header.get("declared_payload_bytes") != intervals[-1][1]:
        raise ValueError("Tensor payload extent differs from Ground")
    required = {"model.embed_tokens.weight": [vocab, hidden],
                "model.norm.weight": [hidden]}
    for i in range(layers):
        prefix = f"model.layers.{i}."
        required.update({prefix + name: shape for name, shape in {
            "input_layernorm.weight": [hidden],
            "post_attention_layernorm.weight": [hidden],
            "self_attn.q_proj.weight": [hidden, hidden],
            "self_attn.k_proj.weight": [kv_dim, hidden],
            "self_attn.v_proj.weight": [kv_dim, hidden],
            "self_attn.o_proj.weight": [hidden, hidden],
            "mlp.gate_proj.weight": [intermediate, hidden],
            "mlp.up_proj.weight": [intermediate, hidden],
            "mlp.down_proj.weight": [hidden, intermediate],
        }.items()})
    tied = arch.get("tie_word_embeddings") is True
    if not tied:
        required["lm_head.weight"] = [vocab, hidden]
    for name, shape in required.items():
        if name not in tensors or tensors[name]["shape"] != shape:
            raise ValueError("Missing/mismatched tensor: " + name)
    # Unknown extra tensors are allowed: they were observed, not invented.
    nodes = []
    def add(id_, op, deps, shape, weight=None):
        nodes.append({"id": id_, "op": op, "inputs": deps,
                      "output_shape": shape, "weight": weight})
        return id_
    previous = add("tokens", "input.token_ids", [], ["sequence"])
    previous = add("embedding", "embedding.lookup", [previous], [hidden],
                   "model.embed_tokens.weight")
    for i in range(layers):
        p = f"layer.{i}."
        w = f"model.layers.{i}."
        norm = add(p+"input_norm", "rms_norm", [previous], [hidden], w+"input_layernorm.weight")
        q = add(p+"q", "linear", [norm], [heads, head_dim], w+"self_attn.q_proj.weight")
        k = add(p+"k", "linear", [norm], [kv, head_dim], w+"self_attn.k_proj.weight")
        v = add(p+"v", "linear", [norm], [kv, head_dim], w+"self_attn.v_proj.weight")
        qr = add(p+"q_rope", "rope", [q], [heads, head_dim])
        kr = add(p+"k_rope", "rope", [k], [kv, head_dim])
        a = add(p+"attention", "causal_gqa", [qr, kr, v], [hidden])
        o = add(p+"o", "linear", [a], [hidden], w+"self_attn.o_proj.weight")
        residual = add(p+"attention_residual", "residual", [previous, o], [hidden])
        post = add(p+"post_norm", "rms_norm", [residual], [hidden], w+"post_attention_layernorm.weight")
        gate = add(p+"gate", "linear", [post], [intermediate], w+"mlp.gate_proj.weight")
        up = add(p+"up", "linear", [post], [intermediate], w+"mlp.up_proj.weight")
        activated = add(p+"swiglu", "silu_mul", [gate, up], [intermediate])
        down = add(p+"down", "linear", [activated], [hidden], w+"mlp.down_proj.weight")
        previous = add(p+"ffn_residual", "residual", [residual, down], [hidden])
    final = add("final_norm", "rms_norm", [previous], [hidden], "model.norm.weight")
    add("vocab_logits", "linear", [final], [vocab],
        "model.embed_tokens.weight" if tied else "lm_head.weight")
    plan = {"schema": "vl64.typed-pretrained-execution-dag.v1",
            "model": observed["model"], "revision": observed["pinned_revision"],
            "model_ground_digest": observed["ground_digest"],
            "tensor_ground_digest": header["ground_digest"],
            "header_bytes": header["header_bytes"],
            "source_file_bytes": source_bytes,
            "architecture": {"hidden": hidden, "layers": layers, "heads": heads,
                "kv_heads": kv, "head_dim": head_dim, "intermediate": intermediate,
                "vocab": vocab, "context": context, "eps": float(eps),
                "theta": float(theta), "tied_embedding": tied},
            "tensors": {k: {"shape": tensors[k]["shape"], "dtype": tensors[k]["dtype"],
                            "data_offsets": tensors[k]["data_offsets"]}
                        for k in sorted(required)},
            "nodes": nodes, "weight_values_observed": False,
            "temporal_state": "per_layer_causal_rope_rotated_key_and_value_cache"}
    plan["dag_digest"] = domain_digest(plan["schema"], plan)
    return plan


def apply_rope(vector, position, theta):
    """Llama rotate_half layout, not the incompatible interleaved pairing."""
    n = len(vector)
    if n % 2 or type(position) is not int or position < 0:
        raise ValueError("Invalid rotary coordinate")
    half = n//2
    output = [0.0]*n
    for i in range(half):
        angle = position/(theta**(2*i/n))
        c, s = math.cos(angle), math.sin(angle)
        a, b = vector[i], vector[i+half]
        output[i], output[i+half] = a*c-b*s, b*c+a*s
    return output


def grouped_query_attention(query, key_history, value_history, *, heads, kv_heads):
    if (not key_history or len(key_history) != len(value_history) or
            heads % kv_heads or len(query) != heads):
        raise ValueError("Invalid causal GQA cache")
    result = []
    repeat = heads//kv_heads
    dim = len(query[0])
    for h in range(heads):
        kv = h//repeat
        scores = [sum(a*b for a,b in zip(query[h], keys[kv]))/math.sqrt(dim)
                  for keys in key_history]
        peak = max(scores)
        probabilities = [math.exp(s-peak) for s in scores]
        total = sum(probabilities)
        result.extend(sum((p/total)*values[kv][j]
                          for p, values in zip(probabilities, value_history))
                      for j in range(dim))
    return result


def swiglu(gate, up):
    if len(gate) != len(up):
        raise ValueError("SwiGLU dimension mismatch")
    return [(g/(1+math.exp(-g)) if g >= 0 else
             g*math.exp(g)/(1+math.exp(g)))*u for g,u in zip(gate,up)]


class QualifiedTensorReader:
    """Data-only observer-host interface. No implicit trust from tensor shapes.

    The host returns: payload(bytes), observer_host, status, model, revision,
    tensor_ground_digest, byte_begin, byte_end_exclusive, sha256. Host's
    claimed source identity must be separately trusted by the caller.
    """
    def __init__(self, plan, observer_host, *, approved_hosts):
        if (plan.get("schema") != "vl64.typed-pretrained-execution-dag.v1" or
                plan.get("dag_digest") != domain_digest(
                    plan.get("schema"), {k:v for k,v in plan.items() if k!="dag_digest"})):
            raise ValueError("Unverified execution DAG")
        if not callable(observer_host) or not approved_hosts:
            raise ValueError("No qualified observer-host")
        self.plan, self.host, self.approved = plan, observer_host, set(approved_hosts)
        self.bytes_read = 0
        self.ranges = 0
        self.receipt_chain = "0"*64

    def values(self, name, start=0, count=None):
        tensor = self.plan["tensors"].get(name)
        if tensor is None:
            raise ValueError("Tensor not admitted to execution DAG: " + name)
        dtype = tensor["dtype"]
        size = DTYPE_SIZE[dtype]
        elements = math.prod(tensor["shape"])
        if count is None:
            count = elements-start
        if (type(start) is not int or type(count) is not int or
                start < 0 or count < 0 or start+count > elements):
            raise ValueError("Tensor request outside observed extent")
        output = []
        remaining = count
        while remaining:
            take = min(remaining, MAX_RANGE//size)
            begin = self.plan["header_bytes"] + tensor["data_offsets"][0] + start*size
            end = begin + take*size
            request = {"model": self.plan["model"], "revision": self.plan["revision"],
                       "tensor_ground_digest": self.plan["tensor_ground_digest"],
                       "tensor": name, "dtype": dtype, "byte_begin": begin,
                       "byte_end_exclusive": end}
            reply = self.host(dict(request))
            if not isinstance(reply, dict):
                raise ValueError("Observer-host returned no range witness")
            data = reply.get("payload")
            if (reply.get("status") != "QUALIFIED_EXACT_RANGE" or
                    reply.get("observer_host") not in self.approved or
                    any(reply.get(k) != v for k, v in request.items()) or
                    not isinstance(data, bytes) or len(data) != end-begin or
                    reply.get("sha256") != _sha(data)):
                raise ValueError("Unqualified/incomplete observer weight range")
            if dtype == "F32":
                batch = struct.unpack("<"+"f"*take, data)
            else:
                batch = [struct.unpack("<f", struct.pack("<I", bits<<16))[0]
                         for (bits,) in struct.iter_unpack("<H",data)]
            if not all(math.isfinite(x) for x in batch):
                raise ValueError("Nonfinite observer tensor payload")
            output.extend(batch)
            # Transcript binds every exact request and response, without
            # retaining raw pretrained weights in the Ground or receipts.
            witness = {k: reply[k] for k in request}
            witness.update({"observer_host": reply["observer_host"],
                            "sha256": reply["sha256"]})
            self.receipt_chain = domain_digest(
                "vl64.observer-range-chain.v1",
                {"previous":self.receipt_chain,"range":witness})
            self.ranges += 1
            self.bytes_read += len(data)
            start += take
            remaining -= take
        return output

    def matrix_vec(self, name, vector):
        rows, cols = self.plan["tensors"][name]["shape"]
        if len(vector) != cols:
            raise ValueError("Matrix/vector dimension mismatch")
        # Row streaming rather than an entire checkpoint or matrix allocation.
        rows_per_batch = max(1, MAX_RANGE//(cols*DTYPE_SIZE[self.plan["tensors"][name]["dtype"]]))
        result=[]
        for row in range(0, rows, rows_per_batch):
            n = min(rows_per_batch, rows-row)
            weights = self.values(name, row*cols, n*cols)
            sparse = {(r,c): weights[r*cols+c]
                      for r in range(n) for c in range(cols)
                      if weights[r*cols+c] != 0.0}
            result.extend(StructuralLinearOperator(rows=n,cols=cols,
                                                  sparse_delta=sparse).matvec(vector))
        return result

    def embedding(self, token):
        name = "model.embed_tokens.weight"
        vocab, hidden = self.plan["tensors"][name]["shape"]
        if type(token) is not int or not 0 <= token < vocab:
            raise ValueError("Token is outside model vocabulary")
        return self.values(name, token*hidden, hidden)


def execute_tokens(plan, tokens, observer_host, *, approved_hosts):
    """Actual, fully weight-backed causal forward execution of provided IDs.

    Tokenization/sampling policy remains external; this does NOT claim text
    generation unless real weight ranges have been supplied for every stage.
    """
    a = plan["architecture"]
    if (not isinstance(tokens, (list, tuple)) or not 1 <= len(tokens) <= a["context"] or
            any(type(t) is not int or not 0 <= t < a["vocab"] for t in tokens)):
        raise ValueError("Unqualified token sequence")
    reader = QualifiedTensorReader(plan, observer_host, approved_hosts=approved_hosts)
    kv_cache = [([], []) for _ in range(a["layers"])]
    hidden, head_dim = a["hidden"], a["head_dim"]
    for position, token in enumerate(tokens):
        state = reader.embedding(token)
        for i in range(a["layers"]):
            p = f"model.layers.{i}."
            x = rms_norm(state, reader.values(p+"input_layernorm.weight"), eps=a["eps"])
            q = reader.matrix_vec(p+"self_attn.q_proj.weight", x)
            k = reader.matrix_vec(p+"self_attn.k_proj.weight", x)
            v = reader.matrix_vec(p+"self_attn.v_proj.weight", x)
            q = [apply_rope(q[j*head_dim:(j+1)*head_dim],position,a["theta"])
                 for j in range(a["heads"])]
            k = [apply_rope(k[j*head_dim:(j+1)*head_dim],position,a["theta"])
                 for j in range(a["kv_heads"])]
            v = [v[j*head_dim:(j+1)*head_dim] for j in range(a["kv_heads"])]
            keys, values = kv_cache[i]
            keys.append(k)
            values.append(v)
            context = grouped_query_attention(q, keys, values,
                                               heads=a["heads"],kv_heads=a["kv_heads"])
            out = reader.matrix_vec(p+"self_attn.o_proj.weight", context)
            residual = [left+right for left,right in zip(state,out)]
            x = rms_norm(residual,reader.values(p+"post_attention_layernorm.weight"),eps=a["eps"])
            gate = reader.matrix_vec(p+"mlp.gate_proj.weight",x)
            up = reader.matrix_vec(p+"mlp.up_proj.weight",x)
            down = reader.matrix_vec(p+"mlp.down_proj.weight",swiglu(gate,up))
            state = [left+right for left,right in zip(residual,down)]
    final = rms_norm(state,reader.values("model.norm.weight"),eps=a["eps"])
    projection = "model.embed_tokens.weight" if a["tied_embedding"] else "lm_head.weight"
    logits = reader.matrix_vec(projection,final)
    chosen = max(range(len(logits)),key=lambda index:(logits[index],-index))
    receipt = {"schema":"vl64.weight-backed-token-dag-receipt.v1",
               "status":"WEIGHT_BACKED_DECODER_FORWARD_EXECUTED",
               "model":plan["model"],"revision":plan["revision"],
               "dag_digest":plan["dag_digest"],
               "tensor_ground_digest":plan["tensor_ground_digest"],
               "tokens":list(tokens),"selected_next_token":chosen,
               "logits":logits,"weight_bytes_acquired":reader.bytes_read,
               "exact_range_count":reader.ranges,
               "range_witness_chain_digest":reader.receipt_chain,
               "claims":"provided_token_ids_forward_only_no_tokenizer_or_full_artifact_digest_proof"}
    receipt["receipt_digest"] = domain_digest(receipt["schema"],receipt)
    return receipt


def huggingface_range_host(plan):
    """Optional strictly ranged pinned HTTPS observer, never a full-file GET."""
    from .hf_glance import _read, _url
    endpoint = _url(plan["model"],plan["revision"],"model.safetensors")
    def observe(request):
        begin,end = request["byte_begin"],request["byte_end_exclusive"]
        data,evidence = _read(endpoint,start=begin,stop=end-1,limit=end-begin)
        if evidence["http_status"] != 206:
            raise ValueError("Remote provider did not honor exact range")
        return dict(request, payload=data,sha256=_sha(data),
                    observer_host="pinned-huggingface-https-range",
                    status="QUALIFIED_EXACT_RANGE")
    return observe
