# Scale-independent pretrained model Grounds: 135M → 27B → MoE

The next VL-64 model-fabric step follows the verified 30-layer SmolLM2 forward
pass and causal recovery. The goal is **the same evidence and execution contract
at every model scale**, not assuming an unfamiliar architecture executes just
because its tensor names are visible.

No change is needed in the 65,093-byte `src/vld/` core. All source files in
this change live in `experiments/model_fabric/` and `tests/`.

## Architecture facts observed from public model configs

| Model | Structural family | Decoder | Distinguishing operators | Index |
| --- | --- | --- | --- | --- |
| `HuggingFaceTB/SmolLM2-135M` | `llama` | 30 layers | RMSNorm, RoPE/GQA, SwiGLU, vocab | Previously verified full BF16 forward pass |
| `Qwen/Qwen3.8-27B` | `qwen3_5` | 64 text layers: 48 linear / 16 full attention | gated delta linear attention, conv state, gated full attention, partial/multimodal RoPE, 27-layer vision encoder | 18 shards, index-advertised 55,562,855,904 payload bytes |
| `deepseek-ai/DeepSeek-V4.1-Flash-1017` | `deepseek_v41` | 40 text layers | compressed/sparse attention, 384 routed experts with 6 selected per token, Engram, mHC, YaRN, FP8 + FP4 experts, vision | 48 shards; very large index (~7.47 MB) |

Model and revision are pinned *before* any numeric weight requests. The
large-model experiments use the immutable revisions in `scaled_model_lab.py`.
The exact representation and special operators must be rechecked against each
pinned config. The numeric math admitted for `llama` cannot substitute for the
other two families.

## Implemented reusable contracts

1. **Sharded index Ground.** Strict JSON duplicate-key rejection; sanitize
   shard names; record exact `index_sha256`, tensor count, shard count,
   advertised bytes, model identity and revision. The index is not weight data.
2. **Shard-header Ground.** Acquire the 8-byte prefix and header body through
   bounded requests; admit every declared shape, dtype and exact byte extent;
   reject overlap, truncated headers and mismatch between the index and the
   shard's tensor inventory. The index/header relationship is checked before
   retrieving any tensor from that shard.
3. **Sparse, qualified byte navigation.** A requested flat tensor span maps to
   the pinned shard and byte offset. Every received range is at most 65,536
   bytes and is bound into a domain-separated evidence chain. Only a trusted
   caller-approved observer-host can satisfy the request. There is no silent
   fallback to a complete shard download.
4. **Architecture-aware schedule.** Typed layer-family list and required
   operator families, including explicit `STRUCTURE_ONLY` status for Qwen3.8
   and DeepSeek-V4.1 until their distinct math and quantizers are implemented.
5. **Optional real BF16/F32 numerical probe.** Take one exact observed
   `input_layernorm.weight` tensor and compute a deterministic finite
   normalization test against a separately expressed scalar reference. A
   successful result means **one real pretrained tensor was executed**, not
   that the corresponding model runs or generates text.

The index may exceed 64 KiB. That is **outside** the compact runtime budget.
The index is bounded to 16 MiB; a per-shard Safetensors header to 32 MiB;
each individual model byte-range request to 64 KiB. The index is read into
host memory but large model weight tensors are never allocated just to locate
one small slice.

The content digest and byte-range receipt establish **integrity relative to
observed bytes**, not whole-checkpoint authenticity, independent signatures,
or the physical independence of two claimed observer-host IDs. HTTPS at an
immutable model revision constrains source attribution but is not an
attestation of all the other shards.

## Reproduction

```bash
python -m pytest -q tests/test_scaled_model_fabric.py tests/test_scaled_model_lab.py

# Optional public HTTPS; index and config only, no model weights:
python -m experiments.model_fabric.scaled_model_lab --model qwen \
  --out artifacts/qwen-scale-ground.json
python -m experiments.model_fabric.scaled_model_lab --model deepseek \
  --out artifacts/deepseek-scale-ground.json

# Optional one genuine pinned BF16/F32 normalization slice per model:
python -m experiments.model_fabric.scaled_model_lab --model qwen --numeric \
  --out artifacts/qwen-pretrained-slice.json
python -m experiments.model_fabric.scaled_model_lab --model deepseek --numeric \
  --out artifacts/deepseek-pretrained-slice.json
```

The live probes exit nonzero on unresolved network evidence and the CI workflow
archives that status instead of inventing a successful observation.

## Next verified execution modules

- **Qwen3.8:** obtain pinned complete per-shard headers, implement and compare
  real gated-delta linear attention/conv-state transitions, partial/multimodal
  RoPE and gated full attention; pass text-only token inputs through all 64
  layers, then admit the vision encoder and cross-modal projections.
- **DeepSeek V4.1:** verify scale tensors alongside FP8/FP4 blocks; implement
  actual scaled quantized matvec, routed sparse MoE composition, Engram,
  compressed sparse attention and mHC transitions; never replace these with
  dense unquantized approximations under a real-model claim.
- **Observer-host scale:** add store-backed caches for verified pages and
  qualified per-op compute assignments, with receiver-owned checkpoints,
  fencing and numerical acceptance. Model size changes byte counts and route
  topology, not admission semantics.

The first new result sought is a genuine **weight-backed subgraph in both
families**, followed by independently verified full text forward passes. The
exact evidence scope remains visible at every step.

## Source references

- Qwen config: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json
- Qwen index: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/model.safetensors.index.json
- DeepSeek config: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/config.json
- DeepSeek index: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/model.safetensors.index.json

## First real large-model observations (GitHub Actions)

[Actions run #44541524537](https://github.com/Kier73/VL-64/actions/runs/44541524537)
completed both pinned real-weight slice experiments successfully.
The exact machine-emitted reports were preserved in
[`2026-11-19-large-model-frontier-observations.json`](../experiments/model_fabric/datasets/2026-11-19-large-model-frontier-observations.json).
`tests/test_saved_scaled_model_frontier.py` guards against scope escalation.

- **Qwen3.8-27B**: 64 text layers (48 linear/16 full attention),
  18 shards, 1,199 mapped tensor names, 55,562,855,904 advertised
  payload bytes. A 5,120-element *real BF16* layer-0
  `input_layernorm.weight` was acquired via three bounded ranges
  (header prefix + header + 10,240 weight bytes), and checked against an
  independent scalar reference with reported max error 0 at 12 decimals.
- **DeepSeek-V4.1-Flash-1017**: 40 text layers, 48 shards, 96,085
  mapped tensor names, 510,286,023,000 advertised payload bytes. A
  512-element *real BF16* `layers.0.attn.kv_norm.weight` was acquired
  via six bounded ranges including its shard header, with 1,024
  weight bytes, and similarly checked at 12-decimal reported precision.
  The pinned source is release revision
  `1bae0e1088993b9b82de08f59de5be5a34e42170`, not the earlier
  pre-release `dfbc4a7...` commit that lacks the final config.

The navigator additionally admits packed `F4` and `F6` tensor
extents in the Safetensors header, with byte-aligned range controls.
**It does not decode or execute quantized FP4/FP6 expert matrices yet.**
These results are a scale-independent **weight-location plus single
numeric-operator milestone**, not an executed 27B or 510GB decoder.

