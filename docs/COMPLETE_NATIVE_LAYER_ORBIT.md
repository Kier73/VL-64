# Complete native numerical layers at model scale — development record

This extension reuses VL-64's qualified sharded tensor navigator and the
existing independently fixture-checked Qwen Gated DeltaNet and DeepSeek MoE
mathematics. **No code in `src/vld` changes**. Every read stays at or below
65,536 bytes. A model-index entry is not proof that numerical work occurred.

## Qwen3.8-27B

`QwenQualifiedLayer` composes the **entire** pinned linear-attention layer:
zero-centered RMSNorm, complete Q/K/V/Z/A/B, 4-tap depthwise convolution,
all 16 Q/K and 48 V recurrent heads, beta/decay gates, gated RMSNorm, full
output projection, attention residual, complete 17,408-channel SwiGLU FFN,
and residual output. The numerical operator is shared with the earlier
mathematical GDN reference; only the tensor source and bounded observation
change. `QwenQualifiedLayer.next()` commits its convolution and recurrent
state only after *all* projections have successfully executed.

`export_qwen_causal_ground` stores exact finite F64 causal coordinates in
SHA-256-addressed pages of at most 64 KiB. A new host
`restore_qwen_causal_ground` must receive an external expected checkpoint
head **and** an external input-sequence digest, plus matching model/revision,
index Ground, layer index, and head dimensions. This is integrity/rollback
fencing by the caller, **not** remote host attestation or distributed quorum.
The checkpoint excludes the huge static weight matrices and does not claim
compressed or affine-collapsed nonlinear attention. The caller must retain
its authenticated high-water pin and arrange reliable page transfer.

The Qwen adaptation applies only to `linear_attention` layers. Full softmax
attention, partial/multimodal RoPE, vision paths, and 64-layer composition are
subsequent work. It does not silently emulate these as gated delta rules.

## DeepSeek-V4.1-Flash-1017

`DeepSeekQualifiedMoE` executes the **whole routed feed-forward subgraph**
for one post-attention residual: exact FFN norm, complete 384-way router,
selection-bias-aware top six, all gate/up/down rows for **only those six**
FP4 experts, one complete shared expert, and full output mix. It enforces the
native training SwiGLU activation clipping: gate clamped only from above,
up clamped on both sides. The bias changes *which* experts run; it does not
change their normalized weights. The arithmetic uses paired per-32 E8M0
scales for native packed I8/FP4 rows, and supports F16/F32/BF16 or E4M3
with 32x32 E8M0 scales in shared expert matrices. Missing scale companions,
nonfinite codes, mismatched shapes, and unselected-expert substitutions fail.

This output is one **FFN branch**, not the full DeepSeek transformer residual;
DeepSeek's mHC, Engram, compressed sparse attention, vision and tokenizer
are separate required operators and remain in active development.

## Evidence and execution stages

Offline tests contain separate scalar oracles for (i) the complete
Qwen 3-position stateful layer and (ii) all six FP4 experts plus the
shared expert; exact range and scale admission tests are run beside them.
These are fixture tests, **not proof of live 27B/510GB full inference**.

Real checkpoint execution is available as an explicit *opt-in* command:

```bash
python -m pytest -q tests/test_complete_native_layers.py tests/test_complete_native_lab.py
python -m experiments.model_fabric.complete_native_lab --model qwen --out artifacts/qwen-layer-plan.json
python -m experiments.model_fabric.complete_native_lab --model deepseek --out artifacts/deepseek-moe-plan.json
# Requires pinned HTTPS HTTP 206 ranges; substantial network time and byte budget:
python -m experiments.model_fabric.complete_native_lab --model qwen --execute --budget-bytes 1500000000 --out artifacts/qwen-complete-layer.json
python -m experiments.model_fabric.complete_native_lab --model deepseek --execute --budget-bytes 1500000000 --out artifacts/deepseek-complete-moe.json
```

A structural result returns `STRUCTURAL_NATIVE_LAYER_PLAN_ONLY`; numerical
success requires all tensor rows, paired scales and the final output. The live
executor labels the result `..._NO_EXTERNAL_ORACLE` until an independently
implemented *real-weight* oracle reproduces the full output. A failed HTTP
range, incomplete tensor, or budget refusal produces `UNRESOLVED` and nonzero
exit status — it is never described as a completed trained layer.

Production-scale execution may require native vectorized kernels and caching
of once-admitted weight ranges. These are not bypassed by declaring an
observed Ground; every trained mathematical operation still requires
computation at a qualified host.

## Reused sources

- [Qwen3.5 Gated DeltaNet decoder implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
- [Qwen3.8 27B tensor dimensions](https://github.com/PhyLiusir/Qwen3.8-27B-Arch/blob/main/docs/layer-by-layer.md)
- [DeepSeek V4.1 shared expert and clamped SwiGLU](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/inference/model.py)
- E-CPU HFG003 handoff, VLD paged KV/state checkpoints, and VL-64 `QualifiedShardNavigator`.
