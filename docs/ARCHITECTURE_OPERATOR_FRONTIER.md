# Native model operators: Qwen3.8 and DeepSeek V4.1

This work extends VL-64's existing qualified sharded tensor navigator with
**actual architecture-specific numerical operations**. The 65,093-byte
`src/vld/` core remains unchanged.

## What is executable

**Qwen3.8-27B:** Qwen3.5-family causal depthwise convolution with recoverable
history; exact per-head beta=sigmoid(b) and
log_decay=-exp(A_log)*softplus(a+dt_bias); L2-normalized grouped-head
delta-rule recurrence; gated RMS norm. The transition is
`S' = exp(g)S + k outer [beta (v - k^T exp(g) S)]` with output `q^T S'`.
The state update returns new memory without mutating the predecessor.

**DeepSeek-V4.1-Flash-1017:** OCP MXFP8 E4M3FN and packed MXFP4 E2M1
element decoders, UE8M0 exponent-only scales, 32x32 FP8 block-scaled
matrix-vector projection, and per-row K=32 FP4 scaled projection. Invalid or
absent scale tensors are rejected. The `noaux_tc` expert selector uses
sqrt(softplus(logit)) plus selection-only bias to choose top-k, then
normalizes **unbiased** scores, applies the route scale, and invokes only
selected experts plus the shared expert. The numerical route does not
grant authority to arbitrary selected external compute hosts.

Every admitted tensor range is constrained to <=65536 bytes by the preexisting
`QualifiedShardNavigator`. Hashes bind exact observed byte ranges and
host-provided identity, but do not constitute whole-checkpoint authenticity
or physical host attestation.

## Pinned pretrained weight probes

`architecture_probe.py` validates actual model-specific slices rather than
inferring weights from config alone:

- **Qwen:** layer-0 `linear_attn.A_log`, `dt_bias`, and one
  `in_proj_a.weight` and `in_proj_b.weight` row. Executes trained
  gating parameters on an openly declared synthetic 5120-wide vector and
  checks beta/log decay against an independent scalar formula.
- **DeepSeek:** first row of `layers.0.attn.wq_a.weight` (FP8 E4M3)
  and the matching `layers.0.attn.wq_a.scale` (UE8M0) grid.
  Performs the scaled projection and separately decodes the same raw
  bytes by bit fields for an independent numerical check.

Both live experiments refuse to claim success if the pinned HTTPS server
does not return the exact required byte ranges, or if the scales, shapes,
or reference result do not agree. Failed cases emit an explicit UNRESOLVED
report. Network jobs in CI are optional; mandatory fixture jobs are not.

```bash
python -m pytest -q tests/test_architecture_ops.py tests/test_architecture_probe.py
python -m experiments.model_fabric.architecture_probe --model qwen \
  --out artifacts/qwen-architecture-operator.json
python -m experiments.model_fabric.architecture_probe --model deepseek \
  --out artifacts/deepseek-architecture-operator.json
```

## Claims this evidence does not make

- Neither probe is a complete transformer layer, an autoregressive language
  generation run, or a numerical verification of all weights.
- Qwen still needs the remaining trained projections, full-attention layers,
  FFNs, vision tower, multimodal RoPE and state checkpoint integration.
- DeepSeek still needs real trained MoE gate logits, complete expert weights,
  sparse/compressed attention, mHC, Engram, vision and DSpark integration.
- Neither proves execution across separate physical processors or proves
  that a configured observer is cryptographically authenticated.

The achievement is an executable, reusable family-specific arithmetic contract
that can attach to qualified observer-host range evidence without replacing
unfamiliar mathematics by Llama approximations.

## References

- https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py
- https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/config.json
- https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/inference/model.py
- https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
