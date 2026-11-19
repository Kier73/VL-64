# Typed trained subgraphs: Qwen Gated DeltaNet + DeepSeek sparse MoE

This follows VL-64's independently verified SmolLM2 30-layer pretrained forward
pass and PR #29's trained Qwen/DeepSeek arithmetic probes. It intentionally adds
**no code to `src/vld/`**, preserves the 65,536-byte core ceiling, and composes
the existing exact-range `QualifiedShardNavigator` rather than a new
communication or model-identity system.

## Native Qwen3.8-27B Gated DeltaNet **complete mathematical layer**

`trained_subgraphs.qwen_gated_delta_layer` executes a whole causal Gated
DeltaNet layer including `input_layernorm -> QKV/Z/A/B -> depthwise conv4
-> gated delta recurrence -> per-V-head gated RMSNorm -> out projection
-> attention residual -> post_attention_layernorm -> SwiGLU -> FFN residual`.
A separate scalar oracle reconstructs the layer across three positions.
Convolution and recurrent states remain immutable until the caller commits a
successful result. The projection and vector interfaces are independent of
model size and can be bound to qualified tensor-range hosts.

**Crucial:** Qwen backbone RMSNorm is *zero-centered* (`1 + weight`). Gated
DeltaNet internal RMSNormGated is the conventional weight (with SiLU gate).
Conflating them creates numerically plausible but invalid inference. The
full-attention layers additionally use a partial multimodal RoPE and sigmoid
output gate and are **not** implemented by this GDN-only layer evaluator.

The `probe_qwen_trained_delta_head` live probe uses the pinned Qwen3.8 model
for a larger **weight-backed subgraph**: the correct input norm, one real
Q/K/V head group, its depthwise convolution, A/B decay and beta gates, causal
Gated DeltaNet update, internal gated norm, and one exact output-projection
contribution. It independently checks that contribution. Other V heads and
the complete FFN remain unexecuted on the real model, so a successful probe
must not be described as a full trained Qwen layer.

## DeepSeek-V4.1-Flash-1017 native MoE **complete mathematical FFN**

`trained_subgraphs.deepseek_moe_layer` performs input normalization, trained
384-way router matrix projection, sqrt(softplus) scores, noaux_tc selection
bias, six selected experts, shared expert, weighted mix and residual.
An independent test uses all 384 distinct router rows and requires that no
unselected expert callable is invoked. The selected expert contract explicitly
reuses the SwiGLU `w1/w3/w2` operator (and demands scale tensors when
building a qualified packed-expert weight plan). A successful FFN fixture
must not be confused with DeepSeek's mHC/Engram/CSA2 or language inference.

The real pinned probe `probe_deepseek_trained_router` executes the trained
384-way router weights with actual selection bias on a declared synthetic
hidden vector. It checks the exact six selected indices and unbiased
contribution weights independently, and returns only the six experts' index
paths. It **does not claim** those selected experts' quantized matrices were
retrieved or multiplied. Weight/scale admission and host reassignment must
precede any selected-expert numerical receipt.

## Observation, transfer and safety contracts

- All real weight bytes come from pinned Safetensors **shard exact ranges**.
- Each range requires caller-qualified host, exact model/revision/index identity,
  byte span and its returned SHA-256. This is byte-integrity/provenance, not an
  independent whole-model signature or verified physical remote execution.
- Tensor shape/dtype, head-group sharing, causal state dimensions and expert
  IDs are checked before their claimed numerical operations.
- Invalid/incomplete weight/scales, stale cursor or host refusal produce
  **UNRESOLVED** receipts, not fabricated model results.
- Larger neural calculations still consume real compute at whichever host
  executes the admitted operators. Virtual Grounds and structural memory
  decrease redundant observation/replay; they do not manufacture weight values
  or eliminate the cost of nontrivial trained mathematics.

## Reproduce

```bash
python -m pytest -q tests/test_trained_subgraphs.py tests/test_qualified_subgraphs.py \
  tests/test_architecture_ops.py tests/test_architecture_probe.py \
  tests/test_scaled_model_fabric.py tests/test_scaled_model_lab.py
# Requires outbound HTTPS and exact 206 Range support:
python -m experiments.model_fabric.trained_subgraph_lab --model qwen \
  --out artifacts/qwen-trained-head.json
python -m experiments.model_fabric.trained_subgraph_lab --model deepseek \
  --out artifacts/deepseek-trained-router.json
```

## Follow-on targets

1. Bind the complete Qwen GDN projection interface to streamed/sharded rows
   and checkpoint its 48 recurrent KV heads. Validate the *entire* pretrained
   layer across multiple positions before composing the 16 full-attention layers.
2. Acquire DeepSeek's six selected experts and their native scale companions
   per token, execute actual quantized SwiGLU paths and independent reference;
   preserve the selector's bias-only effect. Extend to compressed attention,
   hyper-connections and Engram memory without reimplementing transport.
3. Turn successful recovered states into reusable typed producer/consumer
   interfaces and later model-operated research loops.

Sources: [Qwen3.8 architecture/implementation analysis](https://github.com/PhyLiusir/Qwen3.8-27B-Arch/blob/main/docs/layer-by-layer.md), [DeepSeek state-dict tensor naming](https://docs.nvidia.com/nemo/automodel/v0.4/nemo-automodel/nemo_automodel/components/models/deepseek_v41/state_dict_adapter), and original pinned Hugging Face model manifests.
