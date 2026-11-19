# Observed pretrained 30-layer execution — 2026-11-19

GitHub Actions [run #44467793129](https://github.com/Kier73/VL-64/actions/runs/44467793129)
executed the **real pinned SmolLM2-135M** pretrained model across all 30
decoder layers and the complete vocabulary head, using actual BF16 weight
ranges admitted by the existing qualified observer host. This is the first
full-model forward pass evidence after the earlier two-layer and causal recovery
experiments.

## Directly observed result

| Contract | Observation |
|---|---|
| Model | `HuggingFaceTB/SmolLM2-135M` |
| Pinned revision | `4e53f736cbb20a9a0f56b4c4bf378d9f306ff915` |
| Typed DAG | `f89cfc3a1db00ec48658327a7c25bddf1998b54d3619838d0193fb94cb5fa1c6` |
| Input token IDs | `[42, 75]` (not decoded text) |
| Verified pretrained layers | 30 of 30 |
| Full vocabulary size | 49,152 |
| Selected next-token ID | **33** |
| Exact BF16 weight bytes acquired | **269,032,320** |
| Exact <=64 KiB range reads | **4,271** |
| Last-layer checkpoint head | `a88ea7dd338fd3f33a0ca8a47d24975cdcb3e8651194fcd82dc95be569a41618` |
| Complete F64 logits SHA-256 | `b44b804f14098c410ec1b4f547b52cb7ae1731f370251d60015b2d2da9232bc4` |
| NumPy reference max absolute vocabulary error | `0.0` in recorded precision |
| Portable core | 65,093 bytes; unchanged |

The first Python process acquired actual weight ranges and independently
verified layers **0–14**. It committed content-addressed causal pages and
published an exact checkpoint digest to a GitHub Actions step output. A
**distinct second Python interpreter process** resumed with the externally
carried expected digest, a higher owner epoch, and the same admitted DAG, then
independently verified layers **15–29** and all 49,152 final vocabulary logits.
Each layer's numerical deviation against an independently implemented NumPy
oracle remained below `2e-4` absolute. Both CI Python fixture matrices
(3.11 and 3.13) passed (64 passing, 1 skipped each).

## Exact retained evidence

The original GitHub Actions ZIP artifact `vl64-30-layer-pretrained-fabric`
contains three JSON files retained verbatim here:

- [`2026-11-19-real-30-layer-pretrained-partial.json`](../experiments/model_fabric/datasets/2026-11-19-real-30-layer-pretrained-partial.json)
  SHA-256 `8a3fe1b2a8191166de89f2b9c9d1d2a9d8615a8cfbd0fa138f66f064bb095cfc`
- [`2026-11-19-real-30-layer-pretrained-complete.json`](../experiments/model_fabric/datasets/2026-11-19-real-30-layer-pretrained-complete.json)
  SHA-256 `796e0bc6cd1da2bccef47f571a044e9e88acc23c7f4cefff1c2e8911e46c9192`
- [`2026-11-19-real-30-layer-pretrained-checkpoint.json`](../experiments/model_fabric/datasets/2026-11-19-real-30-layer-pretrained-checkpoint.json)
  SHA-256 `be95a2be30fcd73deebc2b36a3adce9989516863eb048df750d8eaa257a92bbd`

`tests/test_saved_real_full_pretrained.py` validates exact file bytes, all
30 per-layer identities, the checkpoint lineage and model/tensor-manifest
accounting offline. The JSON evidence does **not** contain all original weight
bytes or the stored KV/hidden page payloads, so full numerical offline replay
requires reacquiring the weight pages. The digest and range receipts do not
constitute a whole-checkpoint signature, cryptographic remote-host identity,
or proof that numerical work ran on a remote processor.

## Next model-oriented work

The verified forward pass is for *provided token IDs*. It does not yet
verify text tokenization or autoregressive generation from the real model.
`incremental_decoder.py` (following research) extends a pinned complete
model checkpoint with one next token using retained causal KV pages, without
replaying previous tokens. To test **real** pretrained next-token generation,
future runs must preserve all KV pages alongside the checkpoint and compare
successive full-model outputs to a separate reference. This is not assumed
from the first forward pass.
