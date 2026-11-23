# Model Fabric — execute and navigate language-model Grounds

The model fabric is deliberately separate from VL-64's **65,093-byte compact runtime**. It reuses its current `StructuralLinearOperator`, `rms_norm`, `SafetensorsGlance`, `ObserverIntention`, and Ground fingerprint utilities. It does not expand `src/vld/`.

## Source-hosted virtual model-page observer fleet (MPF-001)

[The virtual model-page fleet](../../docs/VIRTUAL_MODEL_PAGE_FLEET.md)
points the existing VL-64 \`ObserverIntention\` directly at externally
hosted Hugging Face model **pages**, through E-CPU's immutable
metadata-only SourceStructure. No tensor payload, shard, checkpoint or
weight range is acquired. Model data stays at its original host.

This separates **one model / many observers** (a shared structural
source with independent causal interpreter contexts) from **many
models / many observers** (distinct pinned source identities).
A 512-observer single-model fixture, 64-model multi-source fixture and
cross-process reconstruction are independently verified.

The first actual live experiment used **three external model pages**:
DeepSeek V4.1 Flash, Qwen3.8-27B and SmolLM2-135M. Their source
hosts declared **566.17 GB** of model data, but the experiment
observed only **73,585 metadata bytes** across six API reads. It
then ran **384 independent structural observer interpretations**,
zero weight payload bytes transferred, no source-local neural
forward calls. Each model's identical catalogue intention shared
one structural position across its many observers, while every
observer retained its own causal receipt.

~~~bash
# With the original E-CPU repository available on PYTHONPATH:
PYTHONPATH=/path/to/E-CPU \
python -m experiments.model_fabric.virtual_page_fleet_lab \
  --observers 128 --workers 16 \
  --out artifacts/vl64-virtual-model-page-fleet.json
~~~

The \`virtual_page_fleet.py\` module and tests run under standard
VL-64 even without E-CPU installed, using independently qualified
HSS-002 source descriptions. The dedicated GitHub Actions workflow
checks out pinned E-CPU to compare the independently implemented
source-position interpreter and to run live model-page observation.

The original \`operational_ground.py\` and \`intention_fabric.py\`
remain the distinct tools for *actual trained model inference at a
source host*. Model-page structural observations do not automatically
grant neural execution authority.

## Executed capabilities

1. **Trained byte language model:** `byte_lm.py` fits a 256-byte vocabulary bigram model to actual README text and performs generated-token inference using `StructuralLinearOperator.matvec`. Held-out predictive quality is measured and compared with a uniform-byte baseline.
2. **Actual causal-attention F32 execution:** `tiny_attention.py` reads real admitted tensor bytes and calculates embeddings, Q/K/V, causal softmax attention, residuals, RMSNorm, and next-token logits. These are seeded demonstration weights, not a pretrained language model.
3. **Hugging Face model Ground:** `hf_glance.py` reads the exact pinned `SmolLM2-135M` model configuration and Safetensors header through **bounded HTTP Range**. The manifest contains tensor names, types, offsets and dimensions but **zero weight-payload bytes**.
4. **Pretrained BF16 subgraph:** `remote_operator_probe.py` fetches 5,760 actual BF16 weight bytes from an authentic pretrained model, executes RMSNorm and four Q-projection rows through VL-64's compact operators, and checks the result against an independent numerical reference.
5. **Portable model-slice Ground:** `slice_ground.py` retains just those exact weight bytes, byte-range provenance and an executable operator description. It is replayable offline with no model server.
6. **A second monolithic seed:** `model_seed.py` packages the complete original VL-64 source, source-derived software Ground and pretrained BF16 Ground into a reproducible **38,360-byte executable model seed**. It runs with only Python 3.11+ on Linux and Windows.

## Run locally

~~~bash
python -m pip install -e ".[dev]"
python -m pytest -q tests/

# Train and execute local model computations.
python -m experiments.model_fabric.model_lab --out artifacts
python -m experiments.observer_intent.arbitrate \
  --intent artifacts/language-intention.json \
  --out artifacts/language-result.json
python -m experiments.observer_intent.arbitrate \
  --intent artifacts/neural-intention.json \
  --out artifacts/neural-result.json

# Observe actual pinned remote model structure with no weight download:
python -m experiments.model_fabric.hf_glance

# Optional network read: execute the REAL pretrained sub-operator with
# two bounded tensor weight ranges, not the complete model:
python -m experiments.model_fabric.remote_operator_probe

# No network is required for the saved pretrained slice:
python - <<'PY'
import json
from pathlib import Path
from experiments.model_fabric.slice_ground import replay_slice_ground
g=json.loads(Path(
 'experiments/model_fabric/datasets/2026-11-18-pretrained-bf16-slice-ground.json'
).read_text())
print(replay_slice_ground(g))
PY

# Generate one portable seed containing both the VL-64 runtime and that
# exact slice of genuine pretrained model computation:
python -m experiments.model_fabric.model_seed \
  --out artifacts/VL64-ModelSeed.pyz
python -I artifacts/VL64-ModelSeed.pyz model-verify
python -I artifacts/VL64-ModelSeed.pyz model-run
~~~

The pinned model's header contains **272 tensors** and represents around **269 MB** of weight payload. VL-64 reads just **30,536 header bytes** to discover their layout; its selected runtime subgraph then reads **5,760 BF16 weight bytes**. It does **not** run the entire SmolLM2 model or generate SmolLM2 text.

## Evidence and portability

[Saved Grounds and measurements](datasets/) include the trained sparse byte LM, the 272-tensor structure Ground, the exact pretrained BF16 slices, and host-stable numerical receipts. The trained-language Ground and the remote tensor Ground are each under 64 KiB in compact JSON representation; the portable model seed is separately below 64 KiB.

The [two-version inference workflow](../../.github/workflows/model-execution.yml) validates the numeric operators, and the [four-host model seed workflow](../../.github/workflows/model-seed.yml) checks Linux and Windows across Python 3.11 and 3.13. Both preserve downloadable experiment artifacts.

For exact provenance, limitations, and the next steps toward complete LLM inference, see [Language Models and Pretrained Tensor Navigation](../../docs/MODEL_EXECUTION_GROUNDS.md).

## Typed pretrained execution DAG (after 6389481)

The independent [typed decoder DAG module](execution_dag.py) compiles the pinned, observed SmolLM2-135M manifest into typed embedding, RoPE, causal grouped-query attention, SwiGLU feed-forward and vocabulary projection nodes without acquiring any weights at compilation. It refuses unknown architectures or divergent Ground/tensor extents. `QualifiedTensorReader` can then acquire each exact BF16/F32 range from an explicitly approved observer-host, reject incomplete/altered data, and record a chained range witness. The original 65,093-byte VL-64 core is unchanged.

```bash
python -m experiments.model_fabric.model_dag_lab
python -m pytest -q tests/test_execution_dag.py
```

Full opt-in ranged execution of provided token IDs is available with `--tokens 42 --allow-remote-weights`. It has not been established as real pretrained full-model generation: obtaining all weight bytes, authenticating them against an external checkpoint digest, and comparing end-to-end logits to an independent pretrained reference remain separate validation steps. Read [the typed DAG contract](../../docs/TYPED_PRETRAINED_EXECUTION_DAG.md).


## Entire pretrained transformer layer (next milestone)

The [cached layer executor](pretrained_layer.py) now binds the typed DAG to
all nine real pretrained layer tensors: input norm, Q/K/V, rotary positions,
causal GQA, output projection, attention residual, post norm, SwiGLU
gate/up/down and feed-forward residual. The selected layer parameters are
loaded once per observer Ground and reused for multiple causal token positions.
All weight ranges remain individually bounded to 65,536 bytes. For
SmolLM2-135M, layer 0 contains **7,080,192 BF16 weight bytes** and each
chosen token embedding row adds **1,152 BF16 bytes**.

```bash
python -m pytest -q tests/test_execution_dag.py tests/test_pretrained_layer.py
# Network-optional genuine pretrained layer, two actual token IDs:
python -m experiments.model_fabric.real_layer_lab \
  --out artifacts/pretrained-layer0-report.json
```

The [independent scalar reference](real_layer_lab.py) reassembles exact
admitted ranges and evaluates every layer operation without VL-64 math
helpers; it compares all 576 final hidden coordinates. The [CI
workflow](../../.github/workflows/pretrained-layer.yml) runs mandatory BF16
and F32 fixture checks and an optional live pinned-model observation. No
result is labeled verified if the remote bytes cannot be obtained.

This is **one complete pretrained layer**, not a full 30-layer pass or
SmolLM2 next-token generation. See the
[pretrained layer evidence contract](../../docs/PRETRAINED_LAYER0_GROUND.md).

## Pretrained layer composition

The [two-layer chain coordinator](pretrained_chain.py) reuses independently
prepared layer weights, retains separate causal KV states, and binds each
`layer.N.ffn_residual` hidden payload to `layer.N+1.input_norm` through a
bounded, digest-checked, temporally ordered handoff. The
[independent reference experiment](real_chain_lab.py) has successfully
executed **two genuine SmolLM2-135M pretrained BF16 layers**, using
14,162,688 selected weight bytes over 228 qualified range requests for
tokens `[42,75]`; all 576 final hidden coordinates matched an independent
scalar reference within 2e-5 (reported error 0.0 at 12 decimal places).
See [the verified chain receipt](datasets/2026-11-18-real-two-layer-pretrained-receipt.json)
and [the evidence contract](../../docs/PRETRAINED_LAYER_CHAIN.md).
The math ran in one Python interpreter; byte providers may be different
approved observer-hosts. Full-model text generation remains a separate goal.

## Recoverable causal model orbit

`pretrained_orbit.py` applies E-CPU's externally pinned model-handoff recovery and Virtual Layer Dynamics' paged causal KV Grounds to the already verified pretrained layer chain. The checkpoint keeps exact F64 intermediate K/V values in <=64-KiB SHA-256-addressed pages and requires a trusted expected checkpoint digest and new monotonic owner epoch before a different approved observer can resume.

```bash
python -m pytest -q tests/test_pretrained_orbit.py
python -m experiments.model_fabric.real_orbit_lab --out artifacts/pretrained-orbit-report.json
```

The network experiment fetches pinned genuine SmolLM2 BF16 layer ranges, executes token 42, checkpoints, restores to a different *logical* observer assignment and continues token 75, checking against independent scalar math. It returns `UNRESOLVED` on unavailable/incomplete ranges. See [Recovery and host rehoming](../../docs/PRETRAINED_ORBIT_RECOVERY.md).


## Scale-independent pretrained Ground navigation

[Scaled model fabric](scaled_model_fabric.py) and
[pinned real-model lab](scaled_model_lab.py) extend the same evidence
contract to the 64-layer Qwen3.8-27B hybrid and 40-layer
DeepSeek-V4.1-Flash-1017 MoE without increasing the VL-64 core.
They admit large sharded Safetensors indices, validate exact tensor
shapes, offsets and BF16/F32/packed low-bit header representations,
and navigate individual ranges of at most 64 KiB.

The first live pinned network run
([Actions #44541524537](https://github.com/Kier73/VL-64/actions/runs/44541524537))
verified real BF16 normalization weights independently in **both** models:
10,240 weight bytes from Qwen and 1,024 from DeepSeek. Their
[exact receipts](datasets/2026-11-19-large-model-frontier-observations.json)
are regression-checked against source identities. These are successful
**real pretrained sub-operations**, not 27B/510GB full-decoder runs:
Qwen's gated linear/full attention and DeepSeek's sparse MoE/FP8/FP4
operators require their own numeric adapters.

See [Scale-independent model Grounds](../../docs/SCALE_INDEPENDENT_MODEL_GROUNDS.md).
