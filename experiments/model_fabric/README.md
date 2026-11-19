# Model Fabric — execute and navigate language-model Grounds

The model fabric is deliberately separate from VL-64's **65,093-byte compact runtime**. It reuses its current `StructuralLinearOperator`, `rms_norm`, `SafetensorsGlance`, `ObserverIntention`, and Ground fingerprint utilities. It does not expand `src/vld/`.

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
