# Consecutive pretrained layers as verified hidden-state handoffs

This follows the real SmolLM2-135M layer-0 experiment (PR #20, commit `ebc2fc6`).
It does not expand the 65,536-byte `src/vld/` source ceiling. The typed DAG,
range reader, and `PreparedLayer` remain the authorities for tensor identity,
weight admission, operator ordering, and causal cache state.

## Executable path

`experiments/model_fabric/pretrained_chain.py` adds a coordinator for consecutive
layers `0..last_layer`, currently exercised at `last_layer=1`. It acquires each
layer's nine tensors exactly once and feeds the real predecessor hidden vector
into the next layer for each causal token position. Every layer independently
maintains its own RoPE-rotated KV history.

A handoff contains the exact little-endian IEEE-754 F64 hidden values, the
manifest-bound typed DAG digest, tensor Ground digest, producer assignment,
source and target node IDs, token position, predecessor handoff digest, and
source weight-range witness chain. Hidden payloads must fit one 65,536-byte
message; bigger hidden vectors require future chunked handoff work. The
receiver checks every field, payload hash, expected prior position, approved
producer, and the value bytes against the *actual upstream calculation*.
The durable report retains hashes/fingerprints instead of full hidden payloads.

On a failure at any layer or handoff in a position, the coordinator restores
local per-layer KV cache steps and publishes no success receipt for that
position. This is **local in-process rollback**. It is not yet a distributed
transaction protocol or a remote-process crash recovery guarantee.

Hosts are distinct logical roles. `range_hosts` are qualified providers of
exact weight bytes. `compute_actors` are execution-assignment labels; all
math is still performed in the same Python interpreter. Changing range hosts
can change witness chains without changing numerical results. This is
observer-routing compatibility, **not demonstrated remote computation**.

## Independent verification

`tests/test_pretrained_chain.py` exercises the complete two-layer path with
synthetic admitted BF16 and F32 tensor weights and compares its final output
against both a separate full-decoder scalar reference (through normalized
vocabulary logits) and a byte-reconstructed two-layer scalar reference. It
also tests tampering with positions, producer identities, payloads, even
recalculated payload/digest pairs, failed transfer rollback, and source
reassignment. The scalar references do not invoke VL64's numerical operators.

```bash
python -m pytest -q tests/test_pretrained_chain.py
```

For an **optional live model** result, the pinned 272-tensor SmolLM2
manifest at `HuggingFaceTB/SmolLM2-135M` revision
`4e53f736cbb20a9a0f56b4c4bf378d9f306ff915` is used.

```bash
python -m experiments.model_fabric.real_chain_lab \
  --out artifacts/pretrained-two-layer-report.json
```

This fetches only the two complete layers' required BF16 tensors and two
selected embedding rows through strictly bounded HTTP 206 requests, without
fetching the full 269-MB checkpoint. The intended measured workload is
14,162,688 BF16 weight bytes, 228 nonempty requests at up to 65,536 bytes,
for token IDs 42 and 75. The independent scalar decoder reconstructs weights
from the *observed range bytes* and checks all 576 final hidden coordinates
within absolute tolerance 2e-5. Actual reported values must come from a
successful live run; these are planning counts, not an invented observation.
If network access or range integrity fails, the experiment exits nonzero and
writes an `UNRESOLVED` record, not a verification claim.

The live CI job is optional and preserves its evidence artifact whether or not
remote access succeeds. Fixture numerical tests and core-size enforcement are
mandatory.

## Next milestone

Continue from the proven two-layer handoff toward full checkpoint coverage.
Separate qualified tensor provision from actual remote execution, add signed or
otherwise independently authenticated compute receipts, durable causal cache
checkpoints, chunked handoffs for larger hidden widths, and partial retry with
quorum fencing. Only claim full language generation after final RMSNorm,
all 30 pretrained layers, vocabulary projection and tokenizer semantics are
independently checked. Receipt digests on their own prove internal consistency,
not authenticity of the complete source checkpoint.
