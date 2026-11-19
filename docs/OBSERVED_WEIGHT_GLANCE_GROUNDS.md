# A weight observation becomes a portable, executable Ground

This experiment follows VL-64's scale-independent sharded model index, native
Qwen and DeepSeek operators, and retained qualified-range cache (PRs #28–34).
The idea is that trained weights need not be retrieved anew for every operation:
a qualified observation can be retained as exact bytes, and a new observer can
read *different portions* of those bytes to evaluate the admitted model math.

**Observation is not inference by metaphor.** Reading a tensor header does not
supply its values; a model needs every required weight value and the requisite
arithmetic. This implementation gives the Virtual Layer a materialized,
verifiable *read surface* for previously acquired values, not an imaginary or
compressed replacement for missing weights.

## The new read surface

`weight_glance_ground.py` introduces:

1. `freeze_weight_glances(...)`: take an **externally pinned** cache catalogue,
   revalidate its content-addressed pages and historical observer identity,
   and freeze a separate manifest with a **new externally retained head**.
   Same-filesystem pages use hardlinks (new names, no second payload copy);
   cross-filesystem pages are streamed and copied. Neither modifies the
   portable VL-64 core.
2. `PinnedWeightGlanceGround(...)`: in any later process, require that separate
   Ground head and the explicit list of trusted historical source observers.
   Index the observed byte intervals by `(shard, purpose)`.
3. **Read/repartition**: reconstruct any **at most 64 KiB** requested interval
   covered by compatible qualified glances. Verify source-page hashes every
   time; reject unseen holes, contradictory overlapping observations, stale
   revision/index identity, invalid headers and inaccessible pages. Only source
   ranges are materialized; never a whole weight tensor or model.
4. **Derivation witness**: expose the precise original source requests and
   digests, the independently pinned Ground head, and the explicitly approved
   *local reconstruction observer*. The local observer is **not** represented
   as the old HTTP host; the execution range chain binds the Ground head and
   the derivation history as well as the reconstructed byte hash.

This is different from the preceding exact-key cache: a fresh request for
`weights[17:79]` may now be derived from previously admitted observations of
`weights[0:64]` and `weights[64:128]`, without requesting either original
range again. **Unobserved weights still cannot be inferred.**

## Observed causal experiment

Offline Qwen fixtures verify three positions against the *independent* scalar
Gated DeltaNet reference, with all weights read from a frozen Ground and no
remote host. Another regression restores the saved convolution/delta state and
frozen weight Ground in a **fresh Python process**, continues at the next
position, and matches the original process numerically.

The same range mechanism works for DeepSeek's packed FP4/scale requests; it
holds byte-exact fragments, not an invented model-neutral numeric decoder.
The optional pinned live experiment replays the previously verified *trained*
Qwen head and DeepSeek router+six-expert **partial slices**, not their full
27B/510GB models. It checks the existing independent numerical reference
before and after the offline replay and requests a new smaller tensor window
that was not the original network request.

```bash
python -m pytest -q tests/test_weight_glance_ground.py tests/test_glance_model_lab.py
# Requires real model access and outbound HTTPS Range 206 support:
python -m experiments.model_fabric.glance_model_lab --model qwen \
  --root artifacts/qwen-glances --out artifacts/qwen-replayed.json
python -m experiments.model_fabric.glance_model_lab --model deepseek \
  --root artifacts/deepseek-glances --out artifacts/deepseek-replayed.json
```

## Authority and fidelity

- Caller-provided externally pinned SHA-256 heads establish **integrity under
  the caller's trust assumptions**, not proof of checkpoint publisher identity.
  The original `QUALIFIED_EXACT_SHARD_RANGE` host already assumes explicit
  caller approval and correct original source bytes. Reconstructed reads are
  bound to original host witnesses and their exact page hashes.
- The frozen view does not enable inference on never-observed spans. If a
  required weight/scale is absent or contradictory, it returns **UNRESOLVED**;
  the caller must acquire new bytes from another approved source rather than
  assert that reconstruction magically supplies them.
- No model-specific memory scale is imposed by the manifest; each source or
  reconstructed data request remains ≤65,536 bytes. Freezing verifies the
  already retained byte pages but never pulls new weights from the network.
- Hardlinked bytes remain writable from elsewhere; every read rehashes the
  original full source page. For hostile concurrent writers, use OS-level
  read-only storage and authenticated external Ground-head retention. This is
  a single-writer snapshot, **not** multi-writer consensus/fencing or a
  whole-checkpoint publisher-signature verification.
- Local mathematical work is still necessary. Ground navigation removes
  redundant acquisition and reinterprets bytes in new contexts; it does not
  eliminate nontrivial trained model arithmetic.

## Next experiment

Prepare a complete Qwen Gated DeltaNet layer under an admitted memory/disk
budget, freeze its ~trained projection observations, then execute multiple
causal positions from that pinned Ground under different observer processes.
After independently verifying the actual full-layer outputs, extend the
operator to the model's remaining full-attention layers. DeepSeek's next
similar step is selected-only expert coverage spanning entire w1/w3/w2 matrices
and all scale pages, rather than row slices. Preserve the same Ground format.