# Retained pretrained ranges — model-independent, cross-position reuse

This experiment follows the verified 30-layer SmolLM2 pass and Qwen/DeepSeek
pretrained arithmetic of VL-64. It addresses an avoidable execution cost:
repeatedly reading identical trained parameter ranges for every token or host
assignment. It intentionally makes **no changes to `src/vld/`**.

## One contract for Llama, GDN, MoE and future adapters

`qualified_range_cache.QualifiedRangeCache` is an optional callable in the
existing `QualifiedShardNavigator(range_host=..., approved_hosts=...)` path.
It never supplies a tensor just because an index contains its name. The first
read requires an **exact approved-host range witness** with the pinned model,
revision, index Ground digest, shard, byte offset, length, purpose and SHA-256.
Only the validated bytes enter content-addressed pages (each <= 65,536 bytes).
The same request on later token positions reuses the validated bytes, with
**the original source observer identity** and a visibly distinct
`cache_disposition=RETAINED_ORIGINAL_OBSERVATION` marker. It does not claim a
new network observation, signature, authorization grant or whole-checkpoint
signature. Model identity is fixed per cache root; another model cannot alias
bytes merely because its offset matches.

The original `QualifiedShardNavigator` continues to enforce exactly the same
byte length, digest and caller-approved host. Its logical `range_count` and
`weight_bytes` count **requested payload use**, including reused bytes; they
are not network bandwidth measurements. The wrapper's `stats()` separately
records **remote physical ranges and bytes**, retained cache hits, evictions,
and requalifications. The optional `cache_disposition` is also bound into the
range transcript when present. Existing noncached transcript hashes retain
their original representation.

## Durability and observer rehoming

The cache uses immutable content-addressed pages and an atomically published
catalogue. `snapshot()` returns the SHA-256 of the exact published catalogue.
The **receiver must retain that head independently**, such as through the
existing E-CPU authority/checkpoint mechanisms. Restoring a catalogue without
an external expected head, or with a mismatching head, is refused. The
checkpoint is a *single-writer* integrity contract, not an mTLS identity,
quorum certificate, or multi-writer ledger. Unpublished observations are not
part of a durable checkpoint. Logical LRU eviction can replace the current
cache head: an older externally pinned head must not be assumed recoverable
after eviction/republication. Cache capacity is an explicit caller budget,
not a promise to retain an entire checkpoint of arbitrary model size.

If an old source is no longer among currently approved hosts, a cache entry
**cannot silently pose as the new host**. Its exact range is freshly acquired
from a new approved host and accepted only if the bytes equal the previously
pinned contents. A changed weight, altered range, changed model/index, corrupt
page or absent checkpoint pin fails closed.

## What was independently tested

`tests/test_qualified_range_cache.py` exercises three-position Qwen GDN
continuation against the separate scalar oracle: the first position acquires
all required matrices; subsequent causal positions reuse the already admitted
ranges with zero extra remote requests. It also tests DeepSeek six selected
FP4 experts plus FP8 shared expert, bounded eviction, tampered ranges, wrong
model/revision, source reassignment, and a real **fresh Python process** which
reloads both the pinned weight cache and Qwen's paged causal Ground to
continue without touching the old weight host.

`retained_model_lab.py` optionally runs the previously independently verified
**real pinned trained subgraphs** twice with one cache. For Qwen this is one
real gated-delta recurrent head and output-projection contribution. For
DeepSeek it is the trained 384-way router and six FP4 selected *row slices*.
The inputs are deliberately repeated: this demonstrates true pretrained
weight reuse and that an independent scalar calculation remains unchanged.
It does **not** claim real full-layer causal continuation, a complete Qwen
27B forward pass, DeepSeek 510 GB inference, or removal of conventional
host compute. All full-layer pretrained claims require additional execution
and numerical reference evidence.

## Reproduce

```bash
python -m pytest -q tests/test_qualified_range_cache.py tests/test_retained_model_lab.py \
  tests/test_complete_native_layers.py tests/test_trained_subgraphs.py \
  tests/test_qualified_subgraphs.py tests/test_scaled_model_fabric.py

# Explicit outbound HTTPS / pinned 206 byte-range provider required:
python -m experiments.model_fabric.retained_model_lab --model qwen \
  --cache-dir artifacts/qwen-range-pages --out artifacts/qwen-retention.json
python -m experiments.model_fabric.retained_model_lab --model deepseek \
  --cache-dir artifacts/deepseek-range-pages --out artifacts/deepseek-retention.json
```

Pretrained requests that fail leave an `UNRESOLVED` result, never a fabricated
successful numerical receipt. In the optional workflow, only the fixture jobs
are mandatory; archived live results must be inspected for actual status.

## Next target: full pretrained layer with amortized weight acquisition

Build an admitted weight-range catalogue once for Qwen's entire GDN layer,
checkpoint that catalogue alongside its recurrent pages, and repeatedly execute
new causal inputs from the same pinned weights using independent scalar/tensor
references. For DeepSeek, only retain **selected** expert bodies after routing;
change of selected set should fetch only the newly needed expert ranges.
Native full-attention and mHC/Engram paths still require dedicated adapters.

## Live transfer accounting

The pinned live retention probe reports **tensor payload bytes**, **shard header bytes**, and **total remote transfer bytes** separately. The previous first-run report used the confusing field name `remote_weight_bytes_total` for the latter; that field included Safetensors headers. The corrected receipt uses `remote_transfer_bytes_total = first_pass_tensor_bytes + remote_header_bytes`. The independent scalar outputs are unchanged. Previous reports remain historical observations; new archived reports use the clarified naming.

## Archived genuine pretrained reuse observation (run #44579893703)

[Real pinned Qwen/DeepSeek range-retention run](https://github.com/Kier73/VL-64/actions/runs/44579893703) completed successfully. The 27B Qwen trained head used **5,277,188 tensor bytes plus 45,400 header bytes** (100 remote requests); the same-input second evaluation reused **98** already admitted requests and issued **no further network request**. The DeepSeek trained router plus six FP4 expert row slices used **3,973,680 tensor bytes plus 258,144 header bytes** (106 requests); the second evaluation reused **101** range requests without new network transfers. Both independently checked scalar subgraphs reported zero difference at 12-decimal display precision. The reported scope remains *two repeated subgraphs*, not an independently verified full native layer.

The original log data fields and job IDs are archived in [real retained-range observations](../experiments/model_fabric/datasets/2026-11-19-real-retained-pretrained-range-observations.json). `tests/test_saved_real_range_retention.py` checks the revisions, bytes, source job IDs, range reuse, selected expert IDs, and numerical status offline.
