# MPF-001 — Many VL-64 observers, many externally hosted model pages

## Research direction

An externally hosted model is a **source of structure**, not a physical
payload to acquire and transfer into the Virtual Layer.

The source host remains responsible for its own model representation.
E-CPU HSS-002 observes its immutable **metadata Ground**: revision,
source file addresses/declared extents, optional content identities and
source-declared architecture. VL-64 compiles that Ground into independently
navigable observer-specific intentions.

The new scaling dimensions are orthogonal:

- **One source model, N observers.** One source description and immutable
  source head; each observer owns a unique causal stream and may issue a
  different intention and interpret the same source from a distinct locus.
- **M source models, N observers.** Many external models with independently
  qualified revision identities can be interpreted in tandem. Neither
  model byte length nor host placement is used to derive a false language
  model behavior.

This integration is in \`experiments/model_fabric/\`, leaving VL-64's
**64 KiB compact runtime unchanged**. It uses the existing native
\`vld.ObserverIntention\` and domain-separated receipt hashes.

## Actual original implementations reused

| Research component | Existing implementation | Role in this experiment |
| --- | --- | --- |
| Source description | E-CPU \`mailcpu.external_virtual_structure.SourceStructure\` | Immutable original externally hosted model's structural address Ground |
| Source observation | E-CPU \`HTTPSMetadataObserver\` and \`MetadataNavigator\` | Only Hugging Face metadata API; never /resolve or weight payload ranges |
| Virtual position oracle | E-CPU \`compile_virtual_read\` | Independent cross-repository structural result and 3-frame control receipt |
| Observer intention | VL-64 \`vld.ObserverIntention\` | Source-bound, typed, checksum-addressed intention without importing remote code |
| Parallel fleet | VL-64 \`VirtualModelPageFleet\` | Multiple source aliases and independent observer state machines |
| Portable continuity | \`VirtualModelPageFleet.snapshot/restore\` | Replay, rehome, receiver-owned result verification from description and receipts |

Implementation: \`experiments/model_fabric/virtual_page_fleet.py\`.
Live benchmark: \`experiments/model_fabric/virtual_page_fleet_lab.py\`.
Tests: \`tests/test_virtual_page_fleet.py\`.
CI: \`.github/workflows/virtual-model-page-fleet.yml\`.

## Formal structure

For source model \`m\`:

\[
S_m=(\mathrm{repository},\mathrm{revision},\mathrm{files},
\mathrm{architecture},\mathrm{observations})
\]

\[
G_m=H(\mathrm{canonical}(S_m)).
\]

The original source's weight or checkpoint data remain at the source host.
The Virtual Layer needs only a descriptive Ground \`S_m\`; for a model with
billions of parameters, this can still be a few dozen kilobytes.

Each observer \`o\` gets an independent lineage:

\[
L_{m,o}^{0}
=H_{\mathrm{VL64.genesis}}(o,m,G_m).
\]

Its intention \`I_{m,o,k}\` is a real VL-64 \`ObserverIntention\` with an
operation, exact options and source-qualified virtual target address.
The interpreted result \`R_{m,o,k}\` must match the original E-CPU
source-local structural law:

\[
R_{m,o,k}=F_{\mathrm{HSS002}}(S_m,I_{m,o,k}).
\]

The receipt's predecessor is the observer's last accepted lineage head:

\[
L_{m,o}^{k+1}=H_{\mathrm{VL64.receipt}}
\big(G_m,o,m,\mathrm{host},\mathrm{epoch},k,
I_{m,o,k},R_{m,o,k},L_{m,o}^{k}\big).
\]

**Two different observers can see the same source-relative position
but do not share a causal identity.**

For one model, all observers' \`source_head\` and structural \`position_head\`
can coincide while their \`receipt_head\` values differ. For distinct
models, each source/revision is separately bound and cross-model
receipt substitution fails validation.

The 3-frame E-CPU HSS-002 position is independently reproduced by VL-64
and compared byte-for-byte against the original E-CPU interpreter:

1. \`ground.source\` — immutable source identity.
2. \`ground.intention\` — requested operation and coordinates.
3. \`ground.realization\` — checksum of source-determined structural result.

**No payload frame exists.**

## Computable intentions

\`source.catalogue\`, \`source.file\`, \`source.extent\`,
\`source.architecture\`, and \`source.layer\` are interpreted
independently of model architecture and weight encoding.

Example using already observed E-CPU metadata:

~~~python
from experiments.model_fabric.virtual_page_fleet import (
    HostedModelSource, VirtualModelPageFleet
)

# ecpu_source is an original HSS-002 SourceStructure, not a model weight file.
model = HostedModelSource(
    ecpu_source.to_dict(), expected_head=ecpu_source.ground_digest
)

fleet = VirtualModelPageFleet()
fleet.register("deepseek", model)
fleet.open("observer-A", "deepseek", host_id="host-A")
fleet.open("observer-B", "deepseek", host_id="host-B")

a = fleet.observe(
    "observer-A", "source.layer", {"index": 0}, request_id="intent-A"
)
b = fleet.observe(
    "observer-B", "source.layer", {"index": 0}, request_id="intent-B"
)
assert a["position_head"] == b["position_head"]
assert a["head"] != b["head"]
assert a["model_payload_bytes"] == b["model_payload_bytes"] == 0
~~~

An observer can also form \`source.extent\` with a very large integer
offset and length; this is a **symbolic virtual coordinate**, not a
range request. Its interpretation cost depends on the source
description and operation contract, not on the number of external
model bytes.

### Multiple observers executing concurrently

\`VirtualModelPageFleet.observe_many\` dispatches independently
identified observer work through a bounded thread pool. Each observer
has its own lock for causal sequencing; a shared ledger lock protects
only source/session registry and receipt commit. Distinct observers
are therefore free to interpret simultaneously.

A request ID is idempotent within its observer; reuse with a different
intention is refused. A stale observer tip cannot publish the next
state in a running coordinator.

\`rehome\` changes an observer's host and advances an explicit epoch.
The full source/intention/receipt stream is reconstructed in another
Python process using its immutable descriptions. **Network-partitioned
multi-coordinator fencing is E-CPU M3's existing responsibility**,
rather than an invented VL-64 quorum.

## Live M-model / N-observer experiment

The executable CLI points the model page observer to actual Hugging Face
sources through E-CPU HSS-002:

~~~bash
PYTHONPATH=/path/to/E-CPU \
python -m experiments.model_fabric.virtual_page_fleet_lab \
  --observers 128 --workers 16 \
  --out artifacts/vl64-virtual-model-page-fleet.json
~~~

Default independently selected model pages:

- [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/main)
- [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/tree/main)
- [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/tree/main)

Use repeated \`--page https://huggingface.co/{owner}/{model}/tree/main\`
to select a different set of sources.

The observer resolves \`main\` to an immutable commit separately for
every source, then observes only that source's bounded metadata tree.
No checkpoint, tokenizer, model shard or partial weight range is read.
Each source Ground is verified against original E-CPU and then handed
to the VL-64 interpreter.

The benchmark systematically evaluates the number of observers per
model at \`1, 8, 64, 128\` (up to user-configured count) and records:

- Immutable per-model source and revision heads, model type and
  declared external source-size sum.
- Cold **source metadata observation count/bytes/time**.
- Warm multi-observer interpretation time and receipt throughput.
- Shared structural-position count versus **independent** observer
  receipt heads.
- Observer migration, replay, checkpoint/reconstruction time.
- Explicit \`model_payload_bytes_downloaded = 0\` and
  \`source_local_trained_inference_calls = 0\`.

The benchmark's checkpoint preserves every description and observer
receipt, allowing an independent auditor to recompute all results.

## What "running" means at this stage

There are now **two different modes of activity** in the model fabric:

1. **Virtual structural processors:** source-relative interpreters
   that are actually running, can scale in tandem, and have complete
   source-bound deterministic structural semantics. These require
   source metadata, not pretrained weights.
2. **Source-local trained neural processors:** the existing VL-64
   \`operational_ground\`, \`operational_hosts\`, \`intention_fabric\`
   and provider/host adapter experiments. Such a processor needs a
   qualified source-local numerical host and actual inference
   evidence. An arbitrary external model page is not automatically
   proof of such a capability.

MPF-001 creates the general shared source-ground and observer identity
space on which **many genuine source-local model operators** can later
be bound; it does not invent model inference from an address.

## To Do — source-local behavior, many-to-many

The next step is to permit a source-resident observer to advertise a
specific **typed callable behavior** alongside its immutable source
Ground and to bind it to an original E-CPU host capability.

A receiver could then express an intention, navigate a source
operator that *already runs at the remote locus*, receive only an
independently qualified behavioral result, and update its own
continuous VL-I state. This would make *many observers per real
trained model* and *many genuine different trained models per
virtual fabric* operational, not merely structural.

Measure source-local host compute time, number of qualified
behaviors served simultaneously, action-specific source evidence,
source-handler replacement, and cross-model result isolation.
Preserve separate semantics for structural equality, numerical
agreement, and actual behavioral intelligence.
