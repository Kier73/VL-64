# MPF-003 — Authenticated access to original-host pretrained behavior

## Objective

Extend [MPF-001](VIRTUAL_MODEL_PAGE_FLEET.md) source-pinned model Grounds
and [MPF-002](SOURCE_PINNED_BEHAVIOR_FLEET.md) independent causal observer
sessions to **real pretrained numerical behavior from a model that is
already running on its original host**, without transferring that model
into the receiver.

The relevant pre-existing parts are:

| Component | Original implementation | Role |
| --- | --- | --- |
| Immutable original source | E-CPU HSS-002 `SourceStructure` | Source repository/revision, metadata Ground, original model address |
| Model-page observer | `VirtualModelPageFleet` | Distinct causal observer tips, source pins, rehome/replay |
| Trained numerical operator | `operational_ground.py` | Token prefixes, F32 logit-tile coverage/commitment, argmax, typed checkpoint |
| Running pretrained model | `AttachedResidentPretrainedLanguage` | Uses **supplied, already-loaded** model/tokenizer/torch objects at the host |
| Authenticated transport | `SignedResidentHostGateway` / `AuthenticatedResidentHostClient` | Verified TLS, source/operator pins, HMAC, nonces, bounded frames, exclusive causal lease |
| Observer-to-model bridge | `SourcePinnedBehaviorFleet` | Independent model conversations, sourced receipts, idempotency, causal rehome and replay |

The compact 64 KiB VL-64 core is unchanged. No `from_pretrained`,
model download, model shard read, weight-range reader or checkpoint
loader is added by this experiment. The source host is responsible for
its existing resident model. Host and receiver exchange **only bounded
control, text/token, F32 logit observations, and authenticated evidence**.

## Exact identities and host qualification

The source Ground `S_m` was already observed by E-CPU HSS-002:

\[
G_m = H(\\operatorname{canonical}(S_m)),\\quad
S_m.\\mathrm{revision} \\in \\{0,\\ldots,9,a,\\ldots,f\\}^{40}.
\]

The original-host numerical descriptor `A_m` is admitted with exact
caller-held pins:

```text
A_m.model           == S_m.repository
A_m.revision        == S_m.revision
A_m.observer_host   in approved_hosts
A_m.artifact_id     == independently_expected_artifact_SHA256
H(A_m)              == expected_operational_ground
```

The digest `G_m` identifies the original source's **metadata**. The
artifact pin identifies the admitted host's **claimed numerical artifact**.
An independent artifact inventory or source-side verification is needed
to connect these claims to exact pretrained checkpoint bytes. Hash
matches on two self-reported strings alone do not prove trained
arithmetic or remote hardware identity.

The host gateway accepts one `source_head`, one
`operational_ground_digest`, and a secret key known only to separately
provisioned, approved participants. Requests include a 128-bit nonce,
UTC issue time, immutable source head, operational head, operation
payload and current numerical lease. Both directions use SHA-256 HMAC
over the complete canonical frame. The client requires verified HTTPS
certificate validation and rejects redirects; the server processes
only exact `/v1/operational` POST frames no larger than 64 KiB.

Only one receiver holds the resident numerical execution lease at a
time; it spans `encode -> open -> all logit tiles -> advance ->
decode`. A bounded host lease expires after 120 seconds without a
valid continuation, which permits recovery after a failed receiver.
The lease is refreshed on every authenticated numerical call.
A shared mutable KV/cache session is never interleaved with a second
receiver's numerical calls. Different original resident hosts can
serve independently in parallel.

## Original-host adapter

On a **pre-existing original model host**, with the model and tokenizer
already alive in its own process:

```python
import os
import ssl

from experiments.model_fabric.attached_resident_pretrained import (
    AttachedResidentPretrainedLanguage,
)
from experiments.model_fabric.authenticated_resident_host import (
    SignedResidentHostGateway, serve_https,
)

# Pre-existing objects, supplied by the existing source host:
# resident_model, resident_tokenizer, resident_torch
# Independently verified source revision, HSS Ground, artifact pin:
# repository, revision, source_head, verified_artifact_sha256

resident = AttachedResidentPretrainedLanguage(
    model=resident_model,
    tokenizer=resident_tokenizer,
    torch=resident_torch,
    repository=repository,
    revision=revision,
    artifact_identity=verified_artifact_sha256,
    observer_host="original-pretrained-host",
    native_chat=True,
)

gateway = SignedResidentHostGateway(
    resident,
    source_head=source_head,
    model=repository,
    revision=revision,
    artifact_identity=verified_artifact_sha256,
    observer_host="original-pretrained-host",
    secret=bytes.fromhex(os.environ["VL64_ORIGINAL_HOST_HMAC_HEX"]),
)

context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain("host.crt", "host.key")
serve_https(gateway, address="0.0.0.0", port=8443, tls_context=context)
```

In particular, this adapter never calls `AutoModel.from_pretrained`,
`AutoTokenizer.from_pretrained`, `snapshot_download`, file reads or
weight-range network requests. The resident model object is passed
by reference. Its native tokenizer performs original model role
templating when `native_chat=True`. The pre-existing
`HostedTransformersLanguage.calculate_logits` performs the actual
forward pass at the host, with host-local exact-prefix KV reuse.

For stronger identity, the source's checkpoint verifier should supply
a hash of the original admitted model artifact. A mere revision-based
hash can serve as a routing claim, but should not be presented as
an independent byte-level checkpoint witness.

## Receiver and multi-turn benchmark

On the **observer machine**, from a previously captured source Ground
JSON, exact source/operator hashes and independently pinned artifact:

```bash
export VL64_ORIGINAL_HOST_HMAC_HEX="...64+ hex characters from private provisioning..."

python -m experiments.model_fabric.authenticated_pretrained_lab \\
  --source-description /secure/observed-original-hss-source.json \\
  --source-head YOUR_HSS_SOURCE_GROUND_SHA256 \\
  --operator-ground YOUR_APPROVED_OPERATIONAL_GROUND_SHA256 \\
  --artifact-head YOUR_INDEPENDENT_ARTIFACT_SHA256 \\
  --endpoint https://original-host.example/v1/operational \\
  --approved-host original-pretrained-host \\
  --observers 8 --workers 8 --tokens 2 \\
  --tokenizer-mode native_chat
```

The `source-description` file is the compact HSS-002 source record,
not a checkpoint. Its SHA-256 is independently recomputed. The first
round opens eight separate `model.behavior.generate` intentions and
the second carries each observer's original assistant output into its
next message. All generated tokens are grounded in the original host's
full logit-tile stream, checked for contiguous complete coverage,
sha256 equality to the host's full logit commitment, and independently
selected argmax at the receiver.

Recorded metrics include independently committed observer heads,
total model-call count, total emitted causal tokens, source identity,
host alias, wall time for each round, replay time and zero model bytes
into the Virtual Layer. The CLI is explicitly opt-in and fails if
there is no usable source-resident host, exact pin or authenticated
endpoint. It will **never substitute a fixture model silently**.

## First reproducible separated-process evidence

`tests/test_authenticated_resident_pretrained.py` establishes an
actual cross-process **trained** numerical host and receiver without
transferring the host's trained numerical parameters. A deterministic
count-trained byte model is instantiated in the original host child
process; the parent exchanges only HMAC-authenticated canonical RPC
frames over a local test pipe. The test also exercises one-host/many-
observers, two-turn causal history, exact-prefix failover challenge,
cross-source substitution refusal, stale/replayed signed frames,
altered signatures, exclusive host leases, and an already-loaded
transformer adapter with no loader calls.

The pipe transport tests the authenticated message protocol and
process separation; production transport uses HTTPS with validated
certificates. The count-trained fixture demonstrates real learned
host arithmetic, **not** the DeepSeek/Qwen/SmolLM2 original weights.

The repository also retains the earlier original
[SmolLM2-135M pretrained causal forward pass](OPERATIONAL_MODEL_GROUNDS.md),
which executed at pinned revision
`4e53f736cbb20a9a0f56b4c4bf378d9f306ff915`
inside a single PyTorch host, with VL-64 receiving logit tiles rather
than weights. MPF-003 supplies the missing authenticated observer
protocol and no-new-load host attachment; **a new live connection to
the existing external pretrained process requires that process's
network endpoint, source witness, and provisioned authorization**.

## Next experiment

Run the opt-in benchmark against a separately identified pretrained
host that already serves the exact original Hugging Face revision;
compare 1, 8, 64 and 128 independent two-turn conversations. Extend
the source-host adapter with per-observer KV namespaces and concurrent
numerical leases to measure true host-side parallel inference, rather
than merely concurrent observer scheduling. E-CPU's existing mTLS
identity, durable prepare/commit and quorum fencing are natural
candidates for multi-host partition recovery.

### Security and measurement scope

Authenticated HMAC plus HTTPS proves that the accepted messages came
from a holder of the approved key over a TLS-verified endpoint; it is
**not** GPU attestation, a proof that a hostile provider used particular
weights, or a guarantee of exactly-once physical computation across
network crashes. The gateway nonce cache is process-local; durable
cross-restart anti-replay/fencing should use E-CPU's authenticated
channel ledger. Logical receipt idempotency remains enforced by the
observer fleet. Successful numerical evidence requires actual
host-generated logits and receiver verification of every tile.
