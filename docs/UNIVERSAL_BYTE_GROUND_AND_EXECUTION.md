# Universal Byte Ground and General Realization Contract

**State:** Implemented first pass with live experiments, November 18, 2026.  
**Source:** [Generic Byte Fabric](../experiments/byte_fabric/README.md).  
**Compact-runtime policy:** The \`vld.ObserverIntention\` contract remains within the 65,536-byte package; acquisition, codec exploration and realizer adapters remain external.

## Principle of generality

**Anything represented by bytes admits an exact byte observation and an abstract Ground.** That claim does not depend on an existing decoder, particular CPU/GPU architecture, browser, model, web page, file extension, application framework or transport.

However, the ability to:
1. **represent** those bytes,
2. **interpret** them as typed variables and relationships,
3. **predict** their future states under interventions,
4. **realize** a requested computation,
5. **independently verify** the realization

is progressively stronger. The implementation must never silently promote (1) to (4) or (5).

The fundamental utility is a *communicable observer-intention protocol*: tell the virtual layer **what information or result is desired**, let a qualified host adapt that request to available byte sources, and return an explicitly supported representation and evidence.

## Execution pipeline

~~~text
ObserverIntention
  operation: bytes.observe | bytes.execute | bytes.capabilities
  target: source identifier / requested host information
  output: JSON, JSONL or text
  options: bounded, non-executable parameters
        ↓
Source capability and admission
  memory | local workspace | HTTPS | registered host adapter
        ↓
Byte observation
  exact bytes or explicitly bounded prefix, SHA-256, source and transport
        ↓
Structural Ground
  source-independent structure identity + per-capture evidence identity
        ↓
Optional typed descriptions
  recognized built-in views + host-registered decoders
        ↓
Optional execution
  explicit pure, bounded, typed realizer; no implicit remote authority
        ↓
Receipt and replay
  Ground identity + observation evidence + actual result + operator scope
~~~

### Independent identities

The Ground carries two fingerprints:
- \`ground_digest\`: identity of the byte content/prefix scope and its declared structural views, independent of where the bytes were acquired or the transport cost.
- \`observation_digest\`: identity of a specific observer's acquisition, including source and transport evidence.

If two hosts obtain the same complete representation and interpret it identically, they can share a Ground identity while preserving distinct source/capture histories. If only a prefix was observed, that identity refers **only to the prefix**, not to the unknown full contents.

This matters when reusing a description in another host or selecting a less expensive network path; latency and observation time belong to the evidence record, not necessarily the represented state.

### Capability discovery

An observer can request:

~~~json
{
  "operation": "bytes.capabilities",
  "target": "*",
  "output": "json",
  "options": {}
}
~~~

The host replies with qualified source identifiers, installed decoder names, typed operator signatures, input limits and permitted HTTPS hosts. No source content needs to be acquired to answer this query.

This is the first capability-negotiation step toward a substrate-independent planner. It is **not** proof that an advertised operator can run on every architecture or remote provider.

### Generic observation

~~~json
{
  "operation": "bytes.observe",
  "target": "https://example.com/",
  "output": "json",
  "options": {
    "max_bytes": 16384,
    "include_data": true
  }
}
~~~

A pure HTTPS acquisition obtains a byte representation from an authorized host, records completeness and digest, and attempts safe local syntactic interpretations. If no codec recognizes it, the outcome remains an \`opaque-bytes\` Ground rather than a failed or fabricated semantic interpretation. \`include_data\` optionally returns up to 4,096 bytes as base64, marked full or prefix.

The system supports public HTTPS host authorization through explicit host policy and local workspace root confinement. A request does not gain permission to connect to arbitrary hosts or read arbitrary local files by changing its \`target\`.

### Generic computation

~~~json
{
  "operation": "bytes.execute",
  "target": "workspace:experiments/byte_fabric/fixtures/state.json",
  "output": "jsonl",
  "options": {
    "operator": "json.pointer",
    "parameters": {
      "pointer": "/observer/constraints/verified"
    }
  }
}
~~~

The executor checks the source and interpretation, ensures the operator is registered and authorized for the acquired input type and completeness, computes its result, and records an execution receipt. A separate replay function can recompute the declared operation given the originally acquired bytes. **Local recomputation is not independent verification of a remote service's truthfulness.**

### Architectural neutrality

A new byte-bearing device or system should be enabled as a **host-installed source adapter**, not as a special case in the core:

~~~python
from experiments.byte_fabric.extensions import (
    SourceRegistry, SourceCapability, DecoderRegistry
)
from experiments.byte_fabric.sources import capture_bytes

sources = SourceRegistry()
sources.register(
    SourceCapability("screen:"),
    lambda target, limit: capture_bytes(frame_bytes[:limit], target)
)
~~~

The code above is illustrative: the host must supply actual \`frame_bytes\` from an authorized screen capture. A separate registered decoder could interpret width/height/pixel format, an instruction stream, a binary record, or a memory map, but those interpretations retain their source evidence and explicit assumptions.

The same abstraction can support:
- software file and memory snapshots;
- screenshots and graphics buffers;
- packet and communication-frame sequences;
- GPU/CPU registers and hardware counters;
- model configuration and Safetensors tensor-header metadata;
- binaries with ELF, PE/MZ or WASM headers;
- types not anticipated when VL-64 was written.

Adapters operate under their own permissions, provenance and capture budgets. Generic byte support does not imply automatic access to devices or networks.

## Actual experiment

A [GitHub Actions experiment](../.github/workflows/generic-byte-fabric.yml) ran with public web resources and local/synthetic files. It demonstrated exact Grounds for text, JSON, HTML and opaque binary, plus typed header recognition for common formats and Safetensors without reading tensor values. It executed SHA-256, JSON-pointer and HTML-link operations, and rejected complete-file operations when only 512 bytes of a larger remote file had been observed.

A second live operation used the **same generic JSON-byte parser** to inspect Hugging Face model metadata and extract \`/id\`; no Hugging Face-specific Ground interpreter was involved. This demonstrated 4,863-byte actual metadata responses as ordinary JSON byte sources.

The experiment subsequently showed **stable structural Ground digests across Python 3.11 and 3.13** for equal byte inputs, while the observation evidence remained separate. The test suite reached 107 passing cases in each environment before the documentation was completed. The original small runtime's source budget stayed intact except for the very small increase in admissible observer target length.

## Information and prediction

The byte-fabric layer should connect next to existing empirical transition/discovery utilities. A raw byte Ground answers *what was observed*; a predictive Ground additionally needs:

- action and clock context, including observation/recovery ordering;
- typed state variables or equivalence quotients with explicit extraction maps;
- transition hypotheses with supporting and contradicted witnesses;
- uncertainty classifications for partial and unobserved states;
- held-out evidence that a candidate description predicts a desired observable;
- cost and authority contracts for an operation that can realize a requested result.

The central research objective is to **discover the representation that offers the most useful, independently validated state transitions at the lowest measured total cost**. Do not maximize superficial byte count or mistake a learned equivalence for a theorem.

## Existing limitations to solve next

1. **Streaming and distributed reconstruction:** current built-in captures are bounded to 1 MiB per request and default to 64 KiB. Large files/models require chunked Merkle evidence, streamed digests, cross-host representation identity and a checkable reconstruction contract.
2. **Generic remote operator admission:** the current \`bytes.execute\` registry runs qualified deterministic local operations over captured bytes; it does not call arbitrary inference APIs or write to remote services. Add remote realizers with side-effect policies, provider/version qualifications, cost ceilings and independent output receipts.
3. **True architecture interpretation:** ELF/WASM, image and tensor-header recognition is shallow by design. Run structural parsers in isolated host adapters, label incomplete traces, and compare candidate executable models against actual output.
4. **Typed multi-hop navigation:** source, decoding and execution presently form an explicit short pipeline. A general planner should discover possible paths, compare cost/evidence, and select one based on the intention rather than the user's known URL.
5. **Selection and verification:** Ground identity is structural, but an operator cannot use that identity as authority. Require independent output checks, temporal freshness and explicit observer-host provenance for any remote action.
6. **Broader output contracts:** the core currently supports JSON, JSONL and text, with binary fields base64-encoded. Raw byte streams and user-defined output schemas should be admitted through typed output adapters rather than arbitrary code.

### Next acceptance experiment

**Without any domain-specific branches inside the VL-64 core**, connect three new sources (e.g. a screen buffer, an instrumented process, and a remote stateful test endpoint), register typed decoders and qualified operations at the hosts, and ask the same external observer protocol for meaningful results. Require a replayable result, correct unknown-state/refusal behavior, and measured cost/evidence across each source.

Success is *not* decoding every binary format in existence. It is the ability for new, independently verified computational interpretations and realizers to become available without changing the universal representation or the observer's communication contract.
