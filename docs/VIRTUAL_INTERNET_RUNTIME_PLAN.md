# Virtual Internet as Computation — VL-64 Findings and VL-N64 Design

**Status:** observed network-ground experiment and follow-up design, November 18, 2026.  
**Lab:** [experiments/internet_lab](../experiments/internet_lab/README.md).  
**Evidence:** [versioned dataset](../experiments/internet_lab/datasets/2026-11-18-github-run-44371133726.json) and [executed workflow](https://github.com/Kier73/VL-64/actions/runs/44371133726).

## 1. Research question

Can a virtual layer use the Internet's available representations and operations as a directly navigable computational environment: observe some reachable state, construct a qualified Ground of permitted state transformations, and satisfy an intention through a verified sequence of existing relations instead of recreating every operation locally?

A host is any permitted *observable computational locus*, including a resolver, Git service, HTTPS endpoint, local TCP receiver, GitHub Actions process, or content-addressed object. The intent is to be **substrate-adaptive**. A service's already-existing behavior can be used as an operator, while the virtual layer remains responsible for representation, routing, admissibility, and proof of the result.

Do not confuse three different claims:
1. A service computed a result in response to a protocol request (observed and testable).
2. A network path forwarded, framed, negotiated or validated a property (observed and testable).
3. The physical transmission medium itself performed an arbitrary operation (not established by the current evidence).

The first two are useful now and can become more expressive with better Grounds. Any stronger claim should have a distinctive measurement and negative control.

## 2. What was actually exercised

The lab ran a fixed, read-only experiment on a GitHub-hosted Ubuntu runner. Python 3.13 executed the unmodified 63,528-byte VL-64 runtime plus a separate network adapter; the larger lab does **not** count toward the 64 KiB runtime budget.

| Operation | Observation | Computation / information relation |
| --- | --- | --- |
| OS \`getaddrinfo\` | Three hostnames produced observed address sets | Name → scoped address-set description |
| Cloudflare DoH JSON | 200 response with 2 A records | Typed DNS question → answer-set record |
| Cloudflare DoH wire | Explicit 29-byte DNS query, 61-byte response | RFC wire packet → parsed A RRset and TTL metadata |
| TLS \`example.com:443\` | TLS 1.3 negotiation, certificate digest | Hostname → session/security properties |
| Git v2 capability discovery | 9 pkt-line frames with capabilities | Git service → advertised protocol operators |
| Git v2 \`ls-refs\` | Bounded POST to upload-pack, HEAD commit | Git command → repository identity at a time |
| GitHub REST | Repo metadata and commit query | Repository description → resolved commit |
| Pinned raw GitHub content | README and linked research-goals document | Revision + path → stable content digest |
| HTTP conditional GET | \`304 Not Modified\`, zero document bytes | Cached content identity + server validator → reuse without 6,745-byte retransmission |
| TCP loopback | 1,024 bytes sent, 32-byte result checked | Client → kernel transport → endpoint's SHA-256 operator |
| Public HTTP | HTTP status, digest, cache headers | URL → content observation and possible freshness hints |

The experiment produced **16 observed operations, 14 Ground nodes, 9 recorded edges**, with **three short protocol response packets retained and replayed independently**. Git wire and REST returned the same commit identifier. JSON DoH and binary DoH reported the same A-answer set in the measured session.

A graph route through \`repository → pinned commit → README → research goals\` was found as three observations, and its evidence digests were checked. The HTTP validator example reused prior knowledge to avoid payload transfer; no data was created ex nihilo and no path-native hash computation was claimed.

A second run can produce different DNS TTLs, ref heads or body/cache metadata. Each Ground is explicitly scoped to the relevant observation time and source.

## 3. Architectural adjustment to VL-64

The existing \`ObservedGround\` has \`bytes → selected offsets → successor counts → cycle navigation\`. That is powerful for small finite state descriptions, but *Internet operations are typed, action-dependent, multi-provider, partially observed and time-qualified*. Simply appending more byte views would destroy useful semantics.

Add the following capabilities as independently tested units, not one oversized core module:

1. **Typed observation contracts:** request kind, source representation, target representation, method, accepted content schema, side-effect classification, observer identity and permitted outcome set.
2. **Versioned Ground graph:** directed edges with a real protocol operation, source/target type, witness digest, confidence and issuer scope. Maintain multiple possible outcomes without erasing ambiguity.
3. **Freshness/lease semantics:** immutable commit/object identity versus expiring DNS answers, conditional HTTP validators and uncertain network status. Distinguish monotonic local time, DNS TTL, cache policy and provider response time.
4. **Strict packet and document decoders:** parse DNS questions/answers and Git pkt-lines with bounded length and explicit refusal on malformed input. Keep large general-purpose network libraries in adapters.
5. **Path search by intention:** query “resolve this name,” “identify the current HEAD,” or “find this document” as a requested observation relation rather than a fixed URL script. Resolve a composed path only when all edges match the requested types and evidence requirements.
6. **Independent checking:** preserve actual wire packets or a reproducible subset, source pins and per-hop digests; replay what is checkable offline. Separate “observed matching output,” “authenticated service,” and “proven semantic equivalence.”
7. **Measured execution and cache reuse:** maintain observation, transfer, operator, verification and recovery costs separately. Count bytes avoided by qualified reuse and refuse stale descriptions when freshness is part of the intention.
8. **Admitted remote realizers:** only after explicit policy, allow a locally qualified HTTPS, Git, DNS, or TCP adapter to perform one permitted, bounded operation on demand. The recorded graph does not itself authorize remote work.

These are candidates for a **network-native specialization** rather than forcing them into the ~2 KiB VL-64 headroom.

## 4. Proposed separate 64 KiB utility: VL-N64

**Name:** VL-N64 — 64 KiB Network-Native Virtual Layer.

**Purpose:** observe, describe, qualify, navigate, and verify permitted Internet relations as a distributed computational state space.

It should be a **standalone Python standard-library utility** with an explicit source-size ceiling of 65,536 bytes. No browser engine, root packet capture driver, heavyweight model, or service credential store belongs inside the core. External adapters provide network acquisition or proprietary API bindings; the core provides portable representation and evidence semantics.

### Source-size envelope (targets, not implementation claims)

| Module | Budget | Responsibility |
| --- | ---: | --- |
| \`codec.py\` | 8,192 B | Exact canonical values, typed representation IDs, bounded packet and JSON primitives |
| \`observe.py\` | 6,144 B | Observation envelopes, digests, clock and source provenance |
| \`ground.py\` | 8,192 B | Versioned typed Ground graph, relations, outcome sets and uncertainty |
| \`protocols.py\` | 8,192 B | Minimal DNS, Git pkt-line and HTTP conditional wire-contract checks |
| \`navigator.py\` | 8,192 B | Evidence-qualified route selection, path composition, cost ceilings |
| \`qualify.py\` | 8,192 B | Capability/admission checks, source pins, freshness and side-effect guards |
| \`verify.py\` | 8,192 B | Independently replayable receipts, proof scopes, witness and outcome checks |
| \`cli.py\` | 4,096 B | Operator requests, inspect, replay, verify and export |
| **Total target** | **59,392 B** | Leaves **6,144 B** for integration and revisions |

Large network adapters, datasets, visual browser tools and tests stay outside that source cap. The source ceiling applies only to the new package; it does not make external computational dependencies disappear.

### Public interface proposal

~~~python
ground = VirtualInternetGround.from_capture("internet-ground-observation.json")
intent = ObservationIntention(
    request="github.repository_document",
    subject="Kier73/VL-64",
    target="docs/RESEARCH_GOALS.md",
    required_freshness="pinned-commit",
    allow_network=True,
    require_evidence=True,
)
plan = Navigator(ground).select(intent, ceilings=CostCeilings(...))
receipt = QualifiedRealizer().execute(plan)
assert IndependentVerifier().verify(receipt).status == "VERIFIED"
~~~

This is a **design sketch**, not a currently callable VL-64 API. Reuse the demonstrated \`InternetGround\` offline replay path as a seed, not as an implicit authority service.

### Operator vocabulary

Use small explicit operators with input/output schemas:
- \`dns.resolve(name, record_type, resolver)\`;
- \`git.refs(repository, prefix, pinned_protocol)\`;
- \`github.object(repository, revision, path)\`;
- \`http.read(url, expected_digest)\`;
- \`http.validate(url, validator, cached_digest)\`;
- \`tls.describe(host, port, trust_scope)\`;
- \`stream.exchange(channel, data, admitted_receiver_operation)\`;
- \`graph.navigate(start_representation, desired_representation)\`;
- \`verify.evidence(receipt, expected_scope)\`.

These operations must be **capability- and scope-checked**. A URL string or a remote advertisement does not grant permission to invoke arbitrary actions. For example, only a declared Git read-only \`ls-refs\` request can pass the demo adapter's POST allowlist.

### Routing model

A qualified network relation is a tuple:

\[
e=(\text{source},\text{operation},\text{target},\text{witness},\text{clock scope},
\text{freshness},\text{authority},\text{cost})
\]

A path is valid only when every edge has compatible representation types, permitted action effects, a satisfied evidence contract, and acceptable freshness. Cost can be a vector:

\[
C=(\text{wall time},\text{bytes moved},\text{local work},
\text{remote calls},\text{verification},\text{recovery risk})
\]

Prefer Pareto-comparable outcomes rather than a universal hidden score. When two routes produce the same required observation, retain both and their witness histories; allow policy to choose based on cost, authority, independent verification or freshness.

One important reusable class is **conditional representation navigation**: a previously observed object and a server-issued validator can yield a verified “unchanged” result with no body transfer. This is a real network protocol operation and can be useful for large cached structures. A stale, unsupported or mismatched validator should not be treated as confirmed identity.

## 5. Research phases for the virtual-only version

### V0 — Reproducible Internet Ground capture (implemented)
Acquire read-only public responses, parse bounded protocols, build an evidence graph, keep a replayable dataset. The initial lab and fixed GitHub Actions runner implement this for the listed operations.

### V1 — Portable capture and exact verifier (partially implemented)
Formalize the v1 dataset/receipt schema, add a second independent implementation of critical packet checks, reject malformed/truncated data and separately identify freshness. Test tamper controls against packets, links, headers, timestamps and digests.

### V2 — Intent-driven network navigation (planned)
Replace the hard-coded three-hop GitHub sequence with typed requests. Find routes among qualified DNS/Git/HTTP relations. Include inability to reach a target as an explicit unresolved outcome; do not invent links or silently follow redirects.

### V3 — Costed reuse and cache-aware navigation (planned)
Compare first acquisition, conditional revalidation, pinned object reuse, repeated resolver queries and fresh content retrieval across at least ten sessions and more than one date. Record preparation costs and uncertainty as well as measured speedup.

### V4 — Multiple observer-hosts (planned)
Replicate read-only observations on a GitHub runner, user-owned machine, and a second provider/host when available. Normalize representation contracts while preserving clock and authority identities. Compare actual independent outcomes, not only repeated API clients at one provider.

### V5 — Verified state realization (planned)
For a safe controlled service, request a semantic state operation rather than merely retrieve content, bind its input/output to a Ground, and verify the result with a separate implementation. Add failure/retry and crash cases. Distinguish computation executed by the service, by the network path, and by the VL-N64 interpreter.

### V6 — Adaptive Ground optimization (planned)
Discover when a richer description can replace repeated remote operations: immutable content addressing, recurrence laws, finite state machines, sufficient observer quotients, algebraic packet/header transitions, or structurally generated results. Promote only if held-out predictions and a defensible end-to-end cost model improve.

## 6. Immediate decisions

- **Keep VL-64 unchanged:** the existing Python source has 63,528 of 65,536 bytes used; a complete network acquisition layer does not belong in its remaining 2,008 bytes.
- **Keep the actual Internet lab in \`experiments/internet_lab\`:** it is executable, reproducible, and can share type and evidence experiments with other projects without polluting the compact runtime.
- **Build VL-N64 only once V1/V2 interfaces settle:** the present results demonstrate a viable surface but not yet a stable, generalized routing contract.
- **Favor path qualification and source-derived computations over speculative packet arithmetic:** test the network's actual transformation and availability rather than assuming its behavior is a general-purpose CPU.
- **Reevaluate after a second independent host and a cost-calibrated intent navigator:** that is the point where separation into a dedicated 64 KiB package becomes useful, not merely aesthetic.

## 7. Reuse without importing entire systems

The experiment has direct conceptual and testable precedent:

- [E-CPU Internet Ground routing](https://github.com/Kier73/E-CPU/blob/f9ed67c495b00f9d26df47e291b40b41070cb609/mailcpu/internet_ground_routing.py): evidence-qualified candidate relations and receiver-installed verifiers.
- [E-CPU path ground probe](https://github.com/Kier73/E-CPU/blob/f9ed67c495b00f9d26df47e291b40b41070cb609/mailcpu/path_ground_probe.py): kernel TCP evidence separate from an asserted in-transit computation.
- [Virtual-Layer-Dynamics Internet paths](https://github.com/Kier73/Virtual-Layer-Dynamics/blob/985ba5c2fbcc40842aed9f45f42587749620b98e/src/vld/substrates/internet_paths.py): bounded HTTP graph and authenticated router-hop research.
- [In-Formation-Space observation lifecycle](https://github.com/Kier73/In-Formation-Space/blob/48d8d3a7b8c8e2adce3fb6067a13b0227adb15da/bridges/observation-path/observation_path.py): distinguish planned, dispatched, admitted, returned, interpreted and applied observations.
- [Git protocol v2](https://git-scm.com/docs/protocol-v2): smart HTTP framing and \`ls-refs\` capability negotiation.
- [Cloudflare DNS over HTTPS](https://developers.cloudflare.com/1.1.1.1/encryption/dns-over-https/make-api-requests/): JSON and DNS wire-message observation requests.

Research code in those repositories is not automatically a verified dependency or a trust authority. Reuse protocols and tests through small explicit adapters.

## 8. Next falsifiable claim

**Can a virtual network Ground satisfy a new, typed observation intention by composing qualified DNS, Git and HTTP relations, with independently checkable output, lower measured transfer or computation work than naive retrieval, and correct refusal when a relevant source or cache relation changes?**

A future version should answer that with live experiments, saved evidence, independent negative controls, and an explicit end-to-end cost report.
