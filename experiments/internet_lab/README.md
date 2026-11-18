# VL-64 Internet Ground Laboratory

This is a **separate experimental tool**, not part of the 65,536-byte VL-64 source package. It uses the installed VL-64 hashing and Git packet-line primitives, and records actual read-only Internet observations from the machine on which it is run.

## What executes

| Source | Operation performed | Interpretation retained |
| --- | --- | --- |
| OS resolver | Resolve example.com, github.com, cloudflare.com | Address set digest, family and timing; no fabricated TTL |
| Cloudflare DNS-over-HTTPS | Query A example.com as JSON | DNS status, answer-set digest, reported TTL |
| Cloudflare DNS-over-HTTPS | Construct a 29-byte DNS wire query and decode the response | Exact retained DNS response packet, field parse and replay |
| Direct TLS to example.com | Establish a certificate-validated TLS connection | Version, negotiated cipher, certificate digest and timing |
| GitHub smart HTTP | Request Git upload-pack protocol v2 capabilities | Retained advertisement packet and decoded capabilities |
| GitHub upload-pack | Issue an actual bounded **read-only ls-refs** command using Git v2 pkt-lines | Retained response packet and reported HEAD |
| GitHub REST | Read the default branch and its commit identity | Remote JSON observations, response digests |
| GitHub raw content | Read pinned README and its selected research-goals link | Checked document content hashes and three-hop Ground route |
| HTTP conditional request | If-None-Match against the pinned README | 304 response: reuse previously observed contents without transfer |
| Cloudflare and example.com | Bounded HTTPS GET | HTTP status, content digest, caching metadata |
| TCP on 127.0.0.1 | Send a framed 1,024-byte request to a separate endpoint | Endpoint SHA-256 operation, checked against local reference |

The Linux GitHub Actions experiment on **November 18, 2026** produced 16 successful network observations, 14 Ground nodes, 9 observed relations, zero unresolved protocol interpretations, and a checked three-edge GitHub repository-to-document route. The DNS wire packet, Git v2 advertisement, and Git v2 ls-refs response were retained for offline re-verification.

- [Live execution](https://github.com/Kier73/VL-64/actions/runs/44371133726)
- [Saved first dataset](datasets/2026-11-18-github-run-44371133726.json)
- [Network-native runtime design](../../docs/VIRTUAL_INTERNET_RUNTIME_PLAN.md)

These are finite observations made from a GitHub runner, **not** universal laws about Internet behavior, physical path instrumentation, or independent attestation of external providers.

## Run from a clone

~~~bash
python -m pip install -e ".[dev]"
python -m pytest -q tests/
python -m experiments.internet_lab.virtual_net --out internet-ground-dataset.json
python -m experiments.internet_lab.virtual_net --verify-file internet-ground-dataset.json
~~~

The first command runs existing package and experiment contract tests. The second invokes live public resources and a local TCP server. The last replays retained protocol packets and validates the dataset and relation-graph fingerprints **without accessing the Internet**.

The automated workflow records the live dataset as a downloadable Actions artifact; the first complete successful observation is also committed as a research record. Later runs may differ because the Internet changes. Keep separate snapshots instead of overwriting prior observations.

## Example: navigation by evidence

The lab requests repository metadata, determines the default branch, asks GitHub for the branch's commit, then retrieves the README **pinned to that commit**. It finds the selected research-goals link, fetches the target pinned to the same commit and constructs a three-edge representation graph:

~~~text
repo:Kier73/VL-64
  --resolve_default_branch_commit--> git-commit:<sha>
  --read_pinned_readme-----------> readme:<content-sha>
  --navigate_markdown_link-------> research-goals:<content-sha>
~~~

Each edge has a digest of its remote observation. A cached route through the graph is **not a fresh request**, and a remote response digest by itself is not independent proof that the provider's claims are true.

The lab also asks Git upload-pack for HEAD via a different protocol. In the first verified execution, the Git wire response and GitHub REST resolution agreed on the same commit. Cloudflare's DNS JSON and DNS wire-format endpoints independently produced the same observed A-address set during that run (they are still two interfaces to the same provider, not independent authorities).

## Distinguish the sources of computation

- **Service-side computation:** DNS resolver interprets a question, Git upload-pack executes ls-refs, GitHub's content API selects a revision, and HTTP handles conditional validators.
- **Path/protocol operation:** TLS negotiation, packet framing, DNS serialization, caching and network transport create and preserve observable relationships. The experiment does not assume these represent arithmetic carried out in transmission media.
- **VL-64 computation:** constructs and navigates a typed graph of observed relations, checks packet interpretations, hashes witnesses, verifies output identities and can reuse cached descriptions.
- **Local endpoint computation:** the loopback worker hashes received payload bytes. The kernel's TCP transport is not credited with computing that hash.

## Evidence and safety

The lab uses fixed public hosts and a strict HTTPS allowlist. It refuses arbitrary addresses, redirects, credentials in URLs, unsafe ports and general POST requests. Its **only POST** is a bounded Git upload-pack v2 ls-refs request (a read-only Git command). It does not scan ports, send packets to private hosts, inspect user traffic, fetch secrets or mutate remote repositories.

Datasets store public endpoint identifiers, results, timings, response content digests, bounded headers and three short protocol packets. They do not store the GitHub runner's public IP, full website bodies or private content. Future capture extensions should use the same disclosure discipline.

Offline verification checks the report content digest, graph identity and edge format, retained DNS/Git packet hashes and decoders, and the known navigation path. This protects against accidental or unsophisticated alterations; the report digest is not a signature, and it cannot establish the honesty of a compromised capture process. It does not independently re-fetch large HTTP bodies or verify an external provider's truthfulness.

## Next investigations

- Compare repeated sessions chronologically to distinguish immutable repository objects from volatile DNS answers, cache validators and provider capabilities.
- Add **typed operation and outcome contracts**, logical lifetimes and freshness checks; do not turn an observed Git ref or DNS RRset into a permanent universal mapping.
- Implement a read-only **cost-aware navigator** that chooses between current retrieval, pinned content reuse, and a qualified conditional request.
- Separate the true costs of observation, state description, remote service execution, transfer, and independent verification. Test the benefit of cache reuse over several requests, including refresh and mutation cases.
- Test whether a *second provider* or independent host can produce equivalent observable results with independent receipts; multiple APIs at the same provider are not independent authority.
- Develop a standalone network-native 64 KiB core if the experiment exposes reusable operations too large or semantically different for the current VL-64 package.
