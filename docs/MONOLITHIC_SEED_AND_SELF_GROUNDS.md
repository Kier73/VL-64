# Monolithic VL-64 Seed and Self-Describing Grounds

**Implemented:** a procedurally generated, independently inspectable, executable **monolithic seed** containing the entire compact VL-64 runtime and a verifiable Ground of its software structure.

**Location:** [Seed Fabric](../experiments/seed_fabric/) · [builder](../experiments/seed_fabric/build.py) · [bootstrap](../experiments/seed_fabric/bootstrap.py) · [Linux/Windows verification workflow](../.github/workflows/monolithic-seed.yml)

## The proposition

A system that can construct Virtual Grounds from the external world should also be able to **construct a Ground of its own computational structure**.

The monolithic seed is a compact *carrier*. It can be moved to another Python-capable host, independently checked, executed there, and used to reconstruct exact copies of the original VL-64 source files. Its internal structural description is a reproducible **Ground of Grounds**, not a manually maintained diagram.

This first seed contains:

1. **All seven existing source modules** of VL-64 in their original exact bytes, without rewriting or minifying them.
2. **A self-checking bootstrap** that audits the exact set of admitted archive members and their SHA-256 before importing code from the seed.
3. **An immutable manifest** stating the source-module paths, byte lengths, content digests, expanded source budget, bootstrap digest, and self-Ground digest.
4. **A generated software Ground** containing each module's syntactic classes, top-level functions, relative imports, exported names, and source fingerprints.
5. **Seven individual module Grounds**, each content-identified from the exact module's source fingerprint and syntactic structure, connected through typed source import relations.

The seed is a standard self-contained **Python zip application** with `__main__.py`. Python's standard library supports execution of such ZIP archives without an additional packaging dependency. The seed **does not bundle the Python interpreter**, the OS, the physical machine, the research experiment modules, or any remote credentials. It therefore requires a working Python 3.11+ host. See the [Python zipapp documentation](https://docs.python.org/3/library/zipapp.html).

## What the Ground records

The software Ground is compiled from the **actual source bytes**, not from the README or a predefined architecture map:

~~~text
VL64.pyz                                  monolithic carrier
├── __main__.py                           self-verifying bootstrap
├── vl64/manifest.json                    exact expected source member list
├── vl64/system_ground.json               generated Ground of Grounds
└── vld/
    ├── __init__.py
    ├── __main__.py
    ├── micro_control.py
    ├── micro_kernel.py
    ├── micro_operators.py
    ├── micro_substrates.py
    └── micro_translation.py
~~~

The Ground compiler parses each module's Python AST. A per-module Ground identifies one exact source artifact and its declared syntactic structure. The root Ground contains the module-Ground digests and relations such as:

~~~json
{
  "source_module": "vld.__init__",
  "target_module": "vld.micro_kernel",
  "imported_names": ["canonical_bytes", "domain_digest"],
  "evidence_kind": "parsed_relative_import"
}
~~~

The example is shortened. The real generated relation contains the full imported-name list from source. A user can inspect and replay every actual relation from the generated file.

The root fingerprint is the SHA-256 of a canonical JSON description with an explicit `vl64.software-ground.v1` domain prefix. Each child module Ground also has its own SHA-256 identity.

The Ground describes **the declared syntax and exact source representation**. It does not prove semantic equivalence between all runtimes, that a particular API's effects are safe, or that another substrate has independently realized a Python-compatible interpreter. A content digest establishes integrity of the bytes that were checked; it does not authenticate the organization or author that supplied them.

## Generate the seed procedurally

From a VL-64 checkout:

~~~bash
python -m experiments.seed_fabric.build build \
  --out artifacts/VL64.pyz
~~~

This writes:

- `artifacts/VL64.pyz` — the complete compressed, executable seed.
- `artifacts/VL64.ground.json` — its independently readable, source-derived software Ground.

The builder enforces **both** of the following:

- The original `src/vld/*.py` source totals no more than **65,536 bytes**.
- The resulting monolithic `.pyz` archive also totals no more than **65,536 bytes**.

The archive is generated with a fixed member order, timestamp, Unix permission bits and ZIP deflate settings; no Git checkout path, build date, runner hostname, user account or random identifier is embedded. A `.gitattributes` rule preserves **LF bytes** on Windows and Linux, so the seed and its Ground are derived from the same committed source rather than different line-ending representations.

**Reproducibility is tested** by generating the archive twice and comparing bytes, and independently building on Linux and Windows with Python 3.11 and 3.13. Build byte equivalence is a property of this pinned source and tested toolchains; the source digests and Ground identity remain independently checkable if future compression libraries change their compressed output format.

## Execute and inspect only the seed

A receiving machine needs only the generated `VL64.pyz` and a compatible Python installation:

~~~bash
python -I VL64.pyz seed-verify
python -I VL64.pyz seed-ground
python -I VL64.pyz seed-modules
python -I VL64.pyz seed-relations
python -I VL64.pyz seed-symbols vld.micro_translation

# Existing VL-64 capabilities, from inside the monolithic carrier:
python -I VL64.pyz verify
python -I VL64.pyz navigate \
  --a 5 --b 1 --x0 3 \
  --steps 1000000000000 --modulus 256
~~~

The `-I` flag requests Python's isolated mode. The seed's own bootstrap audits every archive member before importing the bundled runtime. Its inspection calls do not initiate arbitrary network requests, mutate an external host or install new action operators.

An observer can obtain the full system Ground as JSON and navigate its syntactic relations without downloading the originating Git repository.

### Exact source reconstruction

An authorized operator can explicitly recover the seven original source files:

~~~bash
python -m experiments.seed_fabric.build extract \
  --out artifacts/VL64.pyz \
  --extract-to artifacts/reconstructed
~~~

The reconstruction writes only seven validated paths under `vld/`. It refuses an already occupied output directory and never performs a generic `extractall` operation on untrusted file paths. Tests compare each reconstructed source byte-for-byte against the repository.

The `extract` command is a build-host utility, not an implicit effect of simply executing `VL64.pyz`.

## Measured experiment

The first complete Linux/Windows Python 3.11/3.13 matrix reconstructed and ran the exact original source Ground.

| Metric | First confirmed four-run result |
| --- | ---: |
| Original compact core source | **65,093 bytes** |
| Final monolithic seed (including seven module Grounds) | **30,030 bytes** |
| Module source files / module Grounds | **7 / 7** |
| Source-to-seed size ratio | **2.168×** |
| Independent operating systems | Linux and Windows |
| Python versions | 3.11 and 3.13 |
| Exact reconstruction | All seven modules |
| Cross-platform SHA-256 | Equal across all four builds |
| Required platform substrate | Python 3.11+ |

The final modular software Ground fingerprint is `43f36667ce8326ccb7f8927b55fa99c0a2b826f5e7a6d48c9369a4804393b1a1`. All four tested Linux/Windows/Python toolchains produced the **identical 30,030-byte artifact** with SHA-256 `dbe388c0d2fdc0b13e552e0a7dcf2a32233818cd3ec356e3feacd55ad7d6fd8e`. The preceding baseline without module-Ground graph was 28,966 bytes; the additional 1,064 bytes encodes explicit per-module structural descriptions and their reusable relations. These values refer to the version tested in [the cross-platform workflow](https://github.com/Kier73/VL-64/actions/runs/44426705581). A subsequent source edit changes the derived seed and its identity by design.

Each matrix job verifies the seed, invokes its built-in software Ground inspection, executes the original VL-64 verification and navigation CLI, rebuilds the seed twice to check determinism, and reconstructs all original source bytes. No conventional package installation is required on the receiving path when running the seed itself.

The 64 KiB **source** budget remains unchanged by this work: builder, bootstrap template, tests, documentation and workflows exist outside `src/vld/`.

## How this relates to communication as computation

The seed provides an exact, portable **representation of capacity**:

- **Source capability**: exact bytes from which the Python software can be reconstructed.
- **Structural Ground**: descriptions of exported operations and code relations, without a second copy of the full source code in the Ground itself.
- **Navigation interface**: commands to locate modules, introspect Ground relations and run existing virtual-layer operations from the verified carrier.
- **Bootstrap criterion**: a receiving observer-host that supplies an adequate Python runtime can load and operate the carrier.

It makes a specific falsifiable claim: **the source-derived Ground and monolithic carrier reconstruct exactly the same software bytes and have verified executable behavior on the tested hosts**. It does not claim the abstract software representation is already executing on a passive DNS record, a packet in flight or an uninitialized remote process. For those, a qualified host execution environment and new evidence would be required.

### A future seed of Grounds, rather than only source

The next generation should support:

1. **Intention-specific seed projection**: select only the minimal qualified modules and Ground relations needed for an observer request, with dependency closure and exact provenance.
2. **Constructive interpreter discovery**: identify which abilities an external host already exposes, then compile a compatible bootstrap from the necessary Ground transformations, rather than assuming Python is universally installed.
3. **State and effect Grounds**: preserve live durable virtual state separately from immutable code; permit portable checkpoints with explicit schema, authority and recovery claims.
4. **Lazy acquisition**: transport the small structural Ground first and request only missing source chunks when an intended operation truly requires them. Every chunk remains content-addressed.
5. **Cross-host receipts**: launch the same Ground-described operation on independent qualified observer-hosts and compare result identity, cost and evidence; record differences as contradictions.
6. **Substrate-independent virtual interpreter contracts**: specify the byte-to-operation semantic relation independently of the Python code implementation and admit alternative realizers through reproducible tests.

A seed should ultimately be a **small description from which a qualified host can realize the required computational structure**. This build verifies the first concrete, inspectable implementation of that idea—without expanding the compact runtime.
