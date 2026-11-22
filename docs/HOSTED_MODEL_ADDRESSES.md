# How a VL-64 worker addresses a hosted model

An address lets the worker ask for one trained quantity while the rest of the checkpoint stays on its host. The worker needs an immutable source, a rule mapping tensor coordinates to bytes, and an operation that gives the decoded values meaning. Below is one such address, followed from the public file listing to the actual response bytes.

## Worked address: one original Qwen embedding

| Part of the address | Observed value |
| --- | --- |
| Repository and revision | Qwen/Qwen3.8-27B at 37561816e66b5b9153b1b64c10968c18c1e6f4a0 |
| Tensor | model.language_model.embed_tokens.weight |
| Physical shard | model-00003-of-00018.safetensors |
| Tensor shape and encoding | 248,320 rows × 5,120 columns, BF16, two bytes per element |
| Shard header | 8-byte length field followed by 144 bytes of JSON |
| Tensor extent within the shard payload | $[0,2542796800)$ |
| Requested coordinate | Entire row for token ID 63204 |

The pinned index identifies the shard. Reading its first eight bytes gives the JSON length as a little-endian unsigned integer. The tensor header records offsets relative to the start of the payload. For a contiguous matrix with $d$ columns, $b$ bytes per element, payload offset $o$, and JSON header length $n$, row $r$ occupies $[8+n+o+rdb,\ 8+n+o+(r+1)db)$ in the physical file. [Qwen shard index](https://huggingface.co/Qwen/Qwen3.8-27B/blob/37561816e66b5b9153b1b64c10968c18c1e6f4a0/model.safetensors.index.json) · [Safetensors format](https://github.com/huggingface/safetensors#format)

Here the start is $8+144+0+63204\times5120\times2=647209112$. HTTP range endpoints are inclusive, so the final requested byte is **647219351**. The live observation returned HTTP 206, 10,240 payload bytes and a total shard size of 2,542,796,952 bytes. The SHA-256 of the exact row bytes was **3b6eafbbea137d8ad9abd30e3e376bce9385fb23707ae82e1051920603801047**. The digest of the length field and JSON header together was **b8f9a3cc0294c614390b5722494dcea8082f83c5cbf56da05b8c8f37c4bbe1fb**.

To repeat the row read, run the following in a scratch directory. On Windows PowerShell use the executable name curl.exe. Check for status 206, the exact Content-Range, and the digest above.

~~~sh
curl --fail --silent --show-error --location --max-filesize 10240 --range 647209112-647219351 --dump-header qwen-row.headers --output qwen-row.bf16 https://huggingface.co/Qwen/Qwen3.8-27B/resolve/37561816e66b5b9153b1b64c10968c18c1e6f4a0/model-00003-of-00018.safetensors
python -c "from pathlib import Path; import hashlib; b=Path('qwen-row.bf16').read_bytes(); print(len(b),hashlib.sha256(b).hexdigest())"
~~~

## One logical address space across many shards

The standalone seed reads the config, optional shard index and shard headers. It checks that names are unique, the index agrees with the physical owner of each tensor, and each shard’s declared intervals cover its payload. For shard $j$ in sorted filename order, let $p_j$ be its payload length, $h_j$ its header length including the length field, and $H=\sum_j h_j$. A payload coordinate $x$ in shard $j$ has virtual address $H+\sum_{k<j}p_k+x$ and physical file offset $h_j+x$. The worker translates each request back to one tensor in one shard, respecting the seed’s 65,536-byte weight-request limit. Quantized or packed tensors require their format’s block layout and scales to decode a coordinate. [Address translation and range reader](../experiments/model_fabric/local_seed.py)

For a supported architecture, the typed graph gives those addresses an execution order. The intention “compute vocabulary logits” traverses a different set of dependencies from “read this embedding”. During inspection of the hosted SmolLM-1.7B checkpoint, the seed mapped 218 tensors across two shards using 43,637 metadata bytes and zero weight bytes. In a separate run on SmolLM-135M, it fetched trained values and generated two tokens that agreed with the reference model. The Qwen read above demonstrates one original weight address. Complete Qwen and DeepSeek generation through this profile remains a research objective. [Hosted inspection receipt](../experiments/model_fabric/datasets/2026-11-22-hosted-smollm-1.7b-model-seed-inspect.json) · [Standalone execution and measured costs](STANDALONE_MODEL_SEED.md)

## What another worker can reuse

A worker returning a byte observation should identify the repository and revision, tensor, shard, physical interval, encoding, byte count and content digest. Another worker can repeat the read and compare the bytes. For a computed result it must additionally bind the operator semantics, input state and numerical convention. A routing result also depends on the token’s incoming state. A checkpoint revision alone cannot identify that state.

This distinction matters to the 64 KiB goal. A small program can locate data in a much larger checkpoint. Reducing the arithmetic requires a relation that settles the requested observation early, such as a bound proving which experts can still enter the selected set. The companion VL-I note develops that test and the costs a fleet would need to measure. [Research concept and proposed benchmark](https://github.com/Kier73/VL-I/blob/main/docs/ADDRESSABLE_HOSTED_MODELS.md)
