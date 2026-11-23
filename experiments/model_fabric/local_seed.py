"""Evaluate token IDs on a typed DAG. Shard coordinate x maps to x+Σ_{k<j}p_k."""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
from urllib.error import HTTPError

from vld import domain_digest
from .execution_dag import MAX_RANGE, compile_decoder_dag, compile_pair_readout, execute_tokens
from .hf_glance import _read, _url, range_bytes
from .page_files import PageFiles
from .tensor_ground import HEADER_LIMIT, _distinct_pairs, observe_safetensors_header

INDEX = "model.safetensors.index.json"
INDEX_LIMIT = 16 * 1024 * 1024
SHARD_NAME = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*\.safetensors")


class LocalFiles:
    """A byte source S(j,a,n)=file_j[a:a+n], with at most one open shard."""

    def __init__(self, directory):
        """Initialize the observed file-identity map I=∅."""
        self.directory = Path(directory)
        self.model = self.directory.resolve().name
        self.revision = None
        self.authority = "local-snapshot"
        self.handle = None
        self.current = None
        self.identities = {}

    def close(self):
        """Set the set of retained file descriptors to ∅."""
        if self.handle is not None:
            self.handle.close()

    def metadata(self, name, limit):
        """Admit M only for |M|≤limit."""
        with (self.directory / name).open("rb") as stream:
            data = stream.read(limit + 1)
        if len(data) > limit:
            raise ValueError("Metadata exceeds byte budget")
        return data

    def read_range(self, name, start, length):
        """Read [start,start+length) under equal (device,inode,size,mtime,ctime) witnesses."""
        if self.current != name:
            self.close()
            self.handle = (self.directory / name).open("rb")
            self.current = name
        # Detect file replacement between shard visits and in-place edits during a read.
        for point in ("before", "after"):
            stat = os.fstat(self.handle.fileno())
            identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            if self.identities.setdefault(name, identity) != identity:
                raise ValueError("Local shard changed during observation")
            if point == "before":
                self.handle.seek(start)
                data = self.handle.read(length)
        if len(data) != length:
            raise ValueError("Incomplete local shard range")
        return data, stat.st_size


class HubFiles:
    """S(j,a,n)=HTTP 206 at the immutable (repository,revision,j) coordinate."""

    def __init__(self, model, revision):
        """Admit revision ∈ {0,…,f}⁴⁰ through the existing Hub URL contract."""
        _url(model, revision, "config.json")
        self.model, self.revision = model, revision
        self.authority = "pinned-huggingface-https-range"

    def metadata(self, name, limit):
        """Admit |M|≤limit; a missing shard index alone permits the single-file layout."""
        try:
            return _read(_url(self.model, self.revision, name), limit=limit)[0]
        except HTTPError as exc:
            if name == INDEX and exc.code == 404:
                raise FileNotFoundError(name) from exc
            raise

    def read_range(self, name, start, length):
        """Require Content-Range = [start,start+length−1]/N for N≥start+length."""
        return range_bytes(_url(self.model, self.revision, name), start, length)

    def close(self):
        """Every ranged request closes its response, hence retained connections = 0."""


def metadata_object(data):
    """Decode an object M with exactly one value per key (k↦M[k] is a function)."""
    obj = json.loads(data.decode("utf-8"), object_pairs_hook=_distinct_pairs)
    if not isinstance(obj, dict):
        raise ValueError("Expected JSON object")
    return obj


class Snapshot:
    """A versioned header Ground and the bijection between virtual and shard byte ranges."""

    def __init__(self, files):
        """Lift header_j and map each [lo,hi) to [base_j+lo,base_j+hi)."""
        self.files = files
        config_raw = files.metadata("config.json", HEADER_LIMIT)
        self.config = metadata_object(config_raw)
        self.metadata_bytes = len(config_raw)
        try:
            index_raw = files.metadata(INDEX, INDEX_LIMIT)
        except FileNotFoundError:
            index_raw = None
        weight_map = None
        if index_raw is not None:
            self.metadata_bytes += len(index_raw)
            weight_map = metadata_object(index_raw).get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map or not all(
                    isinstance(s, str) and SHARD_NAME.fullmatch(s) for s in weight_map.values()):
                raise ValueError("Invalid snapshot shard index")
        shards = sorted(set(weight_map.values())) if weight_map is not None else ["model.safetensors"]
        self.locations = {}
        self.headers = {}
        tensors = []
        payload_base = header_bytes = 0
        for name in shards:
            lead, size = files.read_range(name, 0, 8)
            n, = struct.unpack("<Q", lead)
            if not 2 <= n <= HEADER_LIMIT or 8 + n > size:
                raise ValueError("Invalid Safetensors header length")
            body, observed_size = files.read_range(name, 8, n)
            if observed_size != size:
                raise ValueError("Shard size changed while reading header")
            head = observe_safetensors_header(lead + body, source=name, file_size=size)
            prefix = head["header_bytes"]
            cursor = 0
            for tensor in sorted(head["tensors"], key=lambda t: t["data_offsets"]):
                key = tensor["name"]
                lo, hi = tensor["data_offsets"]
                if lo != cursor or key in self.locations:
                    raise ValueError("Gapped payload or duplicate tensor across shards")
                self.locations[key] = (name, prefix, payload_base, lo, hi, size)
                tensors.append(dict(tensor, data_offsets=[payload_base + lo, payload_base + hi]))
                cursor = hi
            if cursor + prefix != size:
                raise ValueError("Tensor inventory differs from shard size")
            self.headers[name] = head["header_sha256"]
            header_bytes += prefix
            payload_base += cursor
        if weight_map is not None and weight_map != {k: loc[0] for k, loc in self.locations.items()}:
            raise ValueError("Shard index differs from headers")
        self.metadata_bytes += header_bytes
        self.header_bytes = header_bytes
        header = {"schema": "vl64.remote-tensor-structure-ground.v1", "tensors": tensors,
                  "tensor_count": len(tensors), "header_bytes": header_bytes,
                  "source_file_bytes": header_bytes + payload_base, "shard_header_sha256": self.headers,
                  "declared_payload_bytes": payload_base, "address_space": "all_headers_then_shard_payloads"}
        header["ground_digest"] = domain_digest(header["schema"], header)
        self.ground = {"schema": "vl64.remote-model-topology-ground.v1", "model": files.model,
                       "pinned_revision": files.revision or "local-structure-" + header["ground_digest"],
                       "config_sha256": hashlib.sha256(config_raw).hexdigest(),
                       "index_sha256": hashlib.sha256(index_raw).hexdigest() if index_raw else None,
                       "architecture_fields": self.config, "tensor_header_status": "OBSERVED_COMPLETE_TENSOR_HEADER",
                       "tensors": header}
        self.ground["ground_digest"] = domain_digest(self.ground["schema"], self.ground)

    def read(self, request):
        """Translate virtual [A,B) to [prefix+A−H−base,prefix+B−H−base) and witness H(bytes)."""
        shard, prefix, base, lo, hi, size = self.locations[request["tensor"]]
        start = request["byte_begin"] - self.header_bytes - base
        length = request["byte_end_exclusive"] - request["byte_begin"]
        if not (0 < length <= MAX_RANGE and lo <= start and start + length <= hi):
            raise ValueError("Read outside admitted tensor")
        data, total = self.files.read_range(shard, prefix + start, length)
        if total != size:
            raise ValueError("Shard size changed during inference")
        return dict(request, payload=data, sha256=hashlib.sha256(data).hexdigest(),
                    status="QUALIFIED_EXACT_RANGE", observer_host=self.files.authority)

    def plan(self):
        """Admit standard Llama operators only where their implemented equations match the config."""
        c = self.config
        if (c.get("architectures") not in (None, ["LlamaForCausalLM"]) or
                any(c.get(k) is not None for k in ("rope_scaling", "rope_parameters", "quantization_config",
                                                   "sliding_window", "layer_types", "logit_scale",
                                                   "final_logit_softcapping", "attn_logit_softcapping")) or
                c.get("partial_rotary_factor", 1) != 1 or c.get("pretraining_tp", 1) != 1):
            raise ValueError("Unimplemented decoder configuration")
        plan = compile_decoder_dag(self.ground)
        if c.get("head_dim", plan["architecture"]["head_dim"]) != plan["architecture"]["head_dim"]:
            raise ValueError("Unimplemented attention head dimension")
        if set(plan["tensors"]) != set(self.locations):
            raise ValueError("Snapshot has tensors outside the admitted decoder")
        return plan

    def describe(self):
        """Report |tensors|, Σ payload lengths and Σ metadata lengths before W is read."""
        header = self.ground["tensors"]
        return {"model": self.files.model, "source_revision": self.files.revision,
                "model_type": self.config.get("model_type"), "structure_digest": self.ground["ground_digest"],
                "shard_header_sha256": self.headers, "tensor_count": header["tensor_count"],
                "declared_payload_bytes": header["declared_payload_bytes"],
                "metadata_bytes_observed": self.metadata_bytes, "weight_bytes_at_inspection": 0}


def generate(snapshot, plan, args):
    """For t=1,…,steps, append argmax_v ℓ_v(prefix) or the sign-selected member of a fixed pair."""
    a = plan["architecture"]
    if (args.steps < 1 or len(args.tokens) + args.steps - 1 > a["context"] or
            any(t < 0 or t >= a["vocab"] for t in args.tokens)):
        raise ValueError("Token IDs or generation length outside the model context")
    if args.pair and args.include_logits:
        raise ValueError("The pair readout yields a sign; full logits require full-vocabulary execution")
    hosts = {snapshot.files.authority}
    readout = compile_pair_readout(plan, snapshot.read, approved_hosts=hosts, pair=args.pair) if args.pair else None
    report = dict(snapshot.describe(), dag_digest=plan["dag_digest"], input_ids=args.tokens,
                  generated_ids=[], steps=[], prefix_policy="recompute supplied and generated prefix for each step",
                  numeric_contract="VL-64 binary64 arithmetic")
    if args.include_logits:
        report["logits"] = []
    if readout:
        report["pair_compile"] = {k: readout[k] for k in
                                  ("ordered_pair", "readout_digest", "cold_weight_bytes_acquired", "cold_exact_range_count")}
    for _ in range(args.steps):
        prefix = args.tokens + report["generated_ids"]
        result = execute_tokens(plan, prefix, snapshot.read, approved_hosts=hosts, pair_readout=readout)
        if readout:
            token = result["selected_token"]
        else:
            logits = result.pop("logits")
            if not all(math.isfinite(x) for x in logits):
                raise ValueError("Nonfinite vocabulary score")
            if args.include_logits:
                report["logits"].append(logits)
            token = result["selected_next_token"]
        report["steps"].append(result)
        if token is None:
            break
        report["generated_ids"].append(token)
    return report


def cli_main():
    """Realize either D=observe(headers) or token_{t+1}=decode(D,W,token_{≤t})."""
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "run"):
        sub = subs.add_parser(name)
        files = sub.add_mutually_exclusive_group(required=True)
        files.add_argument("--snapshot", type=Path, help="Local config and Safetensors directory")
        files.add_argument("--hub", help="Public Hugging Face repository at --revision")
        files.add_argument("--page", help="Public model card or root tree URL")
        files.add_argument("--search", help="Find an exact model ID on the Hub search webpage")
        sub.add_argument("--revision", help="Immutable 40-character Hub commit")
        if name == "run":
            sub.add_argument("--stage-bytes", type=int, default=0, help="Allow a verified temporary shard cache")
            sub.add_argument("--tokens", type=int, nargs="+", required=True)
            sub.add_argument("--steps", type=int, default=1)
            sub.add_argument("--pair", nargs=2, type=int)
            sub.add_argument("--include-logits", action="store_true")
    args = parser.parse_args()
    if bool(args.hub) != bool(args.revision):
        parser.error("--hub and --revision must be supplied together")
    try:
        if args.page or args.search:
            source = PageFiles(args.page or args.search, bool(args.search))
        elif args.hub:
            source = HubFiles(args.hub, args.revision)
        else:
            source = LocalFiles(args.snapshot)
        with closing(source) as files:
            snapshot = Snapshot(files)
            report = snapshot.describe()
            if args.command == "inspect":
                try:
                    plan = snapshot.plan()
                    report.update(executable=True, architecture=plan["architecture"], dag_digest=plan["dag_digest"])
                except (KeyError, TypeError, ValueError) as reason:
                    report.update(executable=False, reason=str(reason))
            else:
                plan = snapshot.plan()
                if args.stage_bytes:
                    if not isinstance(files, PageFiles):
                        raise ValueError("Staging requires a page source")
                    files.stage(snapshot.headers, args.stage_bytes)
                report = generate(snapshot, plan, args)
            if isinstance(files, PageFiles):
                report.update(pages=files.visits, shard_http_bytes=files.transferred,
                              staged_sha256={n: files.entries[n][1]["lfs"]["oid"] for n in files.staged})
            print(json.dumps(report, allow_nan=False, separators=(",", ":")))
    except (ValueError, KeyError, OSError, TypeError) as error:
        sys.exit("VL64 model: " + str(error))
