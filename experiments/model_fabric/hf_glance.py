"""Pinned Hugging Face model observation without downloading tensor payloads.

This fetches a small config and two HTTP Range requests containing only the
Safetensors metadata prefix. Byte ranges MUST be honored; if the provider
cannot supply them, we return an unresolved header observation rather than
silently streaming a large model or inventing tensor information.
"""
from __future__ import annotations

import hashlib
import json
import re
import struct
import urllib.error
import urllib.parse
import urllib.request

from vld import domain_digest
from .tensor_ground import HEADER_LIMIT,observe_safetensors_header

MODEL="HuggingFaceTB/SmolLM2-135M"
REVISION="4e53f736cbb20a9a0f56b4c4bf378d9f306ff915"
MAX_CONFIG=65536
TIMEOUT=18


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _url(model,revision,name):
    """Address (repository, commit, single filename) as an immutable Hub coordinate."""
    if (not isinstance(model,str) or
            not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",model) or
            any(part in (".", "..") for part in model.split("/")) or
            not isinstance(revision,str) or
            not re.fullmatch("[0-9a-f]{40}",revision) or
            not isinstance(name,str) or
            (name not in ("config.json","preprocessor_config.json","model.safetensors.index.json") and
             not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*\.safetensors",name))):
        raise ValueError("Only pinned, qualified Hugging Face paths admitted")
    return "https://huggingface.co/"+model+"/resolve/"+revision+"/"+name


def _read(url,*,start=None,stop=None,limit):
    parsed=urllib.parse.urlparse(url)
    if parsed.scheme!="https" or parsed.hostname!="huggingface.co":
        raise ValueError("Unapproved external model endpoint")
    headers={"User-Agent":"VL64-model-Ground-research/1.0"}
    if start is not None:
        headers["Range"]=f"bytes={start}-{stop}"
        headers["Accept-Encoding"]="identity"
    request=urllib.request.Request(url,headers=headers,method="GET")
    with urllib.request.urlopen(request,timeout=TIMEOUT) as response:
        final=urllib.parse.urlparse(response.geturl())
        if final.scheme!="https" or not final.hostname or not (
                final.hostname=="huggingface.co" or
                final.hostname.endswith(".huggingface.co") or
                final.hostname.endswith(".hf.co")):
            raise ValueError("External model HTTP redirect not qualified")
        status=response.status
        if start is not None and status!=206:
            raise ValueError("Provider did not honor HTTP byte-range constraint")
        content=response.read(limit+1)
        if len(content)>limit:
            raise ValueError("Model metadata read exceeded strict byte limit")
        if start is not None:
            expected=stop-start+1
            content_range=response.headers.get("Content-Range","")
            if (len(content)!=expected or
                    not content_range.startswith(f"bytes {start}-{stop}/")):
                raise ValueError("Incomplete or incorrect ranged header evidence")
        length=response.headers.get("Content-Length")
        return content,{
            "url":url,
            "final_hostname":final.hostname,
            "http_status":status,
            "content_range":response.headers.get("Content-Range"),
            **({"link": response.headers["Link"]} if "Link" in response.headers else {}),
            "content_length":int(length) if length and length.isdigit() else None,
            "sha256":_sha(content),
            "bytes_read":len(content),
            "transport_scope":"public_https_read_only",
        }


def range_bytes(url, start, length):
    """Require Content-Range = [a,a+n−1]/N with N≥a+n."""
    data, receipt = _read(url, start=start, stop=start+length-1, limit=length)
    total = receipt["content_range"].rsplit("/", 1)[1]
    if not total.isdigit() or int(total) < start + length:
        raise ValueError("Invalid shard extent")
    return data, int(total)


def observe_model(*,model=MODEL,revision=REVISION):
    config_url=_url(model,revision,"config.json")
    weights_url=_url(model,revision,"model.safetensors")
    config_raw,config_request=_read(config_url,limit=MAX_CONFIG)
    config=json.loads(config_raw.decode("utf-8"))
    if not isinstance(config,dict):
        raise ValueError("No admitted model configuration object")
    topology={
        key:config[key] for key in (
            "model_type","architectures","hidden_size",
            "num_hidden_layers","num_attention_heads","num_key_value_heads",
            "intermediate_size","vocab_size","max_position_embeddings",
            "hidden_act","tie_word_embeddings","rms_norm_eps",
            "rope_theta","attention_bias","mlp_bias",
        ) if key in config
    }
    ground={
        "schema":"vl64.remote-model-topology-ground.v1",
        "model":model,
        "pinned_revision":revision,
        "config_sha256":_sha(config_raw),
        "config_bytes_observed":len(config_raw),
        "config_request":config_request,
        "architecture_fields":topology,
        "tensor_header_status":"UNRESOLVED",
        "tensors":None,
        "weight_payload_bytes_observed":0,
        "model_execution_status":"NOT_EXECUTED_WEIGHTS_NOT_ADMITTED",
        "scope":"public_pinned_model_structure_not_weight_inference",
    }
    try:
        lead,lead_receipt=_read(
            weights_url,start=0,stop=7,limit=8)
        n=struct.unpack("<Q",lead)[0]
        if not 2<=n<=HEADER_LIMIT:
            raise ValueError("Header exceeds bounded structural observation")
        body,body_receipt=_read(
            weights_url,start=8,stop=7+n,limit=n)
        source_bytes=None
        length=lead_receipt["content_range"] or ""
        if "/" in length and length.split("/")[-1].isdigit():
            source_bytes=int(length.split("/")[-1])
        header=observe_safetensors_header(
            lead+body,source=weights_url,file_size=source_bytes)
        ground["tensor_header_status"]="OBSERVED_COMPLETE_TENSOR_HEADER"
        ground["tensors"]=header
        ground["weight_header_requests"]=[lead_receipt,body_receipt]
        ground["remote_bytes_received"]=(len(config_raw)+len(lead)+len(body))
    except (ValueError,OSError,TimeoutError,urllib.error.URLError) as error:
        ground["tensor_header_status"]="UNRESOLVED"
        ground["tensor_observation_reason"]=type(error).__name__
        ground["remote_bytes_received"]=len(config_raw)
    ground["ground_digest"]=domain_digest(
        "vl64.remote-model-topology-ground.v1",ground)
    return ground


def main():
    report=observe_model()
    print("VL64_HF_MODEL_GROUND "+json.dumps(report,sort_keys=True,
                                          separators=(",",":")))
    if report["tensor_header_status"]!="OBSERVED_COMPLETE_TENSOR_HEADER":
        print("VL64_HF_HEADER_UNRESOLVED "+
              str(report.get("tensor_observation_reason","UNKNOWN")))


if __name__=="__main__":
    main()
