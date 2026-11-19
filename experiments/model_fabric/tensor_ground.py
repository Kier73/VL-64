"""Observe Safetensors tensor metadata without acquiring model weight bytes.

The compact core's SafetensorsGlance parses the standard 8-byte little-endian
header length plus JSON header. This lab adds bounded strict shape/offset and
dtype validation, and a typed Ground that does NOT claim to have the weights.
Local F32 tensors can be read only with separately authorized exact bytes.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct

from vld import SafetensorsGlance,domain_digest

HEADER_LIMIT=2**20
MAX_TENSORS=4096
MAX_ELEMENTS=2**34
DTYPE_BYTES={
    "BOOL":1,"U8":1,"I8":1,"F8_E4M3":1,"F8_E5M2":1,
    "F16":2,"BF16":2,"U16":2,"I16":2,
    "F32":4,"U32":4,"I32":4,
    "F64":8,"I64":8,"U64":8,
}


def _distinct_pairs(pairs):
    result={}
    for k,v in pairs:
        if k in result:
            raise ValueError("Duplicate tensor metadata key")
        result[k]=v
    return result


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def observe_safetensors_header(prefix,*,source="unattributed",file_size=None):
    if (not isinstance(prefix,bytes) or not 8<=len(prefix)<=HEADER_LIMIT+8 or
            not isinstance(source,str) or not source or len(source)>300 or
            (file_size is not None and (
                type(file_size) is not int or file_size<8))):
        raise ValueError("Unqualified bounded model header")
    n=struct.unpack("<Q",prefix[:8])[0]
    if n<2 or n>HEADER_LIMIT or len(prefix)!=n+8:
        raise ValueError("Incomplete/oversized Safetensors metadata header")
    if prefix[8:9]!=b"{":
        raise ValueError("Invalid Safetensors metadata prefix")
    # Reject duplicate JSON keys in the actual bytes before core interpretation.
    strict=json.loads(prefix[8:].decode("utf-8"),object_pairs_hook=_distinct_pairs)
    observed_len,header=SafetensorsGlance.parse_header_bytes(prefix)
    if observed_len!=n or header!=strict or not isinstance(strict,dict):
        raise ValueError("Core Safetensors interpretation diverges")
    meta=strict.pop("__metadata__",{})
    if not isinstance(meta,dict) or not all(
            isinstance(k,str) and isinstance(v,str) for k,v in meta.items()):
        raise ValueError("Unsupported Safetensors metadata map")
    if len(strict)>MAX_TENSORS:
        raise ValueError("Oversized tensor inventory")
    tensors=[]
    for name,desc in sorted(strict.items()):
        if (not isinstance(name,str) or not name or len(name)>256 or
                not isinstance(desc,dict) or
                set(desc)!={"dtype","shape","data_offsets"} or
                desc["dtype"] not in DTYPE_BYTES or
                not isinstance(desc["shape"],list) or
                len(desc["shape"])>8 or
                any(type(x) is not int or not 0<=x<=MAX_ELEMENTS
                    for x in desc["shape"]) or
                not isinstance(desc["data_offsets"],list) or
                len(desc["data_offsets"])!=2 or
                any(type(x) is not int or x<0 for x in desc["data_offsets"])):
            raise ValueError("Malformed typed tensor definition")
        elements=math.prod(desc["shape"])
        if elements>MAX_ELEMENTS:
            raise ValueError("Tensor element count exceeds safe planning limit")
        start,end=desc["data_offsets"]
        expected=elements*DTYPE_BYTES[desc["dtype"]]
        if end<start or end-start!=expected:
            raise ValueError("Tensor shape does not match declared weight byte extent")
        if file_size is not None and end+8+n>file_size:
            raise ValueError("Tensor extends beyond admitted file length")
        tensors.append({
            "name":name,"dtype":desc["dtype"],
            "shape":desc["shape"],
            "data_offsets":[start,end],
            "elements":elements,
            "payload_bytes":expected,
        })
    ordered=sorted(tensors,key=lambda item:item["data_offsets"])
    previous=0
    for item in ordered:
        first,last=item["data_offsets"]
        if first<previous:
            raise ValueError("Overlapping tensor payload intervals")
        previous=last
    ground={
        "schema":"vl64.remote-tensor-structure-ground.v1",
        "source":source,
        "source_file_bytes":file_size,
        "header_sha256":_sha(prefix),
        "header_bytes":len(prefix),
        "tensor_count":len(tensors),
        "tensors":tensors,
        "declared_payload_bytes":previous,
        "weight_payload_bytes_observed":0,
        "tensor_values_known":False,
        "metadata":meta,
        "evidence_scope":"exact_header_only_tensor_payload_unread",
        "inferential_scope":"tensor_shapes_dtypes_offsets_not_executable_weighted_model",
    }
    ground["ground_digest"]=domain_digest(
        "vl64.remote-tensor-structure-ground.v1",ground)
    return ground


def read_f32_tensors(whole_file,names):
    """Admit actual complete local tensor bytes before numeric execution."""
    if (not isinstance(whole_file,bytes) or not 8<=len(whole_file)<=2**20 or
            not isinstance(names,(tuple,list)) or not 1<=len(names)<=24 or
            len(names)!=len(set(names))):
        raise ValueError("Unqualified local F32 weight artifact")
    header_size=struct.unpack("<Q",whole_file[:8])[0]
    if header_size>HEADER_LIMIT or header_size+8>len(whole_file):
        raise ValueError("Incomplete local weight artifact")
    prefix=whole_file[:8+header_size]
    structure=observe_safetensors_header(
        prefix,source="complete-local-byte-artifact",file_size=len(whole_file))
    all_tensors={tensor["name"]:tensor for tensor in structure["tensors"]}
    if any(name not in all_tensors or all_tensors[name]["dtype"]!="F32"
           for name in names):
        raise ValueError("Requested tensor is unknown or not F32")
    observed={}
    offset=8+header_size
    for name in names:
        tensor=all_tensors[name]
        begin,end=tensor["data_offsets"]
        exact=whole_file[offset+begin:offset+end]
        if len(exact)!=tensor["payload_bytes"]:
            raise ValueError("Incomplete F32 payload bytes")
        count=tensor["elements"]
        values=struct.unpack("<"+"f"*count,exact)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Non-finite model weights are inadmissible")
        observed[name]={
            "shape":tensor["shape"],
            "values":list(values),
            "bytes_sha256":_sha(exact),
        }
    return {
        "weights":observed,
        "ground":structure,
        "full_file_sha256":_sha(whole_file),
        "acquired_weight_bytes":sum(
            all_tensors[name]["payload_bytes"] for name in names),
    }


def pack_f32_safetensors(tensors):
    """Produce real, independently parsable tiny F32 model weight bytes."""
    if not isinstance(tensors,dict) or not 1<=len(tensors)<=24:
        raise ValueError("Expected bounded named tensor input")
    records={}
    payload=bytearray()
    for name,(shape,values) in sorted(tensors.items()):
        if (not isinstance(name,str) or not name or len(name)>128 or
                not isinstance(shape,(tuple,list)) or
                any(type(dim) is not int or not 1<=dim<=1024 for dim in shape) or
                math.prod(shape)!=len(values) or
                len(values)>8192 or
                any(not isinstance(v,(float,int)) or not math.isfinite(v)
                    for v in values)):
            raise ValueError("Invalid local F32 tensor structure")
        start=len(payload)
        payload+=struct.pack("<"+"f"*len(values),*values)
        records[name]={
            "dtype":"F32","shape":list(shape),
            "data_offsets":[start,len(payload)],
        }
    header=json.dumps(records,sort_keys=True,separators=(",",":")).encode()
    if len(header)>HEADER_LIMIT:
        raise ValueError("Oversized synthetic header")
    binary=struct.pack("<Q",len(header))+header+payload
    if len(binary)>2**20:
        raise ValueError("Oversized local weight payload")
    return bytes(binary)
