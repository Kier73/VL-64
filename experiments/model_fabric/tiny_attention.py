"""Execute actual local F32 tensor weights through VL-64's compact operators.

This is a deliberately small *single-head causal attention decoder*: token
embeddings, Q/K/V projections, scaled masked softmax, output projection,
residual, RMS norm and vocabulary projection. It is NOT a full transformer
implementation or a pretrained Hugging Face model. No expected outputs are
hardcoded and no tensor is inferred from shape metadata.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from vld import StructuralLinearOperator,rms_norm,domain_digest
from .tensor_ground import pack_f32_safetensors,read_f32_tensors

WEIGHT_NAMES=("embed","lm_head","rms","wk","wo","wq","wv")
VOCAB=8
HIDDEN=4
MAX_CONTEXT=16
MAX_GENERATION=32


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def deterministic_local_neural_weights(seed=73):
    """Host-supplied reproducible demonstration weights; not pretrained."""
    if type(seed) is not int or not 0<=seed<2**32:
        raise ValueError("Invalid deterministic demonstration seed")
    rand=random.Random(seed)
    def matrix(rows,columns,scale):
        return ([rows,columns],
                [round(rand.uniform(-scale,scale),6)
                 for _ in range(rows*columns)])
    weights={
        "embed":matrix(VOCAB,HIDDEN,0.8),
        "wq":matrix(HIDDEN,HIDDEN,0.55),
        "wk":matrix(HIDDEN,HIDDEN,0.55),
        "wv":matrix(HIDDEN,HIDDEN,0.55),
        "wo":matrix(HIDDEN,HIDDEN,0.55),
        "lm_head":matrix(VOCAB,HIDDEN,0.75),
        "rms":([HIDDEN],[round(rand.uniform(0.8,1.2),6)
                           for _ in range(HIDDEN)]),
    }
    return pack_f32_safetensors(weights)


def _operator(weight):
    rows,cols=weight["shape"]
    values=weight["values"]
    return StructuralLinearOperator(
        rows=rows,cols=cols,
        sparse_delta={(r,c):values[r*cols+c]
                      for r in range(rows) for c in range(cols)
                      if values[r*cols+c]!=0})


class TinyCausalAttentionGround:
    def __init__(self,artifact):
        admitted=read_f32_tensors(artifact,WEIGHT_NAMES)
        w=admitted["weights"]
        expected={
            "embed":[VOCAB,HIDDEN],
            "lm_head":[VOCAB,HIDDEN],
            "rms":[HIDDEN],
            "wk":[HIDDEN,HIDDEN],
            "wo":[HIDDEN,HIDDEN],
            "wq":[HIDDEN,HIDDEN],
            "wv":[HIDDEN,HIDDEN],
        }
        if {name:item["shape"] for name,item in w.items()}!=expected:
            raise ValueError("Tiny decoder weights do not match declared topology")
        self.weight_entries=w
        self.file_sha256=admitted["full_file_sha256"]
        self.tensor_ground_digest=admitted["ground"]["ground_digest"]
        self.ops={name:_operator(w[name]) for name in
                  ("wq","wk","wv","wo","lm_head")}
        self.embed=[w["embed"]["values"][i*HIDDEN:(i+1)*HIDDEN]
                    for i in range(VOCAB)]
        self.rms=w["rms"]["values"]
        self.ground={
            "schema":"vl64.tiny-causal-attention-ground.v1",
            "file_sha256":self.file_sha256,
            "tensor_structure_ground_digest":self.tensor_ground_digest,
            "vocab_size":VOCAB,
            "hidden_width":HIDDEN,
            "max_context":MAX_CONTEXT,
            "operator_path":[
                "embedding_lookup","rms_norm",
                "qkv_structural_matvec",
                "scaled_causal_softmax_attention",
                "output_structural_matvec",
                "residual_add",
                "rms_norm",
                "lm_head_structural_matvec",
                "argmax_token"],
            "payload_admission":"full_local_f32_weight_bytes_observed",
            "training_claim":"seeded_untrained_numeric_example",
            "scope":"exact_small_causal_attention_forward_not_full_transformer",
        }
        self.ground["ground_digest"]=domain_digest(
            "vl64.tiny-causal-attention-ground.v1",self.ground)

    def logits(self,tokens):
        if (not isinstance(tokens,(tuple,list)) or
                not 1<=len(tokens)<=MAX_CONTEXT or
                any(type(token) is not int or not 0<=token<VOCAB
                    for token in tokens)):
            raise ValueError("Tokens outside admitted tiny decoder alphabet")
        vectors=[self.embed[token] for token in tokens]
        normalized=[rms_norm(x,self.rms) for x in vectors]
        keys=[self.ops["wk"].matvec(x) for x in normalized]
        values=[self.ops["wv"].matvec(x) for x in normalized]
        query=self.ops["wq"].matvec(normalized[-1])
        scores=[
            sum(a*b for a,b in zip(query,key))/math.sqrt(HIDDEN)
            for key in keys]
        peak=max(scores)
        exp=[math.exp(value-peak) for value in scores]
        probability=[value/sum(exp) for value in exp]
        context=[sum(p*value[d] for p,value in zip(probability,values))
                 for d in range(HIDDEN)]
        output=self.ops["wo"].matvec(context)
        residual=[a+b for a,b in zip(vectors[-1],output)]
        logits=self.ops["lm_head"].matvec(rms_norm(residual,self.rms))
        if any(not math.isfinite(x) for x in logits):
            raise ValueError("Non-finite admitted decoder output")
        return logits,probability

    def next_token(self,tokens):
        logits,attention=self.logits(tokens)
        chosen=max(range(VOCAB),key=lambda idx:(logits[idx],-idx))
        receipt={
            "schema":"vl64.tiny-attention-next-token.v1",
            "status":"ACTUAL_LOCAL_WEIGHT_BACKED_NEURAL_INFERENCE",
            "ground_digest":self.ground["ground_digest"],
            "tensor_source_sha256":self.file_sha256,
            "input_tokens":list(tokens),
            "logits":[round(x,8) for x in logits],
            "causal_attention_probabilities":[round(x,8) for x in attention],
            "selected_next_token":chosen,
            "scope":"seeded_untrained_small_decoder_not_Hugging_Face_model",
        }
        receipt["receipt_digest"]=domain_digest(
            "vl64.tiny-attention-next-token.v1",receipt)
        return receipt

    def generate(self,prompt,steps=8):
        if (not isinstance(prompt,(tuple,list)) or
                not 1<=len(prompt)<=MAX_CONTEXT or
                type(steps) is not int or not 0<=steps<=MAX_GENERATION):
            raise ValueError("Invalid tiny decoder generation request")
        tokens=list(prompt)
        results=[]
        for _ in range(steps):
            next_result=self.next_token(tokens[-MAX_CONTEXT:])
            tokens.append(next_result["selected_next_token"])
            results.append(next_result["receipt_digest"])
        answer={
            "schema":"vl64.tiny-attention-generation.v1",
            "status":"ACTUAL_LOCAL_WEIGHT_BACKED_NEURAL_INFERENCE",
            "ground_digest":self.ground["ground_digest"],
            "input_tokens":list(prompt),
            "generated_tokens":tokens[len(prompt):],
            "step_receipt_digests":results,
            "scope":"untrained_small_model_with_real_f32_weighted_attention",
        }
        answer["receipt_digest"]=domain_digest(
            "vl64.tiny-attention-generation.v1",answer)
        return answer
