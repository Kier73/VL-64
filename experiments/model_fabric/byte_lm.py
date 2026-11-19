"""A genuinely *trained*, weight-backed byte-level language model in VL-64.

This is a count-trained first-order byte model, not a transformer. Its learned
256x256 transition-weight matrix is admitted as a finite sparse Ground.
Inference runs through the compact core's StructuralLinearOperator.matvec.
No remote model file or neural output is approximated by metadata alone.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter

from vld import StructuralLinearOperator, domain_digest

VOCAB=256
MAX_COUNT=2**31-1
MAX_PAIR_KINDS=VOCAB*VOCAB
MAX_PREDICTIONS=512


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fit_byte_lm(training:bytes):
    if not isinstance(training,bytes) or not 16<=len(training)<=2**20:
        raise ValueError("Training requires 16..1,048,576 real bytes")
    pairs=Counter(zip(training,training[1:]))
    weights=[[before,after,count] for (before,after),count in
             sorted(pairs.items())]
    ground={
        "schema":"vl64.trained-byte-language-ground.v1",
        "model_family":"count_trained_first_order_byte_language_model",
        "tokenizer":{"kind":"identity_byte","vocabulary":VOCAB},
        "training_sha256":digest(training),
        "training_byte_count":len(training),
        "sparse_weights":weights,
        "smoothing":"add_one",
        "inference_operator":"vld.StructuralLinearOperator.matvec",
        "scope":"actual_trained_byte_transition_weights_not_neural_transformer",
    }
    ground["ground_digest"]=domain_digest(
        "vl64.trained-byte-language-ground.v1",ground)
    ByteLanguageGround(ground)
    return ground


class ByteLanguageGround:
    def __init__(self,ground):
        if not isinstance(ground,dict):
            raise ValueError("Expected typed byte-language Ground")
        bare={k:v for k,v in ground.items() if k!="ground_digest"}
        if domain_digest("vl64.trained-byte-language-ground.v1",bare)!=ground.get(
                "ground_digest") or bare.get(
                    "schema")!="vl64.trained-byte-language-ground.v1":
            raise ValueError("Language Ground digest or schema mismatch")
        if (bare.get("tokenizer")!={"kind":"identity_byte","vocabulary":VOCAB} or
                bare.get("model_family")!="count_trained_first_order_byte_language_model" or
                bare.get("smoothing")!="add_one" or
                bare.get("inference_operator")!="vld.StructuralLinearOperator.matvec" or
                type(bare.get("training_byte_count")) is not int or
                not 16<=bare["training_byte_count"]<=2**20 or
                not isinstance(bare.get("training_sha256"),str) or
                len(bare["training_sha256"])!=64):
            raise ValueError("Unknown byte-language model contract")
        weights=bare.get("sparse_weights")
        if not isinstance(weights,list) or not 1<=len(weights)<=MAX_PAIR_KINDS:
            raise ValueError("Invalid sparse model weights")
        earlier=(-1,-1)
        sparse={}
        total=0
        rows=[0]*VOCAB
        for item in weights:
            if (not isinstance(item,list) or len(item)!=3 or
                    any(type(x) is not int for x in item)):
                raise ValueError("Non-integer sparse weight entry")
            before,after,count=item
            if (not 0<=before<VOCAB or not 0<=after<VOCAB or
                    not 1<=count<=MAX_COUNT or (before,after)<=earlier):
                raise ValueError("Duplicate, unsorted or invalid token weight")
            earlier=(before,after)
            sparse[after,before]=count
            rows[before]+=count
            total+=count
        if total!=bare["training_byte_count"]-1:
            raise ValueError("Weight counts do not match observed training tokens")
        self.ground=ground
        self.weights=sparse
        self.totals=rows
        self.operator=StructuralLinearOperator(
            rows=VOCAB,cols=VOCAB,sparse_delta=sparse)

    def probabilities(self,previous):
        if type(previous) is not int or not 0<=previous<VOCAB:
            raise ValueError("Token must be a byte")
        one_hot=[0.0]*VOCAB
        one_hot[previous]=1.0
        counts=self.operator.matvec(one_hot)
        denominator=self.totals[previous]+VOCAB
        return [(count+1.0)/denominator for count in counts]

    def predict(self,previous,*,limit=5):
        if type(limit) is not int or not 1<=limit<=VOCAB:
            raise ValueError("Invalid top-k limit")
        probabilities=self.probabilities(previous)
        top=sorted(range(VOCAB),key=lambda i:(-probabilities[i],i))[:limit]
        return {
            "schema":"vl64.byte-language-prediction.v1",
            "status":"MODELED_WEIGHT_BACKED_INFERENCE",
            "ground_digest":self.ground["ground_digest"],
            "previous_byte":previous,
            "ranked_tokens":[
                {"byte":byte,"probability":probabilities[byte]} for byte in top
            ],
            "operator":"StructuralLinearOperator.matvec",
            "scope":"trained_bigram_conditional_probabilities_not_general_text_reasoning",
        }

    def generate(self,prompt,*,steps=32):
        if (not isinstance(prompt,bytes) or not 1<=len(prompt)<=4096 or
                type(steps) is not int or not 0<=steps<=MAX_PREDICTIONS):
            raise ValueError("Invalid byte-language generation request")
        generated=bytearray()
        decisions=[]
        previous=prompt[-1]
        for i in range(steps):
            ranked=self.predict(previous,limit=2)
            next_byte=ranked["ranked_tokens"][0]["byte"]
            decisions.append({
                "position":i,"prior_byte":previous,"chosen_byte":next_byte,
                "p":ranked["ranked_tokens"][0]["probability"]})
            generated.append(next_byte)
            previous=next_byte
        result={
            "schema":"vl64.byte-language-generation.v1",
            "status":"MODELED_WEIGHT_BACKED_INFERENCE",
            "ground_digest":self.ground["ground_digest"],
            "prompt_sha256":digest(prompt),
            "generated_hex":generated.hex(),
            "generated_utf8_replace":bytes(generated).decode("utf-8","replace"),
            "generated_byte_count":len(generated),
            "decision_trace":decisions,
            "scope":"greedy_count_trained_bigram_not_transformer_or_HF_weight_inference",
        }
        result["receipt_digest"]=domain_digest(
            "vl64.byte-language-generation.v1",result)
        return result

    def evaluate(self,holdout):
        if not isinstance(holdout,bytes) or len(holdout)<2:
            raise ValueError("Need independent heldout byte sequence")
        loss=0.0
        matches=0
        for prev,next_byte in zip(holdout,holdout[1:]):
            probs=self.probabilities(prev)
            loss-=math.log2(probs[next_byte])
            if max(range(VOCAB),key=lambda i:(probs[i],-i))==next_byte:
                matches+=1
        n=len(holdout)-1
        loss/=n
        return {
            "schema":"vl64.byte-language-heldout-evaluation.v1",
            "ground_digest":self.ground["ground_digest"],
            "heldout_sha256":digest(holdout),
            "heldout_transitions":n,
            "cross_entropy_bits_per_byte":round(loss,6),
            "perplexity":round(2**loss,6),
            "uniform_baseline_bits_per_byte":8.0,
            "top1_accuracy":round(matches/n,6),
            "status":"MEASURED_ON_HELDOUT",
            "scope":"heldout_next_byte_probabilities_not_inference_equivalence_proof",
        }
