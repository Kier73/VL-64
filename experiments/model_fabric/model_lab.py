"""Use the system: train a real byte LM, run attention tensor inference,
compare heldout data, and export executable ObserverIntention requests.

The trained byte model predicts natural text bytes; the tiny neural model
executes real F32 weights but was not trained for language quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from vld import ObserverIntention
from .byte_lm import ByteLanguageGround,fit_byte_lm
from .tiny_attention import (
    TinyCausalAttentionGround,deterministic_local_neural_weights,
)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def experiment():
    corpus=Path("README.md").read_bytes()
    if len(corpus)<512:
        raise ValueError("Need actual README corpus of at least 512 bytes")
    boundary=int(len(corpus)*0.8)
    training,heldout=corpus[:boundary],corpus[boundary:]
    start=time.perf_counter_ns()
    ground=fit_byte_lm(training)
    model=ByteLanguageGround(ground)
    fit_elapsed=(time.perf_counter_ns()-start)/1e6
    evaluation=model.evaluate(heldout)
    prompt=b"Virtual Layer"
    generated=model.generate(prompt,steps=48)
    # An independent count reference checks the rank of the emitted bytes:
    counts={}
    for a,b in zip(training,training[1:]):
        counts[(a,b)]=counts.get((a,b),0)+1
    first=max(range(256),key=lambda next:(
        counts.get((prompt[-1],next),0),-next))
    if not generated["generated_hex"].startswith(bytes([first]).hex()):
        raise RuntimeError("VL-64 operator result disagrees with independent counts")
    if evaluation["cross_entropy_bits_per_byte"]>=8:
        raise RuntimeError("Learned language law fails basic uniform baseline")

    weights=deterministic_local_neural_weights()
    neural=TinyCausalAttentionGround(weights)
    neural_token=neural.next_token([0,1,2,3])
    neural_generation=neural.generate([0,1,2,3],steps=12)
    if neural_token["status"]!="ACTUAL_LOCAL_WEIGHT_BACKED_NEURAL_INFERENCE":
        raise RuntimeError("Neural operator did not execute local tensor payloads")
    report={
        "schema":"vl64.model-fabric-experiment.v1",
        "status":"EXECUTED_WEIGHT_BACKED_LANGUAGE_AND_NEURAL_MODELS",
        "text_corpus":"VL-64 README real committed UTF-8 bytes",
        "corpus_sha256":_sha(corpus),
        "training_bytes":len(training),
        "training_sha256":_sha(training),
        "heldout_bytes":len(heldout),
        "heldout_sha256":_sha(heldout),
        "learned_weight_entries":len(ground["sparse_weights"]),
        "learned_ground_json_bytes":len(json.dumps(
            ground,sort_keys=True,separators=(",",":")).encode()),
        "heldout_byte_cross_entropy":evaluation["cross_entropy_bits_per_byte"],
        "heldout_perplexity":evaluation["perplexity"],
        "uniform_baseline_bits_per_byte":8.0,
        "heldout_top1_accuracy":evaluation["top1_accuracy"],
        "greedy_generated_hex":generated["generated_hex"],
        "greedy_generated_utf8_replace":generated["generated_utf8_replace"],
        "generated_byte_count":generated["generated_byte_count"],
        "trained_language_ground_digest":ground["ground_digest"],
        "training_ms_one_run":round(fit_elapsed,4),
        "neural_tensor_artifact_sha256":_sha(weights),
        "neural_tensor_artifact_bytes":len(weights),
        "neural_ground_digest":neural.ground["ground_digest"],
        "neural_tensor_structure_digest":neural.tensor_ground_digest,
        "neural_operator_logits":neural_token["logits"],
        "neural_one_step_attention_probabilities":
            neural_token["causal_attention_probabilities"],
        "neural_generated_tokens":neural_generation["generated_tokens"],
        "neural_training_claim":"seeded_untrained_numeric_example",
        "weight_payload_bytes_for_neural_execution":sum(
            4*len(item["values"]) for item in neural.weight_entries.values()),
        "hf_weight_payload_bytes_used":0,
        "core_source_modified":False,
        "scope":"actual_trained_bigram_and_seeded_attention_model_not_pretrained_HF_inference",
    }
    return {"trained_ground":ground,"training_eval":evaluation,
            "weights":weights,"neural_ground":neural.ground,
            "trained_generation":generated,
            "neural_generation":neural_generation,
            "report":report}


def _dump(path,value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True,
                               allow_nan=False)+"\n")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",default="artifacts")
    args=parser.parse_args()
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    exp=experiment()
    _dump(out/"trained-byte-language-ground.json",exp["trained_ground"])
    _dump(out/"language-heldout-evaluation.json",exp["training_eval"])
    _dump(out/"language-generation.json",exp["trained_generation"])
    _dump(out/"local-neural-ground.json",exp["neural_ground"])
    _dump(out/"local-neural-generation.json",exp["neural_generation"])
    _dump(out/"model-execution-report.json",exp["report"])
    (out/"local-neural-weights.safetensors").write_bytes(exp["weights"])
    language=ObserverIntention(
        "model.byte.generate",
        "workspace:"+(out/"trained-byte-language-ground.json").as_posix(),
        output="json",
        options={"prompt_hex":b"Virtual Layer".hex(),"steps":48})
    neural=ObserverIntention(
        "model.neural.generate",
        "workspace:"+(out/"local-neural-weights.safetensors").as_posix(),
        output="json",
        options={"input_tokens":[0,1,2,3],"steps":12})
    _dump(out/"language-intention.json",language.to_dict())
    _dump(out/"neural-intention.json",neural.to_dict())
    print("VL64_MODEL_REPORT "+json.dumps(exp["report"],sort_keys=True))
    print("VL64_MODEL_GROUND "+json.dumps(exp["trained_ground"],sort_keys=True,
                                         separators=(",",":")))
    print("VL64_NEURAL_GROUND "+json.dumps(exp["neural_ground"],sort_keys=True,
                                          separators=(",",":")))


if __name__=="__main__":
    main()
