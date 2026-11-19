"""Two distinct actual model computations: trained bigram and F32 causal attention."""
from __future__ import annotations

import math
import struct
import pytest

from vld import StructuralLinearOperator, SafetensorsGlance
from experiments.model_fabric.byte_lm import (
    fit_byte_lm,ByteLanguageGround,
)
from experiments.model_fabric.tensor_ground import (
    pack_f32_safetensors,read_f32_tensors,observe_safetensors_header,
)
from experiments.model_fabric.tiny_attention import (
    TinyCausalAttentionGround,deterministic_local_neural_weights,
    HIDDEN,VOCAB,
)


def training():
    return (b"ab"*200)+(b"cd"*40)


def test_trained_language_weights_execute_through_compact_linear_operator():
    ground=fit_byte_lm(training())
    model=ByteLanguageGround(ground)
    assert isinstance(model.operator,StructuralLinearOperator)
    assert model.ground["model_family"]=="count_trained_first_order_byte_language_model"
    assert model.predict(ord("a"))["ranked_tokens"][0]["byte"]==ord("b")
    assert model.predict(ord("b"))["ranked_tokens"][0]["byte"]==ord("a")
    assert abs(sum(model.probabilities(ord("a")))-1)<1e-8
    result=model.generate(b"a",steps=6)
    assert bytes.fromhex(result["generated_hex"])==b"bababa"
    assert result["generated_byte_count"]==6
    assert len(result["decision_trace"])==6
    assert result["status"]=="MODELED_WEIGHT_BACKED_INFERENCE"
    assert result["ground_digest"]==ground["ground_digest"]
    assert ground["training_byte_count"]==len(training())


def test_independent_heldout_perplexity_is_better_than_uniform_byte_baseline():
    model=ByteLanguageGround(fit_byte_lm(training()))
    evaluation=model.evaluate(b"abababababababab")
    assert evaluation["heldout_transitions"]==15
    assert evaluation["cross_entropy_bits_per_byte"]<3.0
    assert evaluation["uniform_baseline_bits_per_byte"]==8
    assert evaluation["perplexity"]<8
    assert evaluation["top1_accuracy"]==1.0


def test_untrusted_bigram_ground_is_not_exec_authority_or_fabricated_weights():
    original=fit_byte_lm(training())
    forged=dict(original)
    forged["training_byte_count"]=10**6
    with pytest.raises(ValueError):
        ByteLanguageGround(forged)
    changed=dict(original)
    changed["sparse_weights"]=[list(x) for x in original["sparse_weights"]]
    changed["sparse_weights"][0][2]+=1
    from vld import domain_digest
    changed["ground_digest"]=domain_digest("vl64.trained-byte-language-ground.v1",{
        k:v for k,v in changed.items() if k!="ground_digest"})
    with pytest.raises(ValueError,match="counts"):
        ByteLanguageGround(changed)
    assert original["scope"].endswith("not_neural_transformer")


def test_existing_compact_safetensors_parser_reused_on_exact_local_header():
    artifact=deterministic_local_neural_weights()
    head_n=struct.unpack("<Q",artifact[:8])[0]
    header=artifact[:8+head_n]
    n,metadata=SafetensorsGlance.parse_header_bytes(header)
    assert n==head_n
    assert len(metadata)==7
    ground=observe_safetensors_header(header,source="local-test",
                                        file_size=len(artifact))
    assert ground["tensor_count"]==7
    assert ground["weight_payload_bytes_observed"]==0
    assert ground["tensor_values_known"] is False
    assert ground["declared_payload_bytes"]>0
    assert all(item["payload_bytes"]>0 for item in ground["tensors"])
    admitted=read_f32_tensors(artifact,["embed","wq","wv"])
    assert admitted["acquired_weight_bytes"]>0
    assert admitted["weights"]["embed"]["shape"]==[VOCAB,HIDDEN]


def test_neural_decoder_runs_weighted_causal_attention_from_real_f32_payload():
    artifact=deterministic_local_neural_weights()
    model=TinyCausalAttentionGround(artifact)
    token=model.next_token([0,1,2,3])
    assert token["status"]=="ACTUAL_LOCAL_WEIGHT_BACKED_NEURAL_INFERENCE"
    assert token["ground_digest"]==model.ground["ground_digest"]
    assert len(token["logits"])==VOCAB
    assert len(token["causal_attention_probabilities"])==4
    assert abs(sum(token["causal_attention_probabilities"])-1)<1e-7
    assert 0<=token["selected_next_token"]<VOCAB
    generated=model.generate([0,1,2],steps=12)
    assert len(generated["generated_tokens"])==12
    assert generated["status"]=="ACTUAL_LOCAL_WEIGHT_BACKED_NEURAL_INFERENCE"
    assert generated==TinyCausalAttentionGround(artifact).generate([0,1,2],steps=12)


def reference_logits_from_weight_bytes(artifact,tokens):
    """Independent non-VL64 matrix/reference implementation."""
    weights=read_f32_tensors(artifact,[
        "embed","lm_head","rms","wk","wo","wq","wv"])["weights"]
    def as_matrix(name):
        t=weights[name]
        rows,cols=t["shape"]
        return [t["values"][i*cols:(i+1)*cols] for i in range(rows)]
    def mul(name,x):
        return [sum(a*b for a,b in zip(row,x)) for row in as_matrix(name)]
    embed=as_matrix("embed")
    rms=weights["rms"]["values"]
    def normalize(x):
        scale=1/math.sqrt(sum(v*v for v in x)/len(x)+1e-6)
        return [v*scale*g for v,g in zip(x,rms)]
    representations=[embed[token] for token in tokens]
    h=[normalize(x) for x in representations]
    keys=[mul("wk",x) for x in h]
    vals=[mul("wv",x) for x in h]
    query=mul("wq",h[-1])
    scores=[sum(a*b for a,b in zip(query,k))/math.sqrt(len(query))
            for k in keys]
    normalizer=sum(math.exp(x-max(scores)) for x in scores)
    attention=[math.exp(x-max(scores))/normalizer for x in scores]
    context=[sum(attention[i]*vals[i][j] for i in range(len(vals)))
             for j in range(len(query))]
    output=mul("wo",context)
    residual=[a+b for a,b in zip(representations[-1],output)]
    return mul("lm_head",normalize(residual)),attention


@pytest.mark.parametrize("tokens",[[0],[0,1],[4,3,2,1],[0,1,7,4,3,1,2]])
def test_neural_model_matches_independent_causal_attention_numeric_reference(tokens):
    artifact=deterministic_local_neural_weights()
    model=TinyCausalAttentionGround(artifact)
    actual_logits,actual_attention=model.logits(tokens)
    reference_logits,reference_attention=reference_logits_from_weight_bytes(
        artifact,tokens)
    assert actual_logits==pytest.approx(reference_logits,abs=1e-6)
    assert actual_attention==pytest.approx(reference_attention,abs=1e-6)


def test_neural_weight_bytes_are_required_and_source_digest_changes():
    original=deterministic_local_neural_weights(73)
    other=deterministic_local_neural_weights(74)
    a=TinyCausalAttentionGround(original)
    b=TinyCausalAttentionGround(other)
    assert a.file_sha256!=b.file_sha256
    assert a.ground["ground_digest"]!=b.ground["ground_digest"]
    n=int.from_bytes(original[:8],"little")
    with pytest.raises(ValueError):
        TinyCausalAttentionGround(original[:8+n])
    with pytest.raises(ValueError):
        TinyCausalAttentionGround(b"not weights")
