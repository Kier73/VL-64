"""Byte-exact evidence for the first REAL full 30-layer pretrained forward pass.

The retained compact evidence verifies the claimed model/DAG coordinates, full
vocabulary and causal checkpoint lineage, but intentionally does not purport
to contain the raw 269 MB of pretrained weight ranges or individual KV pages.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from vld import domain_digest
from experiments.model_fabric.execution_dag import compile_decoder_dag, MAX_RANGE
from experiments.model_fabric.pretrained_layer import LAYER_TENSORS

ROOT=Path(__file__).resolve().parents[1]/'experiments/model_fabric/datasets'
REPORTS={
    'partial':('2026-11-19-real-30-layer-pretrained-partial.json',
               '8a3fe1b2a8191166de89f2b9c9d1d2a9d8615a8cfbd0fa138f66f064bb095cfc'),
    'complete':('2026-11-19-real-30-layer-pretrained-complete.json',
                '796e0bc6cd1da2bccef47f571a044e9e88acc23c7f4cefff1c2e8911e46c9192'),
    'checkpoint':('2026-11-19-real-30-layer-pretrained-checkpoint.json',
                  'be95a2be30fcd73deebc2b36a3adce9989516863eb048df750d8eaa257a92bbd'),
}


def read(name):
    file,digest=REPORTS[name]
    exact=(ROOT/file).read_bytes()
    assert hashlib.sha256(exact).hexdigest()==digest
    return json.loads(exact)


def test_pinned_full_pretrained_30_layer_reports_and_checkpoint_are_consistent():
    manifest=json.loads((ROOT/'2026-11-18-pinned-smollm2-tensor-ground.json').read_text())
    plan=compile_decoder_dag(manifest)
    assert plan['architecture']['layers']==30
    assert plan['architecture']['vocab']==49152
    assert plan['architecture']['hidden']==576
    partial,complete,root=read('partial'),read('complete'),read('checkpoint')
    assert partial['status']=='VERIFIED_PRETRAINED_LAYER_FRONTIER'
    assert complete['status']=='REAL_PRETRAINED_30_LAYER_LOGITS_INDEPENDENTLY_VERIFIED'
    assert [partial['layers_completed'],complete['layers_completed']]==[15,30]
    assert [partial['verified_layers_this_process'],complete['verified_layers_this_process']]==[15,15]
    assert [e['index'] for e in partial['verified_this_process']]==list(range(15))
    assert [e['index'] for e in complete['verified_this_process']]==list(range(15,30))
    for report in (partial,complete):
        assert report['model']==plan['model']
        assert report['revision']==plan['revision']
        assert report['dag_digest']==plan['dag_digest']
        assert report['tokens']==[42,75]
        assert report['total_declared_layers']==30
        for stage in report['verified_this_process']:
            assert stage['max_abs_error']<=stage['tolerance_abs']
            assert stage['hidden_sha256'] and len(stage['hidden_sha256'])==64
            assert len(stage['static_weight_identity'])==64
            assert stage['tolerance_abs']>0
            i=stage['index']
            sizes=[math.prod(plan['tensors'][f'model.layers.{i}.'+s]['shape'])*2
                   for s in LAYER_TENSORS]
            assert stage['bytes_acquired']==sum(sizes)==7_080_192
            assert stage['range_reads']==sum(math.ceil(n/MAX_RANGE) for n in sizes)==113
    assert complete['previous_checkpoint_digest']==partial['checkpoint_digest']
    assert complete['checkpoint_digest']==root['checkpoint_digest']
    assert root['previous_checkpoint']==partial['checkpoint_digest']
    assert root['completed_layers']==30 and root['epoch']==1
    assert root['owner']=='observer-process-B'
    assert root['tokens']==[42,75]
    assert root['dag_digest']==plan['dag_digest']
    assert root['tensor_ground_digest']==plan['tensor_ground_digest']
    assert set(root['weight_identities'])=={str(i) for i in range(30)}
    assert set(root['range_witnesses'])=={str(i) for i in range(-1,30)}
    assert len(root['kv_manifests'])==30
    assert all(root['weight_identities'][str(e['index'])]==e['static_weight_identity']
               for e in partial['verified_this_process']+complete['verified_this_process'])
    assert root['checkpoint_digest']==domain_digest(root['schema'],{
        k:v for k,v in root.items() if k!='checkpoint_digest'})
    # 30 x 9 layer tensors + 2 token rows, plus tied embedding and final gamma.
    layer_payload=30*7_080_192
    token_rows=2*576*2
    final_norm=576*2
    full_embeddings=49_152*576*2
    assert partial['weight_payload_bytes_cumulative']==15*7_080_192+token_rows
    assert complete['weight_payload_bytes_cumulative']==root['bytes_acquired']==layer_payload+token_rows==212_408_064
    assert complete['weight_bytes_including_logits']==layer_payload+token_rows+final_norm+full_embeddings==269_032_320
    assert complete['range_reads_cumulative']==root['ranges_acquired']==30*113+2==3392
    assert complete['range_reads_including_logits']==30*113+2+1+math.ceil(49_152/(MAX_RANGE//(576*2)))==4271
    assert complete['full_vocab_size']==49_152
    assert complete['selected_next_token']==33
    assert complete['independent_logit_max_abs_error']<=2e-4
    assert complete['top_token_ids'][0]==complete['selected_next_token']
    assert len(complete['top_token_ids'])==len(complete['top_logits'])==5
    assert all(a>=b for a,b in zip(complete['top_logits'],complete['top_logits'][1:]))
    assert len(complete['logits_f64_sha256'])==64
    assert len(complete['result_receipt_digest'])==64
