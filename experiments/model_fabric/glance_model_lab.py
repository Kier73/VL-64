"""Pinned pretrained model subgraphs executed from frozen weight-glance Grounds.

The independent Qwen/DeepSeek scalar checks are run twice: first on actual
qualified model ranges and then in a new navigator on those exact frozen bytes.
This is NOT full Qwen/DeepSeek inference; gaps or missing sources fail closed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .qualified_range_cache import QualifiedRangeCache
from .weight_glance_ground import freeze_weight_glances, PinnedWeightGlanceGround
from .scaled_model_fabric import QualifiedShardNavigator, MAX_INDEX_BYTES
from .scaled_model_lab import PINNED, _url, _fetch_metadata, pinned_hf_range_host
from .qualified_subgraphs import (
    probe_qwen_trained_delta_head, probe_deepseek_trained_router,
    probe_deepseek_selected_fp4_expert_slices,
)

REMOTE='pinned-huggingface-https-range'
LOCAL='pinned-glance-reconstructor'


def _observe(model, nav):
    if model=='qwen':
        r=probe_qwen_trained_delta_head(nav)
        if r['status']!='REAL_QWEN_GATED_DELTA_TRAINED_HEAD_VERIFIED':
            raise ValueError('Pinned Qwen numerical reference unverified')
        return {'status':r['status'], 'head':r['value_head'],
                'output':r['output_row0_head_contribution'],
                'reference_error':r['max_abs_independent_error']}
    router=probe_deepseek_trained_router(nav)
    fragments=probe_deepseek_selected_fp4_expert_slices(nav,router)
    if (router['status']!='REAL_DEEPSEEK_TRAINED_384_ROUTER_VERIFIED' or
            fragments['status']!='REAL_DEEPSEEK_SELECTED_FP4_SWIGLU_SLICES_VERIFIED'):
        raise ValueError('Pinned DeepSeek selected-expert reference unverified')
    return {'status':fragments['status'],
            'selected_experts':router['selected_experts'],
            'routing_weights':router['routing_weights'],
            'reference_error':max(router['max_abs_independent_error'],
                                  fragments['max_abs_independent_error'])}


def run(model, *, root, budget_bytes=128*1024*1024):
    if model not in ('qwen','deepseek'):
        raise ValueError('Only explicitly pinned model families are qualified')
    model_name,revision=PINNED[model]
    index=_fetch_metadata(_url(model_name,revision,'model.safetensors.index.json'),
                          MAX_INDEX_BYTES)
    base=Path(root)
    cache=QualifiedRangeCache(base/'ranges',pinned_hf_range_host(model_name,revision),
                              approved_hosts={REMOTE},budget_bytes=budget_bytes)
    first_nav=QualifiedShardNavigator(model=model_name,revision=revision,index_raw=index,
                                     range_host=cache,approved_hosts={REMOTE})
    first=_observe(model,first_nav)
    network=cache.stats()
    cache_head=cache.snapshot()
    ground_head=freeze_weight_glances(base/'ranges',expected_cache_head=cache_head,
                                     ground_root=base/'glances',approved_source_hosts={REMOTE})
    ground=PinnedWeightGlanceGround(base/'glances',expected_head=ground_head,
                                   approved_source_hosts={REMOTE},
                                   local_observer_host=LOCAL)
    offline_nav=QualifiedShardNavigator(model=model_name,revision=revision,index_raw=index,
                                        range_host=ground,approved_hosts={LOCAL})
    repeat=_observe(model,offline_nav)
    if first!=repeat or first['reference_error']>2e-5:
        raise ValueError('Observed pretrained computation differs on Ground replay')
    target=('model.language_model.layers.0.input_layernorm.weight' if model=='qwen'
            else 'layers.0.ffn.gate.bias')
    # A different partition than the original whole-vector request.
    sample,_=offline_nav.get_raw(target,flat_start=3,count=5)
    if not sample or len(sample) not in (10,20):
        raise ValueError('Repartitioned weight observation unavailable')
    if ground.stats()['remote_ranges']!=0 or ground.reads<=0:
        raise ValueError('Ground replay mistakenly required network observer')
    return {'schema':'vl64.real-pretrained-glance-replay.v1',
            'status':'REAL_PINNED_GLANCE_NUMERICAL_REPLAY_VERIFIED',
            'model':model_name,'revision':revision,
            'index_ground_digest':first_nav.index['ground_digest'],
            'original_observation':first,'retained_replay':repeat,
            'source_catalogue_head':cache_head,'portable_ground_head':ground_head,
            'original_remote_ranges':network['qualified_remote_ranges'],
            'original_remote_bytes':network['qualified_remote_bytes'],
            'replayed_local_ranges':ground.reads,
            'replayed_local_bytes':ground.bytes_reconstructed,
            'replayed_remote_ranges':0,
            'repartitioned_weight_span_bytes':len(sample),
            'replay_range_chain':offline_nav.chain,
            'scope':'trained-head/router-and-six-expert-fragment replay; not a complete large-model layer'}


def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',choices=('qwen','deepseek'),required=True)
    ap.add_argument('--root',default='artifacts/pinned-model-glances')
    ap.add_argument('--out',default='artifacts/pretrained-glance-result.json')
    ap.add_argument('--budget-bytes',type=int,default=128*1024*1024)
    args=ap.parse_args(argv)
    target=Path(args.out)
    target.parent.mkdir(parents=True,exist_ok=True)
    try:
        result=run(args.model,root=args.root,budget_bytes=args.budget_bytes)
    except (ValueError,OSError,TimeoutError,OverflowError) as error:
        result={'status':'UNRESOLVED','model':PINNED[args.model][0],
                'error_type':type(error).__name__,
                'reason':str(error)[:200],
                'scope':'no pretrained replay verified without full observed numeric subgraph'}
    target.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print('VL64_PINNED_GLANCE_REPLAY '+json.dumps(result,sort_keys=True))
    return 0 if result['status']=='REAL_PINNED_GLANCE_NUMERICAL_REPLAY_VERIFIED' else 1


if __name__=='__main__':
    raise SystemExit(main())
