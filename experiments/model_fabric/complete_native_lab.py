"""Pinned, fail-closed large-model layer experiment.

A plan is structural only. Real execution requires every weight/scale range and
returns a numerical receipt with *no independent real-checkpoint oracle*.
Independent scalar reference coverage is provided separately by fixtures.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .complete_native_layers import QwenQualifiedLayer,DeepSeekQualifiedMoE
from .scaled_model_fabric import MAX_INDEX_BYTES,QualifiedShardNavigator
from .scaled_model_lab import PINNED,_url,_fetch_metadata,pinned_hf_range_host


def run(model, *, execute=False, budget_bytes=1_500_000_000):
    name,revision=PINNED[model]
    index=_fetch_metadata(_url(name,revision,'model.safetensors.index.json'),MAX_INDEX_BYTES)
    nav=QualifiedShardNavigator(model=name,revision=revision,index_raw=index,
          range_host=pinned_hf_range_host(name,revision),
          approved_hosts={'pinned-huggingface-https-range'})
    if model=='qwen':
        layer=QwenQualifiedLayer(nav,layer=0,key_heads=16,value_heads=48,
                   key_dim=128,value_dim=128,budget_bytes=budget_bytes)
        result={'operator':'gated-delta+swiglu','complete_heads':48,
                'complete_layer_weight_bytes':layer.estimated_weight_bytes,
                'native_hidden_width':layer.hidden}
        if execute:
            x=[math.sin((j+1)*.017)*.125 for j in range(layer.hidden)]
            hidden,receipt=layer.next(x)
            result.update({'status':'REAL_PRETRAINED_QWEN_GDN_LAYER_EXECUTED_NO_EXTERNAL_ORACLE',
                           'complete_output_width':len(hidden),
                           'output_first_eight':[round(v,9) for v in hidden[:8]],
                           'receipt':receipt})
    elif model=='deepseek':
        layer=DeepSeekQualifiedMoE(nav,layer=0,budget_bytes=budget_bytes,
                                   swiglu_limit=10.0,norm_eps=1e-20)
        result={'operator':'FP4_6_SELECTED_PLUS_SHARED_FFN',
                'native_hidden_width':layer.width,'router_experts':384,
                'topk':layer.topk}
        if execute:
            x=[math.cos((j+1)*.013)*.0625 for j in range(layer.width)]
            y,receipt=layer.next(x)
            result.update({'status':'REAL_PRETRAINED_DEEPSEEK_SELECTED_MOE_EXECUTED_NO_EXTERNAL_ORACLE',
                           'complete_output_width':len(y),
                           'output_first_eight':[round(v,9) for v in y[:8]],
                           'receipt':receipt})
    else:
        raise ValueError('Unsupported pinned architecture')
    if not execute:
        result['status']='STRUCTURAL_NATIVE_LAYER_PLAN_ONLY'
    result.update({'model':name,'revision':revision,'index_sha256':nav.index['index_sha256'],
                   'index_ground_digest':nav.index['ground_digest'],
                   'actual_weight_bytes_observed':nav.weight_bytes,
                   'exact_range_count':nav.range_count,
                   'range_witness_chain':nav.chain,
                   'scope':'one native GDN layer or MoE FFN only; synthetic hidden input; no full-model inference'})
    return result


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True,choices=sorted(PINNED))
    parser.add_argument('--execute',action='store_true',help='Perform every trained matrix operation, not just structure')
    parser.add_argument('--budget-bytes',type=int,default=1_500_000_000)
    parser.add_argument('--out',default='artifacts/complete-native-layer.json')
    args=parser.parse_args(argv)
    try:
        result=run(args.model,execute=args.execute,budget_bytes=args.budget_bytes)
        code=0
    except (ValueError,OSError,TimeoutError,OverflowError,MemoryError) as error:
        result={'status':'UNRESOLVED',
                'model':PINNED[args.model][0], 'revision':PINNED[args.model][1],
                'error_type':type(error).__name__,'reason':str(error)[:200],
                'scope':'No pretrained complete-layer numeric success claimed'}
        code=1
    file=Path(args.out);file.parent.mkdir(parents=True,exist_ok=True)
    file.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    print('VL64_COMPLETE_NATIVE_LAYER '+json.dumps(result,sort_keys=True))
    return code

if __name__=='__main__':raise SystemExit(main())
