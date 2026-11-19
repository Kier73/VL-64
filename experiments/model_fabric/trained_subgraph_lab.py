"""Optional pinned-weight numerical experiments for two native subgraph families.

These are intentionally *not* full Qwen/DeepSeek model inference. No fake
success on missing HTTP ranges, novel tensor dtypes, or absent named weights.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .scaled_model_fabric import MAX_INDEX_BYTES,QualifiedShardNavigator
from .scaled_model_lab import PINNED,_url,_fetch_metadata,pinned_hf_range_host
from .qualified_subgraphs import probe_qwen_trained_delta_head,probe_deepseek_trained_router


def run(model):
    model_name,revision=PINNED[model]
    index=_fetch_metadata(_url(model_name,revision,'model.safetensors.index.json'),MAX_INDEX_BYTES)
    nav=QualifiedShardNavigator(model=model_name,revision=revision,index_raw=index,
          range_host=pinned_hf_range_host(model_name,revision),
          approved_hosts={'pinned-huggingface-https-range'})
    result=(probe_qwen_trained_delta_head(nav) if model=='qwen'
            else probe_deepseek_trained_router(nav))
    result['index_sha256']=nav.index['index_sha256']
    result['index_ground_digest']=nav.index['ground_digest']
    return result


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True,choices=sorted(PINNED))
    parser.add_argument('--out',default='artifacts/trained-subgraph.json')
    args=parser.parse_args(argv)
    target=Path(args.out);target.parent.mkdir(parents=True,exist_ok=True)
    try:
        result=run(args.model)
    except (ValueError,OSError,TimeoutError,OverflowError) as exc:
        result={'status':'UNRESOLVED','model':PINNED[args.model][0],
                'revision':PINNED[args.model][1],
                'reason':type(exc).__name__,
                'refusal':str(exc)[:200] if isinstance(exc,ValueError) else None,
                'scope':'no trained subgraph verified without exact eligible weights'}
    target.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print('VL64_TRAINED_SUBGRAPH '+json.dumps(result,sort_keys=True))
    return 0 if result['status'].startswith('REAL_') else 1

if __name__=='__main__':raise SystemExit(main())
