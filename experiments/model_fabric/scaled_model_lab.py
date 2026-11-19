"""Pinned Hugging Face metadata/range-only experiment for large architectures.

No model checkpoint is downloaded. This is structural observation and, when
requested, one exact BF16/F32 pretrained norm slice with an independent check.
The CLI never claims complete Qwen3.8 or DeepSeek inference.
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path

from .scaled_model_fabric import (
    MAX_INDEX_BYTES,MAX_RANGE,QualifiedShardNavigator,
    admit_index,admit_architecture,_sha,_identifier,_SHARD,
)

PINNED = {
    'qwen': ('Qwen/Qwen3.8-27B',
             '37561816e66b5b9153b1b64c10968c18c1e6f4a0'),
    'deepseek': ('deepseek-ai/DeepSeek-V4.1-Flash-1017',
                 'dfbc4a7d7fbe69a83a0d945aba10052dd6c9ea2e'),
}


def _url(model,revision,filename):
    _identifier(model,revision)
    if filename not in ('config.json','model.safetensors.index.json') and not _SHARD.fullmatch(filename):
        raise ValueError('Unapproved external model path')
    return 'https://huggingface.co/'+model+'/resolve/'+revision+'/'+filename


def _host_allowed(url):
    p=urllib.parse.urlparse(url)
    return p.scheme=='https' and p.hostname is not None and (
        p.hostname in ('huggingface.co','hf.co') or
        p.hostname.endswith('.huggingface.co') or p.hostname.endswith('.hf.co'))


def _fetch_metadata(url,limit):
    if not _host_allowed(url):raise ValueError('Metadata URL is unapproved')
    req=urllib.request.Request(url,headers={'User-Agent':'VL64-scale-model-observer/1.0',
                                           'Accept-Encoding':'identity'},method='GET')
    with urllib.request.urlopen(req,timeout=24) as res:
        if not _host_allowed(res.geturl()) or res.status!=200:
            raise ValueError('Metadata source or status is unqualified')
        raw=res.read(limit+1)
    if len(raw)>limit:raise ValueError('Metadata exceeds approved byte budget')
    return raw


def pinned_hf_range_host(model,revision):
    _identifier(model,revision)
    def host(request):
        shard=request['shard'];begin=request['byte_begin'];end=request['byte_end_exclusive']
        if not _SHARD.fullmatch(shard) or type(begin) is not int or type(end) is not int or not 0<=begin<end or end-begin>MAX_RANGE:
            raise ValueError('Refused oversized or invalid HTTP range')
        url=_url(model,revision,shard)
        headers={'Range':f'bytes={begin}-{end-1}','Accept-Encoding':'identity',
                 'User-Agent':'VL64-scale-model-observer/1.0'}
        req=urllib.request.Request(url,headers=headers,method='GET')
        with urllib.request.urlopen(req,timeout=30) as res:
            if not _host_allowed(res.geturl()) or res.status!=206:
                raise ValueError('Source did not honor exact authenticated HTTPS range')
            cr=res.headers.get('Content-Range','')
            m=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',cr)
            if not m or (int(m[1]),int(m[2]))!=(begin,end-1) or int(m[3])<end:
                raise ValueError('Source returned different byte-range identity')
            data=res.read(MAX_RANGE+1)
        if len(data)!=end-begin:
            raise ValueError('Incomplete HTTP weight range')
        return {**request,'status':'QUALIFIED_EXACT_SHARD_RANGE',
                'observer_host':'pinned-huggingface-https-range',
                'payload':data,'sha256':_sha(data)}
    return host


def observe_large_model(*,selection, numeric=False):
    model,revision=PINNED[selection]
    config_raw=_fetch_metadata(_url(model,revision,'config.json'),MAX_INDEX_BYTES)
    index_raw=_fetch_metadata(_url(model,revision,'model.safetensors.index.json'),MAX_INDEX_BYTES)
    index,mapping=admit_index(index_raw,model=model,revision=revision)
    ground=admit_architecture(config_raw,model=model,revision=revision,index_ground=index)
    report={'status':'STRUCTURAL_MODEL_GROUND_ADMITTED',
            'model':model,'revision':revision,'ground_digest':ground['ground_digest'],
            'index_digest':index['ground_digest'],'index_sha256':index['index_sha256'],
            'config_sha256':ground['config_sha256'],
            'model_type':ground['model_type'],'layers':ground['num_layers'],
            'layer_families':{f:sum(item['family']==f for item in ground['layer_schedule'])
                              for f in sorted({item['family'] for item in ground['layer_schedule']})},
            'shards':ground['shard_count'],'mapped_tensors':ground['tensor_count'],
            'advertised_payload_bytes':ground['advertised_payload_bytes'],
            'metadata_bytes_read':len(config_raw)+len(index_raw),
            'pretrained_weight_bytes_observed':0,'full_model_executed':False,
            'architecture_status':ground['architecture_status']}
    if numeric:
        nav=QualifiedShardNavigator(model=model,revision=revision,index_raw=index_raw,
                range_host=pinned_hf_range_host(model,revision),
                approved_hosts={'pinned-huggingface-https-range'})
        try:
            report['numeric_probe']=nav.small_numeric_probe()
        except (ValueError,OSError,TimeoutError) as exc:
            report['numeric_probe']={'status':'UNRESOLVED','reason':type(exc).__name__}
        report['pretrained_weight_bytes_observed']=nav.weight_bytes
        report['exact_http_ranges']=nav.range_count
        report['range_witness_chain_digest']=nav.chain
    return report


def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True,choices=sorted(PINNED))
    ap.add_argument('--numeric',action='store_true',help='Optional one-weight normalization probe, never full inference')
    ap.add_argument('--out',default='artifacts/large-model-ground.json')
    args=ap.parse_args(argv)
    try:
        report=observe_large_model(selection=args.model,numeric=args.numeric)
    except (ValueError,OSError,TimeoutError) as exc:
        report={'status':'UNRESOLVED','model':PINNED[args.model][0],
                'revision':PINNED[args.model][1],'reason':type(exc).__name__,
                'full_model_executed':False}
        if isinstance(exc,urllib.error.HTTPError):
            report['http_status']=exc.code
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    print('VL64_LARGE_MODEL_GROUND '+json.dumps(report,sort_keys=True))
    if report['status']!='STRUCTURAL_MODEL_GROUND_ADMITTED' or (
        args.numeric and report.get('numeric_probe',{}).get('status')!='WEIGHT_BACKED_NORMALIZATION_SLICE_VERIFIED'):
        return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
