"""Use VL-64 as a small, weightless operational translator for language tasks.

Default: genuinely trained byte model held in a separate host object.
Optional: pinned HuggingFace pretrained transformer using torch/transformers,
with weights owned by the host rather than moved into the VL navigator.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .operational_ground import OperationalModelNavigator, observe_operational_ground
from .operational_hosts import HostedTrainedByteLanguage, HostedTransformersLanguage

PINNED_SMOLLM2 = ('HuggingFaceTB/SmolLM2-135M',
                   '4e53f736cbb20a9a0f56b4c4bf378d9f306ff915')


def run(*, backend='byte', prompt='Virtual Layer', steps=8,
        training_source='README.md'):
    if backend not in ('byte','transformer'):
        raise ValueError('No typed model operator backend')
    started=time.perf_counter()
    if backend=='byte':
        from .byte_lm import ByteLanguageGround, fit_byte_lm
        corpus=Path(training_source).read_bytes()
        if len(corpus)<16:raise ValueError('Invalid corpus')
        ground_bytes=fit_byte_lm(corpus[:min(len(corpus),2**20)])
        host=HostedTrainedByteLanguage(ground_bytes)
        expected_artifact=ground_bytes['ground_digest']
    else:
        host=HostedTransformersLanguage(*PINNED_SMOLLM2,
                                       max_context=256)
        expected_artifact=host.description['model_artifact_identity']
    meta=host.description
    operational=observe_operational_ground(host,expected_model=meta['model'],
         expected_revision=meta['revision'],approved_hosts={meta['observer_host']},
         expected_artifact=expected_artifact)
    nav=OperationalModelNavigator(operational,host,
               approved_hosts={meta['observer_host']})
    generated=nav.generate(prompt,max_new_tokens=steps)
    elapsed=time.perf_counter()-started
    if backend=='byte':
        oracle=ByteLanguageGround(ground_bytes).generate(prompt.encode('utf-8'),steps=steps)
        if bytes(generated['generated_tokens']).hex()!=oracle['generated_hex']:
            raise ValueError('Operational byte-model output diverges from independent trained oracle')
        status='ACTUAL_TRAINED_BYTE_LM_HOST_ORBIT_VERIFIED'
        reference='independent_byte_transition_model'
    else:
        status='PRETRAINED_TRANSFORMER_HOST_ORBIT_OBSERVED'
        reference='complete_host_logit_tiles_not_independent_transformer_math'
    checkpoint=nav.checkpoint()
    return {'schema':'vl64.operational-language-benchmark.v1','status':status,
        'model':meta['model'],'revision':meta['revision'],
        'backend':backend,'reference':reference,'operational_ground':operational,
        'prompt':prompt,'generated_text':generated['generated_text'],
        'generated_tokens':generated['generated_tokens'],
        'remote_weight_bytes_received':generated['remote_weight_bytes_received'],
        'host_calls':generated['host_calls'],
        'elapsed_s':round(elapsed,5),
        'generated_tokens_per_s':round(len(generated['generated_tokens'])/max(elapsed,1e-9),5),
        'transcript_head':generated['transcript_head'],
        'checkpoint_head':checkpoint['checkpoint_digest'],
        'scope':('trained count-byte LM not conversational transformer' if backend=='byte'
          else 'host-computed real pretrained transformer tokens; remote arithmetic not independently certified')}


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--backend',choices=('byte','transformer'),default='byte')
    parser.add_argument('--prompt',default='Virtual Layer')
    parser.add_argument('--steps',type=int,default=8)
    parser.add_argument('--training-source',default='README.md')
    parser.add_argument('--out',default='artifacts/operational-model-orbit.json')
    args=parser.parse_args(argv)
    target=Path(args.out);target.parent.mkdir(parents=True,exist_ok=True)
    try:
        result=run(backend=args.backend,prompt=args.prompt,steps=args.steps,
                   training_source=args.training_source)
    except (ValueError,OSError,TimeoutError,RuntimeError,ImportError) as exc:
        result={'status':'UNRESOLVED','backend':args.backend,
                'error_type':type(exc).__name__,
                'scope':'no host-resident language execution proved without model and token receipts'}
    target.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
    print('VL64_OPERATIONAL_MODEL '+json.dumps(result,sort_keys=True))
    return 0 if result['status']!='UNRESOLVED' else 1


if __name__=='__main__':raise SystemExit(main())
