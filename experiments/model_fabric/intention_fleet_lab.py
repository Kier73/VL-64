"""Measurable model-fleet use of intent + tokenization + actual trained operators.

The default is two separately trained byte-model hosts, *not* good NLP.
An optional revision-pinned instruction transformer can be evaluated for text,
code or a registered arithmetic tool decision. No network use by default.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from .intention_fabric import (AuthorizedIntentAgent, ModelInstanceFleet,
                               compile_intention)
from .operational_ground import observe_operational_ground
from .operational_hosts import HostedTrainedByteLanguage


def _admit(host):
    meta=host.description
    return observe_operational_ground(host,expected_model=meta['model'],
        expected_revision=meta['revision'],approved_hosts={meta['observer_host']},
        expected_artifact=meta['model_artifact_identity'])


def byte_fleet(training_source='README.md', *, steps=16):
    from .byte_lm import ByteLanguageGround, fit_byte_lm
    corpus=Path(training_source).read_bytes()
    if len(corpus)<64:
        raise ValueError('Need independently trainable model observations')
    a=min(len(corpus)//2,2**20)
    b=min(len(corpus),a+2**20)
    trained=(fit_byte_lm(corpus[:a]),fit_byte_lm(corpus[a:b]))
    hosts=[HostedTrainedByteLanguage(x,observer_host='instance-'+str(i))
           for i,x in enumerate(trained)]
    grounds=[_admit(h) for h in hosts]
    fleet=ModelInstanceFleet({name:{'ground':g,'host':h,
         'approved_hosts':{h.description['observer_host']}}
         for name,h,g in zip(('paragraph','code'),hosts,grounds)})
    requests=[{'instance':'paragraph','intent':compile_intention(grounds[0],
         kind='paragraph',messages=[{'role':'user','content':'Describe a virtual ground.'}],
         max_new_tokens=steps)},
         {'instance':'code','intent':compile_intention(grounds[1],
         kind='python',messages=[{'role':'user','content':'Write a pure Python add function.'}],
         max_new_tokens=steps)}]
    begin=time.perf_counter()
    results=fleet.execute(requests,parallel=True)
    elapsed=time.perf_counter()-begin
    verified=[]
    for j,result in enumerate(results):
        out=result['result']
        tokens=out['generated_tokens']
        prefix=result['checkpoint']['model_checkpoint']['tokens'][:-len(tokens)]
        reference=ByteLanguageGround(trained[j])
        previous=prefix[-1]
        expected=[]
        for _ in tokens:
            next_=reference.predict(previous,limit=1)['ranked_tokens'][0]['byte']
            expected.append(next_);previous=next_
        if tokens!=expected:
            raise ValueError('Trained byte host diverges from independent numeric reference')
        verified.append({
           'instance':result['instance'],'ground_digest':grounds[j]['ground_digest'],
           'model_training_digest':trained[j]['ground_digest'],
           'num_generated_tokens':len(tokens),'output_contract_status':out['status'],
           'generated_text':out['observed_text'],
           'independent_trained_weight_oracle_match':True,
           'host_calls':out['host_calls'],'checkpoint_digest':out['checkpoint_digest'],
           'transferred_weight_bytes':out['remote_weight_bytes_received']})
    return {'schema':'vl64.intent-model-fleet-benchmark.v1',
            'status':'TWO_INDEPENDENT_TRAINED_MODEL_HOSTS_EXECUTED',
            'instances':verified,'elapsed_s':round(elapsed,5),
            'scope':'two_trained_count-byte_model_hosts_not_conversational_quality_or_agent_autonomy'}


def instruction_host(*, model, revision, kind, task, steps):
    from .native_chat_host import HostedNativeChatTransformer
    host=HostedNativeChatTransformer(model,revision,observer_host='instruction-model',max_context=1024)
    g=_admit(host)
    fleet=ModelInstanceFleet({'instruction':{'ground':g,'host':host,
                            'approved_hosts':{'instruction-model'}}})
    if kind=='tool_json':
        result=AuthorizedIntentAgent(fleet,'instruction',
                   tools={'add_numbers':lambda a,b:a+b},max_rounds=2,
                   stop_tokens=()).run(task,max_new_tokens=steps)
        return {'schema':'vl64.intent-instruction-model-benchmark.v1',
                'status':result['status'],'model':model,'revision':revision,
                'agent':result,'scope':'host_resident_model_action_only_caller_authorized_tools'}
    intention=compile_intention(g,kind=kind,
                messages=[{'role':'user','content':task}],
                max_new_tokens=steps,tokenizer_mode='native_chat')
    report=fleet.execute([{'instance':'instruction','intent':intention}])[0]
    return {'schema':'vl64.intent-instruction-model-benchmark.v1',
            'status':report['result']['status'],'model':model,'revision':revision,
            'result':report['result'],
            'scope':'real_host_computed_model_logits_not_independent_neural_math_or_usefulness_proof'}


def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument('--backend',choices=('byte_fleet','instruction'),default='byte_fleet')
    ap.add_argument('--training-source',default='README.md')
    ap.add_argument('--model')
    ap.add_argument('--revision')
    ap.add_argument('--kind',choices=('paragraph','python','tool_json'),default='paragraph')
    ap.add_argument('--task',default='Explain how a virtual ground navigates computation.')
    ap.add_argument('--steps',type=int,default=16)
    ap.add_argument('--out',default='artifacts/intent-fleet.json')
    args=ap.parse_args(argv)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    try:
        result=(byte_fleet(args.training_source,steps=args.steps)
            if args.backend=='byte_fleet'
            else instruction_host(model=args.model,revision=args.revision,
                 kind=args.kind,task=args.task,steps=args.steps))
    except (ValueError,RuntimeError,OSError,TimeoutError,ImportError) as exc:
        result={'status':'UNRESOLVED','error_type':type(exc).__name__,
            'scope':'no model execution asserted without admitted host and causal evidence'}
    out.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print('VL64_INTENTION_FLEET '+json.dumps(result,sort_keys=True))
    return 1 if result['status']=='UNRESOLVED' else 0


if __name__=='__main__':raise SystemExit(main())
