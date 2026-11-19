"""Cross-process large-model Intention Ground benchmark; CPU controls only.

Two distinct model-serving routes, two interpreter process epochs per route.
Provider chat delegation is not VL-native model arithmetic or KV migration.
Tests use fixtures. --live requires provider credentials and performs remote calls.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .operational_ground import canonical, digest
from .remote_large_fleet import (ROUTES, TASKS, HTTPSChatTransport, RemoteLargeSession,
                                 agent_action, fixture_transport, ground, intention)

SCHEMA = 'vl64.remote-large-cross-process-benchmark.v1'
WORKER_SCHEMA = 'vl64.remote-large-process-epoch.v1'


def _atomic_json(path, obj):
    """Do not publish partly serialized checkpoints or measurements."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = canonical(obj) + b'\n'
    temp = target.with_name(target.name + '.pending-' + str(os.getpid()))
    try:
        with open(temp, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)


def _transport(route, live):
    return HTTPSChatTransport(timeout=90) if live else fixture_transport(route)


def _worker(stage, route, live, state_path, out_path, expected=None, max_tokens=128):
    if route not in ROUTES or stage not in ('start', 'continue') or type(max_tokens) is not int or not 1 <= max_tokens <= 384:
        raise ValueError('Unsupported qualified route/epoch/task budget')
    if (stage == 'start') != (expected is None):
        raise ValueError('Recovery requires an externally carried checkpoint head')
    g=ground(route)
    host=_transport(route,live)
    started=time.perf_counter()
    cpu_started=time.process_time()
    if stage=='start':
        if Path(state_path).exists():
            raise ValueError('Start epoch cannot overwrite an existing checkpoint')
        session=RemoteLargeSession(g,host)
        spec=intention(g,'paragraph',[{'role':'user','content':TASKS['paragraph']['prompt']}],max_tokens=max_tokens)
        tasks={'paragraph':session.run(spec)}
        recovered_prior_receipts=0
        epoch=0
    else:
        checkpoint=json.loads(Path(state_path).read_text(encoding='utf-8'))
        session=RemoteLargeSession.restore(g,host,checkpoint,expected_digest=expected,owner='recovered-'+route,epoch=1)
        recovered_prior_receipts=len(session.history)
        if recovered_prior_receipts != 1:
            raise ValueError('Expected exactly one earlier paragraph observation, without replay')
        spec=intention(g,'python',[{'role':'user','content':TASKS['python']['prompt']}],max_tokens=max_tokens)
        tasks={'python':session.run(spec),'agent':agent_action(session,max_tokens=max_tokens)}
        epoch=1
    checkpoint=session.checkpoint()
    expected_events = 1 if stage == 'start' else 2 + len(tasks['agent']['receipts'])
    if len(checkpoint['history']) != expected_events:
        raise ValueError('Recovery history and observed operation count diverge')
    process_cpu=time.process_time()-cpu_started
    result={'schema':WORKER_SCHEMA,'mode':'LIVE_PROVIDER' if live else 'FIXTURE_ONLY',
            'route':route,'epoch':epoch,'stage':stage,'model_architecture':g['model_architecture'],
            'model_ground_digest':g['ground_digest'],'checkpoint_digest':checkpoint['checkpoint_digest'],
            'checkpoint_event_count':len(checkpoint['history']),
            'recovered_prior_receipts':recovered_prior_receipts,
            'prior_provider_requests_reissued':0,'tasks':tasks,
            'interpreter_process_cpu_seconds':round(process_cpu,6),
            'interpreter_process_wall_seconds':round(time.perf_counter()-started,6),
            'local_model_weight_bytes':0,
            'grade':'provider_chat_delegation_no_exact_token_logits_or_KV_recovery'}
    result['result_digest']=digest(WORKER_SCHEMA,result)
    _atomic_json(out_path,result)
    # The start epoch commits the checkpoint last; the caller carries the head.
    if stage=='start':
        _atomic_json(state_path,checkpoint)
    return result


def _launch(route, live, root, max_tokens):
    """A parent process controls epochs; it never runs the remote model or fabricates a successful receipt."""
    base=Path(root)/route
    base.mkdir(parents=True,exist_ok=True)
    cp=base/'checkpoint.json'
    out_start=base/'epoch-0.json';out_end=base/'epoch-1.json'
    if cp.exists() or out_start.exists() or out_end.exists():
        raise ValueError('Benchmark destination must be fresh to avoid stale event reuse')
    args=[sys.executable,'-m','experiments.model_fabric.remote_large_recovery_benchmark',
          '--worker','--route',route,'--max-tokens',str(max_tokens),
          '--checkpoint',str(cp)]
    args.append('--live' if live else '--fixture')
    first=subprocess.run(args+['--stage','start','--out',str(out_start)],
                         capture_output=True,text=True,timeout=390)
    if first.returncode != 0:
        raise RuntimeError('Initial provider/controller epoch unresolved: '+route)
    start=json.loads(out_start.read_text())
    checkpoint=json.loads(cp.read_text())
    if start['checkpoint_digest']!=checkpoint['checkpoint_digest'] or start['checkpoint_event_count']!=1:
        raise ValueError('Parent refused first-epoch checkpoint publication')
    # Externally pinned digest arrives via caller, never from an untrusted new host.
    second=subprocess.run(args+['--stage','continue','--expected-digest',start['checkpoint_digest'],
                          '--out',str(out_end)],capture_output=True,text=True,timeout=450)
    if second.returncode != 0:
        raise RuntimeError('Cross-process provider/controller continuation unresolved: '+route)
    end=json.loads(out_end.read_text())
    if (end['epoch']!=1 or end['recovered_prior_receipts']!=1 or
            end['checkpoint_event_count'] < 3 or end['route']!=route or
            end['prior_provider_requests_reissued']!=0):
        raise ValueError('Cross-process recovery evidence incomplete')
    def check_receipt(obj):
        return (isinstance(obj,dict) and obj.get('receipt_digest')==digest(
            'vl64.remote-large-intention-receipt.v1',
            {k:v for k,v in obj.items() if k!='receipt_digest'}))
    task_receipts=[start['tasks']['paragraph'],end['tasks']['python']]+end['tasks']['agent']['receipts']
    if not all(check_receipt(x) for x in task_receipts):
        raise ValueError('Independently checked task receipt digests differ')
    reported_tokens=[x['usage'].get('completion_tokens') for x in task_receipts]
    tokens=sum(reported_tokens) if all(type(v) is int for v in reported_tokens) else None
    host_wall=sum(x['provider_wall_seconds'] for x in task_receipts)
    cpu=start['interpreter_process_cpu_seconds']+end['interpreter_process_cpu_seconds']
    return {'route':route,'epochs':[start,end],'generated_provider_tokens':tokens,
            'sum_provider_wall_seconds':round(host_wall,6),
            'sum_interpreter_process_cpu_seconds':round(cpu,6),
            'completion_tokens_per_provider_wall_second':round(tokens/host_wall,6) if tokens is not None and host_wall>0 else None,
            'response_shape_passes':sum(x['status']=='OUTPUT_SHAPE_ADMITTED' for x in task_receipts),
            'code_structure_quality_pass':end['tasks']['python']['quality_heuristic_pass'],
            'paragraph_heuristic_pass':start['tasks']['paragraph']['quality_heuristic_pass'],
            'authorized_action_completed':end['tasks']['agent']['status']=='AUTHORIZED_ACTION_COMPLETED',
            'cross_process_checkpoint_verified':True,
            'external_provider_KV_recovered':False}


def run_benchmark(*, live=False, directory, max_tokens=128):
    directory=Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('Fresh benchmark working directory required')
    started=time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(_launch,route,live,directory,max_tokens) for route in ROUTES]
        records=[f.result() for f in futures]
    if len(records)!=2 or {r['route'] for r in records}!=set(ROUTES):
        raise ValueError('Both exact full-scale model Ground routes required')
    report={'schema':SCHEMA,'mode':'LIVE_PROVIDER' if live else 'FIXTURE_ONLY',
        'status':'REAL_PROVIDER_CHAT_BENCHMARK' if live else 'CONTROL_PLANE_FIXTURE_BENCHMARK',
        'fleet_wall_seconds':round(time.perf_counter()-started,6),
        'instances':records,'local_model_weight_bytes':0,
        'evidence_grade':'remote_chat_provider_not_native_model_in_VL',
        'quality_scope':'paragraph_heuristic_and_code_structural_check_not_general_semantic_proof',
        'causal_scope':'VL_transcript_recovery_only_not_provider_KV_restore'}
    report['report_digest']=digest(SCHEMA,report)
    return report


def main(argv=None):
    p=argparse.ArgumentParser()
    modes=p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--fixture',action='store_true')
    modes.add_argument('--live',action='store_true')
    p.add_argument('--worker',action='store_true')
    p.add_argument('--stage',choices=('start','continue'))
    p.add_argument('--route',choices=tuple(ROUTES))
    p.add_argument('--checkpoint')
    p.add_argument('--expected-digest')
    p.add_argument('--max-tokens',type=int,default=128)
    p.add_argument('--out',required=True)
    p.add_argument('--working-dir')
    args=p.parse_args(argv)
    try:
        if args.worker:
            if not all((args.stage,args.route,args.checkpoint)):
                raise ValueError('Unqualified worker invocation')
            _worker(args.stage,args.route,args.live,args.checkpoint,args.out,args.expected_digest,args.max_tokens)
        else:
            if args.working_dir is None:
                raise ValueError('Fresh benchmark working directory required')
            result=run_benchmark(live=args.live,directory=args.working_dir,max_tokens=args.max_tokens)
            _atomic_json(args.out,result)
    except (ValueError,RuntimeError,TimeoutError,OSError,TypeError,KeyError,AssertionError) as exc:
        # Do not leak provider body, credentials or user prompt data into logs.
        result={'schema':SCHEMA,'status':'UNRESOLVED','mode':'LIVE_PROVIDER' if args.live else 'FIXTURE_ONLY',
                'error_type':type(exc).__name__, 'no_real_inference_claim':True,
                'unresolved_route':args.route if args.worker else 'fleet',
                'unresolved_stage':args.stage if args.worker else 'one_or_more_model_appointments'}
        _atomic_json(args.out,result)
        return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
