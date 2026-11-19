"""Real process boundaries and negative controls; fixture != model inference."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from experiments.model_fabric.operational_ground import digest
from experiments.model_fabric.remote_large_fleet import _python_task_quality, ROUTES
from experiments.model_fabric.remote_large_recovery_benchmark import (
    SCHEMA, WORKER_SCHEMA, run_benchmark, _worker,
)


def test_two_full_scale_routes_two_fresh_interpreters_per_route(tmp_path):
    report=run_benchmark(directory=tmp_path/'fresh',max_tokens=64)
    assert report['status']=='CONTROL_PLANE_FIXTURE_BENCHMARK'
    assert report['mode']=='FIXTURE_ONLY'
    assert report['report_digest']==digest(SCHEMA,{k:v for k,v in report.items() if k!='report_digest'})
    assert set(x['route'] for x in report['instances'])==set(ROUTES)
    assert report['local_model_weight_bytes']==0
    for instance in report['instances']:
        assert instance['cross_process_checkpoint_verified']
        assert instance['external_provider_KV_recovered'] is False
        assert instance['generated_provider_tokens']==44
        assert instance['authorized_action_completed']
        assert instance['paragraph_heuristic_pass']
        assert instance['code_structure_quality_pass']
        assert instance['sum_interpreter_process_cpu_seconds']>=0
        first,second=instance['epochs']
        assert first['epoch']==0 and second['epoch']==1
        assert first['checkpoint_event_count']==1 and second['checkpoint_event_count']==4
        assert first['recovered_prior_receipts']==0
        assert second['recovered_prior_receipts']==1
        assert second['prior_provider_requests_reissued']==0
        for step in (first,second):
            assert step['route']==instance['route']
            assert step['result_digest']==digest(WORKER_SCHEMA,{k:v for k,v in step.items() if k!='result_digest'})
    assert (tmp_path/'fresh'/'qwen-27b'/'epoch-0.json').exists()
    assert (tmp_path/'fresh'/'deepseek-v41-flash'/'epoch-1.json').exists()


def test_allows_workload_progress_even_if_tool_not_authorized(tmp_path,monkeypatch):
    # Never convert absence of a confirmed model action into a valid agent completion.
    from experiments.model_fabric import remote_large_recovery_benchmark as benchmark
    from experiments.model_fabric.remote_large_fleet import RemoteLargeSession
    def rejected(session,*,max_tokens):
        from experiments.model_fabric.remote_large_fleet import intention,TASKS
        spec=intention(session.ground,'tool_json',[{'role':'user','content':TASKS['tool_json']['prompt']}],max_tokens=max_tokens)
        first=session.run(spec)
        return {'status':'ACTION_UNRESOLVED','receipts':[first]}
    monkeypatch.setattr(benchmark,'agent_action',rejected)
    cp=tmp_path/'cp.json'
    benchmark._worker('start','qwen-27b',False,cp,tmp_path/'first.json')
    first=json.loads((tmp_path/'first.json').read_text())
    answer=benchmark._worker('continue','qwen-27b',False,cp,tmp_path/'second.json',first['checkpoint_digest'])
    assert answer['checkpoint_event_count']==3
    assert answer['tasks']['agent']['status']=='ACTION_UNRESOLVED'


def test_wrong_model_checkpoint_head_or_stale_epoch_refused(tmp_path):
    cp=tmp_path/'cp.json'
    result=_worker('start','qwen-27b',False,cp,tmp_path/'first.json')
    with pytest.raises(ValueError):
        _worker('continue','deepseek-v41-flash',False,cp,tmp_path/'mismatch.json',result['checkpoint_digest'])
    with pytest.raises(ValueError):
        _worker('continue','qwen-27b',False,cp,tmp_path/'bad.json','0'*64)
    with pytest.raises(ValueError):
        _worker('start','qwen-27b',False,cp,tmp_path/'stale.json')


def test_malicious_model_code_does_not_pass_as_correct_safe_divide():
    good='def safe_divide(a,b):\n if b==0: return None\n return a / b\n'
    assert _python_task_quality(good)
    assert _python_task_quality('def safe_divide(a,b):\n if b != 0: return a/b\n return None')
    invalid=[
        'def safe_divide(a,b):\n if a == 0: return None\n return a / b',
        'def safe_divide(a,b):\n if b == 0: return None\n return b / a',
        'def safe_divide(a,b):\n if b == 1: return None\n return a / b',
        'def safe_divide(a,b):\n if b == 0: return None\n print("bad")\n return a / b',
        'def safe_divide(a,b):\n if b == 0: return None\n return 0',
        'def safe_divide(a,b):\n return a / b',
    ]
    assert all(not _python_task_quality(code) for code in invalid)


def test_main_cli_fails_closed_without_provider_secrets(tmp_path,monkeypatch):
    env=os.environ.copy();env.pop('HF_TOKEN',None);env.pop('DEEPSEEK_API_KEY',None)
    out=tmp_path/'live.json'
    p=subprocess.run([sys.executable,'-m','experiments.model_fabric.remote_large_recovery_benchmark',
        '--live','--working-dir',str(tmp_path/'work'),'--out',str(out)],env=env,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=30)
    assert p.returncode==1
    result=json.loads(out.read_text())
    assert result['status']=='UNRESOLVED' and result['no_real_inference_claim'] is True
    assert not result.get('instances')


def test_incorrect_reuse_of_nonempty_workdir_fails_closed(tmp_path):
    work=tmp_path/'work';work.mkdir();(work/'marker').write_text('preexisting')
    with pytest.raises(ValueError):run_benchmark(directory=work)
