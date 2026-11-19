"""Direct virtual-layer computation over hosted trained operator Grounds."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.model_fabric.operational_ground import digest
from experiments.model_fabric.operational_tensor_graph import (
    HostedMatrixOperators, ProceduralTensorNavigator, compile_operator_graph,
    observe_tensor_ground, OP_RESULT, OP_PLAN,
)

W1=[[.25,-.5,.75],[.3,.1,-.2],[-.1,.4,.6],[.6,-.3,.2]]
W2=[[.5,-.1,.4,.2],[-.2,.3,.1,.7],[.6,.2,-.4,.1]]
MATS={'gated-in':W1,'down-project':W2}
NODES=[{'id':'x','op':'input','width':3},
       {'id':'a','op':'linear','input':'x','operator':'gated-in'},
       {'id':'nonlinear','op':'silu','input':'a'},
       {'id':'y','op':'linear','input':'nonlinear','operator':'down-project'},
       {'id':'result','op':'add','left':'x','right':'y'}]


def setup():
    host=HostedMatrixOperators(model='trained/fixture-mlp',revision='1'*40,
                                observer_host='operator-host-1',matrices=MATS)
    ground=observe_tensor_ground(host,model='trained/fixture-mlp',revision='1'*40,
                                 approved_hosts={'operator-host-1'},
                                 expected_artifact=host.desc['artifact_identity'])
    plan=compile_operator_graph(ground,NODES,'result')
    return host,ground,plan


def oracle(x):
    a=[math.fsum(w*v for w,v in zip(row,x)) for row in W1]
    b=[v/(1+math.exp(-v)) for v in a]
    return [v+math.fsum(w*z for w,z in zip(row,b)) for v,row in zip(x,W2)]


@pytest.mark.parametrize('x',[[1.0,.2,-.4],[.4,.5,-.3],[0.,-1.,2.],[-.3,.0,.7]])
def test_full_procedural_trained_operator_ground(x):
    host,ground,plan=setup()
    nav=ProceduralTensorNavigator(ground,host,approved_hosts={'operator-host-1'})
    got,receipt=nav.execute(plan,x)
    assert got==pytest.approx(oracle(x),abs=1e-6)
    assert receipt['host_weight_bytes_received']==0
    assert receipt['activation_bytes_sent']>0 and receipt['activation_bytes_received']>0
    assert len(receipt['stages'])==len(NODES)
    assert 'gated-in' in ground['operators'] and 'down-project' in ground['operators']
    assert not any('weights' in k for k in ground)
    assert '0.25' not in json.dumps(ground)
    assert len(json.dumps(ground))<1200


def test_logical_operator_work_is_not_a_black_box_model_forward():
    host,ground,plan=setup()
    nav=ProceduralTensorNavigator(ground,host,approved_hosts={'operator-host-1'})
    got,receipt=nav.execute(plan,[.6,.1,.3])
    assert [s['op'] for s in receipt['stages']]==['input','linear','silu','linear','add']
    assert len(host.sessions)==0
    assert receipt['host_calls']==10  # begin, two input tiles, eval/output/close per projection -> 5 each


def test_observed_operator_ground_refuses_wrong_artifact():
    host,ground,plan=setup()
    with pytest.raises(ValueError,match='Ground'):
        observe_tensor_ground(host,model=ground['model'],revision=ground['revision'],
                              approved_hosts={ground['observer_host']},expected_artifact='0'*64)
    with pytest.raises(ValueError,match='observer'):
        ProceduralTensorNavigator(ground,host,approved_hosts={'untrusted'})


def test_reject_invalid_graph_arcs_and_weight_binding():
    host,ground,plan=setup()
    bad=copy.deepcopy(NODES)
    bad[3]['operator']='gated-in'
    with pytest.raises(ValueError,match='shape'):
        compile_operator_graph(ground,bad,'result')
    with pytest.raises(ValueError,match='edge'):
        compile_operator_graph(ground,[{'id':'x','op':'input','width':3},
                                      {'id':'z','op':'silu','input':'z'}], 'z')
    changed=copy.deepcopy(plan);changed['nodes'][1]['operator']='down-project'
    nav=ProceduralTensorNavigator(ground,host,approved_hosts={'operator-host-1'})
    with pytest.raises(ValueError):nav.execute(changed,[1.,2.,3.])


def test_malformed_activated_result_rejected():
    host,ground,plan=setup()
    def tamper(req):
        r=host(req)
        if req['operation']=='output_tile':
            r=copy.deepcopy(r)
            r['result']['start']+=1
            r['result_digest']=digest(OP_RESULT,r['result'])
        return r
    nav=ProceduralTensorNavigator(ground,tamper,approved_hosts={'operator-host-1'})
    with pytest.raises(ValueError,match='Misaddressed'):nav.execute(plan,[1.,2.,3.])


def test_arbitrary_larger_matrix_streamed_with_bounded_frames():
    rows=4100
    matrices={'big':[[((i%17)-8)/10, .75] for i in range(rows)],
              'small':[[1./rows for _ in range(rows)],[-2./rows for _ in range(rows)]]}
    host=HostedMatrixOperators(model='trained/wide-operator',revision='2'*40,
                               observer_host='wide-operator-host',matrices=matrices)
    ground=observe_tensor_ground(host,model=host.desc['model'],revision=host.desc['revision'],
                                 approved_hosts={'wide-operator-host'})
    nodes=[{'id':'x','op':'input','width':2},{'id':'project','op':'linear','input':'x','operator':'big'},
           {'id':'back','op':'linear','input':'project','operator':'small'}]
    plan=compile_operator_graph(ground,nodes,'back')
    nav=ProceduralTensorNavigator(ground,host,approved_hosts={'wide-operator-host'})
    got,receipt=nav.execute(plan,[.2,.4])
    expected=[math.fsum(row[i]*z for row,z in zip(matrices['small'],[])) for i in []]  # no shortcut used
    projected=[r[0]*.2+r[1]*.4 for r in matrices['big']]
    scalar=[math.fsum(w*z for w,z in zip(row,projected)) for row in matrices['small']]
    assert got==pytest.approx(scalar,abs=1e-6)
    assert receipt['activation_bytes_sent'] >= rows*4
    assert receipt['activation_bytes_received'] >= rows*4
    assert receipt['host_weight_bytes_received']==0


def test_operator_ground_portable_across_processes(tmp_path):
    host,ground,plan=setup()
    nav=ProceduralTensorNavigator(ground,host,approved_hosts={'operator-host-1'})
    target,_=nav.execute(plan,[.1,-.2,.3])
    p=tmp_path/'subprocess.json';p.write_text(json.dumps([MATS,ground,plan]))
    script='''
import json,sys
from experiments.model_fabric.operational_tensor_graph import HostedMatrixOperators,ProceduralTensorNavigator
mats,g,plan=json.load(open(sys.argv[1]))
source=HostedMatrixOperators(model=g['model'],revision=g['revision'],observer_host=g['observer_host'],matrices=mats)
nav=ProceduralTensorNavigator(g,source,approved_hosts={g['observer_host']})
print(json.dumps(nav.execute(plan,[.1,-.2,.3])[0]))
'''
    env=dict(os.environ);env['PYTHONPATH']=str(Path(__file__).resolve().parents[1])
    output=subprocess.run([sys.executable,'-c',script,str(p)],env=env,
                          capture_output=True,text=True,check=True)
    assert json.loads(output.stdout)==pytest.approx(target,abs=1e-6)
