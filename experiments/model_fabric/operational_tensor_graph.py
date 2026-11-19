"""Model-independent *procedural* tensor Ground over host-resident weights.

The VL interpreter sequences trained linear projections and nonlinear operators.
No weight entry is transferred to VL; only typed F32 activation windows cross
its observer-host boundary. Arbitrarily large row/column dimensions can be
addressed using bounded input/output tiles; host memory and CPU/GPU do the
parameter-dependent products. Receipts are evidence of reported operation,
not proof of faithful execution by a malicious host.
"""
from __future__ import annotations

import hashlib
import math
import struct

from .operational_ground import digest, MAX_LOGITS_PER_TILE, MAX_FRAME_BYTES

OP_GROUND = 'vl64.procedural-tensor-ground.v1'
OP_PLAN = 'vl64.procedural-tensor-graph.v1'
OP_REQ = 'vl64.procedural-tensor-request.v1'
OP_RESULT = 'vl64.procedural-tensor-result.v1'


def _frame(values):
    if not values or any(not math.isfinite(x) for x in values):
        raise ValueError('Unadmitted tensor values')
    return struct.pack('<'+'f'*len(values),*values)


def _from_frame(raw,count):
    if not isinstance(raw,bytes) or len(raw)!=count*4:
        raise ValueError('Incomplete model activation frame')
    out=[x[0] for x in struct.iter_unpack('<f',raw)]
    if not all(math.isfinite(x) for x in out):
        raise ValueError('Nonfinite activation frame')
    return out


def observe_tensor_ground(host,*,model,revision,approved_hosts,expected_artifact=None):
    info=host({'schema':OP_REQ,'operation':'describe'})
    if not isinstance(info,dict):raise ValueError('Missing procedural tensor host')
    needed={'schema','model','revision','observer_host','artifact_identity','operators','scope'}
    if (set(info)!=needed or info['schema']!=OP_GROUND or
            info['model']!=model or info['revision']!=revision or
            info['observer_host'] not in approved_hosts or
            (expected_artifact is not None and info['artifact_identity']!=expected_artifact) or
            info['scope']!='host_resident_tensor_operators_not_weight_payloads'):
        raise ValueError('Unqualified procedural tensor Ground')
    if (not isinstance(info['operators'],dict) or not info['operators'] or
            len(info['operators'])>100000):
        raise ValueError('Incomplete host operator catalogue')
    for name,op in info['operators'].items():
        if (not isinstance(name,str) or not 1<=len(name)<=256 or
                not isinstance(op,dict) or set(op)!={'kind','shape','sha256'} or
                op['kind']!='matvec' or not isinstance(op['shape'],list) or
                len(op['shape'])!=2 or any(type(v) is not int or not 1<=v<=1_000_000
                                    for v in op['shape']) or
                not isinstance(op['sha256'],str) or len(op['sha256'])!=64):
            raise ValueError('Unadmitted procedural operator')
    info['ground_digest']=digest(OP_GROUND,info)
    return info


def compile_operator_graph(ground,nodes,output):
    if (not isinstance(ground,dict) or
            ground.get('ground_digest') != digest(OP_GROUND,
                {k:v for k,v in ground.items() if k!='ground_digest'}) or
            not isinstance(nodes,list) or not 2<=len(nodes)<=512):
        raise ValueError('Unqualified model operator graph')
    known={}
    for idx,node in enumerate(nodes):
        if (not isinstance(node,dict) or
                not isinstance(node.get('id'),str) or
                not 1<=len(node['id'])<=128 or node['id'] in known or
                node.get('op') not in ('input','linear','silu','add','multiply')):
            raise ValueError('Duplicate/unknown graph node')
        op=node['op']; keys=set(node)
        if op=='input':
            if idx!=0 or keys!={'id','op','width'} or type(node['width']) is not int or not 1<=node['width']<=1_000_000:
                raise ValueError('Graph has no valid first input')
            size=node['width']
        elif op=='linear':
            if keys!={'id','op','input','operator'} or node['input'] not in known:
                raise ValueError('Unadmitted linear edge')
            mat=ground['operators'].get(node['operator'])
            if not mat or mat['shape'][1]!=known[node['input']]:
                raise ValueError('Tensor projection shape differs from graph edge')
            size=mat['shape'][0]
        elif op=='silu':
            if keys!={'id','op','input'} or node['input'] not in known:
                raise ValueError('Unadmitted nonlinear graph edge')
            size=known[node['input']]
        else:
            if keys!={'id','op','left','right'} or node['left'] not in known or node['right'] not in known or known[node['left']]!=known[node['right']]:
                raise ValueError('Unadmitted binary graph edge')
            size=known[node['left']]
        known[node['id']]=size
    if output not in known:raise ValueError('Missing graph output')
    plan={'schema':OP_PLAN,'ground_digest':ground['ground_digest'],
          'nodes':nodes,'output':output,'output_width':known[output]}
    plan['plan_digest']=digest(OP_PLAN,plan)
    return plan


class ProceduralTensorNavigator:
    """Interpret admitted operator DAG by communicating activation tiles."""
    def __init__(self,ground,host,*,approved_hosts):
        if not isinstance(ground,dict) or ground.get('ground_digest')!=digest(OP_GROUND,
                {k:v for k,v in ground.items() if k!='ground_digest'}):
            raise ValueError('Unverified procedural operator Ground')
        if not callable(host) or ground['observer_host'] not in approved_hosts:
            raise ValueError('Unqualified tensor observer')
        self.ground,self.host=ground,host
        self.remote_calls=0
        self.remote_weight_bytes=0
        self.activation_bytes_sent=0
        self.activation_bytes_received=0
        self.transcript='0'*64

    def _call(self,operation,arguments):
        req={'schema':OP_REQ,'operation':operation,'model':self.ground['model'],
             'revision':self.ground['revision'],'ground_digest':self.ground['ground_digest'],
             'arguments':arguments}
        reply=self.host(req)
        if (not isinstance(reply,dict) or reply.get('schema')!=OP_RESULT or
            reply.get('operation')!=operation or
            reply.get('ground_digest')!=self.ground['ground_digest'] or
            reply.get('model')!=self.ground['model'] or
            reply.get('revision')!=self.ground['revision'] or
            reply.get('observer_host')!=self.ground['observer_host'] or
            reply.get('status')!='HOST_OPERATOR_COMPLETED' or
            not isinstance(reply.get('result'),dict) or
            reply.get('result_digest')!=digest(OP_RESULT,reply['result'])):
            raise ValueError('Unqualified procedural operator receipt')
        self.remote_calls+=1
        self.transcript=digest('vl64.procedural-operator-transcript.v1',
             {'previous':self.transcript,'request':req,
              'result_digest':reply['result_digest']})
        return reply['result']

    def linear(self,name,x,*,tile=1024):
        entry=self.ground['operators'].get(name)
        if entry is None or len(x)!=entry['shape'][1] or not 1<=tile<=MAX_LOGITS_PER_TILE:
            raise ValueError('Unadmitted procedural projection dimensions')
        if not all(math.isfinite(z) for z in x):raise ValueError('Nonfinite operator input')
        # Canonical FP32 bytes define a reproducible transfer identity.
        packed=_frame(x)
        opened=self._call('begin',{'operator':name,'input_f32_sha256':hashlib.sha256(packed).hexdigest(),
                                   'input_count':len(x)})
        sid=opened.get('session_id')
        if not isinstance(sid,str) or not sid:raise ValueError('No operator input session')
        for start in range(0,len(x),tile):
            raw=packed[start*4:(start+tile)*4]
            result=self._call('input_tile',{'session_id':sid,'start':start,
                'count':len(raw)//4,'f32le_hex':raw.hex(),
                'sha256':hashlib.sha256(raw).hexdigest()})
            if result.get('accepted_end')!=start+len(raw)//4:
                raise ValueError('Incomplete procedural input partition')
            self.activation_bytes_sent+=len(raw)
        ready=self._call('evaluate',{'session_id':sid})
        if (not isinstance(ready.get('output_sha256'),str) or
                len(ready['output_sha256'])!=64 or
                ready.get('rows')!=entry['shape'][0]):
            raise ValueError('Incomplete procedural numeric execution')
        sha=hashlib.sha256();out=[]
        for start in range(0,entry['shape'][0],tile):
            count=min(tile,entry['shape'][0]-start)
            result=self._call('output_tile',{'session_id':sid,'start':start,'count':count})
            if result.get('start')!=start or result.get('count')!=count or result.get('output_sha256')!=ready['output_sha256']:
                raise ValueError('Misaddressed operator output')
            payload=result.get('f32le_hex')
            if not isinstance(payload,str) or len(payload)!=8*count:
                raise ValueError('Incomplete procedural activation bytes')
            try:raw=bytes.fromhex(payload)
            except ValueError as exc:raise ValueError('Malformed output bytes') from exc
            if hashlib.sha256(raw).hexdigest()!=result.get('sha256'):
                raise ValueError('Modified operator output frame')
            sha.update(raw)
            out.extend(_from_frame(raw,count))
            self.activation_bytes_received+=len(raw)
        if sha.hexdigest()!=ready['output_sha256']:
            raise ValueError('Whole operator output differs from tiles')
        self._call('close',{'session_id':sid})
        return out

    def execute(self,plan,hidden):
        if (not isinstance(plan,dict) or plan.get('schema')!=OP_PLAN or
                plan.get('ground_digest')!=self.ground['ground_digest'] or
                plan.get('plan_digest')!=digest(OP_PLAN,
                    {k:v for k,v in plan.items() if k!='plan_digest'})):
            raise ValueError('Unadmitted procedural execution plan')
        expected=compile_operator_graph(self.ground,plan['nodes'],plan['output'])
        if expected!=plan:raise ValueError('Unadmitted graph stage or tensor binding')
        if len(hidden)!=plan['nodes'][0]['width']:
            raise ValueError('Procedural input vector width differs')
        values={};trace=[]
        for node in plan['nodes']:
            op=node['op']
            if op=='input':out=list(hidden)
            elif op=='linear':out=self.linear(node['operator'],values[node['input']])
            elif op=='silu':
                out=[z/(1+math.exp(-z)) if z>=0 else z*math.exp(z)/(1+math.exp(z))
                     for z in values[node['input']]]
            elif op=='add':out=[a+b for a,b in zip(values[node['left']],values[node['right']])]
            else:out=[a*b for a,b in zip(values[node['left']],values[node['right']])]
            if not out or not all(math.isfinite(z) for z in out):
                raise ValueError('Nonfinite procedural graph result')
            values[node['id']]=out
            trace.append({'node':node['id'],'op':op,
                 'output_f32_sha256':hashlib.sha256(_frame(out)).hexdigest()})
        final=values[plan['output']]
        return final,{'status':'HOST_RESIDENT_OPERATOR_GRAPH_EXECUTED',
                     'model':self.ground['model'],'revision':self.ground['revision'],
                     'ground_digest':self.ground['ground_digest'],
                     'plan_digest':plan['plan_digest'],'stages':trace,
                     'host_calls':self.remote_calls,
                     'host_weight_bytes_received':self.remote_weight_bytes,
                     'activation_bytes_sent':self.activation_bytes_sent,
                     'activation_bytes_received':self.activation_bytes_received,
                     'transcript_head':self.transcript,
                     'scope':'VL_chose_graph_and_nonlinear_math; host_performed_trained_weight_projections'}


class HostedMatrixOperators:
    """Reference host for procedural graphs. Stores weights, never sends them."""
    def __init__(self,*,model,revision,observer_host,matrices):
        if not isinstance(matrices,dict) or not matrices:
            raise ValueError('Need hosted trained operators')
        self.matrices={};description={}
        for name,rows in matrices.items():
            if (not isinstance(name,str) or not rows or
                    any(not isinstance(row,list) or len(row)!=len(rows[0]) for row in rows)):
                raise ValueError('Invalid hosted matrix')
            if not rows[0]:raise ValueError('Empty operator input')
            raw=_frame([v for r in rows for v in r])
            description[name]={'kind':'matvec','shape':[len(rows),len(rows[0])],
                               'sha256':hashlib.sha256(raw).hexdigest()}
            self.matrices[name]=rows
        self.desc={'schema':OP_GROUND,'model':model,'revision':revision,
                   'observer_host':observer_host,
                   'artifact_identity':digest('vl64.host-matrix-catalogue.v1',description),
                   'operators':description,
                   'scope':'host_resident_tensor_operators_not_weight_payloads'}
        self.ground_digest=digest(OP_GROUND,self.desc)
        self.sessions={}

    def __call__(self,req):
        if not isinstance(req,dict) or req.get('schema')!=OP_REQ:
            raise ValueError('Wrong operator communication protocol')
        action=req.get('operation')
        if action=='describe':return dict(self.desc)
        if (req.get('model')!=self.desc['model'] or
                req.get('revision')!=self.desc['revision'] or
                req.get('ground_digest')!=self.ground_digest or
                not isinstance(req.get('arguments'),dict)):
            raise ValueError('Unbound hosted tensor request')
        arg=req['arguments']
        if action=='begin':
            if set(arg)!={'operator','input_f32_sha256','input_count'} or arg['operator'] not in self.matrices:
                raise ValueError('Unadmitted operator begin')
            cols=len(self.matrices[arg['operator']][0]);count=arg['input_count']
            if type(count) is not int or count!=cols:
                raise ValueError('Unadmitted operator width')
            sid=digest('vl64.procedural-operator-session.v1',arg)
            self.sessions[sid]={'name':arg['operator'],'expected':arg['input_f32_sha256'],
                                'cols':cols,'cursor':0,'input':bytearray(),'output':None}
            result={'session_id':sid}
        else:
            sid=arg.get('session_id');state=self.sessions.get(sid)
            if state is None:raise ValueError('Missing hosted operator state')
            if action=='input_tile':
                if set(arg)!={'session_id','start','count','f32le_hex','sha256'}:
                    raise ValueError('Malformed input tile')
                raw=bytes.fromhex(arg['f32le_hex'])
                if (len(raw)!=arg['count']*4 or not 1<=arg['count']<=MAX_LOGITS_PER_TILE or
                        arg['start']!=state['cursor'] or
                        hashlib.sha256(raw).hexdigest()!=arg['sha256']):
                    raise ValueError('Unqualified input range')
                state['input'].extend(raw);state['cursor']+=arg['count']
                result={'accepted_end':state['cursor']}
            elif action=='evaluate':
                if set(arg)!={'session_id'} or state['cursor']!=state['cols'] or\
                        hashlib.sha256(state['input']).hexdigest()!=state['expected']:
                    raise ValueError('Incomplete operator input')
                x=_from_frame(bytes(state['input']),state['cols'])
                output=[math.fsum(a*b for a,b in zip(row,x)) for row in self.matrices[state['name']]]
                state['output']=_frame(output)
                result={'rows':len(output),'output_sha256':hashlib.sha256(state['output']).hexdigest()}
            elif action=='output_tile':
                if set(arg)!={'session_id','start','count'} or state['output'] is None:
                    raise ValueError('No computed trained operator output')
                first,n=arg['start'],arg['count']
                if type(first) is not int or type(n) is not int:
                    raise ValueError('Bad output tile')
                if first<0 or n<1 or n>MAX_LOGITS_PER_TILE or (first+n)*4>len(state['output']):
                    raise ValueError('Out-of-bounds model output tile')
                raw=state['output'][first*4:(first+n)*4]
                result={'start':first,'count':n,'f32le_hex':raw.hex(),
                        'sha256':hashlib.sha256(raw).hexdigest(),
                        'output_sha256':hashlib.sha256(state['output']).hexdigest()}
            elif action=='close':
                if set(arg)!={'session_id'}:raise ValueError('Unknown close request')
                self.sessions.pop(sid)
                result={'closed':True}
            else:raise ValueError('Unsupported hosted tensor operation')
        return {'schema':OP_RESULT,'operation':action,
               'model':self.desc['model'],'revision':self.desc['revision'],
               'ground_digest':self.ground_digest,'observer_host':self.desc['observer_host'],
               'status':'HOST_OPERATOR_COMPLETED','result':result,
               'result_digest':digest(OP_RESULT,result)}
