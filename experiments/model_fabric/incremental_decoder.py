"""Append a genuine causal token after a completed, recoverable full decoder.

Reuses each layer's already verified K/V prefix instead of recomputing any
old token positions. All new token weights are admitted afresh; every decoded
layer's static weight content must match the externally pinned checkpoint.
Different qualified range-source hosts may supply the same weight bytes.

No sampler, tokenizer or remote processor authorization is implied.
"""
from __future__ import annotations

from vld import domain_digest
from .execution_dag import QualifiedTensorReader
from .pretrained_layer import PreparedLayer, _verify_layer_nodes
from .pretrained_orbit import ContentAddressedPages, _static_weight_identity
from .full_decoder import (
    StreamingDecoder, _verify_checkpoint, _get_hidden, _check_kv,
    _put_kv, _put_hidden, GENESIS, MAX_TOKENS,
)


class IncrementalDecoder(StreamingDecoder):
    """Cross-process single-writer next-token state extension.

    Construction requires an external trusted checkpoint head and higher epoch.
    `append_token` is atomic at the logical prefix boundary: no newly produced
    position is visible until all 30 layers and their KV pages are admitted.
    """
    def __init__(self, plan, previous, *, expected_head, host_for,
                 approved_for, store, owner, epoch):
        if (not isinstance(store,ContentAddressedPages) or
                not callable(host_for) or not callable(approved_for) or
                type(epoch) is not int or not isinstance(owner,str) or not owner or
                not isinstance(previous,dict) or epoch<=previous.get('epoch',-1)):
            raise ValueError('A newer authorized compute epoch is required')
        tokens=previous.get('tokens',[])
        completed=_verify_checkpoint(plan,previous,
            expected_head=expected_head,tokens=tokens)
        arch=plan['architecture']
        if completed!=arch['layers'] or len(tokens)>=min(MAX_TOKENS,arch['context']):
            raise ValueError('Only a completed causal model prefix can be extended')
        self.plan=plan
        self.store=store
        self.owner,self.epoch=owner,epoch
        self.host_for,self.approved_for=host_for,approved_for
        self.completed_layers=completed
        self.tokens=list(tokens)
        self.hidden=_get_hidden(store,previous['hidden'],len(tokens),arch['hidden'])
        self.hidden_manifest=previous['hidden']
        self.kv_manifests=list(previous['kv_manifests'])
        # Validate every previous exact KV byte page before any new range read.
        for index,manifest in enumerate(self.kv_manifests):
            _check_kv(store,plan,manifest,index,len(tokens))
        self.weight_identities=dict(previous['weight_identities'])
        self.range_witnesses=dict(previous['range_witnesses'])
        self.bytes_acquired=previous['bytes_acquired']
        self.ranges_acquired=previous['ranges_acquired']
        self.previous_checkpoint=previous['checkpoint_digest']
        self.transcript_head=previous['transcript_head']

    def append_token(self,token):
        arch=self.plan['architecture']
        if (type(token) is not int or not 0<=token<arch['vocab'] or
                len(self.tokens)>=min(MAX_TOKENS,arch['context'])):
            raise ValueError('Invalid next causal token position')
        prior_count=len(self.tokens)
        embed_reader=QualifiedTensorReader(self.plan,self.host_for(-1),
                                           approved_hosts=self.approved_for(-1))
        state=embed_reader.embedding(token)
        new_kv=[]
        new_range_chains={}
        new_byte_count=embed_reader.bytes_read
        new_range_count=embed_reader.ranges
        for index in range(arch['layers']):
            _verify_layer_nodes(self.plan,index)
            reader=QualifiedTensorReader(self.plan,self.host_for(index),
                                         approved_hosts=self.approved_for(index))
            layer=PreparedLayer(reader,index)
            weight_identity=_static_weight_identity(self.plan,layer)
            if weight_identity!=self.weight_identities[str(index)]:
                raise ValueError('Continued model changed pinned static pretrained weights')
            old_k,old_v=_check_kv(self.store,self.plan,
                                 self.kv_manifests[index],index,prior_count)
            d=arch['head_dim']
            kvh=arch['kv_heads']
            layer.keys=[[frame[j*d:(j+1)*d] for j in range(kvh)] for frame in old_k]
            layer.values=[[frame[j*d:(j+1)*d] for j in range(kvh)] for frame in old_v]
            layer.step=prior_count
            state,_=layer.next(state,trace=False)
            new_kv.append(_put_kv(self.store,self.plan,layer))
            new_range_chains[str(index)]=reader.receipt_chain
            new_byte_count+=reader.bytes_read
            new_range_count+=reader.ranges
        new_hidden=list(self.hidden)+[state]
        manifest=_put_hidden(self.store,new_hidden,arch['hidden'])
        new_head=domain_digest('vl64.full-autoregressive-transition.v1',{
            'previous':self.transcript_head,'previous_checkpoint':self.previous_checkpoint,
            'dag_digest':self.plan['dag_digest'],'position':prior_count,
            'token':token,'last_hidden_sha256':manifest['sha256'],
            'kv_manifests':new_kv,'weight_identities':self.weight_identities})
        # Logical commit of the entire new position; a failed tensor host or
        # mathematical stage never mutates a previously committed prefix.
        self.hidden=new_hidden
        self.hidden_manifest=manifest
        self.tokens.append(token)
        self.kv_manifests=new_kv
        self.range_witnesses.update(new_range_chains)
        self.range_witnesses['-1']=embed_reader.receipt_chain
        self.bytes_acquired+=new_byte_count
        self.ranges_acquired+=new_range_count
        self.transcript_head=new_head
        return {'position':prior_count,'tokens':list(self.tokens),
                'hidden_sha256':manifest['sha256'],
                'transcript_head':new_head,'causal_KV_layers':len(new_kv),
                'new_weight_bytes_acquired':new_byte_count,
                'new_exact_ranges':new_range_count}
