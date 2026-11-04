from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import torch
from lm_eval.models.huggingface import HFLM
from . import runtime
def schedule_digest(events):


    normalized=[(stage,q,kv,{kind:{int(layer):rho for layer,rho in mapping.items()}
        for kind,mapping in mask.items()}) for stage,q,kv,mask in events]
    return hashlib.sha256(json.dumps(normalized,sort_keys=True).encode()).hexdigest()

class PairedHFLM(HFLM):
    def __init__(self,*args,configurations,record_path,compare_direct=True,**kwargs):
        super().__init__(*args,**kwargs)
        self.e=runtime
        self.configurations=configurations
        self.record_path=Path(record_path)
        self.variant='compensated'
        self.compare_direct=compare_direct
        self.paired={}
        if self.record_path.exists():
            for line in self.record_path.read_text().splitlines():
                row=json.loads(line)
                assert row['configuration']==self.configurations[row['task']]
                assert 'compensated' in row['outputs']
                events=[(x['stage'],x['q'],x['kv'],x['mask']) for x in row['forward_events']]
                digest=schedule_digest(events)
                assert row['schedule_sha256']['compensated']==digest
                if self.compare_direct or 'direct' in row['outputs']:
                    assert 'direct' in row['outputs'] and row['same_mask']
                    assert len(set(row['schedule_sha256'].values()))==1
                else:
                    assert row['same_mask'] is None
                key=(row['task'],row['doc_id'],row['input_sha256'])
                assert key not in self.paired
                self.paired[key]=row['outputs']

    def _prefix(self,prepared,cfg):
        e=self.e;p,cache,last,end,left=deepcopy(prepared)
        p.cfg=cfg;p.freeze('prefix')
        return p,cache,last,end

    def _scores(self,prefix,continuations,prepared,cfg):
        e=self.e;p,cache,last,end=self._prefix(prepared,cfg)
        with e.install(self.model,p):
            for start in range(end,len(prefix),cfg['prefill_chunk_tokens']):
                stop=min(len(prefix),start+cfg['prefill_chunk_tokens'])
                out=e.forward(self.model,prefix[start:stop],cache,start,p,'prefill',heads=1 if stop==len(prefix) else 0)
                if out is not None:last=out[-1]
        assert last is not None
        scores=[];candidate_events=[]
        for suffix in continuations:
            pp,cc=deepcopy(p),deepcopy(cache)
            before=len(pp.events)
            logits=[last[None]]
            if len(suffix)>1:
                with e.install(self.model,pp):
                    logits.append(e.forward(self.model,suffix[:-1],cc,len(prefix),pp,'candidate'))
            logits=torch.cat(logits).float()
            targets=torch.tensor(suffix,device=logits.device)
            lp=logits.log_softmax(-1).gather(1,targets[:,None])[:,0]
            scores.append((float(lp.sum()),bool((logits.argmax(-1)==targets).all())))
            candidate_events.extend(pp.events[before:])
        p.events.extend(candidate_events)
        return scores,p.evidence()

    def loglikelihood(self,requests,disable_tqdm=False):


        groups=OrderedDict()
        for index,req in enumerate(requests):
            context,suffix=self._encode_pair(*req.args)
            assert context and suffix and len(context)+len(suffix)<=self.max_length+1
            key=(req.task_name,req.doc_id,tuple(context))
            groups.setdefault(key,[]).append((index,req,suffix))
        result=[None]*len(requests)
        self.record_path.parent.mkdir(exist_ok=True,parents=True)
        with torch.inference_mode(),self.record_path.open('a') as log:
            for key,items in groups.items():
                task,doc_id,context=key;prefix=list(context)
                input_hash=hashlib.sha256(json.dumps(prefix).encode()).hexdigest()
                cache_key=(task,doc_id,input_hash)
                if cache_key not in self.paired:
                    cfg=deepcopy(self.configurations[task])
                    start=0

                    if task=='piqa':
                        header=self.tok_encode('Question:')
                        assert prefix[:len(header)]==header
                        start=len(header)
                    prepared=self.e.probe(self.model,prefix,start,cfg)
                    continuations=[x[2] for x in items]
                    outputs={};policies={}
                    routes=[('compensated',True)]+([('direct',False)] if self.compare_direct else [])
                    for name,compensation in routes:
                        route=deepcopy(cfg);route['compensation']=compensation
                        outputs[name],policies[name]=self._scores(prefix,continuations,prepared,route)
                    schedules={name:[(x['stage'],x['q'],x['kv'],x['mask']) for x in policy['events'] if x['event']=='forward'] for name,policy in policies.items()}
                    if self.compare_direct:
                        assert policies['compensated']['prefix_mask']==policies['direct']['prefix_mask']
                        assert schedules['compensated']==schedules['direct']
                    record=dict(task=task,doc_id=doc_id,input_sha256=input_hash,
                        input_tokens=len(prefix),continuation_tokens=[len(x) for x in continuations],configuration=cfg,
                        outputs=outputs,prefix_mask=policies['compensated']['prefix_mask'],same_mask=True if self.compare_direct else None,
                        paired_execution=self.compare_direct,
                        schedule_sha256={name:schedule_digest(events) for name,events in schedules.items()},
                        selection_events=[x for x in policies['compensated']['events'] if x['event']=='selection'],
                        forward_events=[x for x in policies['compensated']['events'] if x['event']=='forward'])
                    self.paired[cache_key]=outputs
                    log.write(json.dumps(record)+'\n');log.flush()
                    if len(self.paired)%25==0:print('PAIRED' if self.compare_direct else 'CALIBRATION',self.variant,task,len(self.paired),flush=True)
                for number,(index,req,suffix) in enumerate(items):
                    result[index]=self.paired[cache_key][self.variant][number]
        assert all(x is not None for x in result)
        return result
