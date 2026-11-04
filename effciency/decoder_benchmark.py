from pathlib import Path
import argparse,hashlib,json,math
import torch,crypten
import crypten.communicator as comm
from crypten.config import cfg
from .timing import measured,configure_cuda_integer_encoding

from . import secure_ops as cost
CONFIG=Path(__file__).resolve().parent/'configs'


def settings(model,smoke):
    c=json.loads((CONFIG/f'{model}.json').read_text())
    if smoke:c.update(hidden_size=32,num_attention_heads=4,num_key_value_heads=2,
        intermediate_size=64,num_hidden_layers=2,vocab_size=64,head_dim=8)
    return c


def operation(label,function,ledger):
    value,record=measured(function)
    ledger[label]=record
    return value


@crypten.mpc.run_multiprocess(world_size=2)
def execute(model,variant,backend,prompt,repeats,smoke):
    rank=comm.get().get_rank();torch.set_num_threads(4)
    device=torch.device('cuda:0' if backend=='cuda' else 'cpu')
    if backend=='cuda':
        torch.cuda.set_device(device)
        configure_cuda_integer_encoding()
    cfg.communicator.verbose=True;cfg.encoder.precision_bits=20
    cost.measured=operation
    torch.manual_seed(20261028)
    c=settings(model,smoke)
    w,heads,kv,mid=(c[k] for k in ['hidden_size','num_attention_heads','num_key_value_heads','intermediate_size'])
    dims=(w,heads,kv,mid);chunk=min(4096,c['vocab_size'])
    shapes={'q':(w,w),'k':(w,kv*(w//heads)),'v':(w,kv*(w//heads)),
        'o':(w,w),'gate':(w,mid),'up':(w,mid),'down':(mid,w)}
    weights={name:cost.share(shape,rank,device,scale=1/math.sqrt(shape[0])) for name,shape in shapes.items()}
    for name in ['norm1','norm2','final_norm']:
        weights[name]=crypten.cryptensor(torch.ones(w,device=device),src=1)
    embedding=cost.share((chunk,w),rank,device,scale=1.)
    head=cost.share((w,chunk),rank,device,scale=1/math.sqrt(w))
    print('PARAMETER_BUFFERS_READY',rank,model,variant,flush=True)
    reports=[]
    with crypten.no_grad():
        for repetition in range(repeats):
            caches=[None]*c['num_hidden_layers']
            for phase,length in [('prefill',prompt),('decode',1)]:
                ids=torch.randint(c['vocab_size'],(1,length),device=device)
                onehot=crypten.cryptensor(torch.nn.functional.one_hot(ids,c['vocab_size']).float(),src=0)
                ledger={}
                def forward():
                    hidden=None
                    for start in range(0,c['vocab_size'],chunk):
                        stop=min(start+chunk,c['vocab_size'])
                        value=operation(f'embedding_{start}',lambda:onehot[...,start:stop].matmul(embedding[:stop-start]),ledger)
                        hidden=value if hidden is None else hidden+value
                    for layer in range(c['num_hidden_layers']):
                        layer_ops={}
                        hidden,caches[layer]=cost.block(hidden,weights,caches[layer],{'puma':'puma_nomax','spark':'spark_nomax'}.get(variant,variant),dims,layer_ops,c)
                        ledger.update({f'layer{layer}.{name}':value for name,value in layer_ops.items()})
                        if (layer+1)%4==0:print('BLOCKS_COMPLETE',rank,repetition,phase,layer+1,flush=True)
                    hidden=operation('final_norm',lambda:cost.norm(hidden[:,-1:,:],weights['final_norm'],c['rms_norm_eps']),ledger)
                    pieces=[]
                    for start in range(0,c['vocab_size'],chunk):
                        stop=min(start+chunk,c['vocab_size'])
                        pieces.append(operation(f'head_{start}',lambda:hidden.matmul(head[:,:stop-start]),ledger))
                    logits=crypten.cat(pieces,dim=-1)
                    return operation('client_output',lambda:logits.get_plain_text(dst=0),ledger)
                output,total=measured(forward)
                assert sum(x['communication_bytes'] for x in ledger.values())==total['communication_bytes']
                assert sum(x['communication_rounds'] for x in ledger.values())==total['communication_rounds']
                if rank==0:assert torch.isfinite(output).all()
                reports.append(dict(repetition=repetition,phase=phase,query_tokens=length,
                    key_tokens=prompt if phase=='prefill' else prompt+1,total=total,operations=ledger))
                print('PHASE_COMPLETE',rank,repetition,phase,total,flush=True)
    return dict(rank=rank,device=str(device),phases=reports)


def main(a):
    prompt=8 if a.smoke else a.prompt_tokens
    parts=execute(a.model,a.variant,a.device,prompt,a.repeats,a.smoke)
    assert parts is not None and len(parts)==2
    checks=dict(two_parties=True,matching_bytes=all(x['total']['communication_bytes']==y['total']['communication_bytes'] for x,y in zip(parts[0]['phases'],parts[1]['phases'])),
        matching_rounds=all(x['total']['communication_rounds']==y['total']['communication_rounds'] for x,y in zip(parts[0]['phases'],parts[1]['phases'])))
    assert all(checks.values())
    result=dict(model=a.model,variant=a.variant,smoke=a.smoke,backend=a.device,config=settings(a.model,a.smoke),
        prompt_tokens=prompt,repeats=a.repeats,fractional_bits=20,parts=parts,checks=checks,
        scope='complete static online circuit: private one-hot embedding, all decoder blocks, final normalization, full vocabulary head and client output',
        cuda_encoding='upstream FP64 limbs (process-local restoration)',
        parameter_storage='shared parameter buffers reused across layer and vocabulary groups; shape-based efficiency fixture',
        network_profile=a.network_profile,gpu_count=1 if a.device=='cuda' else 0,
        softmax_uses_max=a.variant=='crypten_tanh',quality_claim=False,crypten=crypten.__version__,torch=torch.__version__,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(cost.__file__),Path(__file__).with_name('timing.py'),CONFIG/f'{a.model}.json']})
    target=a.output
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2)+'\n')
    print('COMPLETE_CIRCUIT_TIMING_PASS',target,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['llama','mistral'],default='llama')
    p.add_argument('--variant',choices=['crypten_tanh','puma','spark'],default='spark')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--prompt-tokens',type=int,default=512)
    p.add_argument('--network-profile',choices=['local','lan','wan'],default='local',help='Label an independently configured measurement network; this option does not configure it.')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=3);p.add_argument('--smoke',action='store_true');main(p.parse_args())
