import argparse
import json
from pathlib import Path
import time
import torch
import crypten
import crypten.communicator as comm
from crypten.config import cfg
from .controller import secret
from .masked_scale import Engine, prepare


def measured(function):
    before=comm.get().get_communication_stats().copy()
    tick=time.perf_counter()
    value=function()
    after=comm.get().get_communication_stats().copy()
    return value,dict(bytes=after['bytes']-before['bytes'],rounds=after['rounds']-before['rounds'],seconds=time.perf_counter()-tick)


@crypten.mpc.run_multiprocess(world_size=2)
def worker(tokens,width,inner,repeats,rho,coefficients,output):
    torch.set_num_threads(2)
    cfg.communicator.verbose=True
    rank=comm.get().get_rank()
    rows=[]
    torch.manual_seed(20261030)
    cfg.encoder.precision_bits=20
    scales=torch.tensor(coefficients)[:,None,None]
    for q in tokens:
        plain=torch.randn(len(coefficients),q,width)*.1
        x,sharing=measured(lambda:crypten.cryptensor(plain,src=0))
        for repeat in range(repeats+1):
            y,record=measured(lambda:x*scales)
            error=float((y.get_plain_text()-plain*scales).abs().max())
            assert error<1e-5,error
            assert record['bytes']==record['rounds']==0
            rows.append(dict(backend='native',method='public_multiply',tokens=q,width=width,rank=rank,
                repeat=repeat,warmup=repeat==0,max_error=error,sharing=sharing,**record))
    cfg.encoder.precision_bits=0
    provider=crypten.mpc.get_default_provider()
    Q=1<<16
    R=round(Q*rho)
    I=round(Q/rho)
    for q in tokens:
        h=torch.randint(-6000,6001,(q,width))
        x=torch.randint(-512,513,(q,inner))
        w=torch.randint(-512,513,(inner,width))
        raw=x@w
        assert max(int(raw.abs().max())*max(Q,I),int(h.abs().max())*R)<1<<63
        (sh,sx,sw),sharing=measured(lambda:(secret(h),secret(x),crypten.cryptensor(w,src=1,precision=0)))
        for method in ('eager','fused','materialized'):
            requests=[]
            for repeat in range(repeats+1):
                provider.trace(False)
                provider.tuple_cache={}
                mode='M1' if method=='eager' else 'M2'
                (materials,extra),mask_preparation=measured(lambda:(prepare(h.shape,mode),prepare(h.shape,'M1')['compensation'] if method=='materialized' else None))
                provider.tuple_cache={}
                if repeat==0:
                    provider.request_cache=[]
                    provider.trace(True)
                    preprocessing=None
                else:
                    provider.request_cache=list(requests)
                    _,preprocessing=measured(provider.fill_cache)
                engine=Engine(mode,materials,f'{q}:{method}:{repeat}')
                def execute():
                    state=engine.wire(sh)
                    skipped=engine.skip(state,rho)
                    projected=engine.projection(sx,sw,rho)
                    residual=skipped.value+projected
                    if method=='materialized':
                        engine.materials['materialize']=extra
                        residual=engine.truncate(residual*R,16,'materialize',skipped.wire_id)
                    return residual
                result,record=measured(execute)
                if repeat==0:
                    requests=list(provider.request_cache)
                    provider.trace(False)
                else:
                    assert all(len(v)==0 for v in provider.tuple_cache.values())
                expected=h*R//Q+raw//Q if method=='eager' else h+raw*I//(Q*Q)
                if method=='materialized':expected=expected*R//Q
                actual=result.get_plain_text().long()
                assert torch.equal(actual,expected),(q,method)
                rows.append(dict(backend='masked_exact',method=method,tokens=q,width=width,inner=inner,
                    rank=rank,repeat=repeat,warmup=repeat==0,rho=rho,exact_integer_reference=True,
                    representation_scale=rho if method=='fused' else 1.,sharing=sharing,
                    mask_preparation=mask_preparation,preprocessing=preprocessing,trace=engine.trace,**record))
                provider.tuple_cache={}
    (output/f'rank{rank}.json').write_text(json.dumps(rows,indent=2)+'\n')
    return rank


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--tokens',type=int,nargs='+',default=[1,8])
    p.add_argument('--width',type=int,default=4096)
    p.add_argument('--inner',type=int,default=32)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--rho',type=float,default=1.125)
    p.add_argument('--controller-rows',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if min(a.tokens+[a.width,a.inner,a.repeats])<1 or not .5<=a.rho<=2:
        p.error('dimensions and repeats must be positive; rho must be in [0.5, 2]')
    if a.output.exists():p.error('Output directory already exists')
    coefficients=[1.03125,1.0625,1.125,1.25]*4
    if a.controller_rows:
        rows=json.loads(a.controller_rows.read_text())
        row=next(r for r in rows if not r['warmup'] and r['method']=='selected_only' and r['selected']>0)
        coefficients=[v/float(1<<26) for v in row['rho']]
    a.output.mkdir(parents=True)
    result=worker(a.tokens,a.width,a.inner,a.repeats,a.rho,coefficients,a.output)
    assert result is not None and len(result)==2
    rows=[json.loads((a.output/f'rank{r}.json').read_text()) for r in (0,1)]
    assert len(rows[0])==len(rows[1])==len(a.tokens)*4*(a.repeats+1)
    for x,y in zip(*rows):
        for k in ('backend','method','tokens','width','repeat','bytes','rounds'):assert x[k]==y[k],k
    (a.output/'validation.json').write_text(json.dumps(dict(status='passed',rows_per_party=len(rows[0]),
        native_coefficients=coefficients,scope='Native public multiplication and a separate masked exact projection/compensation fragment on synthetic integers. Sharing, mask preparation, cached dealer material and diagnostic openings excluded from online counters. Fused output retains a public scale and requires a compatible consumer. Materialized output charges its extra truncation. No full private request or task quality measured.'),indent=2)+'\n')


if __name__=='__main__':main()
