import argparse
import json
from pathlib import Path

import crypten
import crypten.communicator as comm
from crypten.config import cfg
import torch

from .secure_ops import block, share


CONFIGS = {name: json.loads((Path(__file__).parent/'configs'/f'{name}.json').read_text()) for name in ('llama','mistral')}


SHAPES = {'small':(32,4,2,64), 'llama':(4096,32,8,14336), 'mistral':(4096,32,8,14336)}


@crypten.mpc.run_multiprocess(world_size=2)
def worker(variant, shape, tokens, skip_attention, skip_ffn):
    torch.set_num_threads(2)
    cfg.communicator.verbose = True
    cfg.encoder.precision_bits = 20
    torch.manual_seed(0)
    rank = comm.get().get_rank()
    dimensions = SHAPES[shape]
    public_config = CONFIGS.get(shape)
    width, heads, kv_heads, intermediate = dimensions
    kv_width = width//heads*kv_heads
    shapes = dict(q=(width,width), k=(width,kv_width), v=(width,kv_width), o=(width,width),
                  gate=(width,intermediate), up=(width,intermediate), down=(intermediate,width),
                  norm1=(width,), norm2=(width,))
    weights = {name:share(size,rank,'cpu') for name,size in shapes.items()}
    hidden = share((1,tokens,width),rank,'cpu')
    prefill, decode = {}, {}
    out, cache = block(hidden,weights,None,variant,dimensions,prefill,config=public_config,
                       skip_attention=skip_attention,skip_ffn=skip_ffn)
    if skip_attention and skip_ffn:
        original, restored = hidden.get_plain_text(), out.get_plain_text()
        if not torch.equal(original,restored):
            raise RuntimeError('Bypass changed hidden states')
    one_token = share((1,1,width),rank,'cpu')
    result, _ = block(one_token,weights,cache,variant,dimensions,decode,config=public_config,
                      skip_attention=skip_attention,skip_ffn=skip_ffn)
    if not torch.isfinite(result.get_plain_text()).all():
        raise FloatingPointError('Nonfinite block output')
    return dict(rank=rank, prefill=prefill, decode=decode,
                bytes={phase:sum(v['communication_bytes'] for v in ops.values()) for phase,ops in [('prefill',prefill),('decode',decode)]},passed=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', choices=('crypten_tanh','puma','puma_nomax','east','spark','spark_nomax'),default='spark_nomax')
    parser.add_argument('--shape', choices=SHAPES, default='small')
    parser.add_argument('--tokens', type=int, default=4)
    parser.add_argument('--skip-attention', action='store_true')
    parser.add_argument('--skip-ffn', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.tokens < 1:
        parser.error('tokens must be positive')
    ranks = worker(args.recipe,args.shape,args.tokens,args.skip_attention,args.skip_ffn)
    if not ranks or len(ranks) != 2 or ranks[0]['bytes'] != ranks[1]['bytes']:
        raise RuntimeError('Two-party block test failed')
    result = dict(recipe=args.recipe, dimensions=SHAPES[args.shape], tokens=args.tokens,
                  skip_attention=args.skip_attention,skip_ffn=args.skip_ffn,ranks=ranks,
                  crypten=crypten.__version__,torch=torch.__version__,
                  scope='one shared-weight decoder block; random inputs and weights; embedding, output head, setup and output reconstruction excluded')
    payload = json.dumps(result,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(payload)
    print(payload,end='')


if __name__ == '__main__':
    main()
