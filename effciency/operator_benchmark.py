import argparse
import json
from pathlib import Path
import crypten
import crypten.communicator as comm
from crypten.config import cfg
import torch
from .secure_ops import puma_gelu, spark_gelu, east_gelu, softmax
from .timing import measured, configure_cuda_integer_encoding

@crypten.mpc.run_multiprocess(world_size=2)
def execute(variant, tokens, width, repeats, backend):
    torch.set_num_threads(2)
    torch.manual_seed(20261008)
    cfg.communicator.verbose = True
    cfg.encoder.precision_bits = 20
    device = torch.device('cuda:0' if backend == 'cuda' else 'cpu')
    if backend == 'cuda':
        torch.cuda.set_device(device)
        configure_cuda_integer_encoding()
    inputs = crypten.cryptensor(torch.randn(1, tokens, width, device=device), src=0)
    scores = crypten.cryptensor(torch.randn(1, 1, tokens, tokens, device=device), src=0)
    mask = torch.tril(torch.ones(tokens, tokens, device=device))
    activation = {'puma': puma_gelu, 'spark': spark_gelu, 'east': east_gelu}[variant]
    result = []
    with crypten.no_grad():
        for repeat in range(repeats):
            for name, fn in [('gelu', lambda: activation(inputs)), ('softmax', lambda: softmax(scores, mask, variant))]:
                output, record = measured(fn)
                assert torch.isfinite(output.get_plain_text()).all()
                result.append(dict(operator=name, repeat=repeat, **record))
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=('puma', 'spark', 'east'), default='spark')
    parser.add_argument('--tokens', type=int, default=128)
    parser.add_argument('--width', type=int, default=5120)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    parts = execute(args.variant, args.tokens, args.width, args.repeats, args.device)
    assert parts is not None and len(parts) == 2
    for a, b in zip(*parts):
        assert a['communication_bytes'] == b['communication_bytes']
        assert a['communication_rounds'] == b['communication_rounds']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(variant=args.variant, tokens=args.tokens, width=args.width,
        crypten=crypten.__version__, torch=torch.__version__, fractional_bits=20, parts=parts), indent=2)+'\n')

if __name__ == '__main__':
    main()
