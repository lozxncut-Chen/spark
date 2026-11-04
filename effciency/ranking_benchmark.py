import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
import crypten
import crypten.communicator as comm
from crypten.config import cfg
from . import controller_benchmark as b
S = b.S

def cross(sums, n, serial):
    idx = torch.arange(32)
    diff = sums.unsqueeze(-1) * n.unsqueeze(-2) - n.unsqueeze(-1) * sums.unsqueeze(-2)
    if serial:
        possible = crypten.stack([(diff[..., l] > 0) + (diff[..., l] == 0) * (idx < l).long() for l in range(32)], -1)
    else:
        possible = (diff > 0) + (diff == 0) * (idx[:, None] < idx[None, :]).long()
    possible = possible * (n.unsqueeze(-1) > 0) * (1 - torch.eye(32, dtype=torch.long))
    return (n > 0) * (sums > n * round(0.97 * S)) * (1 + possible.sum(-2) <= 8)

@crypten.mpc.run_multiprocess(world_size=2)
def worker(tokens, repeats, output):
    torch.set_num_threads(2)
    cfg.communicator.verbose = True
    cfg.encoder.precision_bits = 0
    rank = comm.get().get_rank()
    provider = crypten.mpc.get_default_provider()
    rows = []
    for T in tokens:
        for case in ('equal_counts', 'mixed_counts'):
            n = np.full((2, 32), T, dtype=np.int64)
            if case == 'mixed_counts':
                n = np.broadcast_to(np.arange(32) % T + 1, (2, 32)).copy()
                n[:, [2, 12, 31]] = 0
            means = np.rint((0.999 - np.arange(32) * 0.001) * S).astype(np.int64)
            means[5] = means[4]
            sums = n * means
            (ss, nn) = (b.secret(sums), b.secret(n))
            expected = np.zeros((2, 32), dtype=np.int64)
            for g in range(2):
                eligible = [l for l in range(32) if n[g, l] > 0 and means[l] > round(0.97 * S)]
                expected[g, sorted(eligible, key=lambda l: (-means[l], l))[:8]] = 1
            requests = {}
            for repeat in range(repeats + 1):
                methods = ['mean_batch', 'cross_serial', 'cross_batch']
                if repeat % 2:
                    methods = methods[::-1]
                for method in methods:
                    provider.tuple_cache = {}
                    if repeat == 0:
                        provider.request_cache = []
                        provider.trace(True)
                    else:
                        provider.trace(False)
                        provider.request_cache = list(requests[method])
                        provider.fill_cache()
                    before = comm.get().get_communication_stats().copy()
                    tick = time.perf_counter()
                    result = b.mean_select(ss, nn, T) if method == 'mean_batch' else cross(ss, nn, method == 'cross_serial')
                    opened = result.get_plain_text()
                    after = comm.get().get_communication_stats().copy()
                    row = dict(T=T, case=case, method=method, repeat=repeat, warmup=repeat == 0, rank=rank, bytes=after['bytes'] - before['bytes'], rounds=after['rounds'] - before['rounds'], seconds=time.perf_counter() - tick, mask=opened.tolist())
                    assert np.array_equal(opened.numpy(), expected), (T, case, method)
                    if repeat == 0:
                        requests[method] = list(provider.request_cache)
                        provider.trace(False)
                    else:
                        assert all((len(v) == 0 for v in provider.tuple_cache.values()))
                    rows.append(row)
                    (output / f'rank{rank}.json').write_text(json.dumps(rows, indent=2) + '\n')
                    print('BATCH', rank, T, case, repeat, method, row['bytes'], row['rounds'], round(row['seconds'], 4), flush=True)
    return rank

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tokens', type=int, nargs='+', default=[16, 32])
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if min(a.tokens) < 1 or a.repeats < 1:
        p.error('tokens and repeats must be positive')
    if a.output.exists():
        p.error('Output directory already exists')
    a.output.mkdir(parents=True)
    result = worker(a.tokens, a.repeats, a.output)
    assert result is not None and len(result) == 2
    rows = [json.loads((a.output / f'rank{r}.json').read_text()) for r in (0, 1)]
    assert len(rows[0]) == len(rows[1]) == len(a.tokens) * 2 * 3 * (a.repeats + 1)
    for (x, y) in zip(*rows):
        for k in ('T', 'case', 'method', 'repeat', 'bytes', 'rounds', 'mask'):
            assert x[k] == y[k], k
    (a.output / 'validation.json').write_text(json.dumps(dict(status='passed', rows_per_party=len(rows[0]), scope='Private sums/counts to public mask. Equal and varying counts, ties and zero-count controls. Serial mode splits pairwise comparisons only. Collective calls are not measured network RTTs.'), indent=2) + '\n')
if __name__ == '__main__':
    main()
