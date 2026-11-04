import argparse
import csv
import hashlib
import json
import statistics
import time
from pathlib import Path
import numpy as np
import torch
import crypten
import crypten.communicator as comm
from crypten.config import cfg
from crypten.mpc import ptype
from . import controller as cf
from .controller import S, secret, fixed, refine, select_rank, shared_gram
METHODS = ['direct', 'endpoint_gram', 'shared_inverse', 'cross_rank', 'selected_only']

def direct_gram(T, parts, measure):
    rng = np.random.default_rng(20261023 + T)
    base = rng.integers(-1800, 1800, size=(T, 4096), dtype=np.int64)
    states = np.stack([base + rng.integers(-5 * (i + 1), 5 * (i + 1) + 1, size=base.shape) for i in range(65)])
    shared = secret(states)

    def gram():
        (a, b) = (shared[:-1], shared[1:])
        raw = (crypten.stack([a, b, a], 0) * crypten.stack([a, b, b], 0)).sum(-1)
        exponents = torch.arange(65) % 2
        shifts = torch.stack([6 + 2 * exponents[:-1], 6 + 2 * exponents[1:], 6 + exponents[:-1] + exponents[1:]])[:, :, None]
        binary = raw.to(ptype.binary)
        binary._tensor.share = torch.bitwise_right_shift(binary._tensor.share, shifts)
        converted = binary.to(ptype.arithmetic, precision=0)
        return crypten.stack([converted[0], converted[1], converted[2]], -1).reshape(2, 32, T, 3)
    abc = measure('vector_statistics', gram)
    gains = np.broadcast_to((2.0 ** ((np.arange(65) % 2)[1:] - (np.arange(65) % 2)[:-1])).reshape(2, 32, 1), (2, 32, T)).copy()
    return (abc, gains)

def count_inverse(n, T):
    lookup = n.unsqueeze(-1) == torch.arange(1, T + 1)
    return (lookup * torch.tensor(np.rint(S / np.arange(1, T + 1)).astype(np.int64))).sum(-1)

def mean_select(sums, n, T):
    means = fixed(sums, count_inverse(n, T))
    idx = torch.arange(32)
    diff = means.unsqueeze(-1) - means.unsqueeze(-2)
    possible = (diff > 0) + (diff == 0) * (idx[:, None] < idx[None, :]).long()
    possible = possible * (n.unsqueeze(-1) > 0) * (1 - torch.eye(32, dtype=torch.long))
    return (n > 0) * (means > round(0.97 * S)) * (1 + possible.sum(-2) <= 8)

def independent_inverse(abc, repeat, material):
    (m, gain, x, domain, valid) = cf.normalization(abc)
    cf.CONTEXT['material'] = material
    z = fixed(cf.fourier(m, 4, repeat), gain)
    return refine(x, z)

@crypten.mpc.run_multiprocess(world_size=2)
def worker(tokens, repeats, methods, output, serial):
    torch.set_num_threads(2)
    cfg.communicator.verbose = True
    cfg.encoder.precision_bits = 0
    rank = comm.get().get_rank()
    provider = crypten.mpc.get_default_provider()
    rows = []
    for T in tokens:
        requests = {}
        for repeat in range(repeats + 1):
            order = methods if repeat % 2 == 0 else methods[::-1]
            for method in order:
                provider.tuple_cache = {}
                if repeat == 0:
                    provider.request_cache = []
                    provider.trace(True)
                    prep = None
                else:
                    provider.trace(False)
                    provider.request_cache = list(requests[method])
                    before = comm.get().get_communication_stats().copy()
                    tick = time.perf_counter()
                    provider.fill_cache()
                    after = comm.get().get_communication_stats().copy()
                    prep = dict(seconds=time.perf_counter() - tick, bytes=after['bytes'] - before['bytes'], rounds=after['rounds'] - before['rounds'])
                parts = {}

                def measure(name, fn):
                    before = comm.get().get_communication_stats().copy()
                    tick = time.perf_counter()
                    value = fn()
                    after = comm.get().get_communication_stats().copy()
                    parts[name] = dict(seconds=time.perf_counter() - tick, bytes=after['bytes'] - before['bytes'], rounds=after['rounds'] - before['rounds'])
                    return value
                if method == 'direct':
                    (abc, gains) = direct_gram(T, parts, measure)
                else:
                    (abc, gains) = shared_gram(T, parts, measure)
                cf.CONTEXT['material'] = cf.prepare(abc.shape[:-1], 4, repeat, 0, rank)
                (s, z, v, domain, x) = measure('scoring', lambda : cf.normalized_score(abc, 'CIFR4', repeat))

                def aggregate():
                    return (((s * domain - S * (1 - domain)) * v).sum(-1), v.sum(-1))
                (sums, n) = measure('aggregation', aggregate)
                all_scales = method != 'selected_only'
                scale_material = cf.prepare((64, T), 4, repeat, 1, rank) if method in ('direct', 'endpoint_gram') else None

                def scales(g, l):
                    if not len(g):
                        return None
                    zz = independent_inverse(abc[g, l], repeat, scale_material) if method in ('direct', 'endpoint_gram') else refine(x[g, l], z[g, l])
                    ratio = fixed(abc[g, l][..., 1], zz) * v[g, l]
                    ratio = fixed(ratio, torch.from_numpy(np.rint(gains[g, l] * S).astype(np.int64)))
                    return fixed(ratio.sum(-1), count_inverse(n[g, l], T))
                if all_scales:
                    (g, l) = np.indices((2, 32)).reshape(2, -1).tolist()
                    prepared = measure('scale_preparation', lambda : scales(g, l))
                mask = measure('selection_mask_release', lambda : ((serial_select(sums, n) if serial else select_rank(sums, sums, n, round(0.97 * S), 8)) if method in ('cross_rank', 'selected_only') else mean_select(sums, n, T)).get_plain_text())
                ix = np.argwhere(mask.numpy())

                def release():
                    if not len(ix):
                        return None
                    if all_scales:
                        result = prepared[[int(g) * 32 + int(l) for (g, l) in ix]]
                    else:
                        result = scales(ix[:, 0].tolist(), ix[:, 1].tolist())
                    return result.get_plain_text()
                rho = measure('selected_scale_release', release)
                if repeat == 0:
                    requests[method] = list(provider.request_cache)
                    provider.trace(False)
                else:
                    assert all((len(t) == 0 for t in provider.tuple_cache.values())), 'unused tuples'
                ref = abc.get_plain_text().numpy() / S
                exact = ref[..., 2] / np.sqrt(np.maximum(ref[..., 0] * ref[..., 1], 1e-300))
                actual = s.get_plain_text().numpy() / S
                valid = (ref[..., 0] > 0) & (ref[..., 1] > 0)
                expected = np.zeros((2, 32), dtype=np.int64)
                for g in range(2):
                    allowed = [l for l in range(32) if valid[g, l].all() and exact[g, l].mean() > 0.97]
                    for l in sorted(allowed, key=lambda l: (-exact[g, l].mean(), l))[:8]:
                        expected[g, l] = 1
                assert np.array_equal(mask.numpy(), expected), (T, method, 'mask mismatch')
                err = float(np.abs(actual[valid] - exact[valid]).max())
                assert err < 5e-05, (T, method, 'score', err)
                expected_rho = (np.sqrt(ref[ix[:, 0], ix[:, 1], :, 1] / ref[ix[:, 0], ix[:, 1], :, 0]) * gains[ix[:, 0], ix[:, 1]]).mean(-1)
                rhoerr = float(np.abs(rho.numpy() / S - expected_rho).max())
                assert rhoerr < 0.0001, (T, method, 'rho', rhoerr)
                row = dict(T=T, method=method, repeat=repeat, warmup=repeat == 0, rank=rank, bytes=sum((p['bytes'] for p in parts.values())), rounds=sum((p['rounds'] for p in parts.values())), seconds=sum((p['seconds'] for p in parts.values())), parts=parts, preprocessing=prep, selected=int(mask.sum()), mask=mask.tolist(), rho=rho.tolist(), max_score_error=err, max_rho_error=rhoerr, gram_sha256=hashlib.sha256(ref.tobytes()).hexdigest())
                rows.append(row)
                (output / f'rank{rank}.json').write_text(json.dumps(rows, indent=2) + '\n')
                print('ROW', rank, T, repeat, method, row['bytes'], row['rounds'], round(row['seconds'], 3), err, rhoerr, flush=True)
                provider.tuple_cache = {}
        requests.clear()
    return rank

def serial_select(sums, n):
    idx = torch.arange(32)
    diff = sums.unsqueeze(-1) * n.unsqueeze(-2) - n.unsqueeze(-1) * sums.unsqueeze(-2)
    possible = crypten.stack([(diff[..., l] > 0) + (diff[..., l] == 0) * (idx < l).long() for l in range(32)], -1)
    possible = possible * (n.unsqueeze(-1) > 0) * (1 - torch.eye(32, dtype=torch.long))
    return (n > 0) * (sums > n * round(0.97 * S)) * (1 + possible.sum(-2) <= 8)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tokens', type=int, nargs='+', default=[16, 32])
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS)
    p.add_argument('--serial-ranking', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if min(a.tokens) < 1 or a.repeats < 1:
        p.error('tokens and repeats must be positive')
    if a.output.exists():
        p.error('Output directory already exists')
    a.output.mkdir(parents=True)
    refs = worker(a.tokens, a.repeats, a.methods, a.output, a.serial_ranking)
    assert refs is not None and len(refs) == 2, '2PC worker failed'
    rows = [json.loads((a.output / f'rank{r}.json').read_text()) for r in (0, 1)]
    assert len(rows[0]) == len(rows[1]) == len(a.tokens) * len(a.methods) * (a.repeats + 1)
    for (x, y) in zip(*rows):
        for k in ('T', 'method', 'repeat', 'bytes', 'rounds', 'selected', 'mask', 'rho', 'gram_sha256'):
            assert x[k] == y[k], k
    summary = []
    for T in a.tokens:
        rr = [r for r in rows[0] if r['T'] == T]
        assert len({r['gram_sha256'] for r in rr}) == 1
        assert all((r['mask'] == rr[0]['mask'] for r in rr))
        for method in a.methods:
            indices = [i for (i, r) in enumerate(rows[0]) if r['T'] == T and r['method'] == method and (not r['warmup'])]
            record = rows[0][indices[0]]
            summary.append(dict(tokens=T, method=method, serial_ranking=a.serial_ranking, bytes_per_party=record['bytes'], collective_calls=record['rounds'], selected=record['selected'], median_local_seconds=statistics.median((max(rows[0][i]['seconds'], rows[1][i]['seconds']) for i in indices))))
    with (a.output / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    out = dict(status='passed', rows_per_party=len(rows[0]), tokens=a.tokens, methods=a.methods, torch=torch.__version__, crypten=crypten.__version__, backend='gloo CPU, 2 threads per party', summary=summary, scope='Seeded full-width controller fixtures. Online counters exclude input sharing, cached trusted-dealer correlations and diagnostic openings. Final masks and selected coefficients are public. No model execution or task quality measured.')
    (a.output / 'validation.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out), flush=True)
if __name__ == '__main__':
    main()
