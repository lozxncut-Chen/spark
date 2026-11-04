import json
from pathlib import Path
import numpy as np
import torch
import crypten
import crypten.communicator as comm
from crypten.mpc import MPCTensor, ptype
F = 26
S = 1 << F
Q = 4*S
CONTEXT = {}
HERE = Path(__file__).resolve().parent
def trunc(x, bits=F):
    b = x.to(ptype.binary)
    b._tensor = b._tensor.rshift(bits)
    return b.to(ptype.arithmetic, precision=0)

def fixed(x, y):
    return trunc(x * y)

def secret(x):
    return crypten.cryptensor(torch.as_tensor(x, dtype=torch.long), src=0, precision=0)

def refine(x, z):
    return trunc(z * (3 * S - fixed(x, fixed(z, z))), F + 1)

def select_rank(low, high, n, threshold, budget):
    J = n.size(-1)
    idx = torch.arange(J)
    diff = high.unsqueeze(-1) * n.unsqueeze(-2) - n.unsqueeze(-1) * low.unsqueeze(-2)
    eqpreferred = (idx[:, None] < idx[None, :]).long()
    possible = (diff > 0) + (diff == 0) * eqpreferred
    possible = possible * (n.unsqueeze(-1) > 0) * (1 - torch.eye(J, dtype=torch.long))
    worst = 1 + possible.sum(-2)
    return (n > 0) * (low > n * threshold) * (worst <= budget)

def normalization(abc):
    (A, B, C) = (abc[..., 0], abc[..., 1], abc[..., 2])
    x = fixed(A, B)
    domain = (A >= S // 16) * (A <= 4 * S) * (B >= S // 16) * (B <= 4 * S) * (x >= S // 16) * (x < 16 * S)
    safe = domain * (x - S) + S
    exponents = np.arange(-4, 4)
    bounds = torch.tensor(np.rint(2.0 ** np.arange(-4, 5) * S).astype(np.int64))
    above = safe.unsqueeze(-1) >= bounds
    oh = above[..., :-1] - above[..., 1:]
    inv = (oh * torch.tensor(np.rint(2.0 ** (-exponents) * S).astype(np.int64))).sum(-1)
    gain = (oh * torch.tensor(np.rint(2.0 ** (-exponents / 2) * S).astype(np.int64))).sum(-1)
    return (fixed(safe, inv), gain, safe, domain, (A > 0) * (B > 0))

def normalized_score(state, method, repeat):
    (m, gain, x, dom, v) = normalization(state)
    assert method == 'CIFR4'
    p = fourier(m, 4, repeat)
    z = fixed(p, gain)
    s = fixed(state[..., 2], z)
    return (s, z, v, dom, x)

def fourier(m, K, repeat):
    rank = comm.get().get_rank()
    (r, U, V) = CONTEXT['material']
    delta = (comm.get().all_reduce(m._tensor.share - torch.from_numpy(r) & Q - 1) & Q - 1).numpy() / S
    coeff = next((r for r in json.loads((HERE / 'controller_coefficients.json').read_text())['rows'] if r['K'] == K and r['base_degree'] == 1))
    w = 2 * np.pi * np.arange(1, K + 1) / 4
    cp = torch.from_numpy(np.rint(np.cos(delta[..., None] * w) * S).astype(np.int64))
    sp = torch.from_numpy(np.rint(np.sin(delta[..., None] * w) * S).astype(np.int64))
    sh = (cp * torch.from_numpy(U) + sp * torch.from_numpy(V)).sum(-1)
    (b0, b1) = np.rint(np.array(coeff['base']) * S).astype(np.int64)
    beta = round(coeff['coeff'][0] * S)
    sh += m._tensor.share * int(b1)
    if rank == 0:
        sh += int((b0 + beta) * S)
    return trunc(MPCTensor.from_shares(sh, precision=0))

def prepare(shape, K, repeat, stage, rank):
    rng = np.random.default_rng(202610213 + repeat * 100 + K * 10 + stage)
    r = rng.integers(0, Q, size=shape, dtype=np.int64)
    coeff = next((r for r in json.loads((HERE / 'controller_coefficients.json').read_text())['rows'] if r['K'] == K and r['base_degree'] == 1))
    a = np.array(coeff['coeff'][1:K + 1])
    g = np.array(coeff['coeff'][K + 1:])
    w = 2 * np.pi * np.arange(1, K + 1) / 4
    phase = r[..., None] / S * w
    U = np.rint((a * np.cos(phase) + g * np.sin(phase)) * S).astype(np.int64)
    V = np.rint((-a * np.sin(phase) + g * np.cos(phase)) * S).astype(np.int64)
    return tuple(((lambda z, v: z if rank == 0 else v - z)(rng.integers(0, 2 ** 64, size=v.shape, dtype=np.uint64).view(np.int64), v) for v in (r, U, V)))

def shared_gram(T, parts, measure):
    rng = np.random.default_rng(20261023 + T)
    base = rng.integers(-1800, 1800, size=(T, 4096), dtype=np.int64)
    states = np.stack([base + rng.integers(-5 * (i + 1), 5 * (i + 1) + 1, size=base.shape) for i in range(65)])
    a = np.arange(64)
    b = a + 1
    own = np.arange(len(states))
    pa = np.r_[own, a]
    pb = np.r_[own, b]
    raw_mask = rng.integers(0, 2 ** 64, size=states.shape, dtype=np.uint64).view(np.int64)
    correlation = (raw_mask[pa] * raw_mask[pb]).sum(-1)

    def split(v):
        r = rng.integers(0, 2 ** 64, size=v.shape, dtype=np.uint64).view(np.int64)
        return torch.from_numpy(r if comm.get().get_rank() == 0 else v - r)
    (xs, rs, cs) = (split(states), split(raw_mask), split(correlation))

    def gram():
        delta = MPCTensor.from_shares(xs - rs, precision=0)._tensor.reveal()
        local = cs + (delta[pa] * rs[pb]).sum(-1) + (rs[pa] * delta[pb]).sum(-1)
        if comm.get().get_rank() == 0:
            local += (delta[pa] * delta[pb]).sum(-1)
        raw = MPCTensor.from_shares(local, precision=0)
        exponents = torch.arange(len(states)) % 2
        shifts = torch.cat([6 + 2 * exponents, 6 + exponents[a] + exponents[b]])[:, None]
        binary = raw.to(ptype.binary)
        binary._tensor.share = torch.bitwise_right_shift(binary._tensor.share, shifts)
        converted = binary.to(ptype.arithmetic, precision=0)
        abc = crypten.stack([converted[:len(states)][a], converted[:len(states)][b], converted[len(states):]], -1)
        gains = 2.0 ** (exponents[b] - exponents[a]).numpy()
        return (abc, gains)
    (abc, gains) = measure('vector_statistics', gram)
    abc = abc.reshape(2, 32, T, 3)
    gains = np.broadcast_to(gains.reshape(2, 32, 1), (2, 32, T)).copy()
    return (abc, gains)
