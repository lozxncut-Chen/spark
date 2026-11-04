from copy import deepcopy
import hashlib
import json
import random
import torch
from transformers import DynamicCache
from .policy import Policy, install, select
SEED = 20261023
def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def configuration(ta=.97, tf=.97, td=.97, a=32, f=32, d=32,
                  span=2, samples=None, offset=0, df=12, protect=(), mode='similarity',
                  compensation=True, seed=0, fixed=None, match=None):
    c = dict(tau_a=ta, tau_f=tf, tau_d=td, a=a, f=f, d=max(f, d),
             pf_span=span, pf_samples=samples or span, pf_offset=offset,
             df_forwards=df, protected=list(protect), selection=mode,
             compensation=compensation, seed=seed, prefill_chunk_tokens=256)
    if fixed is not None:
        c['fixed'] = fixed
    if match is not None:
        c['match'] = match
    c['id'] = digest(c)[:14]
    return c

class JointPolicy(Policy):
    def __init__(self, cfg, layers=32):
        super().__init__(layers, max(cfg['a'], cfg['d']), 1., True, cfg['pf_samples'])
        self.cfg = cfg
        self.prefix_closed = {'attn': {}, 'ffn': {}}
        self.prefix_frozen = False
        self.decode_frozen = False
        self.rng = random.Random(SEED + cfg['seed'])
        self.observation_positions = []

    def measure(self, kind, index, before, after):
        if self.measure_type == 'prefix':
            indices = torch.tensor(self.observation_positions, device=before.device)
        else:
            indices = torch.tensor([0], device=before.device)
        a, b = before[:, indices].float(), after[:, indices].float()
        norm = a.norm(dim=-1)
        sim = torch.nn.functional.cosine_similarity(a, b, dim=-1)
        ratio = b.norm(dim=-1) / norm.clamp_min(1e-12)
        valid = (norm > 1e-12) & torch.isfinite(sim) & torch.isfinite(ratio)
        entry = self.samples[kind].setdefault(index, [0., 0., 0])
        entry[0] += float(sim[valid].sum())
        entry[1] += float(ratio[valid].sum())
        entry[2] += int(valid.sum())
        self.events.append(dict(event='assessment', stage=self.measure_type,
                                kind=kind, layer=index, pairs=len(indices),
                                valid_pairs=int(valid.sum()), step=self.step))

    def call(self, kind, index, function, hidden):
        rho = self.closed[kind].get(index)
        if rho is not None:
            return hidden * rho if self.cfg['compensation'] else hidden
        out = function(hidden)
        if self.measure_type == 'prefix' or (self.measure_type == 'decode' and kind == 'ffn'):
            self.measure(kind, index, hidden, out)
        return out

    def freeze(self, window):
        for kind in (('attn', 'ffn') if window == 'prefix' else ('ffn',)):
            field = 'a' if kind == 'attn' else ('f' if window == 'prefix' else 'd')
            tau = self.cfg['tau_a' if kind == 'attn' else ('tau_f' if window == 'prefix' else 'tau_d')]
            means = {i: (s/n, r/n) for i, (s, r, n) in self.samples[kind].items()
                     if n and i not in self.cfg['protected']}
            selected = select(means, tau, self.cfg[field], self.closed[kind], strict=self.cfg.get("strict_threshold", False))
            mode = self.cfg['selection']
            if mode == 'random':

                pool = [i for i in means if i not in self.closed[kind]]
                self.rng.shuffle(pool)
                count = self.target_counts[window][kind] if getattr(self,'target_counts',None) else len(selected)
                assert len(pool)>=count
                selected = {i: means[i][1] for i in pool[:count]}
            elif mode == 'fixed':
                ids = self.cfg['fixed'][window][kind]
                selected = {i: means[i][1] for i in ids if i in means and i not in self.closed[kind]}
            self.closed[kind].update(selected)
            self.events.append(dict(event='selection', stage=window, kind=kind,
                                    step=self.step, means=means, added=selected,
                                    closed=dict(self.closed[kind]), budget=self.cfg[field], threshold=tau))
            self.samples[kind] = {}
        if window == 'prefix':
            self.prefix_frozen = True
            self.prefix_closed = deepcopy(self.closed)
        else:
            self.decode_frozen = True
        self.measure_type = None

    def evidence(self):
        return dict(prefix_mask=self.prefix_closed, final_mask=deepcopy(self.closed),
                    decode_window_completed=self.decode_frozen, events=deepcopy(self.events))

def forward(model, ids, cache, position, p, stage, step=0, measure=None, heads=None):
    p.stage, p.step, p.q, p.kv = stage, step, len(ids), position+len(ids)
    p.measure_type = measure
    p.events.append(dict(event='forward', stage=stage, q=p.q, kv=p.kv, step=step,
                         head_positions=len(ids) if heads is None else heads,
                         mask=deepcopy(p.closed)))
    pos = torch.arange(position, p.kv, device=model.device)


    original_norm = model.model.norm.forward
    if heads==0:
        model.model.norm.forward=lambda value:value
    try:
        hidden = model.model(input_ids=torch.tensor([ids], device=model.device),
                             attention_mask=torch.ones((1,p.kv), dtype=torch.long, device=model.device),
                             position_ids=pos[None], cache_position=pos,
                             past_key_values=cache, use_cache=True).last_hidden_state
    finally:
        model.model.norm.forward=original_norm
    if heads==0:
        return None
    if heads is not None:
        hidden = hidden[:, -heads:]
    logits = model.lm_head(hidden)[0]
    assert torch.isfinite(logits).all(), 'Nonfinite logits'
    for i in range(len(model.model.layers)):
        if i not in p.closed['attn']:
            assert cache.get_seq_length(i) == p.kv
    return logits

def probe(model, prefix, start, cfg):
    p = JointPolicy(cfg, len(model.model.layers))
    cache = DynamicCache()
    left = min(len(prefix)-1, start+cfg['pf_offset'])
    end = min(len(prefix), left+cfg['pf_span'])
    p.observation_positions = sorted(set(round(i*(end-left-1)/max(1,min(cfg['pf_samples'],end-left)-1))
                                         for i in range(min(cfg['pf_samples'],end-left))))
    with install(model,p):
        if left:
            for position in range(0,left,cfg['prefill_chunk_tokens']):
                stop = min(left, position+cfg['prefill_chunk_tokens'])
                forward(model,prefix[position:stop],cache,position,p,'prefill',heads=0)
        logits = forward(model,prefix[left:end],cache,left,p,'prefill',measure='prefix',heads=1 if end==len(prefix) else 0)
    return p, cache, None if logits is None else logits[-1], end, left


def score_candidates(model, prefix, continuations, prepared, cfg):
    policy, cache, last, end, left = deepcopy(prepared)
    policy.cfg = deepcopy(cfg)
    policy.freeze('prefix')
    with install(model, policy):
        for start in range(end, len(prefix), cfg['prefill_chunk_tokens']):
            stop = min(len(prefix), start + cfg['prefill_chunk_tokens'])
            output = forward(model, prefix[start:stop], cache, start, policy, 'prefill', heads=1 if stop == len(prefix) else 0)
            if output is not None:
                last = output[-1]
    scores = []
    for suffix in continuations:
        branch, branch_cache = deepcopy(policy), deepcopy(cache)
        before = len(branch.events)
        logits = [last[None]]
        if len(suffix) > 1:
            with install(model, branch):
                logits.append(forward(model, suffix[:-1], branch_cache, len(prefix), branch, 'candidate'))
        logits = torch.cat(logits).float()
        targets = torch.tensor(suffix, device=model.device)
        values = logits.log_softmax(-1).gather(1, targets[:, None])[:, 0]
        scores.append((float(values.sum()), bool((logits.argmax(-1) == targets).all())))
        policy.events.extend(branch.events[before:])
    return scores, policy.evidence()
