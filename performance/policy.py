from contextlib import contextmanager
import math
import types

import torch
import torch.nn.functional as F


def select(means, threshold, budget, closed, strict=False):
    candidates = [i for i, (similarity, scale) in means.items()
                  if i not in closed and math.isfinite(similarity)
                  and math.isfinite(scale) and scale > 0 and (similarity > threshold if strict else similarity >= threshold)]
    candidates.sort(key=lambda i: (-means[i][0], i))
    return {i: means[i][1] for i in candidates[:max(0, budget-len(closed))]}


class Policy:
    def __init__(self, layers, budget, threshold, decode=True, sample_positions=8):
        self.layers, self.budget, self.threshold = layers, budget, threshold
        self.decode, self.sample_positions = decode, sample_positions
        self.closed = {"attn": {}, "ffn": {}}
        self.samples = {"attn": {}, "ffn": {}}
        self.events = []
        self.measure_type = None
        self.stage = "prefill"
        self.q = self.kv = self.step = 0

    def measure(self, kind, index, before, after):

        n = before.shape[1]
        indices = torch.linspace(0, n-1, min(n, self.sample_positions), device=before.device).round().long().unique()
        a, b = before[:, indices].float(), after[:, indices].float()
        norms = torch.linalg.vector_norm(a, dim=-1)
        ratios = torch.linalg.vector_norm(b, dim=-1) / norms.clamp_min(1e-12)
        similarities = F.cosine_similarity(a, b, dim=-1)
        valid = (norms > 1e-12) & torch.isfinite(ratios) & torch.isfinite(similarities)
        values = self.samples[kind].setdefault(index, [0.0, 0.0, 0])
        values[0] += float(similarities[valid].sum())
        values[1] += float(ratios[valid].sum())
        values[2] += int(valid.sum())
        self.events.append({"event": "assessment", "stage": self.measure_type,
                            "kind": kind, "layer": index, "pairs": len(indices),
                            "valid_pairs": int(valid.sum()), "step": self.step})

    def freeze(self, window):
        kinds = ("attn", "ffn") if window == "prefix" else ("ffn",)
        for kind in kinds:
            means = {i: (s/n, r/n) for i, (s, r, n) in self.samples[kind].items() if n}
            added = select(means, self.threshold, self.budget, self.closed[kind])
            self.closed[kind].update(added)
            self.events.append({"event": "selection", "stage": window, "kind": kind,
                                "step": self.step, "means": means, "added": added,
                                "closed": dict(self.closed[kind]), "budget": self.budget})
            self.samples[kind] = {}
        self.measure_type = None

    def call(self, kind, index, function, hidden):
        scale = self.closed[kind].get(index)
        self.events.append({"event": "sublayer", "stage": self.stage, "kind": kind,
                            "layer": index, "skipped": scale is not None,
                            "q": self.q, "kv": self.kv, "step": self.step})
        if scale is not None:
            return hidden * scale
        result = function(hidden)
        if self.measure_type == "prefix" or (self.measure_type == "decode" and kind == "ffn"):
            self.measure(kind, index, hidden, result)
        return result


@contextmanager
def install(model, policy):
    originals = [layer.forward for layer in model.model.layers]
    for index, layer in enumerate(model.model.layers):
        def forward(this, hidden_states, attention_mask=None, position_ids=None,
                    past_key_value=None, output_attentions=False, use_cache=False,
                    cache_position=None, _index=index, **kwargs):
            assert not output_attentions
            def attention(hidden):
                output, _, _ = this.self_attn(hidden_states=this.input_layernorm(hidden),
                    attention_mask=attention_mask, position_ids=position_ids,
                    past_key_value=past_key_value, output_attentions=False,
                    use_cache=use_cache, cache_position=cache_position, **kwargs)
                return hidden + output
            hidden_states = policy.call("attn", _index, attention, hidden_states)
            hidden_states = policy.call("ffn", _index,
                lambda hidden: hidden + this.mlp(this.post_attention_layernorm(hidden)), hidden_states)
            outputs = (hidden_states,)
            if use_cache:
                outputs += (past_key_value,)
            return outputs
        layer.forward = types.MethodType(forward, layer)
    try:
        yield
    finally:
        for layer, original in zip(model.model.layers, originals):
            layer.forward = original
