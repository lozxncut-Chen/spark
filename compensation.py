import argparse
from copy import deepcopy
import json
from pathlib import Path
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from performance.model import recipe
from performance.observation import work_metrics
from performance.runtime import configuration, probe, score_candidates

def simulate(window=4):
    torch.set_num_threads(2)
    torch.manual_seed(7)
    config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, hidden_act='gelu', max_position_embeddings=128)
    config._attn_implementation = 'eager'
    model = LlamaForCausalLM(config).eval()
    prefix = torch.randint(3, 64, (24,)).tolist()
    suffixes = [[5, 6], [7, 8, 9]]
    cfg = configuration(.9, .9, .9, a=2, f=2, d=2, span=window, samples=window)
    results = {}
    with recipe(model, 'spark'), torch.inference_mode():
        prepared = probe(model, prefix, 2, cfg)
        for name in ('compensated', 'direct', 'dense'):
            route = deepcopy(cfg)
            if name == 'direct':
                route['compensation'] = False
            elif name == 'dense':
                route.update(a=0, f=0, d=0)
            scores, policy = score_candidates(model, prefix, suffixes, prepared, route)
            results[name] = dict(choice_loglikelihood=[s[0] for s in scores], policy=policy)
    assert results['compensated']['policy']['prefix_mask'] == results['direct']['policy']['prefix_mask']
    assert results['compensated']['policy']['events'] == results['direct']['policy']['events']
    return dict(observation_start=2, observation_end=prepared[3],
        observation=work_metrics(len(prefix), prepared[4], prepared[3], results['compensated']['policy'], len(model.model.layers)), results=results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', type=int, default=4)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 1 <= args.window <= 22:
        parser.error('--window must be between 1 and 22')
    payload = json.dumps(simulate(args.window), indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    else:
        print(payload, end='')

if __name__ == '__main__':
    main()
