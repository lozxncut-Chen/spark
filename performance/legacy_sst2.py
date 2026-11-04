import argparse
from copy import deepcopy
import json
from pathlib import Path
import torch
from .model import load, recipe
from .runtime import configuration, probe, score_candidates
from .observation import work_metrics, summarize, compare_masks

def prepare(tokenizer, row):
    text = 'Classify the sentiment of this sentence. Reply positive or negative.\nSentence: ' + row['sentence'] + '\nSentiment:'
    ids = tokenizer.apply_chat_template([dict(role='user', content=text)], tokenize=True, add_generation_prompt=True)
    if len(ids) > 4096:
        ids = ids[:2048] + ids[-2048:]
    head = text[:text.index('Sentence:') + len('Sentence:')]
    head_ids = tokenizer.encode(head, add_special_tokens=False)
    start = next(i+len(head_ids) for i in range(len(ids)) if ids[i:i+len(head_ids)] == head_ids)
    return ids, start

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--adapter')
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--windows', type=int, nargs='+', default=[4, 8, 16])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.output.exists():
        parser.error('Output directory already exists')
    if any(n < 1 for n in args.windows):
        parser.error('Observation windows must be positive')
    if args.data.suffix == '.parquet':
        import pyarrow.parquet as pq
        rows = pq.read_table(args.data).to_pylist()
    else:
        rows = [json.loads(s) for s in args.data.read_text().splitlines() if s.strip()]
    rows = rows[:args.limit] if args.limit else rows
    torch.set_num_threads(4)
    torch.manual_seed(20261023)
    model, tokenizer = load(args.model, args.adapter, args.device)
    args.output.mkdir(parents=True)
    suffixes = [tokenizer.encode(s, add_special_tokens=False) for s in (' negative', ' positive')]
    summary = {}
    observations = {}
    masks = {}
    previous = None
    with recipe(model, 'spark'), torch.inference_mode():
        for window in args.windows:
            cfg = configuration(.95, .92, .95, a=32, f=24, d=24, span=window, samples=window, df=20)
            counts = dict(compensated=0, direct=0, dense=0)
            records = []
            with (args.output/f'p{window}.jsonl').open('w') as log:
                for index, row in enumerate(rows):
                    prefix, start = prepare(tokenizer, row)
                    prepared = probe(model, prefix, start, cfg)
                    record = dict(index=index, gold=int(row['label']), input_ids=prefix, observation_start=start, observation_end=prepared[3])
                    for variant in counts:
                        route = deepcopy(cfg)
                        if variant == 'direct':
                            route['compensation'] = False
                        elif variant == 'dense':
                            route.update(a=0, f=0, d=0)
                        scores, policy = score_candidates(model, prefix, suffixes, prepared, route)
                        values = [s[0]/len(ids) for s, ids in zip(scores, suffixes)]
                        pred = max(range(len(values)), key=values.__getitem__)
                        counts[variant] += int(pred == record['gold'])
                        record[variant] = dict(prediction=pred, choice_scores=values, policy=policy)
                    assert record['compensated']['policy']['prefix_mask'] == record['direct']['policy']['prefix_mask']
                    assert record['compensated']['policy']['events'] == record['direct']['policy']['events']
                    record['observation'] = work_metrics(len(prefix), prepared[4], prepared[3], record['compensated']['policy'], len(model.model.layers))
                    records.append(dict(gold=record['gold'], observation=record['observation'],
                        compensated=dict(prediction=record['compensated']['prediction'], policy=dict(prefix_mask=record['compensated']['policy']['prefix_mask'])),
                        direct=dict(prediction=record['direct']['prediction'])))
                    log.write(json.dumps(record)+'\n')
                    log.flush()
            summary[str(window)] = {key:100*value/len(rows) for key, value in counts.items()}
            observations[str(window)] = summarize(records)
            if previous is not None:
                masks[str(previous[0])+'->'+str(window)] = compare_masks(previous[1], records)
            previous = (window, records)
    (args.output/'summary.json').write_text(json.dumps(dict(count=len(rows), accuracy=summary, observations=observations, mask_comparisons=masks), indent=2)+'\n')

if __name__ == '__main__':
    main()
