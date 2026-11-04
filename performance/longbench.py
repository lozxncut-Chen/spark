import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import torch
from rouge import Rouge
from . import runtime
from .model import load, recipe, provenance, FiniteLogits

PROMPT = 'You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:'

def rouge_score(prediction, answer):
    try:
        return Rouge().get_scores([prediction], [answer], avg=True)['rouge-l']['f']
    except ValueError:
        return 0.0

def prompt_input(tokenizer, row):
    text = PROMPT.format(**row)
    rendered = tokenizer.apply_chat_template([dict(role='user', content=text)], tokenize=False, add_generation_prompt=True)
    character = rendered.index(text) + text.index(row['context'])
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    content = next(i for i, (left, right) in enumerate(encoded['offset_mapping']) if right > character)
    full = encoded['input_ids']
    return (full if len(full) <= 4096 else full[:2048] + full[-2048:]), content

def generate(e,model,tok,prefix,prepared,configuration,maximum,replay=None,smoke=False):
    p,cache,last,end,_=deepcopy(prepared);p.cfg=deepcopy(configuration)
    def freeze(window):
        if replay is None:p.freeze(window)
        else:
            p.closed=deepcopy(replay['prefix_mask' if window=='prefix' else 'final_mask'])
            p.measure_type=None
            if window=='prefix':p.prefix_closed=deepcopy(p.closed);p.prefix_frozen=True
            else:p.decode_frozen=True
            p.events.append(dict(event='policy_replay',stage=window,step=p.step,mask=deepcopy(p.closed)))
    freeze('prefix')
    with e.install(model,p):
        for start in range(end,len(prefix),configuration['prefill_chunk_tokens']):
            stop=min(len(prefix),start+configuration['prefill_chunk_tokens'])
            output=e.forward(model,prefix[start:stop],cache,start,p,'prefill',heads=1 if stop==len(prefix) else 0)
            if output is not None:last=output[-1]
        assert last is not None
        eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
        if smoke:eos=set()
        token=int(last.argmax());tokens=[]
        for _ in range(maximum):
            tokens.append(token)
            if token in eos or len(tokens)==maximum:break
            step=len(tokens)
            learning=replay is None and not p.decode_frozen and configuration['d']>len(p.closed['ffn']) and step<=configuration['df_forwards']
            output=e.forward(model,[token],cache,len(prefix)+step-1,p,'decode',step,'decode' if learning else None,heads=1)
            token=int(output[-1].argmax())
            if learning and step==configuration['df_forwards']:freeze('decode')
            if replay is not None and replay['decode_window_completed'] and step==configuration['df_forwards']:freeze('decode')
    return dict(prediction=tok.decode(tokens,skip_special_tokens=True).strip(),generated_ids=tokens,
        generated_tokens=len(tokens),policy=p.evidence())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--adapter')
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recipe', choices=('exact', 'crypten', 'puma', 'east', 'spark'), default='spark')
    parser.add_argument('--policy', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error('--max-new-tokens must be positive')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.output.exists():
        parser.error('Output directory already exists')
    rows = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
    if args.limit is None and len(rows) != 200:
        parser.error('Expected 200 LongBench Multi-News examples')
    if args.limit:
        rows = rows[:args.limit]
    torch.set_num_threads(4)
    torch.manual_seed(20261028)
    model, tokenizer = load(args.model, args.adapter, args.device)
    configuration = json.loads(args.policy.read_text()) if args.policy else None
    if configuration:
        configuration = configuration.get('selected', configuration)
        configuration = configuration.get('multi_news', configuration)
    args.output.mkdir(parents=True)
    scores = {'static': []} if configuration is None else {'compensated': [], 'direct': []}
    metadata = dict(model=provenance(args.model, args.adapter), recipe=args.recipe, policy=configuration,
        data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(), input_cap=4096,
        max_new_tokens=args.max_new_tokens, torch=torch.__version__)
    (args.output/'configuration.json').write_text(json.dumps(metadata, indent=2)+'\n')
    with recipe(model, args.recipe), torch.inference_mode(), (args.output/'samples.jsonl').open('w') as log:
        for index, row in enumerate(rows):
            prefix, start = prompt_input(tokenizer, row)
            record = dict(index=index, input_tokens=len(prefix), answers=row['answers'])
            if configuration is None:
                inputs = torch.tensor([prefix], device=model.device)
                output = model.generate(inputs, attention_mask=torch.ones_like(inputs), do_sample=False,
                    num_beams=1, max_new_tokens=args.max_new_tokens, use_cache=True, pad_token_id=tokenizer.pad_token_id, logits_processor=[FiniteLogits()])
                tokens = output[0, len(prefix):].tolist()
                record['static'] = dict(prediction=tokenizer.decode(tokens, skip_special_tokens=True).strip(), generated_ids=tokens)
            else:
                prepared = runtime.probe(model, prefix, start, configuration)
                record['compensated'] = generate(runtime, model, tokenizer, prefix, prepared, configuration, args.max_new_tokens)
                direct_cfg = dict(configuration, compensation=False)
                record['direct'] = generate(runtime, model, tokenizer, prefix, prepared, direct_cfg, args.max_new_tokens,
                    replay=record['compensated']['policy'])
                for key in ('prefix_mask', 'final_mask', 'decode_window_completed'):
                    assert record['compensated']['policy'][key] == record['direct']['policy'][key]
            for variant in scores:
                value = max(rouge_score(record[variant]['prediction'], answer) for answer in row['answers'])
                record[variant]['score'] = value
                scores[variant].append(value)
            log.write(json.dumps(record, ensure_ascii=False)+'\n')
            log.flush()
            print(index+1, len(rows), flush=True)
    summary = dict(completed=args.limit is None and args.max_new_tokens == 512, count=len(rows),
        rouge_l={key:100*sum(values)/len(values) for key, values in scores.items()})
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')

if __name__ == '__main__':
    main()
