import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import datasets
import lm_eval
import torch
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from .model import load, recipe, provenance
from .paired import PairedHFLM

COUNTS = {'sst2': 872, 'piqa': 1838, 'boolq': 3270, 'rte': 277}
TASK_DIRS = ('glue/sst2', 'glue/rte', 'super_glue/boolq', 'piqa')
SEED = 20261028

@contextmanager
def local_data(root):
    original = datasets.load_dataset
    def loader(path, name=None, **kwargs):
        task = name if name in COUNTS else ('piqa' if str(path).endswith('/piqa') else None)
        if root is None or task is None:
            return original(path, name, **kwargs)
        splits = {}
        for split in ('train', 'validation'):
            candidates = sorted((root / task).glob(split + '*.parquet'))
            if not candidates:
                raise FileNotFoundError(str(root / task / (split + '*.parquet')))
            rows = datasets.Dataset.from_parquet([str(p) for p in candidates])
            if task == 'piqa' and 'goal' not in rows.column_names:
                rows = rows.map(lambda x: {'goal': x['question'], 'sol1': x['choices'][0], 'sol2': x['choices'][1], 'label': x['answer_index']})
            elif task == 'boolq' and 'label' not in rows.column_names:
                rows = rows.rename_column('answer', 'label')
            splits[split] = rows
        return datasets.DatasetDict(splits)
    datasets.load_dataset = loader
    try:
        yield
    finally:
        datasets.load_dataset = original

class FiniteHFLM(HFLM):
    def _model_call(self, *args, **kwargs):
        logits = super()._model_call(*args, **kwargs)
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite logits')
        return logits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--adapter')
    parser.add_argument('--tasks', nargs='+', choices=COUNTS, default=list(COUNTS))
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--mode', choices=('static', 'paired'), default='static')
    parser.add_argument('--recipe', choices=('exact', 'crypten', 'puma', 'east', 'spark'), default='spark')
    parser.add_argument('--policy', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output directory already exists')
    if args.mode == 'paired' and args.policy is None:
        parser.error('paired mode requires --policy')
    if args.mode == 'static' and args.policy is not None:
        parser.error('--policy requires --mode paired')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    model, tokenizer = load(args.model, args.adapter, args.device)
    args.output.mkdir(parents=True)
    root = Path(lm_eval.__file__).parent / 'tasks'
    manager = TaskManager(include_defaults=False, include_path=[str(root / p) for p in TASK_DIRS])
    common = dict(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size, device=args.device, max_length=4096)
    cfg = json.loads(args.policy.read_text()) if args.policy else None
    if cfg:
        cfg = cfg.get('selected', cfg)
        cfg = {t: dict(cfg[t], strict_threshold=True) for t in args.tasks}
    metadata = dict(model=provenance(args.model, args.adapter), recipe=args.recipe,
        mode=args.mode, policy=cfg, torch=torch.__version__, task_source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for directory in TASK_DIRS for p in (root / directory).glob('*.yaml')},
        num_fewshot=0, apply_chat_template=False, limit=args.limit)
    (args.output / 'configuration.json').write_text(json.dumps(metadata, indent=2)+'\n')
    runner = PairedHFLM(**common, configurations=cfg, record_path=args.output/'pairs.jsonl') if cfg else FiniteHFLM(**common)
    with local_data(args.data_root), recipe(model, args.recipe), torch.inference_mode():
        for variant in ('compensated', 'direct') if cfg else ('static',):
            if cfg:
                runner.variant = variant
            result = simple_evaluate(model=runner, tasks=args.tasks, num_fewshot=0,
                batch_size=args.batch_size, limit=args.limit, log_samples=True, bootstrap_iters=0,
                task_manager=manager, random_seed=SEED, numpy_random_seed=SEED,
                torch_random_seed=SEED, fewshot_random_seed=SEED)
            samples = result.pop('samples')
            for task, rows in samples.items():
                if args.limit is None and len(rows) != COUNTS[task]:
                    raise ValueError(f'{task}: expected {COUNTS[task]} validation rows, got {len(rows)}')
                (args.output / f'{variant}_{task}.jsonl').write_text(''.join(json.dumps(r, default=str)+'\n' for r in rows))
            payload = dict(completed=args.limit is None, results=result)
            (args.output / f'{variant}.json').write_text(json.dumps(payload, indent=2, default=str)+'\n')
            print(json.dumps(result['results']), flush=True)

if __name__ == '__main__':
    main()
