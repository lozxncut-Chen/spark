from contextlib import contextmanager
import hashlib
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LogitsProcessor
from .operators import ACTIVATION, SOFTMAX

class FiniteLogits(LogitsProcessor):
    def __call__(self, input_ids, scores):
        if not torch.isfinite(scores).all():
            raise FloatingPointError('Nonfinite generation logits')
        return scores


class Activation(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, x):
        return ACTIVATION[self.name](x)


def load(model_path, adapter=None, device='cuda:0'):
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    if config.model_type not in ('llama', 'mistral'):
        raise ValueError('Supported model types: llama, mistral')
    config.hidden_act = 'gelu'
    dtype = torch.bfloat16 if device.startswith('cuda') else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=dtype,
        attn_implementation='eager', local_files_only=True).to(device).eval()
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, local_files_only=True).merge_and_unload().eval()
    return model, tokenizer


def provenance(model_path, adapter=None):
    paths = [('model', Path(model_path))] + ([('adapter', Path(adapter))] if adapter else [])
    return {name: {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()
        if p.is_file() and p.suffix in ('.json', '.safetensors', '.bin', '.model')} for name, root in paths}
@contextmanager
def _recipe(model, activation, softmax, audit=False):
    layers = model.model.layers
    originals = [x.mlp.act_fn for x in layers]
    counts = [0] * len(layers)
    handles = [layer.register_forward_hook(lambda m, a, o, i=i: counts.__setitem__(i, counts[i]+1))
               for i, layer in enumerate(layers)]
    for layer in layers:
        layer.mlp.act_fn = torch.nn.GELU() if activation == "exact" else Activation(activation)
    original_softmax = F.softmax
    stats = {"attention_calls": 0, "nonfinite_calls": 0, "first_prefill_ranges": []}

    def patched(x, *args, **kwargs):
        dim = kwargs.get("dim", args[0] if args else None)
        if x.ndim == 4 and dim == -1:
            stats["attention_calls"] += 1
            public_padding = stats.get("_public_query_padding")
            if softmax == "spark" and public_padding is not None and x.shape[-2] == public_padding.shape[-2]:


                x = x.masked_fill(public_padding, 0.0)
            if audit and x.shape[-2] > 1 and len(stats["first_prefill_ranges"]) < 32:
                z = x.float() - x.float().amax(-1, keepdim=True)
                valid = x.float() > -10000
                if public_padding is not None and x.shape[-2] == public_padding.shape[-2]:
                    valid = valid & ~public_padding
                values = z[valid]
                stats["first_prefill_ranges"].append({"min_shift": float(values.min()),
                    "fraction_below_minus32": float((values < -32).float().mean()),
                    "fraction_below_minus64": float((values < -64).float().mean())})
            result = original_softmax(x, *args, **kwargs) if softmax == "exact" else SOFTMAX[softmax](x)
            if audit and len(stats["first_prefill_ranges"]) < 32 and not torch.isfinite(result).all():
                stats["nonfinite_calls"] += 1
            return result
        return original_softmax(x, *args, **kwargs)

    F.softmax = patched
    try:
        yield stats, counts
    finally:
        F.softmax = original_softmax
        for layer, original in zip(layers, originals):
            layer.mlp.act_fn = original
        for handle in handles:
            handle.remove()


@contextmanager
def recipe(model, name):
    originals = SOFTMAX.copy()
    for key, function in originals.items():
        def chunked(x, fn=function):
            if x.shape[-2] <= 128:
                return fn(x)
            output = torch.empty_like(x)
            for start in range(0, x.shape[-2], 128):
                output[..., start:start+128, :] = fn(x[..., start:start+128, :])
            return output
        SOFTMAX[key] = chunked
    try:
        with _recipe(model, name, name) as stats:
            yield stats
    finally:
        SOFTMAX.clear()
        SOFTMAX.update(originals)
