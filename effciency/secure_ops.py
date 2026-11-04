import json
import math
from pathlib import Path

import crypten
import crypten.communicator as comm
from crypten.config import cfg
import torch

EAST = json.loads(Path(__file__).with_name("east_coefficients.json").read_text())

def share(shape, rank, device, scale=0.02):
    source = torch.randn(shape, device=device) * scale if rank == 1 else torch.empty(shape, device=device)
    result = crypten.cryptensor(source, src=1)
    del source
    return result


def norm(x, weight, eps=1e-05):
    return x * (x.square().mean(dim=-1, keepdim=True) + eps).inv_sqrt() * weight

def puma_gelu(x):
    x2 = x.square()
    x3 = x2 * x
    x4 = x2.square()
    x6 = x3.square()
    low = -0.011034134030615728 * x3 - 0.11807612951181953 * x2 - 0.42226581151983866 * x - 0.5054031199708174
    mid = 0.0018067462606141187 * x6 - 0.037688200365904236 * x4 + 0.3603292692789629 * x2 + 0.5 * x + 0.008526321541038084
    b0 = x < -4
    b1 = x < -1.95
    b2 = x > 3
    return (b1 - b0) * low + (1 - b1 - b2) * mid + b2 * x

def east_oppe(x, edges, coefficients):
    x2 = x.square()
    x3 = x2 * x
    result = x * 0
    previous = x * 0
    for (edge, coeff) in zip(edges, coefficients):
        value = coeff[0] + coeff[1] * x + coeff[2] * x2 + coeff[3] * x3
        result = result + (x > edge) * (value - previous)
        previous = value
    return result

def east_gelu(x):
    if x.size(1) > 64:
        return crypten.cat([east_gelu(part) for part in x.split(64, dim=1)], dim=1)
    return east_oppe(x, EAST['gelu_edges'], EAST['gelu_coefficients'])

def spark_gelu(x):
    boundary = 1.768699454455912
    erf_input = x / math.sqrt(2.0)
    below = erf_input < -boundary
    above = erf_input > boundary
    u = erf_input / boundary
    t1 = u
    t2 = 2 * u * t1 - 1
    t3 = 2 * u * t2 - t1
    t4 = 2 * u * t3 - t2
    t5 = 2 * u * t4 - t3
    middle = 1.154338547453227 * t1 - 0.19315335830413669 * t3 + 0.038814810850909626 * t5
    erf = -below + (1 - below - above) * middle + above
    return 0.5 * x * (1 + erf)

def softmax(scores, mask, variant):
    if variant in ('crypten', 'crypten_tanh'):
        masked = scores * mask + (1 - mask) * -32.0
        if variant == 'crypten_tanh':
            with cfg.temp_override({'functions.exp_iterations': 8, 'functions.reciprocal_initial': 1 / scores.size(-1), 'functions.reciprocal_nr_iters': 16}):
                return masked.softmax(dim=-1) * mask
        return masked.softmax(dim=-1) * mask
    if variant == 'east' and scores.size(-2) > 64:
        parts = []
        for start in range(0, scores.size(-2), 64):
            part = scores[:, :, start:start + 64]
            parts.append(softmax(part, mask[start:start + part.size(-2)], variant))
        return crypten.cat(parts, dim=-2)
    if variant in ('puma_nomax', 'spark_nomax'):
        shifted = scores * mask + (1 - mask) * -32.0
    else:
        masked = scores * mask + (1 - mask) * -1000.0
        maximum = masked.max(dim=-1, keepdim=True)[0]
        shifted = (scores - maximum) * mask + (1 - mask) * -32.0
    if variant == 'native':
        numerator = shifted.exp() * mask
    elif variant == 'east':
        numerator = east_oppe(shifted, EAST['exp_edges'], EAST['exp_coefficients']) * mask
    else:
        numerator = 1 + shifted / 32
        for _ in range(5):
            numerator = numerator.square()
        if variant in ('puma', 'puma_nomax'):
            numerator = numerator * (shifted > -14)
        numerator = numerator * mask
    denominator = numerator.sum(dim=-1, keepdim=True)
    if variant == 'east':
        inverse = denominator * 0 + 2.0 / (scores.size(-1) + 1)
        for _ in range((2 * scores.size(-1)).bit_length() + 6):
            inverse = inverse * (2 - denominator * inverse)
    else:
        inverse = denominator.reciprocal()
    return numerator * inverse

def repeated_kv(x, groups):
    return crypten.cat([x[:, i:i + 1].repeat(1, groups, 1, 1) for i in range(x.size(1))], dim=1)

def rope(x, positions, config):
    width = x.size(-1)
    freq = 1.0 / config['rope_theta'] ** (torch.arange(0, width, 2, device=x.device).float() / width)
    scaling = config.get('rope_scaling')
    if scaling and scaling.get('rope_type') == 'llama3':
        wavelength = 2 * math.pi / freq
        old = scaling['original_max_position_embeddings']
        (low, high) = (scaling['low_freq_factor'], scaling['high_freq_factor'])
        smooth = (old / wavelength - low) / (high - low)
        medium = (1 - smooth) * freq / scaling['factor'] + smooth * freq
        freq = torch.where(wavelength > old / low, freq / scaling['factor'], freq)
        freq = torch.where((wavelength <= old / low) & (wavelength >= old / high), medium, freq)
    angles = torch.outer(positions.float(), freq)
    angles = torch.cat([angles, angles], dim=-1)[None, None]
    half = width // 2
    rotated = crypten.cat([-x[..., half:], x[..., :half]], dim=-1) if crypten.is_encrypted_tensor(x) else torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * angles.cos() + rotated * angles.sin()

def block(hidden, weights, past, variant, dimensions, result, config=None, skip_attention=False, skip_ffn=False, attention_scale=1.0, ffn_scale=1.0):
    (width, heads, kv_heads, intermediate) = dimensions
    head_width = width // heads
    query_length = hidden.size(1)
    key_length = query_length + (0 if past is None else past[0].size(-2))
    eps = 1e-05 if config is None else config['rms_norm_eps']
    if skip_attention:
        (residual, cache) = (hidden * attention_scale, past)
    else:
        state = measured('rmsnorm_input', lambda : norm(hidden, weights['norm1'], eps), result)
        q = measured('q_projection', lambda : state.matmul(weights['q']), result)
        k = measured('k_projection', lambda : state.matmul(weights['k']), result)
        v = measured('v_projection', lambda : state.matmul(weights['v']), result)
        q = q.reshape(1, query_length, heads, head_width).transpose(1, 2)
        k = k.reshape(1, query_length, kv_heads, head_width).transpose(1, 2)
        v = v.reshape(1, query_length, kv_heads, head_width).transpose(1, 2)
        if config is not None:
            positions = torch.arange(key_length - query_length, key_length, device=hidden.device)
            q = measured('rope_query', lambda : rope(q, positions, config), result)
            k = measured('rope_key', lambda : rope(k, positions, config), result)
        if past is not None:
            k = crypten.cat([past[0], k], dim=-2)
            v = crypten.cat([past[1], v], dim=-2)
        cache = (k, v)
        k = repeated_kv(k, heads // kv_heads)
        v = repeated_kv(v, heads // kv_heads)
        scores = measured('attention_scores', lambda : q.matmul(k.transpose(-1, -2)) / head_width ** 0.5, result)
        mask = torch.tril(torch.ones((query_length, key_length), device=hidden.device), diagonal=key_length - query_length)
        if config is not None and config.get('sliding_window'):
            queries = torch.arange(key_length - query_length, key_length, device=hidden.device)[:, None]
            keys = torch.arange(key_length, device=hidden.device)[None, :]
            mask = mask * (keys > queries - config['sliding_window'])
        probs = measured('attention_softmax', lambda : softmax(scores, mask, variant), result)
        context = measured('attention_values', lambda : probs.matmul(v), result)
        context = context.transpose(1, 2).reshape(1, query_length, width)
        attn = measured('o_projection', lambda : context.matmul(weights['o']), result)
        residual = hidden + attn
    if skip_ffn:
        return (residual * ffn_scale, cache)
    state = measured('rmsnorm_ffn', lambda : norm(residual, weights['norm2'], eps), result)
    gate = measured('gate_projection', lambda : state.matmul(weights['gate']), result)
    up = measured('up_projection', lambda : state.matmul(weights['up']), result)
    activation = {'native': lambda x: x * x.sigmoid(), 'crypten': lambda x: 0.5 * x * (1 + (x / math.sqrt(2.0)).erf()), 'crypten_tanh': lambda x: 0.5 * x * (1 + (math.sqrt(2 / math.pi) * (x + 0.044715 * x.square() * x)).tanh()), 'puma': puma_gelu, 'puma_nomax': puma_gelu, 'east': east_gelu, 'spark': spark_gelu, 'spark_nomax': spark_gelu}[variant]
    active = measured('activation', lambda : activation(gate), result)
    combined = measured('gated_product', lambda : active * up, result)
    down = measured('down_projection', lambda : combined.matmul(weights['down']), result)
    return (residual + down, cache)
from .timing import operation as measured
