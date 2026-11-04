from __future__ import annotations

import torch
import torch.nn.functional as F
import json
import math
from pathlib import Path


EAST_COEFFS = json.loads(Path(__file__).with_name("east_coefficients.json").read_text())


def puma_gelu(x: torch.Tensor) -> torch.Tensor:
    z = x.float()
    z2 = z * z
    z3 = z2 * z
    z4 = z2 * z2
    z6 = z3 * z3
    low = (-0.011034134030615728 * z3 - 0.11807612951181953 * z2
           - 0.42226581151983866 * z - 0.5054031199708174)
    middle = (0.0018067462606141187 * z6 - 0.037688200365904236 * z4
              + 0.3603292692789629 * z2 + 0.5 * z + 0.008526321541038084)
    y = torch.where(z < -4, 0.0, torch.where(z < -1.95, low,
                    torch.where(z > 3, z, middle)))
    return y.to(x.dtype)


def _shift(x: torch.Tensor) -> torch.Tensor:
    z = x.float()
    return z - z.amax(dim=-1, keepdim=True)


def puma_softmax(x: torch.Tensor, **_: object) -> torch.Tensor:
    shifted = _shift(x)
    y = 1 + shifted / 32
    for _ in range(5):
        y = y.square()
    y = torch.where(shifted > -14, y, 0.0)
    return (y / y.sum(dim=-1, keepdim=True)).to(x.dtype)


def spark_softmax(x: torch.Tensor, **_: object) -> torch.Tensor:

    shifted = _shift(x)
    shifted = torch.where(x.float() < -10000, -32.0, shifted)
    y = 1 + shifted / 32
    for _ in range(5):
        y = y.square()
    return (y / y.sum(dim=-1, keepdim=True)).to(x.dtype)


def _east_poly(x: torch.Tensor, edges: list[float], coefficients: list[list[float]]) -> torch.Tensor:
    boundaries = torch.tensor(edges, device=x.device, dtype=x.dtype)
    table = torch.tensor(coefficients, device=x.device, dtype=x.dtype)
    piece = torch.bucketize(x, boundaries) - 1
    chosen = table[piece.clamp(0, len(coefficients) - 1)]
    result = ((chosen[..., 3] * x + chosen[..., 2]) * x + chosen[..., 1]) * x + chosen[..., 0]
    return torch.where(piece < 0, 0.0, result)


def east_gelu(x: torch.Tensor) -> torch.Tensor:
    z = x.float()
    return _east_poly(z, EAST_COEFFS["gelu_edges"], EAST_COEFFS["gelu_coefficients"]).to(x.dtype)


def spark_gelu(x: torch.Tensor) -> torch.Tensor:
    z = x.float()
    erf_input = z / math.sqrt(2.0)
    boundary = 1.768699454455912
    u = erf_input / boundary
    t1 = u
    t2 = 2 * u * t1 - 1
    t3 = 2 * u * t2 - t1
    t4 = 2 * u * t3 - t2
    t5 = 2 * u * t4 - t3
    middle = (1.154338547453227 * t1 - 0.19315335830413669 * t3
              + 0.038814810850909626 * t5)
    erf = torch.where(erf_input < -boundary, -1.0,
                      torch.where(erf_input > boundary, 1.0, middle))
    return (0.5 * z * (1 + erf)).to(x.dtype)


def _east_softmax_chunk(x: torch.Tensor) -> torch.Tensor:
    shifted = _shift(x)
    numerator = _east_poly(shifted, EAST_COEFFS["exp_edges"], EAST_COEFFS["exp_coefficients"])
    numerator = torch.where(x.float() < -10000, 0.0, numerator)
    denominator = numerator.sum(dim=-1, keepdim=True)
    width = x.shape[-1]
    inverse = torch.full_like(denominator, 2.0 / (width + 1))
    for _ in range((2 * width).bit_length() + 6):
        inverse = inverse * (2 - denominator * inverse)
    return (numerator * inverse).to(x.dtype)


def east_softmax(x: torch.Tensor, **_: object) -> torch.Tensor:
    if x.ndim != 4 or x.shape[-2] <= 128:
        return _east_softmax_chunk(x)
    return torch.cat([_east_softmax_chunk(part) for part in x.split(128, dim=-2)], dim=-2)


ACTIVATION = {"native": F.silu, "puma": puma_gelu, "spark": spark_gelu, "east": east_gelu}
SOFTMAX = {"puma": puma_softmax, "spark": spark_softmax, "east": east_softmax}

def euler_exp(x, iterations=8):
    value = 1 + x / (2 ** iterations)
    for _ in range(iterations):
        value = value.square()
    return value

def crypten_gelu(x):


    z = x.float()
    argument = math.sqrt(2 / math.pi) * (z + 0.044715 * z.square() * z)
    denominator = 1 + euler_exp(-2 * argument.abs())
    inverse = torch.full_like(denominator, 0.75)
    for _ in range(3):
        inverse = inverse * (2 - denominator * inverse)
    sigmoid = torch.where(argument < 0, 1 - inverse, inverse)
    tanh = 2 * sigmoid - 1
    return (0.5 * z * (1 + tanh)).to(x.dtype)

def crypten_softmax(x):
    z = x.float()
    valid = z > -10000
    shifted = z - z.amax(-1, keepdim=True)
    shifted = torch.where(valid, shifted, 0.0)
    numerator = euler_exp(shifted) * valid
    denominator = numerator.sum(-1, keepdim=True)
    inverse = torch.full_like(denominator, 1.0 / x.shape[-1])
    for _ in range(16):
        inverse = inverse * (2 - denominator * inverse)
    return (numerator * inverse).to(x.dtype)

ACTIVATION["crypten"] = crypten_gelu
SOFTMAX["crypten"] = crypten_softmax
