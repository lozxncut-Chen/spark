import secrets
import time
import uuid
from dataclasses import dataclass
import torch
import crypten
import crypten.communicator as comm
from crypten.mpc import MPCTensor, ptype
from crypten.mpc.primitives.binary import BinarySharedTensor as BST
def random64(shape):
    return torch.frombuffer(bytearray(secrets.token_bytes(8 * int(torch.tensor(shape).prod()))), dtype=torch.int64).reshape(shape)

@dataclass(frozen=True)
class Wire:
    value: object
    wire_id: str
    version: int = 0
    scale: float = 1.0

class Engine:

    def __init__(self, mode, materials, namespace):
        self.mode = mode
        self.namespace = namespace
        self.counter = 0
        self.materials = materials
        self.used = set()
        self.trace = []
        self.stat_cache = {}

    def wire(self, x):
        self.counter += 1
        return Wire(x, f'{self.namespace}:wire:{self.counter}')

    def skip(self, x, rho):
        if self.mode != 'M1':
            return Wire(x.value, x.wire_id, x.version, x.scale * rho)
        y = self.truncate(x.value * round(rho * 65536), 16, 'compensation', x.wire_id)
        return self.wire(y)

    def truncate(self, raw, bits, slot, source_wire):
        material = self.materials[slot]
        assert material['mask_id'] not in self.used
        self.used.add(material['mask_id'])
        before = comm.get().get_communication_stats().copy()
        tick = time.perf_counter()
        opened = (raw + material['r'])._tensor.reveal()
        bind = comm.get().get_communication_stats().copy()
        db = BST.from_shares(opened if comm.get().get_rank() == 0 else torch.zeros_like(opened))
        recovered = db + material['minus_r_binary']._tensor
        reduced = recovered.rshift(bits)
        out = MPCTensor.from_shares(reduced.share, ptype=ptype.binary).to(ptype.arithmetic, precision=0)
        after = comm.get().get_communication_stats().copy()
        self.trace.append(dict(operation=slot, source_wire=source_wire, mask_id=material['mask_id'], mask_dependency='fresh random shares independent of reciprocal and data', bits=bits, vector_bindings=1, vector_truncations=1, binding_bytes=bind['bytes'] - before['bytes'], conversion_and_truncation_bytes=after['bytes'] - bind['bytes'], bytes=after['bytes'] - before['bytes'], collective_calls=after['rounds'] - before['rounds'], seconds=time.perf_counter() - tick))
        return out

    def projection(self, x, w, scale):
        start = comm.get().get_communication_stats().copy()
        tick = time.perf_counter()
        raw = x.matmul(w)
        stats = comm.get().get_communication_stats().copy()
        self.trace.append(dict(operation='secret_projection', bytes=stats['bytes'] - start['bytes'], collective_calls=stats['rounds'] - start['rounds'], seconds=time.perf_counter() - tick))
        I = 65536 if self.mode == 'M1' else round(65536 / scale)
        return self.truncate(raw if self.mode == 'M1' else raw * I, 16 if self.mode == 'M1' else 32, 'projection_output', 'raw_projection')

def prepare(shape, mode):
    slots = ['projection_output'] + (['compensation'] if mode == 'M1' else [])
    result = {}
    for slot in slots:
        r = MPCTensor.from_shares(random64(shape), precision=0)
        result[slot] = dict(r=r, minus_r_binary=(-r).to(ptype.binary), mask_id=uuid.uuid4().hex)
    return result
