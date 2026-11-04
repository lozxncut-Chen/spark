import time
import torch
import crypten.communicator as comm
def configure_cuda_integer_encoding():


    from crypten.cuda.cuda_tensor import CUDALongTensor
    def encode(x,num_blocks=3):
        bits=CUDALongTensor._CUDALongTensor__BLOCK_SIZE[num_blocks]
        return CUDALongTensor.stack([(x>>(bits*i))&((1<<bits)-1) for i in range(num_blocks)]).double()
    CUDALongTensor._CUDALongTensor__encode_as_fp64=staticmethod(encode)

def measured(function):

    if torch.cuda.is_initialized():torch.cuda.synchronize()
    before=comm.get().get_communication_stats().copy()
    start=time.perf_counter()
    value=function()
    if torch.cuda.is_initialized():torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    after=comm.get().get_communication_stats().copy()
    record=dict(rank=comm.get().get_rank(),end_to_end_seconds=elapsed,
        communication_seconds=after['time']-before['time'],
        communication_bytes=after['bytes']-before['bytes'],
        communication_rounds=after['rounds']-before['rounds'])
    assert 0<=record['communication_seconds']<=elapsed+1e-6
    return value,record


def operation(label, function, ledger):
    value, record = measured(function)
    ledger[label] = record
    return value
