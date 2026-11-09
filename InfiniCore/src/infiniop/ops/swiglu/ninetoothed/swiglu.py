import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(output, up, gate, block_size):
    return (
        output.tile((block_size,)),
        up.tile((block_size,)),
        gate.tile((block_size,)),
    )


def application(output, up, gate):
    gate_f32 = ntl.cast(gate, ntl.float32)
    sigmoid = ntl.cast(ntl.sigmoid(gate_f32), gate.dtype)

    if gate.dtype == ninetoothed.bfloat16:
        sigmoid = ntl.cast(sigmoid, ntl.float32)
        output = ntl.cast(
            (gate_f32 * sigmoid) * ntl.cast(up, ntl.float32), gate.dtype
        )
    else:
        output = (gate * sigmoid) * up


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)
    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors
