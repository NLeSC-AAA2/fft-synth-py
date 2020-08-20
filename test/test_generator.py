# ~\~ language=Python filename=test/test_generator.py
# ~\~ begin <<lit/code-generator.md|test/test_generator.py>>[0]
import pytest
import numpy as np
import pyopencl as cl
from fftsynth.parity import ParitySplitting

cases = [
    ParitySplitting(64, 4),
    ParitySplitting(81, 3),
    ParitySplitting(125, 5)
]

mf = cl.mem_flags

# ~\~ begin <<lit/code-generator.md|test-parity>>[0]
@pytest.mark.parametrize('ps', cases)
def test_parity_4(cl_context, ps):
    from fftsynth.generator import gen_parity_fn
    from fftsynth.parity import parity
    source = gen_parity_fn(ps)
    kernel = f"""
    {source}

    __kernel void test_parity(__global const int *x, __global int *y) {{
        int i = get_global_id(0);
        y[i] = parity_{ps.radix}(x[i]);
    }}
    """
    program = cl.Program(cl_context, kernel).build(["-DTESTING"])
    queue = cl.CommandQueue(cl_context)
    x = np.arange(ps.N, dtype=cl.cltypes.int)
    y = np.zeros_like(x)
    x_g = cl.Buffer(cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
    y_g = cl.Buffer(cl_context, mf.WRITE_ONLY, x.nbytes)
    program.test_parity(queue, (ps.N,), None, x_g, y_g)
    cl.enqueue_copy(queue, y, y_g)

    y_ref = np.array([parity(ps.radix, i) for i in range(ps.N)]) 
    assert np.all(y == y_ref)
# ~\~ end
# ~\~ begin <<lit/code-generator.md|test-transpose>>[0]
@pytest.mark.parametrize('ps', cases)
def test_transpose_4(cl_context, ps):
    from fftsynth.generator import gen_transpose_fn
    source = gen_transpose_fn(ps)
    kernel = f"""
    {source}

    __kernel void test_transpose(__global const int *x, __global int *y) {{
        int i = get_global_id(0);
        y[i] = transpose_{ps.radix}(x[i]);
    }}
    """
    program = cl.Program(cl_context, kernel).build(["-DTESTING"])
    queue = cl.CommandQueue(cl_context)
    x = np.arange(ps.N, dtype=cl.cltypes.int)
    y = np.zeros_like(x)
    x_g = cl.Buffer(cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
    y_g = cl.Buffer(cl_context, mf.WRITE_ONLY, x.nbytes)
    program.test_transpose(queue, (ps.N,), None, x_g, y_g)
    cl.enqueue_copy(queue, y, y_g)

    y_ref = x.reshape(ps.factors).T.flatten()
    assert np.all(y == y_ref)
# ~\~ end
# ~\~ end
