# ~\~ language=Python filename=test/test_generator.py
# ~\~ begin <<lit/code-generator.md|test/test_generator.py>>[0]
import pytest
import numpy as np
from kernel_tuner import run_kernel  # type: ignore

from fftsynth.parity import ParitySplitting, parity
from fftsynth.generator import gen_parity_fn, gen_transpose_fn

cases = [
    ParitySplitting(64, 4),
    ParitySplitting(81, 3),
    ParitySplitting(125, 5)
]

# ~\~ begin <<lit/code-generator.md|test-parity>>[0]
@pytest.mark.parametrize('ps', cases)
def test_parity_4(ps: ParitySplitting):
    source = gen_parity_fn(ps)
    kernel = f"""
    #define DIVR(x) ((x) / {ps.radix})
    #define MODR(x) ((x) % {ps.radix})
    #define MULR(x) ((x) * {ps.radix})
    {source}

    __kernel void test_parity(__global const int *x, __global int *y) {{
        int i = get_global_id(0);
        y[i] = parity_{ps.radix}(x[i]);
    }}
    """
    x = np.arange(ps.N, dtype=np.int32)
    y = np.zeros_like(x)
    kernel_args = [x, y]

    results = run_kernel("test_parity", kernel, ps.N, kernel_args, {}, compiler_options=["-DTESTING"])
    y_ref = np.array([parity(ps.radix, i) for i in range(ps.N)]) 

    assert np.all(results[1] == y_ref)
# ~\~ end

# ~\~ begin <<lit/code-generator.md|test-transpose>>[0]
@pytest.mark.parametrize('ps', cases)
def test_transpose_4(ps: ParitySplitting):
    source = gen_transpose_fn(ps)
    kernel = f"""
    #define DIVR(x) ((x) / {ps.radix})
    #define MODR(x) ((x) % {ps.radix})
    #define MULR(x) ((x) * {ps.radix})
    {source}

    __kernel void test_transpose(__global const int *x, __global int *y) {{
        int i = get_global_id(0);
        y[i] = transpose_{ps.radix}(x[i]);
    }}
    """
    x = np.arange(ps.N, dtype=np.int32)
    y = np.zeros_like(x)
    kernel_args = [x, y]

    results = run_kernel("test_transpose", kernel, ps.N, kernel_args, {}, compiler_options=["-DTESTING"])
    y_ref = x.reshape(ps.factors).T.flatten()

    assert np.all(results[1] == y_ref)
# ~\~ end
# ~\~ end
