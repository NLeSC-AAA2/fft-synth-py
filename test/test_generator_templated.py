import pytest
import numpy as np
from kernel_tuner import run_kernel # type: ignore

from fftsynth.parity import ParitySplitting, parity
from fftsynth.twiddle import make_twiddle
from fftsynth.generator_templated import generate_transpose_function, generate_parity_function, generate_twiddle_array

cases = [
    ParitySplitting(64, 4),
    ParitySplitting(81, 3),
    ParitySplitting(125, 5)
]


@pytest.mark.parametrize("parity_splitting", cases)
def test_parity_4(parity_splitting):
    kernel = generate_parity_function(parity_splitting)
    x = np.arange(parity_splitting.N, dtype=np.int32)
    y = np.zeros_like(x)
    kernel_args = [x, y]

    results = run_kernel("test_parity", kernel, parity_splitting.N, kernel_args, {}, compiler_options=["-DTESTING"])
    y_ref = np.array([parity(parity_splitting.radix, i) for i in range(parity_splitting.N)])

    assert np.all(results[1] == y_ref)


@pytest.mark.parametrize('parity_splitting', cases)
def test_transpose_4(parity_splitting):
    kernel = generate_transpose_function(parity_splitting)
    x = np.arange(parity_splitting.N, dtype=np.int32)
    y = np.zeros_like(x)
    kernel_args = [x, y]

    results = run_kernel("test_transpose", kernel, parity_splitting.N, kernel_args, {}, compiler_options=["-DTESTING"])
    y_ref = x.reshape(parity_splitting.factors).T.flatten()

    assert np.all(results[1] == y_ref)


@pytest.mark.parametrize('parity_splitting', cases)
def test_generate_twiddle_array(parity_splitting):
    print(generate_twiddle_array(parity_splitting, make_twiddle(parity_splitting.N, parity_splitting.N)))
    assert(True)
