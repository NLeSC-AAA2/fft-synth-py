import pytest
import numpy as np
from kernel_tuner import run_kernel  # type: ignore

from fftsynth.parity import ParitySplitting, parity
from fftsynth.generator_templated import generate_preprocessor, generate_transpose_function, generate_parity_function

cases = [
    ParitySplitting(64, 4),
    ParitySplitting(81, 3),
    ParitySplitting(125, 5)
]


@pytest.mark.parametrize("parity_splitting", cases)
def test_parity(parity_splitting: ParitySplitting):
    kernel = generate_preprocessor(parity_splitting) + generate_parity_function(parity_splitting)
    x = np.arange(parity_splitting.N, dtype=np.int32)
    y = np.zeros_like(x)
    kernel_args = [x, y]

    results = run_kernel("test_parity_{}".format(parity_splitting.radix), kernel, parity_splitting.N, kernel_args, {}, compiler_options=["-DTESTING"])
    y_ref = np.array([parity(parity_splitting.radix, i) for i in range(parity_splitting.N)])

    assert np.all(results[1] == y_ref)


@pytest.mark.parametrize('parity_splitting', cases)
def test_transpose(parity_splitting: ParitySplitting):
    kernel = generate_preprocessor(parity_splitting) + generate_transpose_function(parity_splitting)
    x = np.arange(parity_splitting.N, dtype=np.int32)
    y = np.zeros_like(x)
    kernel_args = [x, y]

    results = run_kernel("test_transpose_{}".format(parity_splitting.radix), kernel, parity_splitting.N, kernel_args, {}, compiler_options=["-DTESTING"])
    y_ref = x.reshape(parity_splitting.factors).T.flatten()

    assert np.all(results[1] == y_ref)
