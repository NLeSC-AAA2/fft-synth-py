# ~\~ language=Python filename=test/test_generator.py
# ~\~ begin <<lit/code-generator.md|test/test_generator.py>>[0]
import pytest
import numpy as np
from kernel_tuner import run_kernel  # type: ignore

from fftsynth.parity import ParitySplitting, parity
from fftsynth.generator import generate_preprocessor, generate_transpose_function, generate_parity_function, generate_fft, generate_fma_fft

cases = [
    ParitySplitting(128, 2),
    ParitySplitting(64, 4)
]


# ~\~ begin <<lit/code-generator.md|test-parity>>[0]
@pytest.mark.parametrize("parity_splitting", cases)
def test_parity(parity_splitting: ParitySplitting):
    kernel = generate_preprocessor(parity_splitting, False) + generate_parity_function(parity_splitting)
    x = np.arange(parity_splitting.N, dtype=np.int32)
    y = np.zeros_like(x)
    kernel_args = [x, y]

    results = run_kernel("test_parity_{}".format(parity_splitting.radix), kernel, parity_splitting.N, kernel_args, {}, compiler_options=["-DTESTING"])
    y_ref = np.array([parity(parity_splitting.radix, i) for i in range(parity_splitting.N)])

    assert np.all(results[1] == y_ref)
# ~\~ end

# ~\~ begin <<lit/code-generator.md|test-transpose>>[0]
@pytest.mark.parametrize('parity_splitting', cases)
def test_transpose(parity_splitting: ParitySplitting):
    kernel = generate_preprocessor(parity_splitting, False) + generate_transpose_function(parity_splitting)
    x = np.arange(parity_splitting.N, dtype=np.int32)
    y = np.zeros_like(x)
    kernel_args = [x, y]

    results = run_kernel("test_transpose_{}".format(parity_splitting.radix), kernel, parity_splitting.N, kernel_args, {}, compiler_options=["-DTESTING"])
    y_ref = x.reshape(parity_splitting.factors).T.flatten()

    assert np.all(results[1] == y_ref)
# ~\~ end

# ~\~ begin <<lit/code-generator.md|test-fft>>[0]
@pytest.mark.parametrize('parity_splitting', cases)
def test_fft(parity_splitting: ParitySplitting):
    kernel = generate_fft(parity_splitting, False)

    x = np.random.normal(size=(parity_splitting.N, 2)).astype(np.float32)
    y = np.zeros_like(x)

    results = run_kernel(
        f"fft_{parity_splitting.N}", kernel, parity_splitting.N, [x, y], {})
    y_ref = np.fft.fft(x[:, 0] + 1j * x[:, 1])
    y = results[1][:, 0] + 1j * results[1][:, 1]
    np.testing.assert_almost_equal(y, y_ref, decimal=4)
# ~\~ end

# ~\~ begin <<lit/code-generator.md|test-fft-fma>>[0]
@pytest.mark.parametrize('parity_splitting', cases)
def test_fft_fma(parity_splitting: ParitySplitting):
    kernel = generate_fma_fft(parity_splitting, False)

    x = np.random.normal(size=(parity_splitting.N, 2)).astype(np.float32)
    y = np.zeros_like(x)

    results = run_kernel(
        f"fft_{parity_splitting.N}", kernel, parity_splitting.N, [x, y], {})
    y_ref = np.fft.fft(x[:, 0] + 1j * x[:, 1])
    y = results[1][:, 0] + 1j * results[1][:, 1]
    np.testing.assert_almost_equal(y, y_ref, decimal=4)
# ~\~ end

# ~\~ end
