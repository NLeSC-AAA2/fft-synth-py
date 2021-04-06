import numpy as np

import pytest
from fftsynth import generator, parity
from kernel_tuner import run_kernel     # type: ignore


@pytest.mark.parametrize('radix', [2, 4])
def test_radix(radix):
    # this test runs 256 instances of the radix n function
    # it does not use twiddle factors, so as a test
    # it's not to be relied upon fully
    n = np.int32(256)
    x = np.random.normal(size=(n, radix, 2)).astype(np.float32)
    y = np.zeros_like(x)

    y_ref = np.fft.fft(x[..., 0]+1j*x[..., 1])

    ps = parity.ParitySplitting(radix * n, radix)
    codelets = "{} {} {}".format(generator.generate_preprocessor(ps, False), generator.generate_twiddle_array(ps), generator.generate_codelets(False))
    args = [x, y, n]
    answer = run_kernel("test_radix_" + str(radix), codelets, 1, args, {}, compiler_options=["-DTESTING"])

    y = answer[1]
    y = y[..., 0]+1j*y[..., 1]

    assert abs(y - y_ref).max() < 1e-4
