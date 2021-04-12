# ~\~ language=Python filename=test/test_fma_codelets.py
# ~\~ begin <<lit/fma-codelets.md|test/test_fma_codelets.py>>[0]
import numpy

import pytest
from fftsynth import generator, parity
from kernel_tuner import run_kernel     # type: ignore


@pytest.mark.parametrize('radix', [2, 3, 4])
def test_radix(radix):
    # this test runs 256 instances of the radix n function
    # it does not use twiddle factors, so as a test
    # it's not to be relied upon fully
    n = numpy.int32(256)
    x = numpy.random.normal(size=(n, radix, 2)).astype(numpy.float32)
    y = numpy.zeros_like(x)

    y_ref = numpy.fft.fft(x[..., 0]+1j*x[..., 1])

    parity_splitting = parity.ParitySplitting(radix * n, radix)
    codelets = "{}\n{}".format(generator.generate_preprocessor(parity_splitting, False),
                               generator.generate_fma_codelets(parity_splitting, False))
    args = [x, y, n]
    answer = run_kernel(f"test_radix_{radix}", codelets, 1, args, {}, compiler_options=["-DTESTING"])

    y = answer[1]
    y = y[..., 0] + 1j * y[..., 1]

    numpy.testing.assert_almost_equal(y, y_ref, decimal=5)
# ~\~ end
