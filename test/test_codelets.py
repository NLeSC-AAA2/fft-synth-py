# ~\~ language=Python filename=test/test_codelets.py
# ~\~ begin <<lit/code-generator.md|test/test_codelets.py>>[0]
import numpy

import pytest
from fftsynth import generator, parity
from kernel_tuner import run_kernel     # type: ignore


@pytest.mark.parametrize('radix,c_type', [(2, 'float2'), (4, 'float2'),
                                          (2, 'float4'), (4, 'float4'),
                                          (2, 'float8'), (4, 'float8')])
def test_radix(radix, c_type):
    # this test runs 256 instances of the radix n function
    # it does not use twiddle factors, so as a test
    # it's not to be relied upon fully
    n = numpy.int32(256)
    m = {'float2': 1, 'float4': 2, 'float8': 4}[c_type]
    x = numpy.random.normal(size=(n, radix, m, 2)).astype(numpy.float32)
    y = numpy.zeros_like(x)

    y_ref = numpy.fft.fft(x[..., 0] + 1j * x[..., 1], axis=1)

    parity_splitting = parity.ParitySplitting(radix * n, radix)
    codelets = "{}\n{}\n{}".format(generator.generate_preprocessor(parity_splitting, False, c_type=c_type),
                                   generator.generate_twiddle_array(parity_splitting, True),
                                   generator.generate_codelets(parity_splitting, False, c_type=c_type))
    args = [x, y, n]
    answer = run_kernel(f"test_radix_{radix}", codelets, 1, args, {}, compiler_options=["-DTESTING_RADIX"])

    y = answer[1]
    y = y[..., 0] + 1j * y[..., 1]

    numpy.testing.assert_almost_equal(y, y_ref, decimal=5)
# ~\~ end
