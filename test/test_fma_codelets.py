# ~\~ language=Python filename=test/test_fma_codelets.py
# ~\~ begin <<lit/fma-codelets.md|test/test_fma_codelets.py>>[0]
import numpy as np

import pytest
from kernel_tuner import run_kernel     # type: ignore

from pkg_resources import resource_filename

filename = resource_filename("fftsynth", "opencl/fma-codelets.cl")


@pytest.mark.parametrize('radix', [2, 3, 4, 5])
def test_radix(radix):
    #this test runs 256 instances of the radix n function
    #it does not use twiddle factors, so as a test
    #it's not to be relied upon fully
    n = np.int32(256)
    x = np.random.normal(size=(n, radix, 2)).astype(np.float32)
    y = np.zeros_like(x)

    y_ref = np.fft.fft(x[...,0]+1j*x[...,1])

    args = [x, y, n]
    answer = run_kernel("test_radix" + str(radix), filename, 1, args, {})

    print(answer)
    y = answer[1]
    y = y[...,0]+1j*y[...,1]
    print(y)
    print(y_ref)

    assert abs(y - y_ref).max() < 1e-4
# ~\~ end