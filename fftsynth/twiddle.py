## ------ language="Python" file="fftsynth/twiddle.py" project://lit/parity-splitting.md#88
import numpy as np

## ------ begin <<make-twiddle>>[0] project://lit/parity-splitting.md#94
def make_twiddle(n1, n2):
    def w(k, n):
        return np.exp(2j * np.pi * k / n)

    I1 = np.arange(n1)
    I2 = np.arange(n2)
    return w(I1[:,None] * I2[None,:], n1*n2).astype('complex64')
## ------ end
## ------ end
