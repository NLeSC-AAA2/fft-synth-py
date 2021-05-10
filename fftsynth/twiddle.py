# ~\~ language=Python filename=fftsynth/twiddle.py
# ~\~ begin <<lit/parity-splitting.md|fftsynth/twiddle.py>>[0]
import numpy as np

# ~\~ begin <<lit/parity-splitting.md|make-twiddle>>[0]
def make_twiddle(n1: int, n2: int, forward: bool):
    def w(k, n):
        if forward:
            return np.exp(2j * np.pi * k / n)
        else:
            return np.exp(-2j * np.pi * k / n)

    I1 = np.arange(n1)
    I2 = np.arange(n2)
    return w(I1[:, None] * I2[None, :], n1 * n2).astype('complex64')
# ~\~ end
# ~\~ end
