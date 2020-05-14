## ------ language="Python" file="fftsynth/parity.py" project://lit/parity-splitting.md#24
from numba import jit, vectorize
import numpy as np
import math

from functools import partial
from itertools import groupby
from dataclasses import dataclass

from typing import List, Mapping, Iterator, Sequence


@jit(nopython=True)
def digits(n: int, i: int) -> List[int]:
    """Generates the n-numbered digits of i, in reverse order."""
    while True:
        if i == 0:
            return
        else:
            i, q = divmod(i, n)
            yield q


@vectorize(nopython=True)
def parity(n: int, i: int) -> int:
    """Computes the n-parity of a number i."""
    x = 0
    for j in digits(n, i):
        x += j
    return x % n

## ------ begin <<parity-channels>>[0] project://lit/parity-splitting.md#62
def channels(N: int, radix: int) -> Mapping[int, Iterator[int]]:
    """Given a 1-d contiguous array of size N, and a FFT of given radix,
    this returns a map of iterators of the different memory channels."""
    parity_r = partial(parity, radix)
    return groupby(sorted(range(N), key=parity_r), parity_r)
## ------ end
## ------ begin <<parity-index-functions>>[0] project://lit/parity-splitting.md#158
def comp_idx(radix: int, i: int, j: int, k: int) -> int:
    base = (i & ~(radix**k - 1))
    rem  = (i &  (radix**k - 1))
    return rem + j * radix**k + base * radix


def comp_perm(radix: int, i: int) -> int:
    base = (i & ~(radix - 1))
    rem = (i & (radix - 1))
    p = parity(radix, base)
    return base | ((rem - p) % radix)
## ------ end
## ------ begin <<parity-splitting-interface>>[0] project://lit/parity-splitting.md#176
@dataclass
class ParitySplitting:
    """Collects a lot of properties on the parity-splitting FFT algorithm.
    This algorithm splits memory access for radix-n algorithms into n
    chunks, such that each chunk is only accessed once for each butterfly
    operation. Pseudo code:

        for k in range(depth):
            for i in range(L):
                for j in range(radix):
                    fft(*permute(chunks), W)
    """
    N: int
    radix: int
    
    @property
    def depth(self) -> int:
        """radix-log of FFT size"""
        return int(math.log(self.N, self.radix))
    
    @property
    def M(self) -> int:
        """N // radix"""
        return self.N//self.radix
    
    @property
    def L(self) -> int:
        """N // radix**2"""
        return self.M//self.radix
    
    @property
    def factors(self) -> List[int]:
        """Shape of FFT"""
        return [self.radix] * self.depth
    
    @property
    def channels(self) -> Mapping[int, Iterator[int]]:
        """Pattern for memory chunks"""
        return channels(self.N, self.radix)
    
    @property
    def channel_loc(self) -> np.ndarray:
        x = np.zeros(shape=(self.N,), dtype=int)
        for g, i in self.channels:
            x[list(i)] = g
        return x.reshape(self.factors)
    
    @property
    def index_loc(self) -> np.ndarray:
        x = np.zeros(shape=(self.N,), dtype=int)
        for g, i in self.channels:
            x[list(i)] = np.arange(self.M, dtype=int)
        return x.reshape(self.factors)
    
    def mix(self, x: np.ndarray) -> List[np.ndarray]:
        return [x[list(i)].copy() for g, i in self.channels]

    def unmix(self, s: Sequence[np.ndarray]) -> np.ndarray:
        x = np.zeros(shape=(self.N,), dtype='complex64')
        for g, i in self.channels:
            x[list(i)] = s[g]
        return x
## ------ end
## ------ end
