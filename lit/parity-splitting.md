# Parity splitting

## Radix-n data streamlining

For a radix-2 FFT, it is possible to divide input data in such a way that every butterfly operation reads and writes to two different arrays in memory. This could be advantageous for an implementation on the FPGA. The trick is to separate data locations based on the parity of their index.

### Parity

The index of each element in the array can be written in binary. In the case of a radix-2 FFT, we can see the array being reshaped in a $[2, 2, 2, \dots]$ shape. Every stage of the $n=2^k$ sized radix-2 FFT is being performed in a different dimension of the $k$-dimensional reshaped array. The indexing in this multi-dimensional array is the same as the binary notation for the linear index. That means that each time the 2-FFT is performed, the multi-index of the elements involved will differ by one bit in the linear index. Separating the array based on the parity of the index will guarantee that each 2-FFT operation reads and writes one (complex) number from each set. The parity is the sum of all the individual bits, modulo 2.

$$P_2(i) = \left(\sum_k b_k\right) \mod 2,\quad {\rm where}\, i := \sum_k b_k 2^k$$

We can extend this concept to the radix-4 FFT, on 4 data channels.

### 4-parity

In the radix-4 FFT we can view the array as being reshaped to a $[4, 4, 4, \dots]$ shape. The multi-index into this array is equivalent to the quarternary number notation of the linear index. Similar to the radix-2 parity, we can define the radix-4 parity as the sum of each quarternary digit in the index, modulo 4.

$$P_4(i) = \left(\sum_k q_k\right) \mod 4,\quad {\rm where}\, i := \sum_k q_k 4^k$$

This would ensure that each 4-FFT reads its data from the 4 different input channels. We define the `parity` function to work with any radix:

``` {.python file=fftsynth/parity.py}
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

<<parity-channels>>
<<parity-index-functions>>
<<parity-splitting-interface>>
```

Using the parity function we can determine which index belongs to which channel

``` {.python #parity-channels}
def channels(N: int, radix: int) -> Mapping[int, Iterator[int]]:
    """Given a 1-d contiguous array of size N, and a FFT of given radix,
    this returns a map of iterators of the different memory channels."""
    parity_r = partial(parity, radix)
    return groupby(sorted(range(N), key=parity_r), parity_r)
```

For instance, for radix-2, size 16, this creates the following channels

``` {.python .doctest #parity}
from fftsynth.parity import channels

for (g, i) in channels(16, 2):
    print(g, list(i))
---
0 [0, 3, 5, 6, 9, 10, 12, 15]
1 [1, 2, 4, 7, 8, 11, 13, 14]
```

and for a radix-4, size 64 fft, this creates the following channels

``` {.python .doctest #parity}
from fftsynth.parity import channels

for (g, i) in channels(16, 2):
    print(g, list(i))
---
0 [0, 7, 10, 13, 19, 22, 25, 28, 34, 37, 40, 47, 49, 52, 59, 62]
1 [1, 4, 11, 14, 16, 23, 26, 29, 35, 38, 41, 44, 50, 53, 56, 63]
2 [2, 5, 8, 15, 17, 20, 27, 30, 32, 39, 42, 45, 51, 54, 57, 60]
3 [3, 6, 9, 12, 18, 21, 24, 31, 33, 36, 43, 46, 48, 55, 58, 61]
```

## Index functions

``` {.python #parity-index-functions}
def comp_idx(radix: int, i: int, j: int, k: int) -> int:
    base = (i & ~(radix**k - 1))
    rem  = (i &  (radix**k - 1))
    return rem + j * radix**k + base * radix


def comp_perm(radix: int, i: int) -> int:
    base = (i & ~(radix - 1))
    rem = (i & (radix - 1))
    p = parity(radix, base)
    return base | ((rem - p) % radix)
```

## Interface

This interface should ease the implementation of the parity-splitting FFT.

``` {.python .bootstrap-fold #parity-splitting-interface}
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
```

