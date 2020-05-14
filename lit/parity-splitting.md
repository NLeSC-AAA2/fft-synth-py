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

``` {.python .eval #parity}
from fftsynth.parity import channels

for (g, i) in channels(16, 2):
    print(g, list(i))
```

and for a radix-4, size 64 fft, this creates the following channels

``` {.python .eval #parity}
for (g, i) in channels(16, 2):
    print(g, list(i))
```

## Twiddles

``` {.python file=fftsynth/twiddle.py}
import numpy as np

<<make-twiddle>>
```

``` {.python #make-twiddle}
def make_twiddle(n1, n2):
    def w(k, n):
        return np.exp(2j * np.pi * k / n)

    I1 = np.arange(n1)
    I2 = np.arange(n2)
    return w(I1[:,None] * I2[None,:], n1*n2).astype('complex64')
```

## Radix-4 FFT example

The radix-4 FFT, takes four elements in every butterfly operation, along with four weights.

``` {.python #fft-ps-4}
def fft4(x0, x1, x2, x3, w0=1, w1=1, w2=1, w3=1):
    a = w0*x0 + w2*x2
    b = w1*x1 + w3*x3
    c = w0*x0 - w2*x2
    d = w1*x1 - w3*x3
    x0[:] = a + b
    x1[:] = c - 1j*d
    x2[:] = a - b
    x3[:] = c + 1j*d
```

We'll perform a $N=64$ FFT,

``` {.python #fft-ps-4}
import numpy as np
np.set_printoptions(precision=4)

from fftsynth.parity import ParitySplitting
from fftsynth.twiddle import make_twiddle

PS = ParitySplitting(64, 4)
x = np.fft.ifft(np.arange(0, PS.N, dtype='complex64'))
```

We reshape the input data, and transpose the result

``` {.python #fft-ps-4}
s = x.copy().reshape(PS.factors).transpose()
```

Then the transform looks like

``` {.python #fft-ps-4}
for k in range(0, len(PS.factors)):
    w = make_twiddle(4, 4**k).conj()[:,:]
    z = s.reshape([-1, 4, 4**k])
    z *= w
    fft4(*(z[..., l, :] for l in range(4)))
    s = z
```

In every step, we multiply with twiddles, select a dimension to transform over and perform the butterfly.

``` {.python .eval #fft-ps-4}
print(s.flatten().real)
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

## Properties

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

## Testing

``` {.python .bootstrap-fold file=test/test_radix_4_ps.py}
import pytest
import pyopencl as cl
import pyopencl.cltypes
import numpy as np

from fft.util import parity
from fft.multi_channel import MultiChannel, comp_idx, comp_perm


N = 1024
radix = 4
mc = MultiChannel(N, radix)
mf = cl.mem_flags


@pytest.fixture
def program(cl_context):
    kernel = open("test/fft1024.cl", "r").read()
    prog = cl.Program(cl_context, kernel).build(["-DTESTING"])
    return prog


def test_parity_4(cl_context, program):
    queue = cl.CommandQueue(cl_context)
    x = np.arange(N, dtype=cl.cltypes.int)
    y = np.zeros_like(x)
    x_g = cl.Buffer(cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
    y_g = cl.Buffer(cl_context, mf.WRITE_ONLY, x.nbytes)
    program.test_parity_4(queue, (N,), None, x_g, y_g)
    cl.enqueue_copy(queue, y, y_g)

    y_ref = np.array([parity(radix, i) for i in range(N)]) 
    assert np.all(y == y_ref)


def test_comp_idx_4(cl_context, program):
    queue = cl.CommandQueue(cl_context)
    x = np.arange(mc.L, dtype=cl.cltypes.int)
    y = np.zeros_like(x)
    x_g = cl.Buffer(cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
    y_g = cl.Buffer(cl_context, mf.WRITE_ONLY, x.nbytes)

    for k in range(1, 5):
        for j in range(4):
            program.test_comp_idx_4(
                queue, (mc.L,), None,
                cl.cltypes.int(k), cl.cltypes.int(j), x_g, y_g)
            cl.enqueue_copy(queue, y, y_g)

            y_ref = mc.index_loc \
                .reshape([-1, 4, 4**k]) \
                .transpose([0, 2, 1]) \
                .reshape([-1, 4])[::4, j]
            y_ref_2 = np.array([comp_idx(4, i, j, k-1) for i in range(mc.L)])
            assert np.all(y == y_ref)
            assert np.all(y == y_ref_2)


def test_comp_perm_4(cl_context, program):
    queue = cl.CommandQueue(cl_context)
    x = np.arange(mc.L, dtype=cl.cltypes.int)
    y = np.zeros_like(x)
    x_g = cl.Buffer(cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
    y_g = cl.Buffer(cl_context, mf.WRITE_ONLY, x.nbytes)

    for r in range(4):
        program.test_comp_perm_4(queue, (mc.L,), None, cl.cltypes.int(r), x_g, y_g)
        cl.enqueue_copy(queue, y, y_g)

        y_ref = np.array([comp_perm(radix, i*radix + r) for i in range(mc.L)])
        assert np.all(y == y_ref)


def test_transpose_4(cl_context, program):
    queue = cl.CommandQueue(cl_context)
    x = np.arange(N, dtype=cl.cltypes.int)
    y = np.zeros_like(x)
    x_g = cl.Buffer(cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
    y_g = cl.Buffer(cl_context, mf.WRITE_ONLY, x.nbytes)
    program.test_transpose_4(queue, (N,), None, x_g, y_g)
    cl.enqueue_copy(queue, y, y_g)

    y_ref = x.reshape(mc.factors).T.flatten()
    assert np.all(y == y_ref)


def test_fft4_4(cl_context, program):
    queue = cl.CommandQueue(cl_context)
    x = np.random.normal(size=(1024, 4, 2)).astype(cl.cltypes.float)
    y = np.zeros_like(x)
    x_g = cl.Buffer(cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
    y_g = cl.Buffer(cl_context, mf.WRITE_ONLY, x.nbytes)

    for cycle in range(4):
        y_ref = np.fft.fft(np.roll(x[...,0]+1j*x[...,1], -cycle, axis=1))
        program.test_fft_4(queue, (N,), None, cl.cltypes.int(cycle), x_g, y_g)
        cl.enqueue_copy(queue, y, y_g)
        y_Z = np.roll(y[...,0]+1j*y[...,1], -cycle, axis=1)

        assert abs(y_Z - y_ref).max() < 1e-4


def test_fft_1024(cl_context, program):
    queue = cl.CommandQueue(cl_context)

    x = np.random.normal(size=(1024, 2)).astype(cl.cltypes.float)
    y = np.zeros_like(x)
    x_g = cl.Buffer(cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
    y_g = cl.Buffer(cl_context, mf.WRITE_ONLY, x.nbytes)
    program.fft_1024(queue, (1,), None, x_g, y_g)
    cl.enqueue_copy(queue, y, y_g)

    y_Z = y[...,0]+1j*y[...,1]
    y_ref = np.fft.fft(x[...,0]+1j*x[...,1])
    assert abs(y_Z - y_ref).max() < 1e-3
```

