# Code generator

``` {.python file=fftsynth/generator.py}
from .indent import indent

<<generate-twiddles>>

<<generate-transpose>>

<<generate-parity>>

<<generate-fft>>
```

## Testing
To test we need to define small `__kernel` functions.

``` {.python file=test/test_generator.py}
import pytest
import numpy as np
import pyopencl as cl
from fftsynth.parity import ParitySplitting

cases = [
    ParitySplitting(64, 4),
    ParitySplitting(81, 3),
    ParitySplitting(125, 5)
]

mf = cl.mem_flags

<<test-parity>>
<<test-transpose>>
```

## Bit math
The `transpose_{r}` function should reverse the digits in a base-n representation of the integer.

``` {.python #generate-transpose}
def gen_transpose_fn(ps):
    return f"""
inline int transpose_{ps.radix}(int j) {{
    int x = 0;
    for (int l = 0; l < {ps.depth}; ++l) {{
        x *= {ps.radix};
        x += j % {ps.radix};
        j /= {ps.radix};
    }}
    return x;
}}"""
```

The `parity_{r}` function should compute the parity of the index.

``` {.python #generate-parity}
def gen_parity_fn(ps):
    return f"""
inline int parity_{ps.radix}(int i) {{
    int x = i % {ps.radix};
    for (int a = 0; a < {ps.depth}; ++a) {{
        i /= {ps.radix};
        x += i % {ps.radix};
    }}
    return x % {ps.radix};
}}"""
```

### Testing

``` {.python #test-transpose}
@pytest.mark.parametrize('ps', cases)
def test_transpose_4(cl_context, ps):
    from fftsynth.generator import gen_transpose_fn
    source = gen_transpose_fn(ps)
    kernel = f"""
    {source}

    __kernel void test_transpose(__global const int *x, __global int *y) {{
        int i = get_global_id(0);
        y[i] = transpose_{ps.radix}(x[i]);
    }}
    """
    program = cl.Program(cl_context, kernel).build(["-DTESTING"])
    queue = cl.CommandQueue(cl_context)
    x = np.arange(ps.N, dtype=cl.cltypes.int)
    y = np.zeros_like(x)
    x_g = cl.Buffer(cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
    y_g = cl.Buffer(cl_context, mf.WRITE_ONLY, x.nbytes)
    program.test_transpose(queue, (ps.N,), None, x_g, y_g)
    cl.enqueue_copy(queue, y, y_g)

    y_ref = x.reshape(ps.factors).T.flatten()
    assert np.all(y == y_ref)
```

#### Parity

``` {.python #test-parity}
@pytest.mark.parametrize('ps', cases)
def test_parity_4(cl_context, ps):
    from fftsynth.generator import gen_parity_fn
    from fftsynth.parity import parity
    source = gen_parity_fn(ps)
    kernel = f"""
    {source}

    __kernel void test_parity(__global const int *x, __global int *y) {{
        int i = get_global_id(0);
        y[i] = parity_{ps.radix}(x[i]);
    }}
    """
    program = cl.Program(cl_context, kernel).build(["-DTESTING"])
    queue = cl.CommandQueue(cl_context)
    x = np.arange(ps.N, dtype=cl.cltypes.int)
    y = np.zeros_like(x)
    x_g = cl.Buffer(cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
    y_g = cl.Buffer(cl_context, mf.WRITE_ONLY, x.nbytes)
    program.test_parity(queue, (ps.N,), None, x_g, y_g)
    cl.enqueue_copy(queue, y, y_g)

    y_ref = np.array([parity(ps.radix, i) for i in range(ps.N)]) 
    assert np.all(y == y_ref)
```

## Outer loop

``` {.python #generate-fft}
def write_outer_loop_fn(ps):
    print(f"void fft_{ps.N}_mc(__restrict float2 *s0, __restrict float2 *s1, __restrict float2 *s2, __restrict float2 *s3)")
    print( "{")
    print( "    int wp = 0;")
    print(f"    for (int k = 0; k < {ps.depth}; ++k) {{")
    with indent("         "):
        <<fft-inner-loop>>
    print( "    }")
    print( "}")
```

### Inner loop
Here we have `k` being the index of the outer loop. We now 

``` {.python #fft-inner-loop}
print(f"int j = (k == 0 ? 0 : ipow({ps.radix}, k - 1));")
print(f"for (int i = 0; i < {ps.L}; ++i) {{")
```

The next bit is only still implemented for radix-4.

``` {.python #fft-inner-loop}
print( "if (k != 0) {")
print( "    a = comp_idx_4(i >> 2, k-1);")
print( "} else {")
print( "    a = comp_perm_4(i >> 2, i&3);")
print( "}")
print( "fft_4(s0, s1, s2, s3, i&3, a, a+j, a+2*j, a+3*j, wp);")
print( "if (k != 0) ++wp;")
```

### Outer kernel
The outer kernel reads the data, puts it in the correct parity-channel and then calls the respective `fft_{n}_mc` kernel.

``` {.python #generate-outer-fft}
declare_arrays = "\n".join(
    f"float2 s{i}[{ps.L}];" for i in range(ps.radix))
mc_args = ", ".join(
    f"s{i}" for i in range(ps.radix))
read_cases = "\n".join(
    f"case {p}: s{p}[i/{ps.radix}] = x[j]; break;" for p in range(ps.radix))
write_cases = "\n".join(
    f"case {p}: y[i] = s{p}[i/{ps.radix}]; break;" for p in range(ps.radix))
print(f"""
__kernel void fft_{ps.N}(__global const float2 * restrict x, __global float2 * restrict y)
{{
    {declare_arrays}
    float2 s{ps.radix}[{ps.L}];

    for (int j = 0; j < {ps.N}; ++j) {{
        int i = transpose_{ps.radix}(j);
        int p = parity_{ps.radix}(i);
        switch (p) {{
            {read_cases}
        }}
    }}

    fft_1024_mc({mc_args});

    for (int i = 0; i < {ps.N}; ++i) {{
        int p = parity_{ps.radix}(i);
        switch (p) {{
            {write_cases}
        }}
    }}
}}""")
```

## Twiddles

``` {.python #generate-twiddles}
def write_twiddles(ps, W):
    print(f"__constant float2 W[{W.shape[0]}][{ps.radix-1}] {{")
    print( "    {" + "},\n    {".join(", ".join(
                f"(float2) ({w.real: f}f, {w.imag: f}f)"
                for w in ws[1:])
            for ws in W) + "}};")
```

## Utility

``` {.python file=fftsynth/indent.py}
from contextlib import (contextmanager, redirect_stdout)
import io
import textwrap


@contextmanager
def indent(prefix: str):
    f = io.StringIO()
    with redirect_stdout(f):
        yield
    output = f.getvalue()
    print(textwrap.indent(output, prefix), end="")
```