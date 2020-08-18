# Code generator

``` {.python file=fftsynth/generator.py}
<<generate-twiddles>>

<<generate-transpose>>

<<generate-parity>>

<<generate-fft>>
```

## Bit math
The `transpose_{r}` function should reverse the digits in a base-n representation of the integer.

``` {.python #generate-transpose}
def write_transpose_fn(ps):
    print(f"""
inline int transpose_{ps.radix}(int j) {{
    int x = 0;
    for (int l = 0; l < {ps.depth}; ++l) {{
        x *= {ps.radix};
        x += j % {ps.radix};
        j /= {ps.radix};
    }}
    return x;
}}""")
```

The `parity_{r}` function should compute the parity of the index.

``` {.python #generate-parity}
def write_parity_fn(ps):
    print(f"""
inline int parity_{ps.radix}(int i) {{
    int x = i % {ps.radix};
    for (int a = 0; a < {ps.depth}; ++a) {{
        i /= {ps.radix};
        x += i % {ps.radix};
    }}
    return x % {ps.radix};
}}
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
