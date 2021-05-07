# Code generator

``` {.python file=fftsynth/generator.py}
from jinja2 import Environment, FileSystemLoader
import numpy
from pkg_resources import resource_filename

from .parity import ParitySplitting, comp_perm
from .twiddle import make_twiddle

template_loader = FileSystemLoader(resource_filename("fftsynth", "templates"))
template_environment = Environment(loader=template_loader)


def get_n_twiddles(radix):
    """Gives the width of the twiddle array as a function of radix."""
    return [0, 1, 1, 2, 2, 4][radix]


<<generate-preprocessor>>


<<generate-twiddle-array>>


<<generate-fpga-functions>>


<<generate-transpose-function>>


<<generate-parity-function>>


<<generate-ipow-function>>


<<generate-index-functions>>


<<generate-fft-functions>>


<<generate-codelets>>


def generate_fft(parity_splitting: ParitySplitting, fpga: bool, c_type: str = "float2"):
    """
    Generate the complete OpenCL FFT code.
    """
    code = "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n".format(generate_preprocessor(parity_splitting, fpga, c_type),
                                                     generate_twiddle_array(parity_splitting),
                                                     generate_parity_function(parity_splitting),
                                                     generate_transpose_function(parity_splitting),
                                                     generate_ipow_function(parity_splitting),
                                                     generate_index_functions(parity_splitting),
                                                     generate_codelets(parity_splitting, fpga, c_type),
                                                     generate_fft_functions(parity_splitting, fpga, c_type))
    if fpga:
        code = "{}\n{}\n".format(code, generate_fpga_functions(c_type=c_type))
    return code


def generate_fma_twiddle_array(parity_splitting: ParitySplitting):
    """
    Generate OpenCL constant array for twiddle factors (FMA version).
    """
    template = template_environment.get_template("fma-twiddles.cl")
    twiddles = numpy.ones(shape=[parity_splitting.radix, parity_splitting.radix])
    perm = numpy.array([comp_perm(parity_splitting.radix, i) for i in range(parity_splitting.M)])

    n = parity_splitting.radix
    for k in range(parity_splitting.depth - 1):
        w = make_twiddle(parity_splitting.radix, n).conj()
        w_r_x = (numpy.ones(shape=[parity_splitting.M // n, parity_splitting.radix, n]) * w) \
            .transpose([0, 2, 1]) \
            .reshape([-1, parity_splitting.radix])[perm]
        twiddles = numpy.r_[twiddles, w_r_x]
        n *= parity_splitting.radix

    return template.render(radix=parity_splitting.radix,
                           n_twiddles=get_n_twiddles(parity_splitting.radix),
                           W=twiddles)


def generate_fma_codelets(parity_splitting: ParitySplitting, fpga: bool, c_type: str = "float2"):
    """
    Generate OpenCL codelets for FFT (FMA version).
    """
    template = template_environment.get_template("fma-codelets.cl")

    return template.render(radix=parity_splitting.radix, fpga=fpga, c_type=c_type)


def generate_fma_fft_functions(parity_splitting: ParitySplitting, fpga: bool, c_type: str = "float2"):
    """
    Generate outer loop for OpenCL FFT (FMA version).
    """
    template = template_environment.get_template("fma-fft.cl")
    depth_type = "unsigned int"
    m_type = "unsigned int"
    n_type = "unsigned int"
    if fpga:
        depth_type = "uint{}_t".format(int(numpy.ceil(numpy.log2(parity_splitting.depth + 0.5))))
        m_type = "uint{}_t".format(int(numpy.ceil(numpy.log2(parity_splitting.M + 0.5))))
        n_type = "uint{}_t".format(int(numpy.ceil(numpy.log2(parity_splitting.N + 0.5))))

    return template.render(N=parity_splitting.N,
                           depth=parity_splitting.depth,
                           radix=parity_splitting.radix,
                           n_twiddles=get_n_twiddles(parity_splitting.radix),
                           M=parity_splitting.M,
                           fpga=fpga,
                           depth_type=depth_type,
                           m_type=m_type,
                           n_type=n_type,
                           c_type=c_type)


def generate_fma_fft(parity_splitting: ParitySplitting, fpga: bool, c_type: str = "float2"):
    """
    Generate the complete OpenCL FFT code (FMA version).
    """
    code = "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n".format(generate_preprocessor(parity_splitting, fpga, c_type=c_type),
                                                     generate_fma_twiddle_array(parity_splitting),
                                                     generate_parity_function(parity_splitting),
                                                     generate_transpose_function(parity_splitting),
                                                     generate_ipow_function(parity_splitting),
                                                     generate_index_functions(parity_splitting),
                                                     generate_fma_codelets(parity_splitting, fpga, c_type=c_type),
                                                     generate_fma_fft_functions(parity_splitting, fpga, c_type=c_type))
    if fpga:
        code = "{}\n{}\n".format(code, generate_fpga_functions(c_type=c_type))
    return code


if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser(description="Generate an OpenCL FFT kernel")
    parser.add_argument("--radix", type=int, default=4, help="FFT radix")
    parser.add_argument("--depth", type=int, default=3, help="FFT depth")
    parser.add_argument("--fpga", action="store_true")
    parser.add_argument("--fma", action="store_true")
    parser.add_argument("--ctype", type=str, default="float2", help="complex type")
    args = parser.parse_args()
    print("/* FFT")
    print(f" * command: python -m fftsynth.generator {' '.join(sys.argv[1:])}")
    print(" */")
    N = args.radix**args.depth
    ps = ParitySplitting(N, args.radix)
    if args.fma:
        print(generate_fma_fft(ps, args.fpga, args.ctype))
    else:
        print(generate_fft(ps, args.fpga, args.ctype))
```

### Codelets

These are the codelets for specific radix FFT.

```{.opencl file=fftsynth/templates/codelets.cl}
{% if radix == 2%}
void fft_2({{c_type}} * restrict s0, {{c_type}} * restrict s1,{% if fpga %} {{c_type}} * restrict s0_in, {{c_type}} * restrict s1_in, {{c_type}} * restrict s0_out, {{c_type}} * restrict s1_out, bool first_iteration, bool last_iteration,{% endif %} int cycle, int i0, int i1, int iw)
{
    {{c_type}} t0, t1, ws0, ws1;
    __constant float2 *w = W[iw];

    {% if fpga %}
    switch (cycle) {
        case 1: SWAP(int, i0, i1); break;
    }
    if ( first_iteration )
    {
        t0 = s0_in[i0]; t1 = s1_in[i1];
    }
    else
    {
        t0 = s0[i0]; t1 = s1[i1];
    }
    switch (cycle) {
        case 1: SWAP({{c_type}}, t0, t1); break;
    }
    {% else %}
    switch (cycle) {
        case 0: t0 = s0[i0]; t1 = s1[i1]; break;
        case 1: t0 = s1[i0]; t1 = s0[i1]; break;
    }
    {% endif %}

    ws0 = t0;
    ws1.even = w[0].even * t1.even - w[0].odd * t1.odd;
    ws1.odd = w[0].even * t1.odd + w[0].odd * t1.even;

    t0 = ws0 + ws1;
    t1 = ws0 - ws1;

    {% if fpga %}
    switch (cycle) {
        case 1: SWAP({{c_type}}, t0, t1); break;
    }
    if ( last_iteration )
    {
        s0_out[i0] = t0; s1_out[i1] = t1;
    }
    else
    {
        s0[i0] = t0; s1[i1] = t1;
    }
    {% else %}
    switch (cycle) {
        case 0: s0[i0] = t0; s1[i1] = t1; break;
        case 1: s1[i0] = t0; s0[i1] = t1; break;
    }
    {% endif %}
}

#ifdef TESTING_RADIX
__kernel void test_radix_2(__global {{c_type}} *x, __global {{c_type}} *y, int n) {
    int i = get_global_id(0) * 2;

    //n is the number of radix2 ffts to perform
    if (i < 2 * n) {
        {{c_type}} s0 = x[i];
        {{c_type}} s1 = x[i + 1];

        fft_2(&s0, &s1, 0, 0, 0, 0);

        y[i] = s0; y[i + 1] = s1;
    }
}
#endif // TESTING_RADIX

{% elif radix == 4 %}
void fft_4({{c_type}} * restrict s0, {{c_type}} * restrict s1, {{c_type}} * restrict s2, {{c_type}} * restrict s3,{% if fpga %} {{c_type}} * restrict s0_in, {{c_type}} * restrict s1_in, {{c_type}} * restrict s2_in, {{c_type}} * restrict s3_in, {{c_type}} * restrict s0_out, {{c_type}} * restrict s1_out, {{c_type}} * restrict s2_out, {{c_type}} * restrict s3_out, bool first_iteration, bool last_iteration,{% endif %} int cycle, int i0, int i1, int i2, int i3, int iw)
{
    {{c_type}} t0, t1, t2, t3, ws0, ws1, ws2, ws3, a, b, c, d;
    __constant float2 *w = W[iw];

    {% if fpga %}
    switch (cycle) {
        case 1: SWAP(int, i0, i1); SWAP(int, i2, i3); SWAP(int, i0, i2); break;
        case 2: SWAP(int, i0, i2); SWAP(int, i1, i3); break;
        case 3: SWAP(int, i0, i1); SWAP(int, i1, i3); SWAP(int, i1, i2); break;
    }
    if ( first_iteration )
    {
        t0 = s0_in[i0]; t1 = s1_in[i1]; t2 = s2_in[i2]; t3 = s3_in[i3];
    }
    else
    {
        t0 = s0[i0]; t1 = s1[i1]; t2 = s2[i2]; t3 = s3[i3];
    }
    switch (cycle) {
        case 1: SWAP({{c_type}}, t0, t1); SWAP({{c_type}}, t1, t2); SWAP({{c_type}}, t2, t3); break;
        case 2: SWAP({{c_type}}, t0, t2); SWAP({{c_type}}, t1, t3); break;
        case 3: SWAP({{c_type}}, t2, t3); SWAP({{c_type}}, t0, t1); SWAP({{c_type}}, t0, t2); break;
    }
    {% else %}
    switch (cycle) {
        case 0: t0 = s0[i0]; t1 = s1[i1]; t2 = s2[i2]; t3 = s3[i3]; break;
        case 1: t0 = s1[i0]; t1 = s2[i1]; t2 = s3[i2]; t3 = s0[i3]; break;
        case 2: t0 = s2[i0]; t1 = s3[i1]; t2 = s0[i2]; t3 = s1[i3]; break;
        case 3: t0 = s3[i0]; t1 = s0[i1]; t2 = s1[i2]; t3 = s2[i3]; break;
    }
    {% endif %}

    ws0 = t0;
    ws1.even = w[0].even * t1.even - w[0].odd * t1.odd;
    ws1.odd = w[0].even * t1.odd + w[0].odd * t1.even;
    ws2.even = w[1].even * t2.even - w[1].odd * t2.odd;
    ws2.odd = w[1].even * t2.odd + w[1].odd * t2.even;
    ws3.even = w[2].even * t3.even - w[2].odd * t3.odd;
    ws3.odd = w[2].even * t3.odd + w[2].odd * t3.even;

    a = ws0 + ws2;
    b = ws1 + ws3;
    c = ws0 - ws2;
    d = ws1 - ws3;
    t0 = a + b;
    t1.even = c.even + d.odd;
    t1.odd = c.odd - d.even;
    t2 = a - b;
    t3.even = c.even - d.odd;
    t3.odd = c.odd + d.even;

    {% if fpga %}
    switch (cycle) {
        case 1: SWAP({{c_type}}, t2, t3); SWAP({{c_type}}, t1, t2); SWAP({{c_type}}, t0, t1); break;
        case 2: SWAP({{c_type}}, t1, t3); SWAP({{c_type}}, t0, t2); break;
        case 3: SWAP({{c_type}}, t0, t2); SWAP({{c_type}}, t0, t1); SWAP({{c_type}}, t2, t3); break;
    }
    if ( last_iteration )
    {
        s0_out[i0] = t0; s1_out[i1] = t1; s2_out[i2] = t2; s3_out[i3] = t3;
    }
    else
    {
        s0[i0] = t0; s1[i1] = t1; s2[i2] = t2; s3[i3] = t3;
    }
    {% else %}
    switch (cycle) {
        case 0: s0[i0] = t0; s1[i1] = t1; s2[i2] = t2; s3[i3] = t3; break;
        case 1: s1[i0] = t0; s2[i1] = t1; s3[i2] = t2; s0[i3] = t3; break;
        case 2: s2[i0] = t0; s3[i1] = t1; s0[i2] = t2; s1[i3] = t3; break;
        case 3: s3[i0] = t0; s0[i1] = t1; s1[i2] = t2; s2[i3] = t3; break;
    }
    {% endif %}
}

#ifdef TESTING_RADIX
__kernel void test_radix_4(__global {{c_type}} *x, __global {{c_type}} *y, int n) {
    int i = get_global_id(0) * 4;

    //n is the number of radix4 ffts to perform
    if (i < 4 * n) {
        {{c_type}} s0 = x[i];
        {{c_type}} s1 = x[i + 1];
        {{c_type}} s2 = x[i + 2];
        {{c_type}} s3 = x[i + 3];
        fft_4(&s0, &s1, &s2, &s3, 0, 0, 0, 0, 0, 0);

        y[i] = s0;    y[i + 1] = s1;    y[i + 2] = s2;    y[i + 3] = s3;
    }
}
#endif // TESTING_RADIX
{% endif %}
```

What follows is the Python function used to generate the OpenCL code.

```{.python #generate-codelets}
def generate_codelets(parity_splitting: ParitySplitting, fpga: bool, c_type: str = "float2"):
    """
    Generate OpenCL codelets for FFT.
    """
    template = template_environment.get_template("codelets.cl")

    return template.render(radix=parity_splitting.radix, fpga=fpga, c_type=c_type)
```

Here we also have a test for the codelets.

```{.python file=test/test_codelets.py}
import numpy

import pytest
from fftsynth import generator, parity
from kernel_tuner import run_kernel     # type: ignore


@pytest.mark.parametrize('radix,c_type', [(2,'float2'), (4,'float2'),
                                          (2, 'float4'), (4,'float4'),
                                          (2, 'float8'), (4,'float8')])
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
                                   generator.generate_twiddle_array(parity_splitting),
                                   generator.generate_codelets(parity_splitting, False, c_type=c_type))
    args = [x, y, n]
    answer = run_kernel(f"test_radix_{radix}", codelets, 1, args, {}, compiler_options=["-DTESTING_RADIX"])

    y = answer[1]
    y = y[..., 0] + 1j * y[..., 1]

    numpy.testing.assert_almost_equal(y, y_ref, decimal=5)
```

### Preprocessor

This is the parameterized OpenCL code used to write the preprocessor directives.

```{.opencl file=fftsynth/templates/preprocessor.cl}
{% if fpga -%}
#pragma OPENCL EXTENSION cl_intel_channels : enable
#include <ihc_apint.h>
#ifdef TESTING
channel {{c_type}} in_channel, out_channel;
#endif // TESTING
#define SWAP(type, x, y) do { type temp = x; x = y, y = temp; } while ( false );
{% endif -%}
{%if radix == 2 %}
#define DIVR(x) ((x) >> 1)
#define MODR(x) ((x) & 1)
#define MULR(x) ((x) << 1)
{% elif radix == 4 %}
#define DIVR(x) ((x) >> 2)
#define MODR(x) ((x) & 3)
#define MULR(x) ((x) << 2)
{% else %}
#define DIVR(x) ((x) / {{ radix }})
#define MODR(x) ((x) % {{ radix }})
#define MULR(x) ((x) * {{ radix }})
{% endif %}
```

What follows is the Python function used to generate the OpenCL code.

```{.python #generate-preprocessor}
def generate_preprocessor(parity_splitting: ParitySplitting, fpga: bool, c_type: str = "float2"):
    """
    Generate the preprocessor directives necessary for the FFT.
    """
    template = template_environment.get_template("preprocessor.cl")

    return template.render(radix=parity_splitting.radix, fpga=fpga, c_type=c_type)
```

### FPGA specific code

This is the parameterized OpenCL code used to write the FPGA specific functions.

```{.opencl file=fftsynth/templates/fpga.cl}
#ifdef TESTING
__kernel __attribute__((max_global_work_dim(0)))
void source(__global const volatile {{c_type}} * in, unsigned count)
{
    #pragma ii 1
    for ( unsigned i = 0; i < count; i++ )
    {
        write_channel_intel(in_channel, in[i]);
    }
}

__kernel __attribute__((max_global_work_dim(0)))
void sink(__global {{c_type}} *out, unsigned count)
{
    #pragma ii 1
    for ( unsigned i = 0; i < count; i++ )
    {
        out[i] = read_channel_intel(out_channel);
    }
}
#endif // TESTING
```

What follows is the Python function used to generate the OpenCL code.

```{.python #generate-fpga-functions}
def generate_fpga_functions(c_type: str = "float2"):
    """
    Generate OpenCL code for FPGA functions.
    """
    template = template_environment.get_template("fpga.cl")

    return template.render(c_type=c_type)
```

## Testing

To test we need to define small `__kernel` functions.

```{.python file=test/test_generator.py}
import pytest
import numpy as np
from kernel_tuner import run_kernel  # type: ignore

from fftsynth.parity import ParitySplitting, parity
from fftsynth.generator import generate_preprocessor, generate_transpose_function, generate_parity_function, generate_fft, generate_fma_fft

cases = [
    ParitySplitting(128, 2),
    ParitySplitting(64, 4)
]


<<test-parity>>

<<test-transpose>>

<<test-fft>>

<<test-fft-fma>>

```

```{.python #test-fft}
c_types = [f"float{n}" for n in [2, 4, 8]]
test_matrix = [(p, c) for p in cases for c in c_types]

@pytest.mark.parametrize('parity_splitting,c_type', test_matrix)
def test_fft(parity_splitting: ParitySplitting, c_type: str):
    kernel = generate_fft(parity_splitting, False, c_type=c_type)

    m = {'float2': 1, 'float4': 2, 'float8': 4}[c_type]
    x = np.random.normal(size=(parity_splitting.N, m, 2)).astype(np.float32)
    y = np.zeros_like(x)

    results = run_kernel(
        f"fft_{parity_splitting.N}", kernel, parity_splitting.N, [x, y], {}, compiler_options=["-DTESTING"])
    y_ref = np.fft.fft(x[..., 0] + 1j * x[..., 1], axis=0)
    y = results[1][..., 0] + 1j * results[1][..., 1]
    np.testing.assert_almost_equal(y, y_ref, decimal=4)
```

```{.python #test-fft-fma}
@pytest.mark.parametrize('parity_splitting,c_type', test_matrix)
def test_fft_fma(parity_splitting: ParitySplitting, c_type: str):
    kernel = generate_fma_fft(parity_splitting, False, c_type=c_type)

    m = {'float2': 1, 'float4': 2, 'float8': 4}[c_type]
    x = np.random.normal(size=(parity_splitting.N, m, 2)).astype(np.float32)
    y = np.zeros_like(x)

    results = run_kernel(
        f"fft_{parity_splitting.N}", kernel, parity_splitting.N, [x, y], {}, compiler_options=["-DTESTING"])
    y_ref = np.fft.fft(x[..., 0] + 1j * x[..., 1], axis=0)
    y = results[1][..., 0] + 1j * results[1][..., 1]
    np.testing.assert_almost_equal(y, y_ref, decimal=4)
```

## Bit math

The `transpose_{r}` function should reverse the digits in a base-n representation of the integer.

```{.opencl file=fftsynth/templates/transpose.cl}
inline int transpose_{{ radix }}(int j)
{
    int x = 0;

    {% for item in range(depth) %}
    x = MULR(x) + MODR(j);
    j = DIVR(j);
    {%- endfor %}

    return x;
}
```

``` {.python #generate-transpose-function}
def generate_transpose_function(parity_splitting: ParitySplitting):
    """
    Generate inline OpenCL function to reverse the digits in base-n representation.
    """
    template = template_environment.get_template("transpose.cl")

    return template.render(radix=parity_splitting.radix,
                           depth=parity_splitting.depth)
```

The `parity_{r}` function should compute the parity of the index.

```{.opencl file=fftsynth/templates/parity.cl}
inline int parity_{{ radix }}(int i)
{
    int x = MODR(i);

    {% for item in range(depth) %}
    i = DIVR(i);
    x += MODR(i);
    {%- endfor %}

    return MODR(x);
}
```

``` {.python #generate-parity-function}
def generate_parity_function(parity_splitting: ParitySplitting):
    """
    Generate inline OpenCL function to compute the parity of the index.
    """
    template = template_environment.get_template("parity.cl")

    return template.render(radix=parity_splitting.radix,
                           depth=parity_splitting.depth)
```

### Testing

```{.opencl file=fftsynth/templates/transpose.cl}
#ifdef TESTING
__kernel void test_transpose_{{ radix }}(__global const int * x, __global int * y)
{
    int i = get_global_id(0);
    
    y[i] = transpose_{{ radix }}(x[i]);
}
#endif // TESTING
```

``` {.python #test-transpose}
@pytest.mark.parametrize('parity_splitting', cases)
def test_transpose(parity_splitting: ParitySplitting):
    kernel = generate_preprocessor(parity_splitting, False) + generate_transpose_function(parity_splitting)
    x = np.arange(parity_splitting.N, dtype=np.int32)
    y = np.zeros_like(x)
    kernel_args = [x, y]

    results = run_kernel("test_transpose_{}".format(parity_splitting.radix), kernel, parity_splitting.N, kernel_args, {}, compiler_options=["-DTESTING"])
    y_ref = x.reshape(parity_splitting.factors).T.flatten()

    assert np.all(results[1] == y_ref)
```

#### Parity

```{.opencl file=fftsynth/templates/parity.cl}
#ifdef TESTING
__kernel void test_parity_{{ radix }}(__global const int * x, __global int * y)
{
    int i = get_global_id(0);
    
    y[i] = parity_{{ radix }}(x[i]);
}
#endif // TESTING
```

``` {.python #test-parity}
@pytest.mark.parametrize("parity_splitting", cases)
def test_parity(parity_splitting: ParitySplitting):
    kernel = generate_preprocessor(parity_splitting, False) + generate_parity_function(parity_splitting)
    x = np.arange(parity_splitting.N, dtype=np.int32)
    y = np.zeros_like(x)
    kernel_args = [x, y]

    results = run_kernel("test_parity_{}".format(parity_splitting.radix), kernel, parity_splitting.N, kernel_args, {}, compiler_options=["-DTESTING"])
    y_ref = np.array([parity(parity_splitting.radix, i) for i in range(parity_splitting.N)])

    assert np.all(results[1] == y_ref)
```

## FFT kernel

``` {.python #generate-fft-functions}
def generate_fft_functions(parity_splitting: ParitySplitting, fpga: bool, c_type: str):
    """
    Generate outer and inner loop for OpenCL FFT.
    """
    template = template_environment.get_template("fft.cl")
    depth_type = "unsigned int"
    m_type = "unsigned int"
    n_type = "unsigned int"
    if fpga:
        depth_type = "uint{}_t".format(int(numpy.ceil(numpy.log2(parity_splitting.depth + 0.5))))
        m_type = "uint{}_t".format(int(numpy.ceil(numpy.log2(parity_splitting.M + 0.5))))
        n_type = "uint{}_t".format(int(numpy.ceil(numpy.log2(parity_splitting.N + 0.5))))

    return template.render(N=parity_splitting.N,
                           depth=parity_splitting.depth,
                           radix=parity_splitting.radix,
                           M=parity_splitting.M,
                           fpga=fpga,
                           depth_type=depth_type,
                           m_type=m_type,
                           n_type=n_type,
                           c_type=c_type)
```

### Inner loop

Here we have `k` being the index of the outer loop.
The next bit is only  implemented for radix-2 and radix-4.

```{.opencl file=fftsynth/templates/fft.cl}
void fft_{{ N }}_ps({% for i in range(radix) %} {{c_type}} * restrict s{{ i }}{%- if not loop.last %},{% endif %}{% endfor %}{% if fpga %},{% for i in range(radix) %} {{c_type}} * restrict s{{ i }}_in,{% endfor %}{% for i in range(radix) %} {{c_type}} * restrict s{{ i }}_out{%- if not loop.last %},{% endif %}{% endfor %}{% endif %})
{
    int wp = 0;

    for ( {{ depth_type }} k = 0; k != {{ depth }}; ++k )
    {
        int j = (k == 0 ? 0 : ipow(k - 1));

        {% if fpga -%}
        #pragma ivdep
        {% endif -%}
        for ( {{ m_type }} i = 0; i != {{ M }}; ++i )
        {
            int a;
            if ( k != 0 )
            {
                a = comp_idx_{{ radix }}(DIVR(i), k-1);
            }
            else
            {
                a = comp_perm_{{ radix }}(DIVR(i), MODR(i));
            }
            {% if fpga %}
            fft_{{ radix }}({% for i in range(radix) %} s{{ i }},{% endfor %}{% for i in range(radix) %} s{{ i }}_in,{% endfor %}{% for i in range(radix) %} s{{ i }}_out,{% endfor %} k == 0, k == {{ depth - 1 }}, MODR(i), {% for i in range(radix) %} a + {{ i }} * j,{% endfor %} wp);
            {% else %}
            fft_{{ radix }}({% for i in range(radix) %} s{{ i }},{% endfor %} MODR(i), {% for i in range(radix) %} a + {{ i }} * j,{% endfor %} wp);
            {% endif %}
            if ( k != 0 )
            {
                ++wp;
            }
        }
    }
}
```

FMA version.

```{.opencl file=fftsynth/templates/fma-fft.cl}
void fft_{{ N }}_ps({% for i in range(radix) %} {{c_type}} * restrict s{{ i }}{%- if not loop.last %},{% endif %}{% endfor %}{% if fpga %},{% for i in range(radix) %} {{c_type}} * restrict s{{ i }}_in,{% endfor %}{% for i in range(radix) %} {{c_type}} * restrict s{{ i }}_out{%- if not loop.last %},{% endif %}{% endfor %}{% endif %})
{
    int wp = 0;

    for ( {{ depth_type }} k = 0; k != {{ depth }}; ++k )
    {
        int j = (k == 0 ? 0 : ipow(k - 1));

        {% if fpga -%}
        #pragma ivdep
        {% endif -%}
        for ( {{ m_type }} i = 0; i != {{ M }}; ++i )
        {
            int a;
            if ( k != 0 )
            {
                a = comp_idx_{{ radix }}(DIVR(i), k-1);
            }
            else
            {
                a = comp_perm_{{ radix }}(DIVR(i), MODR(i));
            }
            {% if fpga %}
            fft_{{ radix }}({% for i in range(radix) %} s{{ i }},{% endfor %}{% for i in range(radix) %} s{{ i }}_in,{% endfor %}{% for i in range(radix) %} s{{ i }}_out,{% endfor %} k == 0, k == {{ depth - 1 }}, MODR(i), {% for i in range(radix) %} a + {{ i }} * j,{% endfor %} wp);
            {% else %}
            fft_{{ radix }}({% for i in range(radix) %} s{{ i }},{% endfor %} MODR(i), {% for i in range(radix) %} a + {{ i }} * j,{% endfor %} wp);
            {% endif %}
            if ( k != 0 )
            {
                ++wp;
            }
        }
    }
}
```

### Outer loop

The outer kernel reads the data, puts it in the correct parity-channel and then calls the respective `fft_{n}_ps` kernel.

```{.opencl file=fftsynth/templates/fft.cl}
#ifdef TESTING
__kernel {%if fpga %}__attribute__((autorun)) __attribute__((max_global_work_dim(0))){% endif %}
void fft_{{ N }}({% if not fpga %}__global const {{c_type}} * restrict x, __global {{c_type}} * restrict y{% endif %})
{
    {% if fpga -%}
    while ( true )
    {
    {% endif -%}
    {%- for i in range(radix) %}
    {{c_type}} s{{ i }}[{{ M }}];
    {% if fpga -%}
    {{c_type}} s{{ i }}_in[{{ M }}], s{{ i }}_out[{{ M }}];
    {%- endif -%}
    {%- endfor %}

    for ( {{ n_type }} j = 0; j != {{ N }}; ++j )
    {
        int i = transpose_{{ radix }}(j);
        int p = parity_{{ radix }}(i);

        {% if fpga -%}
        {{c_type}} x = read_channel_intel(in_channel);
        {% endif -%}
        switch ( p )
        {
            {%- if fpga %}
            {%- for p in range(radix) %}
            case {{ p }}: s{{ p }}_in[DIVR(i)] = x; break;
            {%- endfor -%}
            {% else %}
            {%- for p in range(radix) %}
            case {{ p }}: s{{ p }}[DIVR(i)] = x[j]; break;
            {%- endfor -%}
            {%- endif %}
        }
    }

    {% if fpga %}
    fft_{{ N }}_ps({%- for i in range(radix) %} s{{ i }},{% endfor %} {%- for i in range(radix) %} s{{ i }}_in,{% endfor %} {%- for i in range(radix) %} s{{ i }}_out{%- if not loop.last %},{% endif %}{% endfor %});
    {% else %}
    fft_{{ N }}_ps({%- for i in range(radix) %} s{{ i }}{%- if not loop.last %},{% endif %}{% endfor %});
    {% endif %}

    for ( {{ n_type }} i = 0; i != {{ N }}; ++i )
    {
        int p = parity_{{ radix }}(i);
        {% if fpga -%}
        {{c_type}} y;
        {% endif -%}

        switch ( p )
        {
            {%- if fpga -%}
            {%- for p in range(radix) %}
            case {{ p }}: y = s{{ p }}_out[DIVR(i)]; break;
            {%- endfor -%}
            {% else %}
            {%- for p in range(radix) %}
            case {{ p }}: y[i] = s{{ p }}[DIVR(i)]; break;
            {%- endfor -%}
            {%- endif %}
        }
        {%- if fpga %}
        write_channel_intel(out_channel, y);
        {%- endif %}
    }
    {%- if fpga %}
    }
    {%- endif %}
}
#endif // TESTING
```

FMA version.

```{.opencl file=fftsynth/templates/fma-fft.cl}
#ifdef TESTING
__kernel {%if fpga %}__attribute__((autorun)) __attribute__((max_global_work_dim(0))){% endif %}
void fft_{{ N }}({% if not fpga %}__global const {{c_type}} * restrict x, __global {{c_type}} * restrict y{% endif %})
{
    {% if fpga -%}
    while ( true )
    {
    {% endif -%}
    {%- for i in range(radix) %}
    {{c_type}} s{{ i }}[{{ M }}];
    {% if fpga -%}
    {{c_type}} s{{ i }}_in[{{ M }}], s{{ i }}_out[{{ M }}];
    {%- endif -%}
    {%- endfor %}

    for ( {{ n_type }} j = 0; j != {{ N }}; ++j )
    {
        int i = transpose_{{ radix }}(j);
        int p = parity_{{ radix }}(i);

        {% if fpga -%}
        {{c_type}} x = read_channel_intel(in_channel);
        {% endif -%}
        switch ( p )
        {
            {%- if fpga %}
            {%- for p in range(radix) %}
            case {{ p }}: s{{ p }}_in[DIVR(i)] = x; break;
            {%- endfor -%}
            {% else %}
            {%- for p in range(radix) %}
            case {{ p }}: s{{ p }}[DIVR(i)] = x[j]; break;
            {%- endfor -%}
            {%- endif %}
        }
    }

    {% if fpga %}
    fft_{{ N }}_ps({%- for i in range(radix) %} s{{ i }},{% endfor %} {%- for i in range(radix) %} s{{ i }}_in,{% endfor %} {%- for i in range(radix) %} s{{ i }}_out{%- if not loop.last %},{% endif %}{% endfor %});
    {% else %}
    fft_{{ N }}_ps({%- for i in range(radix) %} s{{ i }}{%- if not loop.last %},{% endif %}{% endfor %});
    {% endif %}

    for ( {{ n_type }} i = 0; i != {{ N }}; ++i )
    {
        int p = parity_{{ radix }}(i);
        {% if fpga -%}
        {{c_type}} y;
        {% endif -%}

        switch ( p )
        {
            {%- if fpga -%}
            {%- for p in range(radix) %}
            case {{ p }}: y = s{{ p }}_out[DIVR(i)]; break;
            {%- endfor -%}
            {% else %}
            {%- for p in range(radix) %}
            case {{ p }}: y[i] = s{{ p }}[DIVR(i)]; break;
            {%- endfor -%}
            {%- endif %}
        }
        {%- if fpga %}
        write_channel_intel(out_channel, y);
        {%- endif %}
    }
    {%- if fpga %}
    }
    {%- endif %}
}
#endif // TESTING
```

## Twiddles

This is the parameterized OpenCL code containing the constant array of the twiddle factors.

```{.opencl file=fftsynth/templates/twiddles.cl}
__constant float2 W[{{ W.shape[0] - radix }}][{{ radix - 1 }}] = {
{% for ws in W[radix:] %}
{ {% for w in ws[1:] -%}
(float2)({{ "%0.6f" | format(w.real) }}f, {{ "%0.6f" | format(w.imag) }}f) {%- if not loop.last %}, {% endif %}
{%- endfor %} } {%- if not loop.last %},{% endif %}
{%- endfor %}
};
```

What follows is the Python function used to generate the OpenCL code.

``` {.python #generate-twiddle-array}
def generate_twiddle_array(parity_splitting: ParitySplitting):
    """
    Generate OpenCL constant array for twiddle factors
    """
    template = template_environment.get_template("twiddles.cl")
    twiddles = numpy.ones(shape=[parity_splitting.radix, parity_splitting.radix])
    perm = numpy.array([comp_perm(parity_splitting.radix, i) for i in range(parity_splitting.M)])

    n = parity_splitting.radix
    for k in range(parity_splitting.depth - 1):
        w = make_twiddle(parity_splitting.radix, n).conj()
        w_r_x = (numpy.ones(shape=[parity_splitting.M // n, parity_splitting.radix, n]) * w) \
            .transpose([0, 2, 1]) \
            .reshape([-1, parity_splitting.radix])[perm]
        twiddles = numpy.r_[twiddles, w_r_x]
        n *= parity_splitting.radix

    return template.render(radix=parity_splitting.radix,
                           W=twiddles)
```

## Index functions

This is the parameterized OpenCL code used to compute indices.

```{.opencl file=fftsynth/templates/indices.cl}
inline int comp_idx_{{ radix }}(int i, int k)
{
    int rem = i % ipow(k);
    int base = i - rem;
    return MULR(base) + rem;
}


inline int comp_perm_{{ radix }}(int i, int rem)
{
    int p = parity_{{ radix }}(i);
    return MULR(i) + MODR(rem + {{ radix }} - p);
}
```

What follows is the Python function used to generate the OpenCL code.

``` {.python #generate-index-functions}
def generate_index_functions(parity_splitting: ParitySplitting):
    """
    Generate inline OpenCL function to compute indices and permutations of the indices.
    """
    template = template_environment.get_template("indices.cl")

    return template.render(radix=parity_splitting.radix)
```

## Utilities

This is the parameterized OpenCL code used to compute the integer power of a radix.

```{.opencl file=fftsynth/templates/ipow.cl}
{% if radix == 2 %}
inline int ipow(int b)
{
    return 1 << b;
}
{% elif radix == 4 %}
inline int ipow(int b)
{
    return 1 << (2*b);
}
{% else %}
inline int ipow(int b)
{
    int i, j;
    int a = {{ radix }};
    #pragma unroll
    for (i = 0, j = 1; i < b; ++i, j*=a);
    return j;
}
{% endif %}
```

What follows is the Python function used to generate the OpenCL code.

```{.python #generate-ipow-function}
def generate_ipow_function(parity_splitting: ParitySplitting):
    """
    Generate inline OpenCL function to compute the integer power of a radix.
    """
    template = template_environment.get_template("ipow.cl")

    return template.render(radix=parity_splitting.radix)
```
