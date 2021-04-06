# Code generator

``` {.python file=fftsynth/generator.py}
from jinja2 import Environment, FileSystemLoader
import numpy
from pkg_resources import resource_filename

from .parity import ParitySplitting, comp_perm
from .twiddle import make_twiddle

template_loader = FileSystemLoader(resource_filename("fftsynth", "templates"))
template_environment = Environment(loader=template_loader)


<<generate-preprocessor>>


<<generate-twiddle-array>>


<<generate-fpga-functions>>


<<generate-transpose-function>>


<<generate-parity-function>>


<<generate-ipow-function>>


<<generate-index-functions>>


<<generate-fft-functions>>


<<generate-codelets>>


def generate_fft(parity_splitting: ParitySplitting, fpga: bool):
    """
    Generate and print the complete OpenCL FFT.
    """
    print(generate_preprocessor(parity_splitting, fpga))
    print("\n")
    print(generate_twiddle_array(parity_splitting))
    print("\n")
    if fpga:
        print(generate_fpga_functions())
        print("\n")
    print(generate_parity_function(parity_splitting))
    print("\n")
    print(generate_transpose_function(parity_splitting))
    print("\n")
    print(generate_ipow_function(parity_splitting))
    print("\n")
    print(generate_index_functions(parity_splitting))
    print("\n")
    print(generate_codelets(fpga))
    print("\n")
    print(generate_fft_functions(parity_splitting, fpga))


def generate_fma_twiddle_array(parity_splitting: ParitySplitting):
    """
    Generate OpenCL constant array for twiddle factors
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
                           W=twiddles)


def generate_fma_codelets(fpga: bool):
    """
    Generate OpenCL codelets for FFT.
    """
    template = template_environment.get_template("fma-codelets.cl")

    return template.render(fpga=fpga)


def generate_fma_fft_functions(parity_splitting: ParitySplitting, fpga: bool):
    """
    Generate outer loop for OpenCL FFT.
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
                           M=parity_splitting.M,
                           fpga=fpga,
                           depth_type=depth_type,
                           m_type=m_type,
                           n_type=n_type)


def generate_fma_fft(parity_splitting: ParitySplitting, fpga: bool):
    """
    Generate and print the complete OpenCL FFT (FMA version).
    """
    print(generate_preprocessor(parity_splitting, fpga))
    print("\n")
    print(generate_fma_twiddle_array(parity_splitting))
    print("\n")
    if fpga:
        print(generate_fpga_functions())
        print("\n")
    print(generate_parity_function(parity_splitting))
    print("\n")
    print(generate_transpose_function(parity_splitting))
    print("\n")
    print(generate_ipow_function(parity_splitting))
    print("\n")
    print(generate_index_functions(parity_splitting))
    print("\n")
    print(generate_fma_codelets(fpga))
    print("\n")
    print(generate_fma_fft_functions(parity_splitting, fpga))


if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser(description="Generate an OpenCL FFT kernel")
    parser.add_argument("--radix", type=int, default=4, help="FFT radix")
    parser.add_argument("--depth", type=int, default=3, help="FFT depth")
    parser.add_argument("--fpga", action="store_true")
    parser.add_argument("--fma", action="store_true")
    args = parser.parse_args()
    print("/* FFT")
    print(f" * command: python -m fftsynth.generator {' '.join(sys.argv[1:])}")
    print(" */")
    N = args.radix**args.depth
    ps = ParitySplitting(N, args.radix)
    if args.fma:
        generate_fma_fft(ps, args.fpga)
    else:
        generate_fft(ps, args.fpga)
```

### Codelets

These are the codelets for specific radix FFT.

```{.opencl file=fftsynth/templates/codelets.cl}
void fft_2(__global float2 * restrict s0, __global float2 * restrict s1,{% if fpga %} float2 * restrict s0_in, float2 * restrict s1_in, float2 * restrict s0_out, float2 * restrict s1_out, bool first_iteration, bool last_iteration,{% endif %} int cycle, int i0, int i1, int iw)
{
    float2 t0, t1, ws0, ws1;
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
        case 1: SWAP(float2, t0, t1); break;
    }
    {% else %}
    switch (cycle) {
        case 0: t0 = s0[i0]; t1 = s1[i1]; break;
        case 1: t0 = s1[i0]; t1 = s0[i1]; break;
    }
    {% endif %}

    ws0 = t0;
    ws1 = (float2) (w[0].x * t1.x - w[0].y * t1.y,
                    w[0].x * t1.y + w[0].y * t1.x);

    t0 = ws0 + ws1;
    t1 = ws0 - ws1;

    {% if fpga %}
    switch (cycle) {
        case 1: SWAP(float2, t0, t1); break;
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

#ifdef TESTING
__kernel void test_radix_2(__global float2 *x, __global float2 *y, int n) {

    float2 w = (float2) (1.0, 0.0);
    int i = get_global_id(0)*2;

    //n is the number of radix2 ffts to perform
    if (i<2*n) {
        float2 y0, y1;
        fft_2(x, x, 0, i, i + 1, 0);

        y[i] = x[i]; y[i+1] = x[i+1];
    }
}
#endif // TESTING

void fft_4(float2 * restrict s0, float2 * restrict s1, float2 * restrict s2, float2 * restrict s3,{% if fpga %} float2 * restrict s0_in, float2 * restrict s1_in, float2 * restrict s2_in, float2 * restrict s3_in, float2 * restrict s0_out, float2 * restrict s1_out, float2 * restrict s2_out, float2 * restrict s3_out, bool first_iteration, bool last_iteration,{% endif %} int cycle, int i0, int i1, int i2, int i3, int iw)
{
    float2 t0, t1, t2, t3, ws0, ws1, ws2, ws3, a, b, c, d;
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
        case 1: SWAP(float2, t0, t1); SWAP(float2, t1, t2); SWAP(float2, t2, t3); break;
        case 2: SWAP(float2, t0, t2); SWAP(float2, t1, t3); break;
        case 3: SWAP(float2, t2, t3); SWAP(float2, t0, t1); SWAP(float2, t0, t2); break;
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
    ws1 = (float2) (w[0].x * t1.x - w[0].y * t1.y,
                    w[0].x * t1.y + w[0].y * t1.x);
    ws2 = (float2) (w[1].x * t2.x - w[1].y * t2.y,
                    w[1].x * t2.y + w[1].y * t2.x);
    ws3 = (float2) (w[2].x * t3.x - w[2].y * t3.y,
                    w[2].x * t3.y + w[2].y * t3.x);

    a = ws0 + ws2;
    b = ws1 + ws3;
    c = ws0 - ws2;
    d = ws1 - ws3;
    t0 = a + b;
    t1 = (float2) (c.x + d.y, c.y - d.x);
    t2 = a - b;
    t3 = (float2) (c.x - d.y, c.y + d.x);

    {% if fpga %}
    switch (cycle) {
        case 1: SWAP(float2, t2, t3); SWAP(float2, t1, t2); SWAP(float2, t0, t1); break;
        case 2: SWAP(float2, t1, t3); SWAP(float2, t0, t2); break;
        case 3: SWAP(float2, t0, t2); SWAP(float2, t0, t1); SWAP(float2, t2, t3); break;
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
```

What follows is the Python function used to generate the OpenCL code.

```{.python #generate-codelets}
def generate_codelets(fpga: bool):
    """
    Generate OpenCL codelets for FFT.
    """
    template = template_environment.get_template("codelets.cl")

    return template.render(fpga=fpga)
```

### Preprocessor

This is the parameterized OpenCL code used to write the preprocessor directives.

```{.opencl file=fftsynth/templates/preprocessor.cl}
{% if fpga -%}
#pragma OPENCL EXTENSION cl_intel_channels : enable
#include <ihc_apint.h>
channel float2 in_channel, out_channel;
#define SWAP(type, x, y) do { type temp = x; x = y, y = temp; } while ( false );
{% endif -%}
{% if radix == 4 %}
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
def generate_preprocessor(parity_splitting: ParitySplitting, fpga: bool):
    """
    Generate the preprocessor directives necessary for the FFT.
    """
    template = template_environment.get_template("preprocessor.cl")

    return template.render(radix=parity_splitting.radix, fpga=fpga)
```

### FPGA specific code

This is the parameterized OpenCL code used to write the FPGA specific functions.

```{.opencl file=fftsynth/templates/fpga.cl}
__kernel __attribute__((max_global_work_dim(0)))
void source(__global const volatile float2 * in, unsigned count)
{
    #pragma ii 1
    for ( unsigned i = 0; i < count; i++ )
    {
        write_channel_intel(in_channel, in[i]);
    }
}

__kernel __attribute__((max_global_work_dim(0)))
void sink(__global float2 *out, unsigned count)
{
    #pragma ii 1
    for ( unsigned i = 0; i < count; i++ )
    {
        out[i] = read_channel_intel(out_channel);
    }
}
```

What follows is the Python function used to generate the OpenCL code.

```{.python #generate-fpga-functions}
def generate_fpga_functions():
    """
    Generate OpenCL code for FPGA functions.
    """
    template = template_environment.get_template("fpga.cl")

    return template.render()
```

## Testing

To test we need to define small `__kernel` functions.

``` {.python file=test/test_generator.py}
import pytest
import numpy as np
from kernel_tuner import run_kernel  # type: ignore

from fftsynth.parity import ParitySplitting, parity
from fftsynth.generator import generate_preprocessor, generate_transpose_function, generate_parity_function

cases = [
    ParitySplitting(64, 4),
    ParitySplitting(81, 3),
    ParitySplitting(125, 5)
]


<<test-parity>>

<<test-transpose>>
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
def generate_fft_functions(parity_splitting: ParitySplitting, fpga: bool):
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
                           n_type=n_type)
```

### Inner loop

Here we have `k` being the index of the outer loop.
The next bit is only still implemented for radix-4.

```{.opencl file=fftsynth/templates/fft.cl}
void fft_{{ N }}_ps({% for i in range(radix) %} float2 * restrict s{{ i }}{%- if not loop.last %},{% endif %}{% endfor %}{% if fpga %},{% for i in range(radix) %} float2 * restrict s{{ i }}_in,{% endfor %}{% for i in range(radix) %} float2 * restrict s{{ i }}_out{%- if not loop.last %},{% endif %}{% endfor %}{% endif %})
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
__kernel {%if fpga %}__attribute__((autorun)) __attribute__((max_global_work_dim(0))){% endif %}
void fft_{{ N }}({% if not fpga %}__global const float2 * restrict x, __global float2 * restrict y{% endif %})
{
    {% if fpga -%}
    while ( true )
    {
    {% endif -%}
    {%- for i in range(radix) %}
    float2 s{{ i }}[{{ M }}];
    {% if fpga -%}
    float2 s{{ i }}_in[{{ M }}], s{{ i }}_out[{{ M }}];
    {%- endif -%}
    {%- endfor %}

    for ( {{ n_type }} j = 0; j != {{ N }}; ++j )
    {
        int i = transpose_{{ radix }}(j);
        int p = parity_{{ radix }}(i);

        {% if fpga -%}
        float2 x = read_channel_intel(in_channel);
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
        float2 y;
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
{% if radix == 4 %}
inline int ipow(int b)
{
    return 1 << (2*b);
}
{% else %}
inline int ipow(int b)
{
    int i, j;
    int a = {{ radix }};
    #pragma unroll 10
    for (i = 1, j = a; i < b; ++i, j*=a);
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
