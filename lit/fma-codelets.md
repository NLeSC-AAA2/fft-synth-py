# Fused-Multiply-Add codelets

Rewriting FFT codelets to contain **more** floating-point operations, but increase the fraction of FP operations inside an FMA operation can increase the efficiency on some architectures.

## Twiddle factors

Only half of the twiddle factors are used.

```{.opencl file=fftsynth/templates/fma-twiddles.cl}
__constant float2 W[{{ W.shape[0] - radix }}][{{ n_twiddles }}] = {
{% for ws in W[radix:] %}
{ {% for w in ws[1:n_twiddles+1] -%}
(float2)({{ "%0.6f" | format(w.real) }}f, {{ "%0.6f" | format(w.imag) }}f) {%- if not loop.last %}, {% endif %}
{%- endfor %} } {%- if not loop.last %},{% endif %}
{%- endfor %}
};

```

## Radix 2

A normal implementation of the radix-2 butterfly

``` {.opencl}
void radix2(float2 e, float2 o, float2 w, float2* xa, float2* xb)
{
    float2 t = w * o;
    *xa = e + t;
    *xb = e - t;
}
```

This contains one complex multiplication and two complex additions, totaling four floating point multiplications and six additions.

``` {.opencl #fma-radix2}
{% if radix == 2 %}
void fft_2({{c_type}} * restrict s0, {{c_type}} * restrict s1,{% if fpga %} {{c_type}} * restrict s0_in, {{c_type}} * restrict s1_in, {{c_type}} * restrict s0_out, {{c_type}} * restrict s1_out, bool first_iteration, bool last_iteration,{% endif %} int cycle, int i0, int i1, int iw)
{
    {{c_type}} t0, t1, a, b;
    #ifndef TESTING_RADIX
    __constant float2 *w = W[iw];
    #endif // !TESTING_RADIX
    #ifdef TESTING_RADIX
    float2 w[] = {(float2)(1.0, 0.0)};
    #endif // TESTING_RADIX

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

    a = ({{c_type}}) (-w[0].odd * t1.odd + t0.even, w[0].odd * t1.even + t0.odd);
    a += ({{c_type}}) (w[0].even * t1.even, w[0].even * t1.odd);
    b = 2 * t0 - a;

    t0 = a;
    t1 = b;

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
{% endif %}
```

Now we have six multiplications and six additions, but they are completely contained in six FMA operations.

## Higher radix

``` {.opencl .bootstrap-fold file=fftsynth/templates/fma-codelets.cl}
<<fma-radix2>>

{% if radix == 3 %}
void fft_3({{c_type}} * restrict s0, {{c_type}} * restrict s1, {{c_type}} * restrict s2,{% if fpga %} {{c_type}} * restrict s0_in, {{c_type}} * restrict s1_in, {{c_type}} * restrict s2_in, {{c_type}} * restrict s0_out, {{c_type}} * restrict s1_out, {{c_type}} * restrict s2_out, bool first_iteration, bool last_iteration,{% endif %} int cycle, int i0, int i1, int i2, int iw)
{
    {{c_type}} t0, t1, t2, z1, a, b, c, d, e, f;
    const float c1 = -0.5;
    const float c2 = -0.8660254037844386;
    #ifndef TESTING_RADIX
    __constant float2 *w = W[iw];
    #endif // !TESTING_RADIX
    #ifdef TESTING_RADIX
    float2 w[] = {(float2)(1.0, 0.0), (float2)(1.0, 0.0)};
    #endif // TESTING_RADIX

    {% if fpga %}
    switch (cycle) {
        case 1: SWAP(int, i0, i1); SWAP(int, i0, i2); break;
        case 2: SWAP(int, i0, i2); break;
    }
    if ( first_iteration )
    {
        t0 = s0_in[i0]; t1 = s1_in[i1]; t2 = s2_in[i2];
    }
    else
    {
        t0 = s0[i0]; t1 = s1[i1]; t2 = s2[i2];
    }
    switch (cycle) {
        case 1: SWAP({{c_type}}, t0, t1); SWAP({{c_type}}, t1, t2); break;
        case 2: SWAP({{c_type}}, t0, t2); break;
    }
    {% else %}
    switch (cycle) {
        case 0: t0 = s0[i0]; t1 = s1[i1]; t2 = s2[i2]; break;
        case 1: t0 = s1[i0]; t1 = s2[i1]; t2 = s0[i2]; break;
        case 2: t0 = s2[i0]; t1 = s0[i1]; t2 = s1[i2]; break;
    }
    {% endif %}

    // z1 = w0 * t1
    z1 = ({{c_type}}) (w[0].even * t1.even - w[0].odd * t1.odd,
                   w[0].even * t1.odd + w[0].odd * t1.even);

    // a = z1 - w1 * t2
    a = ({{c_type}}) (z1.even - w[1].even * t2.even + w[1].odd * t2.odd,
                  z1.odd - w[1].even * t2.odd - w[1].odd * t2.even);

    // b = 2 * z1 - a
    b = ({{c_type}}) (2 * z1.even - a.even,
                  2 * z1.odd - a.odd);

    // c = b + t0
    c = ({{c_type}}) (b.even + t0.even,
                  b.odd + t0.odd);

    // d = t0 + c1 * b
    d = ({{c_type}}) (t0.even + c1 * b.even,
                  t0.odd + c1 * b.odd);

    // e = d - i * c2 * a
    e = ({{c_type}}) (d.even - c2 * a.odd,
                  d.odd + c2 * a.even);

    // f = 2 * d - e
    f = ({{c_type}}) (2 * d.even - e.even,
                  2 * d.odd - e.odd);

    t0 = c;
    t1 = e; // pseudo code in karner2001multiply paper says s6
    t2 = f; // pseudo code in karner2001multiply paper says s5

    {% if fpga %}
    switch (cycle) {
        case 1: SWAP({{c_type}}, t1, t2); SWAP({{c_type}}, t0, t1); break;
        case 2: SWAP({{c_type}}, t0, t2); break;
    }
    if ( last_iteration )
    {
        s0_out[i0] = t0; s1_out[i1] = t1; s2_out[i2] = t2;
    }
    else
    {
        s0[i0] = t0; s1[i1] = t1; s2[i2] = t2;
    }
    {% else %}
    switch (cycle) {
        case 0: s0[i0] = t0; s1[i1] = t1; s2[i2] = t2; break;
        case 1: s1[i0] = t0; s2[i1] = t1; s0[i2] = t2; break;
        case 2: s2[i0] = t0; s0[i1] = t1; s1[i2] = t2; break;
    }
    {% endif %}
}
{% endif %}

{% if radix == 4 %}
void fft_4({{c_type}} * restrict s0, {{c_type}} * restrict s1, {{c_type}} * restrict s2, {{c_type}} * restrict s3,{% if fpga %} {{c_type}} * restrict s0_in, {{c_type}} * restrict s1_in, {{c_type}} * restrict s2_in, {{c_type}} * restrict s3_in, {{c_type}} * restrict s0_out, {{c_type}} * restrict s1_out, {{c_type}} * restrict s2_out, {{c_type}} * restrict s3_out, bool first_iteration, bool last_iteration,{% endif %} int cycle, int i0, int i1, int i2, int i3, int iw)
{
     {{c_type}} t0, t1, t2, t3, a, b, c, d;
    #ifndef TESTING_RADIX
    __constant float2 *w = W[iw];
    #endif // !TESTING_RADIX
    #ifdef TESTING_RADIX
    float2 w[] = {(float2)(1.0, 0.0), (float2)(1.0, 0.0)};
    #endif // TESTING_RADIX

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

   // adapted from pedram2013transforming, however
   // some versions of the pedram2013transforming paper, including the one hosted by IEEE and the one hosted here:
   // https://www.cs.utexas.edu/users/flame/pubs/LAC_fft.pdf
   // contains serious errors in the FMA-optimized radix-4 pseudocode
   // finally corrected based on the pseudocode reported in karner1998top
    a = t0;     b = t2;     c = t1;     d = t3;

    b = ({{c_type}}) (a.even - w[1].even * b.even + w[1].odd * b.odd,
                  a.odd - w[1].even * b.odd - w[1].odd * b.even);
    a = ({{c_type}}) (2*a.even - b.even,
                  2*a.odd - b.odd);
    d = ({{c_type}}) (c.even - w[1].even * d.even + w[1].odd * d.odd,
                  c.odd - w[1].even * d.odd - w[1].odd * d.even);
    c = ({{c_type}}) (2*c.even - d.even,
                  2*c.odd - d.odd);

    c = ({{c_type}}) (a.even - w[0].even * c.even + w[0].odd * c.odd,
                  a.odd - w[0].even * c.odd - w[0].odd * c.even);
    t2 = c;
    t0 = ({{c_type}}) (2*a.even - c.even,
                    2*a.odd - c.odd);

    //d = b - i*w0*d
    d = ({{c_type}}) (b.even + w[0].even * d.odd + w[0].odd * d.even,
                  b.odd - w[0].even * d.even + w[0].odd * d.odd);
    t1 = d;
    t3 = ({{c_type}}) (2*b.even - d.even,
                    2*b.odd - d.odd);

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
{% endif %}


{% if radix == 5 %}
void fft_5({{c_type}} * restrict s0, {{c_type}} * restrict s1, {{c_type}} * restrict s2, {{c_type}} * restrict s3, {{c_type}} * restrict s4,{% if fpga %} {{c_type}} * restrict s0_in, {{c_type}} * restrict s1_in, {{c_type}} * restrict s2_in, {{c_type}} * restrict s3_in, {{c_type}} * restrict s4_in, {{c_type}} * restrict s0_out, {{c_type}} * restrict s1_out, {{c_type}} * restrict s2_out, {{c_type}} * restrict s3_out, {{c_type}} * restrict s4_out, bool first_iteration, bool last_iteration,{% endif %} int cycle, int i0, int i1, int i2, int i3, int i4, int iw)
 {

    const float c1 = 0.25;                  // 1/4
    const float c2 = 0.5590169943749475;    // sqrt(5)/4
    const float c3 = 0.6180339887498949;    // sqrt( (5-sqrt(5))/(5+sqrt(5)) )
    const float c4 = 0.9510565162951535;    // 1/2 * np.sqrt(5/2 + np.sqrt(5)/2)

    {{c_type}} z0, z1, z2;
    {{c_type}} s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11;
    {{c_type}} q1, q2;

    //z0 = t0
    z0 = t0;

    //z1 = w0*t1
    z1 = ({{c_type}}) (w0.even * t1.even - w0.odd * t1.odd,
                   w0.even * t1.odd + w0.odd * t1.even);

    //z2 = w1*t2
    z2 = ({{c_type}}) (w1.even * t2.even - w1.odd * t2.odd,
                   w1.even * t2.odd + w1.odd * t2.even);

    //s1 = z1 - w3*t4
    s1 = ({{c_type}}) (z1.even - w3.even * t4.even + w3.odd * t4.odd,
                   z1.odd - w3.even * t4.odd - w3.odd * t4.even);

    //s2 = 2*z1-s1
    s2 = ({{c_type}}) (2*z1.even - s1.even,
                   2*z1.odd - s1.odd);

    //s3 = z2 - w2*t3
    s3 = ({{c_type}}) (z2.even - w2.even * t3.even + w2.odd * t3.odd,
                   z2.odd - w2.even * t3.odd - w2.odd * t3.even);

    //s4 = 2*z2 - s3
    s4 = ({{c_type}}) (2*z2.even - s3.even,
                   2*z2.odd - s3.odd);

    //s5 = s2+s4
    s5 = ({{c_type}}) (s2.even + s4.even,
                   s2.odd + s4.odd);

    //s6 = s2-s4
    s6 = ({{c_type}}) (s2.even - s4.even,
                   s2.odd - s4.odd);

    //s7 = z0 - c1*s5
    s7 = ({{c_type}}) (z0.even - c1*s5.even,
                   z0.odd - c1*s5.odd);

    //s8 = s7 - c2*s6
    s8 = ({{c_type}}) (s7.even - c2*s6.even,
                   s7.odd - c2*s6.odd);

    //s9 = 2*s7 - s8
    s9 = ({{c_type}}) (2*s7.even - s8.even,
                   2*s7.odd - s8.odd);

    //s10 = s1 + c3*s3
    s10 = ({{c_type}}) (s1.even + c3*s3.even,
                    s1.odd + c3*s3.odd);

    //s11 = c3*s1 - s3
    s11 = ({{c_type}}) (c3*s1.even - s3.even,
                    c3*s1.odd - s3.odd);

    // *x0 = z0 + s5
    *x0 = ({{c_type}}) (z0.even + s5.even,
                    z0.odd + s5.odd);

    //q1 = s9 - i*c4*s10
    q1 = ({{c_type}}) (s9.even - c4*s10.odd,
                   s9.odd + c4*s10.even);

    // *x1 = 2*s9 - q1
    *x1 = ({{c_type}}) (2*s9.even - q1.even,
                    2*s9.odd - q1.odd);

    //q2 = s8 - i*c4*s11
    q2 = ({{c_type}}) (s8.even - c4*s11.odd,
                   s8.odd + c4*s11.even);

    // *x2 = 2*s8-q2
    *x2 = ({{c_type}}) (2*s8.even - q2.even,
                    2*s8.odd - q2.odd);

    // *x3 = q2
    *x3 = q2;

    // *x4 = q1 
    *x4 = q1;

    return 0.0;
}
{% endif %}

<<fma-codelet-tests>>
```

## Testing

``` {.opencl .bootstrap-fold #fma-codelet-tests}
#ifdef TESTING

{% if radix == 2 %}
__kernel void test_radix_2(__global {{c_type}} *x, __global {{c_type}} *y, int n)
{
    int i = get_global_id(0) * 2;

    // n is the number of radix2 ffts to perform
    if ( i < 2 * n ) {
        {{c_type}} s0 = x[i];
        {{c_type}} s1 = x[i + 1];

        fft_2(&s0, &s1, 0, 0, 0, 0);

        y[i] = s0; y[i + 1] = s1;
    }
}
{% endif %}

{% if radix == 3 %}
__kernel void test_radix_3(__global {{c_type}} *x, __global {{c_type}} *y, int n)
{
    int i = get_global_id(0) * 3;

    // n is the number of radix3 ffts to perform
    if ( i < 3 * n ) {
        {{c_type}} s0 = x[i];
        {{c_type}} s1 = x[i + 1];
        {{c_type}} s2 = x[i + 2];

        fft_3(&s0, &s1, &s2, 0, 0, 0, 0, 0);

        y[i] = s0;    y[i+1] = s1;    y[i+2] = s2;
    }
}
{% endif %}

{% if radix == 4 %}
__kernel void test_radix_4(__global {{c_type}} *x, __global {{c_type}} *y, int n)
{
    int i = get_global_id(0) * 4;

    // n is the number of radix4 ffts to perform
    if (i < 4 * n) {
        {{c_type}} s0 = x[i];
        {{c_type}} s1 = x[i + 1];
        {{c_type}} s2 = x[i + 2];
        {{c_type}} s3 = x[i + 3];
        fft_4(&s0, &s1, &s2, &s3, 0, 0, 0, 0, 0, 0);

        y[i] = s0;    y[i + 1] = s1;    y[i + 2] = s2;    y[i + 3] = s3;
    }
}
{% endif %}

{% if radix == 5 %}
__kernel void test_radix_5(__global {{c_type}} *x, __global {{c_type}} *y, int n)
{
    int i = get_global_id(0)*5;

    // n is the number of radix5 ffts to perform
    if ( i < 5 * n ) {
        {{c_type}} s0 = x[i];
        {{c_type}} s1 = x[i + 1];
        {{c_type}} s2 = x[i + 2];
        {{c_type}} s3 = x[i + 3];
        {{c_type}} s4 = x[i + 4];

        fft_5(&s0, &s1, &s2, &s3, &s4, 0, 0, 0, 0, 0, 0, 0);

        y[i] = s0;    y[i+1] = s1;    y[i+2] = s2;    y[i+3] = s3;    y[i+4] = s4;
    }
}
{% endif %}
#endif // TESTING
```

``` {.python file=test/test_fma_codelets.py}
import numpy

import pytest
from fftsynth import generator, parity
from kernel_tuner import run_kernel     # type: ignore


@pytest.mark.parametrize('radix', [2, 3, 4])
def test_radix(radix):
    # this test runs 256 instances of the radix n function
    # it does not use twiddle factors, so as a test
    # it's not to be relied upon fully
    n = numpy.int32(256)
    x = numpy.random.normal(size=(n, radix, 2)).astype(numpy.float32)
    y = numpy.zeros_like(x)

    y_ref = numpy.fft.fft(x[..., 0]+1j*x[..., 1])

    parity_splitting = parity.ParitySplitting(radix * n, radix)
    codelets = "{}\n{}".format(generator.generate_preprocessor(parity_splitting, False),
                               generator.generate_fma_codelets(parity_splitting, False))
    args = [x, y, n]
    answer = run_kernel(f"test_radix_{radix}", codelets, 1, args, {}, compiler_options=["-DTESTING", "-DTESTING_RADIX"])

    y = answer[1]
    y = y[..., 0] + 1j * y[..., 1]

    numpy.testing.assert_almost_equal(y, y_ref, decimal=5)
```
