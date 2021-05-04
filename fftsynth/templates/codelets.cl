/* ~\~ language=OpenCL filename=fftsynth/templates/codelets.cl */
/* ~\~ begin <<lit/code-generator.md|fftsynth/templates/codelets.cl>>[0] */
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

#ifdef TESTING
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
#endif // TESTING

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

#ifdef TESTING
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
#endif // TESTING
{% endif %}
/* ~\~ end */
