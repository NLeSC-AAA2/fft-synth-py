/* ~\~ language=OpenCL filename=fftsynth/templates/fma-codelets.cl */
/* ~\~ begin <<lit/fma-codelets.md|fftsynth/templates/fma-codelets.cl>>[0] */
/* ~\~ begin <<lit/fma-codelets.md|fma-radix2>>[0] */
{% if radix == 2 %}
void fft_2(float2 * restrict s0, float2 * restrict s1,{% if fpga %} float2 * restrict s0_in, float2 * restrict s1_in, float2 * restrict s0_out, float2 * restrict s1_out, bool first_iteration, bool last_iteration,{% endif %} int cycle, int i0, int i1, int iw)
{
    float2 t0, t1, a, b;
    #ifndef TESTING
    __constant float2 *w = W[iw];
    #endif
    #ifdef TESTING
    float2 w[] = {(float2)(1.0, 0.0)};
    #endif // TESTING

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

    a = (float2) (-w[0].y * t1.y + t0.x, w[0].y * t1.x + t0.y);
    a += (float2) (w[0].x * t1.x, w[0].x * t1.y);
    b = 2 * t0 - a;

    t0 = a;
    t1 = b;

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
{% endif %}
/* ~\~ end */

{% if radix == 3 %}
void fft_3(float2 * restrict s0, float2 * restrict s1, float2 * restrict s2,{% if fpga %} float2 * restrict s0_in, float2 * restrict s1_in, float2 * restrict s2_in, float2 * restrict s0_out, float2 * restrict s1_out, float2 * restrict s2_out, bool first_iteration, bool last_iteration,{% endif %} int cycle, int i0, int i1, int i2, int iw)
{
    float2 t0, t1, t2, z1, a, b, c, d, e, f;
    const float c1 = -0.5;
    const float c2 = -0.8660254037844386;
    #ifndef TESTING
    __constant float2 *w = W[iw];
    #endif
    #ifdef TESTING
    float2 w[] = {(float2)(1.0, 0.0), (float2)(1.0, 0.0)};
    #endif // TESTING

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
        case 1: SWAP(float2, t0, t1); SWAP(float2, t1, t2); break;
        case 2: SWAP(float2, t0, t2); break;
    }
    {% else %}
    switch (cycle) {
        case 0: t0 = s0[i0]; t1 = s1[i1]; t2 = s2[i2]; break;
        case 1: t0 = s1[i0]; t1 = s2[i1]; t2 = s0[i2]; break;
        case 2: t0 = s2[i0]; t1 = s0[i1]; t2 = s1[i2]; break;
    }
    {% endif %}

    // z1 = w0 * t1
    z1 = (float2) (w[0].x * t1.x - w[0].y * t1.y,
                   w[0].x * t1.y + w[0].y * t1.x);

    // a = z1 - w1 * t2
    a = (float2) (z1.x - w[1].x * t2.x + w[1].y * t2.y,
                  z1.y - w[1].x * t2.y - w[1].y * t2.x);

    // b = 2 * z1 - a
    b = (float2) (2 * z1.x - a.x,
                  2 * z1.y - a.y);

    // c = b + t0
    c = (float2) (b.x + t0.x,
                  b.y + t0.y);

    // d = t0 + c1 * b
    d = (float2) (t0.x + c1 * b.x,
                  t0.y + c1 * b.y);

    // e = d - i * c2 * a
    e = (float2) (d.x - c2 * a.y,
                  d.y + c2 * a.x);

    // f = 2 * d - e
    f = (float2) (2 * d.x - e.x,
                  2 * d.y - e.y);

    t0 = c;
    t1 = e; // pseudo code in karner2001multiply paper says s6
    t2 = f; // pseudo code in karner2001multiply paper says s5

    {% if fpga %}
    switch (cycle) {
        case 1: SWAP(float2, t1, t2); SWAP(float2, t0, t1); break;
        case 2: SWAP(float2, t0, t2); break;
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
void fft_4(float2 * restrict s0, float2 * restrict s1, float2 * restrict s2, float2 * restrict s3,{% if fpga %} float2 * restrict s0_in, float2 * restrict s1_in, float2 * restrict s2_in, float2 * restrict s3_in, float2 * restrict s0_out, float2 * restrict s1_out, float2 * restrict s2_out, float2 * restrict s3_out, bool first_iteration, bool last_iteration,{% endif %} int cycle, int i0, int i1, int i2, int i3, int iw)
{
     float2 t0, t1, t2, t3, a, b, c, d;
    #ifndef TESTING
    __constant float2 *w = W[iw];
    #endif
    #ifdef TESTING
    float2 w[] = {(float2)(1.0, 0.0), (float2)(1.0, 0.0)};
    #endif // TESTING

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

   // adapted from pedram2013transforming, however
   // some versions of the pedram2013transforming paper, including the one hosted by IEEE and the one hosted here:
   // https://www.cs.utexas.edu/users/flame/pubs/LAC_fft.pdf
   // contains serious errors in the FMA-optimized radix-4 pseudocode
   // finally corrected based on the pseudocode reported in karner1998top
    a = t0;     b = t2;     c = t1;     d = t3;

    b = (float2) (a.x - w[1].x * b.x + w[1].y * b.y,
                  a.y - w[1].x * b.y - w[1].y * b.x);
    a = (float2) (2*a.x - b.x,
                  2*a.y - b.y);
    d = (float2) (c.x - w[1].x * d.x + w[1].y * d.y,
                  c.y - w[1].x * d.y - w[1].y * d.x);
    c = (float2) (2*c.x - d.x,
                  2*c.y - d.y);

    c = (float2) (a.x - w[0].x * c.x + w[0].y * c.y,
                  a.y - w[0].x * c.y - w[0].y * c.x);
    t2 = c;
    t0 = (float2) (2*a.x - c.x,
                    2*a.y - c.y);

    //d = b - i*w0*d
    d = (float2) (b.x + w[0].x * d.y + w[0].y * d.x,
                  b.y - w[0].x * d.x + w[0].y * d.y);
    t1 = d;
    t3 = (float2) (2*b.x - d.x,
                    2*b.y - d.y);

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
{% endif %}


{% if radix == 5 %}
void fft_5(float2 * restrict s0, float2 * restrict s1, float2 * restrict s2, float2 * restrict s3, float2 * restrict s4,{% if fpga %} float2 * restrict s0_in, float2 * restrict s1_in, float2 * restrict s2_in, float2 * restrict s3_in, float2 * restrict s4_in, float2 * restrict s0_out, float2 * restrict s1_out, float2 * restrict s2_out, float2 * restrict s3_out, float2 * restrict s4_out, bool first_iteration, bool last_iteration,{% endif %} int cycle, int i0, int i1, int i2, int i3, int i4, int iw)
 {

    const float c1 = 0.25;                  // 1/4
    const float c2 = 0.5590169943749475;    // sqrt(5)/4
    const float c3 = 0.6180339887498949;    // sqrt( (5-sqrt(5))/(5+sqrt(5)) )
    const float c4 = 0.9510565162951535;    // 1/2 * np.sqrt(5/2 + np.sqrt(5)/2)

    float2 z0, z1, z2;
    float2 s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11;
    float2 q1, q2;

    //z0 = t0
    z0 = t0;

    //z1 = w0*t1
    z1 = (float2) (w0.x * t1.x - w0.y * t1.y,
                   w0.x * t1.y + w0.y * t1.x);

    //z2 = w1*t2
    z2 = (float2) (w1.x * t2.x - w1.y * t2.y,
                   w1.x * t2.y + w1.y * t2.x);

    //s1 = z1 - w3*t4
    s1 = (float2) (z1.x - w3.x * t4.x + w3.y * t4.y,
                   z1.y - w3.x * t4.y - w3.y * t4.x);

    //s2 = 2*z1-s1
    s2 = (float2) (2*z1.x - s1.x,
                   2*z1.y - s1.y);

    //s3 = z2 - w2*t3
    s3 = (float2) (z2.x - w2.x * t3.x + w2.y * t3.y,
                   z2.y - w2.x * t3.y - w2.y * t3.x);

    //s4 = 2*z2 - s3
    s4 = (float2) (2*z2.x - s3.x,
                   2*z2.y - s3.y);

    //s5 = s2+s4
    s5 = (float2) (s2.x + s4.x,
                   s2.y + s4.y);

    //s6 = s2-s4
    s6 = (float2) (s2.x - s4.x,
                   s2.y - s4.y);

    //s7 = z0 - c1*s5
    s7 = (float2) (z0.x - c1*s5.x,
                   z0.y - c1*s5.y);

    //s8 = s7 - c2*s6
    s8 = (float2) (s7.x - c2*s6.x,
                   s7.y - c2*s6.y);

    //s9 = 2*s7 - s8
    s9 = (float2) (2*s7.x - s8.x,
                   2*s7.y - s8.y);

    //s10 = s1 + c3*s3
    s10 = (float2) (s1.x + c3*s3.x,
                    s1.y + c3*s3.y);

    //s11 = c3*s1 - s3
    s11 = (float2) (c3*s1.x - s3.x,
                    c3*s1.y - s3.y);

    // *x0 = z0 + s5
    *x0 = (float2) (z0.x + s5.x,
                    z0.y + s5.y);

    //q1 = s9 - i*c4*s10
    q1 = (float2) (s9.x - c4*s10.y,
                   s9.y + c4*s10.x);

    // *x1 = 2*s9 - q1
    *x1 = (float2) (2*s9.x - q1.x,
                    2*s9.y - q1.y);

    //q2 = s8 - i*c4*s11
    q2 = (float2) (s8.x - c4*s11.y,
                   s8.y + c4*s11.x);

    // *x2 = 2*s8-q2
    *x2 = (float2) (2*s8.x - q2.x,
                    2*s8.y - q2.y);

    // *x3 = q2
    *x3 = q2;

    // *x4 = q1 
    *x4 = q1;

    return 0.0;
}
{% endif %}

/* ~\~ begin <<lit/fma-codelets.md|fma-codelet-tests>>[0] */
#ifdef TESTING

{% if radix == 2 %}
__kernel void test_radix_2(__global float2 *x, __global float2 *y, int n)
{
    int i = get_global_id(0) * 2;

    // n is the number of radix2 ffts to perform
    if ( i < 2 * n ) {
        float2 s0 = x[i];
        float2 s1 = x[i + 1];

        fft_2(&s0, &s1, 0, 0, 0, 0);

        y[i] = s0; y[i + 1] = s1;
    }
}
{% endif %}

{% if radix == 3 %}
__kernel void test_radix_3(__global float2 *x, __global float2 *y, int n)
{
    int i = get_global_id(0) * 3;

    // n is the number of radix3 ffts to perform
    if ( i < 3 * n ) {
        float2 s0 = x[i];
        float2 s1 = x[i + 1];
        float2 s2 = x[i + 2];

        fft_3(&s0, &s1, &s2, 0, 0, 0, 0, 0);

        y[i] = s0;    y[i+1] = s1;    y[i+2] = s2;
    }
}
{% endif %}

{% if radix == 4 %}
__kernel void test_radix_4(__global float2 *x, __global float2 *y, int n)
{
    int i = get_global_id(0) * 4;

    // n is the number of radix4 ffts to perform
    if (i < 4 * n) {
        float2 s0 = x[i];
        float2 s1 = x[i + 1];
        float2 s2 = x[i + 2];
        float2 s3 = x[i + 3];
        fft_4(&s0, &s1, &s2, &s3, 0, 0, 0, 0, 0, 0);

        y[i] = s0;    y[i + 1] = s1;    y[i + 2] = s2;    y[i + 3] = s3;
    }
}
{% endif %}

{% if radix == 5 %}
__kernel void test_radix_5(__global float2 *x, __global float2 *y, int n)
{
    int i = get_global_id(0)*5;

    // n is the number of radix5 ffts to perform
    if ( i < 5 * n ) {
        float2 s0 = x[i];
        float2 s1 = x[i + 1];
        float2 s2 = x[i + 2];
        float2 s3 = x[i + 3];
        float2 s4 = x[i + 4];

        fft_5(&s0, &s1, &s2, &s3, &s4, 0, 0, 0, 0, 0, 0, 0);

        y[i] = s0;    y[i+1] = s1;    y[i+2] = s2;    y[i+3] = s3;    y[i+4] = s4;
    }
}
{% endif %}
#endif // TESTING
/* ~\~ end */
/* ~\~ end */
