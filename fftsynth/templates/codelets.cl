/* ~\~ language=OpenCL filename=fftsynth/templates/codelets.cl */
/* ~\~ begin <<lit/code-generator.md|fftsynth/templates/codelets.cl>>[0] */
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
    int i = get_global_id(0) * 2;

    //n is the number of radix2 ffts to perform
    if (i < 2 * n) {
        fft_2(x, x, 0, i, i + 1, 0);

        y[i] = x[i]; y[i + 1] = x[i + 1];
    }
}
#endif // TESTING

void fft_4(__global float2 * restrict s0, __global float2 * restrict s1, __global float2 * restrict s2, __global float2 * restrict s3,{% if fpga %} float2 * restrict s0_in, float2 * restrict s1_in, float2 * restrict s2_in, float2 * restrict s3_in, float2 * restrict s0_out, float2 * restrict s1_out, float2 * restrict s2_out, float2 * restrict s3_out, bool first_iteration, bool last_iteration,{% endif %} int cycle, int i0, int i1, int i2, int i3, int iw)
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

#ifdef TESTING
__kernel void test_radix_4(__global float2 *x, __global float2 *y, int n) {

    float2 w0 = (float2) (1.0, 0.0);
    float2 w1 = (float2) (1.0, 0.0);
    int i = get_global_id(0) * 4;

    //n is the number of radix4 ffts to perform
    if (i < 4 * n) {
        fft_4(x, x, x, x, 0, i, i + 1, i + 2, i + 3, 0);

        y[i] = x[i];    y[i + 1] = x[i + 1];    y[i + 2] = x[i + 2];    y[i + 3] = x[i + 3];
    }
}
#endif // TESTING

/* ~\~ end */
