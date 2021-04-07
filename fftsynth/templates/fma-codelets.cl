/* ~\~ language=OpenCL filename=fftsynth/templates/fma-codelets.cl */
/* ~\~ begin <<lit/fma-codelets.md|fftsynth/templates/fma-codelets.cl>>[0] */
/* ~\~ begin <<lit/fma-codelets.md|fma-radix2>>[0] */
{% if radix == 2 %}
void fft_2(float2 t0, float2 t1, float2 w0, float2* xa, float2* xb)
{
    float2 a = (float2) (-w0.y * t1.y + t0.x, w0.y * t1.x + t0.y);
    a += (float2) (w0.x * t1.x, w0.x * t1.y);
    float2 b = 2 * t0 - a;

    *xa = a;
    *xb = b;
}
{% endif %}
/* ~\~ end */

{% if radix == 3 %}
void fft_3(float2 t0, float2 t1, float2 t2, float2 w0, float2 w1, float2* xa, float2* xb, float2* xc) {

    float2 z1, s1, s2, s3, s4, s5, s6;
    const float c1 = -0.5;
    const float c2 = -0.8660254037844386;

    //z1 = w0 * t1
    z1 = (float2) (w0.x * t1.x - w0.y * t1.y,
                   w0.x * t1.y + w0.y * t1.x);

    //s1 = z1 - w1 * t2
    s1 = (float2) (z1.x - w1.x * t2.x + w1.y * t2.y,
                   z1.y - w1.x * t2.y - w1.y * t2.x);

    //s2 = 2*z1 - s1
    s2 = (float2) (2*z1.x - s1.x,
                   2*z1.y - s1.y);

    //s3 = s2 + t0
    s3 = (float2) (s2.x + t0.x,
                   s2.y + t0.y);

    //s4 = t0 + c1*s2
    s4 = (float2) (t0.x + c1*s2.x,
                   t0.y + c1*s2.y);

    //s5 = s4-i*c2*s1
    s5 = (float2) (s4.x - c2*s1.y,
                   s4.y + c2*s1.x);

    //s6 = 2*s4-s5
    s6 = (float2) (2*s4.x - s5.x,
                   2*s4.y - s5.y);

    *xa = s3;
    *xb = s5; //pseudo code in karner2001multiply paper says s6
    *xc = s6; //pseudo code in karner2001multiply paper says s5
}
{% endif %}

{% if radix == 4 %}
float fft_4(float2 t0, float2 t1, float2 t2, float2 t3,
                float2 w0, float2 w1,
                float2* x0, float2* x1, float2 *x2, float2* x3) {

   //adapted from pedram2013transforming, however
   //some versions of the pedram2013transforming paper, including the one hosted by IEEE and the one hosted here:
   //https://www.cs.utexas.edu/users/flame/pubs/LAC_fft.pdf
   //contains serious errors in the FMA-optimized radix-4 pseudocode
   //finally corrected based on the pseudocode reported in karner1998top

    float2 a, b, c, d;

    a = t0;     b = t2;     c = t1;     d = t3;

    b = (float2) (a.x - w1.x * b.x + w1.y * b.y,
                  a.y - w1.x * b.y - w1.y * b.x);
    a = (float2) (2*a.x - b.x,
                  2*a.y - b.y);
    d = (float2) (c.x - w1.x * d.x + w1.y * d.y,
                  c.y - w1.x * d.y - w1.y * d.x);
    c = (float2) (2*c.x - d.x,
                  2*c.y - d.y);

    c = (float2) (a.x - w0.x * c.x + w0.y * c.y,
                  a.y - w0.x * c.y - w0.y * c.x);
    *x2 = c;
    *x0 = (float2) (2*a.x - c.x,
                    2*a.y - c.y);

    //d = b - i*w0*d
    d = (float2) (b.x + w0.x * d.y + w0.y * d.x,
                  b.y - w0.x * d.x + w0.y * d.y);
    *x1 = d;
    *x3 = (float2) (2*b.x - d.x,
                    2*b.y - d.y);

    return 0.0;
}
{% endif %}


{% if radix == 5 %}
float fft_5(float2 t0, float2 t1, float2 t2, float2 t3, float2 t4,
                 float2 w0, float2 w1, float2 w2, float2 w3,
                 float2* x0, float2* x1, float2 *x2, float2* x3, float2 *x4) {

    const float c1 = 0.25;                  // 1/4
    const float c2 = 0.5590169943749475;      // sqrt(5)/4
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
{% if radix == 2 %}
__kernel void test_radix_2(__global float2 *x, __global float2 *y, int n) {

    float2 w = (float2) (1.0, 0.0);
    int i = get_global_id(0)*2;

    //n is the number of radix2 ffts to perform
    if (i<2*n) {
        float2 y0, y1;
        fft_2(x[i], x[i+1], w, &y0, &y1);

        y[i] = y0; y[i+1] = y1;
    }
}
{% endif %}

{% if radix == 3 %}
__kernel void test_radix_3(__global float2 *x, __global float2 *y, int n) {

    float2 w0 = (float2) (1.0, 0.0);
    float2 w1 = (float2) (1.0, 0.0);
    int i = get_global_id(0)*3;

    //n is the number of radix3 ffts to perform
    if (i<3*n) {
        float2 y0, y1, y2;

        fft_3(x[i], x[i+1], x[i+2], w0, w1, &y0, &y1, &y2);

        y[i] = y0;    y[i+1] = y1;    y[i+2] = y2;
    }
}
{% endif %}

{% if radix == 4 %}
__kernel void test_radix_4(__global float2 *x, __global float2 *y, int n) {

    float2 w0 = (float2) (1.0, 0.0);
    float2 w1 = (float2) (1.0, 0.0);
    int i = get_global_id(0)*4;

    //n is the number of radix4 ffts to perform
    if (i<4*n) {
        float2 y0, y1, y2, y3;

        fft_4(x[i], x[i+1], x[i+2], x[i+3], w0, w1, &y0, &y1, &y2, &y3);

        y[i] = y0;    y[i+1] = y1;    y[i+2] = y2;    y[i+3] = y3;
    }
}
{% endif %}

{% if radix == 5 %}
__kernel void test_radix_5(__global float2 *x, __global float2 *y, int n) {

    float2 w0 = (float2) (1.0, 0.0);
    float2 w1 = (float2) (1.0, 0.0);
    float2 w2 = (float2) (1.0, 0.0);
    float2 w3 = (float2) (1.0, 0.0);
    int i = get_global_id(0)*5;

    //n is the number of radix5 ffts to perform
    if (i<5*n) {
        float2 y0, y1, y2, y3, y4;

        fft_5(x[i], x[i+1], x[i+2], x[i+3], x[i+4], w0, w1, w2, w3, &y0, &y1, &y2, &y3, &y4);

        y[i] = y0;    y[i+1] = y1;    y[i+2] = y2;    y[i+3] = y3;    y[i+4] = y4;
    }
}
{% endif %}
/* ~\~ end */
/* ~\~ end */
