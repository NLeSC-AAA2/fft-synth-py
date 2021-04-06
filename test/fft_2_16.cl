/* FFT
 * command: python -m fftsynth.generator --radix 2 --depth 4
 */
/* ~\~ language=OpenCL filename=fftsynth/templates/preprocessor.cl */
/* ~\~ begin <<lit/code-generator.md|fftsynth/templates/preprocessor.cl>>[0] */

#define DIVR(x) ((x) / 2)
#define MODR(x) ((x) % 2)
#define MULR(x) ((x) * 2)

/* ~\~ end */


/* ~\~ language=OpenCL filename=fftsynth/templates/twiddles.cl */
/* ~\~ begin <<lit/code-generator.md|fftsynth/templates/twiddles.cl>>[0] */
__constant float2 W[24][1] = {

{ (float2)(1.000000f, 0.000000f) },
{ (float2)(0.000000f, -1.000000f) },
{ (float2)(0.000000f, -1.000000f) },
{ (float2)(1.000000f, 0.000000f) },
{ (float2)(0.000000f, -1.000000f) },
{ (float2)(1.000000f, 0.000000f) },
{ (float2)(1.000000f, 0.000000f) },
{ (float2)(0.000000f, -1.000000f) },
{ (float2)(1.000000f, 0.000000f) },
{ (float2)(0.707107f, -0.707107f) },
{ (float2)(-0.707107f, -0.707107f) },
{ (float2)(0.000000f, -1.000000f) },
{ (float2)(0.707107f, -0.707107f) },
{ (float2)(1.000000f, 0.000000f) },
{ (float2)(0.000000f, -1.000000f) },
{ (float2)(-0.707107f, -0.707107f) },
{ (float2)(1.000000f, 0.000000f) },
{ (float2)(0.923880f, -0.382683f) },
{ (float2)(0.382683f, -0.923880f) },
{ (float2)(0.707107f, -0.707107f) },
{ (float2)(-0.382683f, -0.923880f) },
{ (float2)(0.000000f, -1.000000f) },
{ (float2)(-0.707107f, -0.707107f) },
{ (float2)(-0.923880f, -0.382683f) }
};
/* ~\~ end */


/* ~\~ language=OpenCL filename=fftsynth/templates/parity.cl */
/* ~\~ begin <<lit/code-generator.md|fftsynth/templates/parity.cl>>[0] */
inline int parity_2(int i)
{
    int x = MODR(i);

    
    i = DIVR(i);
    x += MODR(i);
    i = DIVR(i);
    x += MODR(i);
    i = DIVR(i);
    x += MODR(i);
    i = DIVR(i);
    x += MODR(i);

    return MODR(x);
}
/* ~\~ end */
/* ~\~ begin <<lit/code-generator.md|fftsynth/templates/parity.cl>>[1] */
#ifdef TESTING
__kernel void test_parity_2(__global const int * x, __global int * y)
{
    int i = get_global_id(0);
    
    y[i] = parity_2(x[i]);
}
#endif // TESTING
/* ~\~ end */


/* ~\~ language=OpenCL filename=fftsynth/templates/transpose.cl */
/* ~\~ begin <<lit/code-generator.md|fftsynth/templates/transpose.cl>>[0] */
inline int transpose_2(int j)
{
    int x = 0;

    
    x = MULR(x) + MODR(j);
    j = DIVR(j);
    x = MULR(x) + MODR(j);
    j = DIVR(j);
    x = MULR(x) + MODR(j);
    j = DIVR(j);
    x = MULR(x) + MODR(j);
    j = DIVR(j);

    return x;
}
/* ~\~ end */
/* ~\~ begin <<lit/code-generator.md|fftsynth/templates/transpose.cl>>[1] */
#ifdef TESTING
__kernel void test_transpose_2(__global const int * x, __global int * y)
{
    int i = get_global_id(0);
    
    y[i] = transpose_2(x[i]);
}
#endif // TESTING
/* ~\~ end */


/* ~\~ language=OpenCL filename=fftsynth/templates/ipow.cl */
/* ~\~ begin <<lit/code-generator.md|fftsynth/templates/ipow.cl>>[0] */

inline int ipow(int b)
{
    int i, j;
    int a = 2;
    #pragma unroll 10
    for (i = 1, j = a; i < b; ++i, j*=a);
    return j;
}

/* ~\~ end */


/* ~\~ language=OpenCL filename=fftsynth/templates/indices.cl */
/* ~\~ begin <<lit/code-generator.md|fftsynth/templates/indices.cl>>[0] */
inline int comp_idx_2(int i, int k)
{
    int rem = i % ipow(k);
    int base = i - rem;
    return MULR(base) + rem;
}


inline int comp_perm_2(int i, int rem)
{
    int p = parity_2(i);
    return MULR(i) + MODR(rem + 2 - p);
}
/* ~\~ end */


/* ~\~ language=OpenCL filename=fftsynth/templates/codelets.cl */
/* ~\~ begin <<lit/code-generator.md|fftsynth/templates/codelets.cl>>[0] */
void fft_2(float2 * restrict s0, float2 * restrict s1, int cycle, int i0, int i1, int iw)
{
    float2 t0, t1, ws0, ws1;
    __constant float2 *w = W[iw];

    
    switch (cycle) {
        case 0: t0 = s0[i0]; t1 = s1[i1]; break;
        case 1: t0 = s1[i0]; t1 = s0[i1]; break;
    }
    

    ws0 = t0;
    ws1 = (float2) (w[0].x * t1.x - w[0].y * t1.y,
                    w[0].x * t1.y + w[0].y * t1.x);

    t0 = ws0 + ws1;
    t1 = ws0 - ws1;

    
    switch (cycle) {
        case 0: s0[i0] = t0; s1[i1] = t1; break;
        case 1: s1[i0] = t0; s0[i1] = t1; break;
    }
    
}

void fft_4(float2 * restrict s0, float2 * restrict s1, float2 * restrict s2, float2 * restrict s3, int cycle, int i0, int i1, int i2, int i3, int iw)
{
    float2 t0, t1, t2, t3, ws0, ws1, ws2, ws3, a, b, c, d;
    __constant float2 *w = W[iw];

    
    switch (cycle) {
        case 0: t0 = s0[i0]; t1 = s1[i1]; t2 = s2[i2]; t3 = s3[i3]; break;
        case 1: t0 = s1[i0]; t1 = s2[i1]; t2 = s3[i2]; t3 = s0[i3]; break;
        case 2: t0 = s2[i0]; t1 = s3[i1]; t2 = s0[i2]; t3 = s1[i3]; break;
        case 3: t0 = s3[i0]; t1 = s0[i1]; t2 = s1[i2]; t3 = s2[i3]; break;
    }
    

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

    
    switch (cycle) {
        case 0: s0[i0] = t0; s1[i1] = t1; s2[i2] = t2; s3[i3] = t3; break;
        case 1: s1[i0] = t0; s2[i1] = t1; s3[i2] = t2; s0[i3] = t3; break;
        case 2: s2[i0] = t0; s3[i1] = t1; s0[i2] = t2; s1[i3] = t3; break;
        case 3: s3[i0] = t0; s0[i1] = t1; s1[i2] = t2; s2[i3] = t3; break;
    }
    
}
/* ~\~ end */


/* ~\~ language=OpenCL filename=fftsynth/templates/fft.cl */
/* ~\~ begin <<lit/code-generator.md|fftsynth/templates/fft.cl>>[0] */
void fft_16_ps( float2 * restrict s0, float2 * restrict s1)
{
    int wp = 0;

    for ( unsigned int k = 0; k != 4; ++k )
    {
        int j = (k == 0 ? 0 : ipow(k - 1));

        for ( unsigned int i = 0; i != 8; ++i )
        {
            int a;
            if ( k != 0 )
            {
                a = comp_idx_2(DIVR(i), k-1);
            }
            else
            {
                a = comp_perm_2(DIVR(i), MODR(i));
            }
            
            fft_2( s0, s1, MODR(i),  a + 0 * j, a + 1 * j, wp);
            
            if ( k != 0 )
            {
                ++wp;
            }
        }
    }
}
/* ~\~ end */
/* ~\~ begin <<lit/code-generator.md|fftsynth/templates/fft.cl>>[1] */
__kernel 
void fft_16(__global const float2 * restrict x, __global float2 * restrict y)
{
    
    float2 s0[8];
    
    float2 s1[8];
    

    for ( unsigned int j = 0; j != 16; ++j )
    {
        int i = transpose_2(j);
        int p = parity_2(i);

        switch ( p )
        {
            case 0: s0[DIVR(i)] = x[j]; break;
            case 1: s1[DIVR(i)] = x[j]; break;
        }
    }

    
    fft_16_ps( s0, s1);
    

    for ( unsigned int i = 0; i != 16; ++i )
    {
        int p = parity_2(i);
        switch ( p )
        {
            case 0: y[i] = s0[DIVR(i)]; break;
            case 1: y[i] = s1[DIVR(i)]; break;
        }
    }
}
/* ~\~ end */
