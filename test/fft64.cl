/* FFT
 * command: python -m fftsynth.generate 
 */
__constant float2 W[36][3] = {
    {(float2) ( 1.000000f,  0.000000f), (float2) ( 1.000000f,  0.000000f), (float2) ( 1.000000f,  0.000000f)},
    {(float2) ( 1.000000f,  0.000000f), (float2) ( 1.000000f,  0.000000f), (float2) ( 1.000000f,  0.000000f)},
    {(float2) ( 1.000000f,  0.000000f), (float2) ( 1.000000f,  0.000000f), (float2) ( 1.000000f,  0.000000f)},
    {(float2) ( 1.000000f,  0.000000f), (float2) ( 1.000000f,  0.000000f), (float2) ( 1.000000f,  0.000000f)},
    {(float2) ( 1.000000f,  0.000000f), (float2) ( 1.000000f,  0.000000f), (float2) ( 1.000000f,  0.000000f)},
    {(float2) ( 0.923880f, -0.382683f), (float2) ( 0.707107f, -0.707107f), (float2) ( 0.382683f, -0.923880f)},
    {(float2) ( 0.707107f, -0.707107f), (float2) ( 0.000000f, -1.000000f), (float2) (-0.707107f, -0.707107f)},
    {(float2) ( 0.382683f, -0.923880f), (float2) (-0.707107f, -0.707107f), (float2) (-0.923880f,  0.382683f)},
    {(float2) ( 0.382683f, -0.923880f), (float2) (-0.707107f, -0.707107f), (float2) (-0.923880f,  0.382683f)},
    {(float2) ( 1.000000f,  0.000000f), (float2) ( 1.000000f,  0.000000f), (float2) ( 1.000000f,  0.000000f)},
    {(float2) ( 0.923880f, -0.382683f), (float2) ( 0.707107f, -0.707107f), (float2) ( 0.382683f, -0.923880f)},
    {(float2) ( 0.707107f, -0.707107f), (float2) ( 0.000000f, -1.000000f), (float2) (-0.707107f, -0.707107f)},
    {(float2) ( 0.707107f, -0.707107f), (float2) ( 0.000000f, -1.000000f), (float2) (-0.707107f, -0.707107f)},
    {(float2) ( 0.382683f, -0.923880f), (float2) (-0.707107f, -0.707107f), (float2) (-0.923880f,  0.382683f)},
    {(float2) ( 1.000000f,  0.000000f), (float2) ( 1.000000f,  0.000000f), (float2) ( 1.000000f,  0.000000f)},
    {(float2) ( 0.923880f, -0.382683f), (float2) ( 0.707107f, -0.707107f), (float2) ( 0.382683f, -0.923880f)},
    {(float2) ( 0.923880f, -0.382683f), (float2) ( 0.707107f, -0.707107f), (float2) ( 0.382683f, -0.923880f)},
    {(float2) ( 0.707107f, -0.707107f), (float2) ( 0.000000f, -1.000000f), (float2) (-0.707107f, -0.707107f)},
    {(float2) ( 0.382683f, -0.923880f), (float2) (-0.707107f, -0.707107f), (float2) (-0.923880f,  0.382683f)},
    {(float2) ( 1.000000f,  0.000000f), (float2) ( 1.000000f,  0.000000f), (float2) ( 1.000000f,  0.000000f)},
    {(float2) ( 1.000000f,  0.000000f), (float2) ( 1.000000f,  0.000000f), (float2) ( 1.000000f,  0.000000f)},
    {(float2) ( 0.995185f, -0.098017f), (float2) ( 0.980785f, -0.195090f), (float2) ( 0.956940f, -0.290285f)},
    {(float2) ( 0.980785f, -0.195090f), (float2) ( 0.923880f, -0.382683f), (float2) ( 0.831470f, -0.555570f)},
    {(float2) ( 0.956940f, -0.290285f), (float2) ( 0.831470f, -0.555570f), (float2) ( 0.634393f, -0.773010f)},
    {(float2) ( 0.773010f, -0.634393f), (float2) ( 0.195090f, -0.980785f), (float2) (-0.471397f, -0.881921f)},
    {(float2) ( 0.923880f, -0.382683f), (float2) ( 0.707107f, -0.707107f), (float2) ( 0.382683f, -0.923880f)},
    {(float2) ( 0.881921f, -0.471397f), (float2) ( 0.555570f, -0.831470f), (float2) ( 0.098017f, -0.995185f)},
    {(float2) ( 0.831470f, -0.555570f), (float2) ( 0.382683f, -0.923880f), (float2) (-0.195090f, -0.980785f)},
    {(float2) ( 0.555570f, -0.831470f), (float2) (-0.382683f, -0.923880f), (float2) (-0.980785f, -0.195090f)},
    {(float2) ( 0.471397f, -0.881921f), (float2) (-0.555570f, -0.831470f), (float2) (-0.995185f,  0.098017f)},
    {(float2) ( 0.707107f, -0.707107f), (float2) ( 0.000000f, -1.000000f), (float2) (-0.707107f, -0.707107f)},
    {(float2) ( 0.634393f, -0.773010f), (float2) (-0.195090f, -0.980785f), (float2) (-0.881921f, -0.471397f)},
    {(float2) ( 0.290285f, -0.956940f), (float2) (-0.831470f, -0.555570f), (float2) (-0.773010f,  0.634393f)},
    {(float2) ( 0.195090f, -0.980785f), (float2) (-0.923880f, -0.382683f), (float2) (-0.555570f,  0.831470f)},
    {(float2) ( 0.098017f, -0.995185f), (float2) (-0.980785f, -0.195090f), (float2) (-0.290285f,  0.956940f)},
    {(float2) ( 0.382683f, -0.923880f), (float2) (-0.707107f, -0.707107f), (float2) (-0.923880f,  0.382683f)}};

void fft_4(
    float2 * restrict s0, float2 * restrict s1,
    float2 * restrict s2, float2 * restrict s3,
    int cycle, int i0, int i1, int i2, int i3, int iw)
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

inline int parity_4(int i) {
    int x = i % 4;
    for (int a = 0; a < 3; ++a) {
        i /= 4;
        x += i % 4;
    }
    return x % 4;
}

inline int transpose_4(int j) {
    int x = 0;
    for (int l = 0; l < 3; ++l) {
        x *= 4;
        x += j % 4;
        j /= 4;
    }
    return x;
}

int ipow(int a, int b) {
    int i, j;
    #pragma unroll 10
    for (i = 1, j = a; i < b; ++i, j*=a);
    return j;
}


int comp_perm_4(int i, int rem) {
    int p = parity_4(i);
    return i * 4 + (rem + 4 - p) % 4;
}


int comp_idx_4(int i, int k) {
    int rem = i % ipow(4, k);
    int base = rem - i;
    return base * 4 + rem;
}

void fft_64_ps(float2 * restrict s0, float2 * restrict s1, float2 * restrict s2, float2 * restrict s3)
{
    int wp = 0;
    for (int k = 0; k < 3; ++k) {
         int j = (k == 0 ? 0 : ipow(4, k - 1));
         for (int i = 0; i < 16; ++i) {
         int a;
         if (k != 0) {
             a = comp_idx_4(i / 4, k-1);
         } else {
             a = comp_perm_4(i / 4, i % 4);
         }
         fft_4(s0, s1, s2, s3, i % 4, a + 0*j, a + 1*j, a + 2*j, a + 3*j, wp);
         if (k != 0) ++wp;
         }
    }
}

    __kernel void fft_64(__global const float2 * restrict x, __global float2 * restrict y)
    {
        float2 s0[4];
float2 s1[4];
float2 s2[4];
float2 s3[4];

        for (int j = 0; j < 64; ++j) {
            int i = transpose_4(j);
            int p = parity_4(i);
            switch (p) {
                case 0: s0[i/4] = x[j]; break;
case 1: s1[i/4] = x[j]; break;
case 2: s2[i/4] = x[j]; break;
case 3: s3[i/4] = x[j]; break;
            }
        }

        fft_64_ps(s0, s1, s2, s3);

        for (int i = 0; i < 64; ++i) {
            int p = parity_4(i);
            switch (p) {
                case 0: y[i] = s0[i/4]; break;
case 1: y[i] = s1[i/4]; break;
case 2: y[i] = s2[i/4]; break;
case 3: y[i] = s3[i/4]; break;
            }
        }
    }
