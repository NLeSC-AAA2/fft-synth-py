# ~\~ language=Python filename=fftsynth/generator.py
# ~\~ begin <<lit/code-generator.md|fftsynth/generator.py>>[0]
from .indent import indent
from .parity import (ParitySplitting, comp_idx, comp_perm)
from .twiddle import (make_twiddle)
import numpy as np

# ~\~ begin <<lit/code-generator.md|generate-twiddles>>[0]
def write_twiddles(ps: ParitySplitting):
    W = np.ones(shape=[ps.radix, ps.radix])
    perm = np.array([comp_perm(ps.radix, i) for i in range(ps.M)])

    n = ps.radix
    for k in range(ps.depth - 1):
        w = make_twiddle(ps.radix, n).conj()
        w_r_x = (np.ones(shape=[ps.M//n,ps.radix,n]) * w) \
                    .transpose([0,2,1]) \
                    .reshape([-1,ps.radix])[perm]
        W = np.r_[W, w_r_x]
        n *= ps.radix

    print(f"__constant float2 W[{W.shape[0]-ps.radix}][{ps.radix-1}] = {{")
    print( "    {" + "},\n    {".join(", ".join(
                f"(float2) ({w.real: f}f, {w.imag: f}f)"
                for w in ws[1:])
            for ws in W[ps.radix:]) + "}};")
# ~\~ end

# ~\~ begin <<lit/code-generator.md|generate-transpose>>[0]
def gen_transpose_fn(ps: ParitySplitting):
    return f"""
inline int transpose_{ps.radix}(int j) {{
    int x = 0;
    for (int l = 0; l < {ps.depth}; ++l) {{
        x = MULR(x) + MODR(j);
        j = DIVR(j);
    }}
    return x;
}}"""
# ~\~ end

# ~\~ begin <<lit/code-generator.md|generate-parity>>[0]
def gen_parity_fn(ps: ParitySplitting):
    return f"""
inline int parity_{ps.radix}(int i) {{
    int x = MODR(i);
    for (int a = 0; a < {ps.depth}; ++a) {{
        i = DIVR(i);
        x += MODR(i);
    }}
    return MODR(x);
}}"""
# ~\~ end

# ~\~ begin <<lit/code-generator.md|generate-index-fn>>[0]
def write_ipow(ps):
    if ps.radix == 4:
        print("inline int ipow(int b) { return 1 << (2*b); }")
    else:
        print(f"""
inline int ipow(int b) {{
    int i, j;
    int a = {ps.radix};
    #pragma unroll 10
    for (i = 1, j = a; i < b; ++i, j*=a);
    return j;
}}
""")
# ~\~ end
# ~\~ begin <<lit/code-generator.md|generate-index-fn>>[1]
def write_comp_idx(ps: ParitySplitting):
    print(f"""
inline int comp_idx_{ps.radix}(int i, int k) {{
    int rem = i % ipow(k);
    int base = i - rem;
    return MULR(base) + rem;
}}
""")
# ~\~ end
# ~\~ begin <<lit/code-generator.md|generate-index-fn>>[2]
def write_comp_perm(ps: ParitySplitting):
    print(f"""
inline int comp_perm_{ps.radix}(int i, int rem) {{
    int p = parity_{ps.radix}(i);
    return MULR(i) + MODR(rem + {ps.radix} - p);
}}
""")
# ~\~ end

# ~\~ begin <<lit/code-generator.md|generate-fft>>[0]
def write_fft_fn(ps: ParitySplitting):
    # ~\~ begin <<lit/code-generator.md|generate-outer-fft>>[0]
    declare_arrays = "\n".join(
        f"float2 s{i}[{ps.M}];" for i in range(ps.radix))
    mc_args = ", ".join(
        f"s{i}" for i in range(ps.radix))
    read_cases = "\n".join(
        f"case {p}: s{p}[DIVR(i)] = x[j]; break;" for p in range(ps.radix))
    write_cases = "\n".join(
        f"case {p}: y[i] = s{p}[DIVR(i)]; break;" for p in range(ps.radix))
    print(f"""
    __kernel void fft_{ps.N}(__global const float2 * restrict x, __global float2 * restrict y)
    {{
        {declare_arrays}

        for (int j = 0; j < {ps.N}; ++j) {{
            int i = transpose_{ps.radix}(j);
            int p = parity_{ps.radix}(i);
            switch (p) {{
                {read_cases}
            }}
        }}

        fft_{ps.N}_ps({mc_args});

        for (int i = 0; i < {ps.N}; ++i) {{
            int p = parity_{ps.radix}(i);
            switch (p) {{
                {write_cases}
            }}
        }}
    }}""")
    # ~\~ end

def write_outer_loop_fn(ps: ParitySplitting):
    args = [f"float2 * restrict s{i}" for i in range(ps.radix)]
    print(f"void fft_{ps.N}_ps({', '.join(args)})")
    print( "{")
    print( "    int wp = 0;")
    print(f"    for (int k = 0; k < {ps.depth}; ++k) {{")
    with indent("         "):
        # ~\~ begin <<lit/code-generator.md|fft-inner-loop>>[0]
        print(f"int j = (k == 0 ? 0 : ipow(k - 1));")
        print(f"for (int i = 0; i < {ps.M}; ++i) {{")
        # ~\~ end
        # ~\~ begin <<lit/code-generator.md|fft-inner-loop>>[1]
        print( "int a;")
        print( "if (k != 0) {")
        print(f"    a = comp_idx_{ps.radix}(DIVR(i), k-1);")
        print( "} else {")
        print(f"    a = comp_perm_{ps.radix}(DIVR(i), MODR(i));")
        print( "}")
        fft_args = [f"s{i}" for i in range(ps.radix)] \
                 + [ "MODR(i)"] \
                 + [f"a + {n}*j" for n in range(ps.radix)] \
                 + [ "wp"]
        print(f"fft_{ps.radix}({', '.join(fft_args)});")
        print( "if (k != 0) ++wp;")
        print( "}")
        # ~\~ end
    print( "    }")
    print( "}")
# ~\~ end

def write_macros(ps):
    if ps.radix == 4:
        print("#define DIVR(x) ((x) >> 2)")
        print("#define MODR(x) ((x) & 3)")
        print("#define MULR(x) ((x) << 2)")
    else:
        print(f"#define DIVR(x) ((x) / {ps.radix})")
        print(f"#define MODR(x) ((x) % {ps.radix})")
        print(f"#define MULR(x) ((x) * {ps.radix})")

def write_codelet():
    print("""
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
}""")

def write_fft(ps: ParitySplitting):
    write_macros(ps)
    write_twiddles(ps)
    write_codelet()
    print(gen_parity_fn(ps))
    print(gen_transpose_fn(ps))
    write_ipow(ps)
    write_comp_perm(ps)
    write_comp_idx(ps)
    write_outer_loop_fn(ps)
    write_fft_fn(ps)

if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser(description="Generate an OpenCL FFT kernel")
    parser.add_argument("--radix", type=int, default=4, help="FFT radix")
    parser.add_argument("--depth", type=int, default=3, help="FFT depth")
    args = parser.parse_args()
    print( "/* FFT")
    print(f" * command: python -m fftsynth.generate {' '.join(sys.argv[1:])}")
    print( " */")
    N = args.radix**args.depth
    ps = ParitySplitting(N, args.radix)
    write_fft(ps)
# ~\~ end
