# ~\~ language=Python filename=fftsynth/generator.py
# ~\~ begin <<lit/code-generator.md|fftsynth/generator.py>>[0]
from .indent import indent

# ~\~ begin <<lit/code-generator.md|generate-twiddles>>[0]
def write_twiddles(ps, W):
    print(f"__constant float2 W[{W.shape[0]}][{ps.radix-1}] {{")
    print( "    {" + "},\n    {".join(", ".join(
                f"(float2) ({w.real: f}f, {w.imag: f}f)"
                for w in ws[1:])
            for ws in W) + "}};")
# ~\~ end

# ~\~ begin <<lit/code-generator.md|generate-transpose>>[0]
def gen_transpose_fn(ps):
    return f"""
inline int transpose_{ps.radix}(int j) {{
    int x = 0;
    for (int l = 0; l < {ps.depth}; ++l) {{
        x *= {ps.radix};
        x += j % {ps.radix};
        j /= {ps.radix};
    }}
    return x;
}}"""
# ~\~ end

# ~\~ begin <<lit/code-generator.md|generate-parity>>[0]
def gen_parity_fn(ps):
    return f"""
inline int parity_{ps.radix}(int i) {{
    int x = i % {ps.radix};
    for (int a = 0; a < {ps.depth}; ++a) {{
        i /= {ps.radix};
        x += i % {ps.radix};
    }}
    return x % {ps.radix};
}}"""
# ~\~ end

# ~\~ begin <<lit/code-generator.md|generate-fft>>[0]
def write_outer_loop_fn(ps):
    print(f"void fft_{ps.N}_mc(__restrict float2 *s0, __restrict float2 *s1, __restrict float2 *s2, __restrict float2 *s3)")
    print( "{")
    print( "    int wp = 0;")
    print(f"    for (int k = 0; k < {ps.depth}; ++k) {{")
    with indent("         "):
        # ~\~ begin <<lit/code-generator.md|fft-inner-loop>>[0]
        print(f"int j = (k == 0 ? 0 : ipow({ps.radix}, k - 1));")
        print(f"for (int i = 0; i < {ps.L}; ++i) {{")
        # ~\~ end
        # ~\~ begin <<lit/code-generator.md|fft-inner-loop>>[1]
        print( "if (k != 0) {")
        print( "    a = comp_idx_4(i >> 2, k-1);")
        print( "} else {")
        print( "    a = comp_perm_4(i >> 2, i&3);")
        print( "}")
        print( "fft_4(s0, s1, s2, s3, i&3, a, a+j, a+2*j, a+3*j, wp);")
        print( "if (k != 0) ++wp;")
        # ~\~ end
    print( "    }")
    print( "}")
# ~\~ end
# ~\~ end
