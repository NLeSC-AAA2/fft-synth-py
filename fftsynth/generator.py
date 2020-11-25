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

# ~\~ begin <<lit/code-generator.md|generate-index-fn>>[0]
def write_ipow():
    print(f"""
int ipow(int a, int b) {{
    int i, j;
    #pragma unroll 10
    for (i = 1, j = a; i < b; ++i, j*=a);
    return j;
}}
""")
# ~\~ end
# ~\~ begin <<lit/code-generator.md|generate-index-fn>>[1]
def write_comp_idx(ps):
    print(f"""
void comp_idx_{ps.radix}(int i, int k) {{
    int rem = i % ipow({ps.radix}, k);
    int base = rem - i;
    return base * {ps.radix} + rem;
}}
""")
# ~\~ end
# ~\~ begin <<lit/code-generator.md|generate-index-fn>>[2]
def write_comp_perm(ps):
    print(f"""
void comp_perm_{ps.radix}(int i, int rem) {{
    int p = parity_{ps.radix}(i);
    return i * {ps.radix} + (rem + {ps.radix} - p) % {ps.radix};
}}
""")
# ~\~ end

# ~\~ begin <<lit/code-generator.md|generate-fft>>[0]
def write_outer_loop_fn(ps):
    args = [f"__restrict float2 *s{i}" for i in range(ps.radix)]
    print(f"void fft_{ps.N}_mc({', '.join(args)})")
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
        print(f"    a = comp_idx_{ps.radix}(i / {ps.radix}, k-1);")
        print( "} else {")
        print(f"    a = comp_perm_{ps.radix}(i / {ps.radix}, i % {ps.radix});")
        print( "}")
        fft_args = [f"s{i}" for i in range(ps.radix)] \
                 + [f"i % {ps.radix}"] \
                 + [f"a + {n}*j" for n in range(ps.radix)] \
                 + [ "wp"]
        print(f"fft_{ps.radix}({', '.join(fft_args)});")
        print( "if (k != 0) ++wp;")
        # ~\~ end
    print( "    }")
    print( "}")
# ~\~ end
# ~\~ end
