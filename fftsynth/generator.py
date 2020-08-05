# ~\~ language=Python filename=fftsynth/generator.py
# ~\~ begin <<lit/code-generator.md|fftsynth/generator.py>>[0]
# ~\~ begin <<lit/code-generator.md|generate-twiddles>>[0]
def write_twiddles(ps, W):
    print(f"__constant float2 W[{W.shape[0]}][{ps.radix-1}] {{")
    print( "    {" + "},\n    {".join(", ".join(f"(float2) ({w.real: f}f, {w.imag: f}f)"
            for w in ws[1:]) for ws in W) + "}};")
# ~\~ end

# ~\~ begin <<lit/code-generator.md|generate-fft>>[0]
def write_outer_loop_fn(ps):
    print(f"void fft_{ps.N}(__restrict float2 *s0, __restrict float2 *s1, __restrict float2 *s2, __restrict float2 *s3)")
    print( "{")
    print(f"    for (int k = 0; k < {ps.depth}; ++k) {{")
    print(f"        for (int i = 0; i < {ps.L}; ++i) {{")        
    print( "        }")
    print( "    }")
    print( "}")
# ~\~ end
# ~\~ end
