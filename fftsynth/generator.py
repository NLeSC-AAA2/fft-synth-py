## ------ language="Python" file="fftsynth/generator.py" project://lit/code-generator.md#4
def write_twiddles(w):
    print(f"__constant float2 W[{W.shape[0]}][{mc.radix-1}] {{")
print( "    {" + "},\n    {".join(", ".join(f"(float2) ({w.real: f}f, {w.imag: f}f)" for w in ws[1:]) for ws in W) + "}};")

## ------ end
