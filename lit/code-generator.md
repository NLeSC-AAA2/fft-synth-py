# Code generator

``` {.python file=fftsynth/generator.py}
<<generate-twiddles>>


```

## Twiddles

``` {.python #generate-twiddles}
def write_twiddles(ps, W):
    print(f"__constant float2 W[{W.shape[0]}][{ps.radix-1}] {{")
    print( "    {" + "},\n    {".join(", ".join(f"(float2) ({w.real: f}f, {w.imag: f}f)"
            for w in ws[1:]) for ws in W) + "}};")
```

