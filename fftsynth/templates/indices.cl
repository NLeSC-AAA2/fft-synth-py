/* ~\~ language=OpenCL filename=fftsynth/templates/indices.cl */
/* ~\~ begin <<lit/code-generator.md|fftsynth/templates/indices.cl>>[0] */
inline int comp_idx_{{ radix }}(int i, int k)
{
    int rem = i % ipow(k);
    int base = i - rem;
    return MULR(base) + rem;
}


inline int comp_perm_{{ radix }}(int i, int rem)
{
    int p = parity_{{ radix }}(i);
    return MULR(i) + MODR(rem + {{ radix }} - p);
}
/* ~\~ end */
