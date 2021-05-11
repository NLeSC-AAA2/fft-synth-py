/* ~\~ language=OpenCL filename=fftsynth/templates/parity.cl */
/* ~\~ begin <<lit/code-generator.md|fftsynth/templates/parity.cl>>[0] */
inline int parity_{{ radix }}(int i)
{
    int x = MODR(i);

    {% for item in range(depth) %}
    i = DIVR(i);
    x += MODR(i);
    {%- endfor %}

    return MODR(x);
}
/* ~\~ end */
/* ~\~ begin <<lit/code-generator.md|fftsynth/templates/parity.cl>>[1] */
#ifdef TESTING
__kernel void test_parity_{{ radix }}(__global const int * restrict x, __global int * restrict y)
{
    int i = get_global_id(0);

    y[i] = parity_{{ radix }}(x[i]);
}
#endif // TESTING
/* ~\~ end */
