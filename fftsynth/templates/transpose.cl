/* ~\~ language=OpenCL filename=fftsynth/templates/transpose.cl */
/* ~\~ begin <<lit/code-generator.md|fftsynth/templates/transpose.cl>>[0] */
inline int transpose_{{ radix }}(int j)
{
    int x = 0;

    {% for item in range(depth) %}
    x = MULR(x) + MODR(j);
    j = DIVR(j);
    {%- endfor %}

    return x;
}
/* ~\~ end */
/* ~\~ begin <<lit/code-generator.md|fftsynth/templates/transpose.cl>>[1] */
#ifdef TESTING
__kernel void test_transpose_{{ radix }}(__global const int * restrict x, __global int * restrict y)
{
    int i = get_global_id(0);

    y[i] = transpose_{{ radix }}(x[i]);
}
#endif // TESTING
/* ~\~ end */
