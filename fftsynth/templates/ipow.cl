/* ~\~ language=OpenCL filename=fftsynth/templates/ipow.cl */
/* ~\~ begin <<lit/code-generator.md|fftsynth/templates/ipow.cl>>[0] */
{% if radix == 2 %}
inline int ipow(int b)
{
    return 1 << b;
}
{% elif radix == 4 %}
inline int ipow(int b)
{
    return 1 << (2*b);
}
{% else %}
inline int ipow(int b)
{
    int i, j;
    int a = {{ radix }};
    #pragma unroll 10
    for (i = 0, j = 1; i < b; ++i, j*=a);
    return j;
}
{% endif %}
/* ~\~ end */
