/* ~\~ language=OpenCL filename=fftsynth/templates/ipow.cl */
/* ~\~ begin <<lit/code-generator.md|fftsynth/templates/ipow.cl>>[0] */
{% if radix == 4 %}
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
    for (i = 1, j = a; i < b; ++i, j*=a);
    return j;
}
{% endif %}
/* ~\~ end */
