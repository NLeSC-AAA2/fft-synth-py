inline int parity_{{ radix }}(int i)
{
    int x = MODR(i);

    {% for item in range(depth) %}
    i = DIVR(i);
    x += MODR(i);
    {%- endfor %}

    return MODR(x);
}

#ifdef TESTING
__kernel void test_parity_{{ radix }}(__global const int * x, __global int * y)
{
    int i = get_global_id(0);
    
    y[i] = parity_{{ radix }}(x[i]);
}
#endif // TESTING
