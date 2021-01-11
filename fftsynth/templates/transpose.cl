inline int transpose_{{ radix }}(int j)
{
    int x = 0;
    
    for ( int l = 0; l < {{ depth }}; ++l )
    {
        x = MULR(x) + MODR(j);
        j = DIVR(j);
    }
    return x;
}

#ifdef TESTING
__kernel void test_transpose_{{ radix }}(__global const int * x, __global int * y)
{
    int i = get_global_id(0);
    
    y[i] = transpose_{{ radix }}(x[i]);
}
#endif // TESTING
