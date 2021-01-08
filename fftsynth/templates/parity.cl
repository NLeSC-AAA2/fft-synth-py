inline int parity_{{ radix }}(int i)
{
    int x = MODR(i);

    for ( int a = 0; a < {{ depth }}; ++a )
    {
        i = DIVR(i);
        x += MODR(i);
    }
    return MODR(x);
}

#ifdef TESTING
__kernel void test_parity(__global const int * x, __global int * y)
{
    int i = get_global_id(0);
    
    y[i] = parity_{{ radix }}(x[i]);
}
#endif // TESTING
