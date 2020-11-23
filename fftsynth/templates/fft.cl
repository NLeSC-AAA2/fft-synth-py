void fft_{{ N }}_mc(__restrict float2 * s0, __restrict float2 * s1, __restrict float2 * s2, __restrict float2 * s3)
{
    int wp = 0;

    for ( int k = 0; k < {{ depth }}; ++k )
    {
        int j = (k == 0 ? 0 : ipow({{ radix }}, k - 1));

        for ( int i = 0; i < {{ L }}; ++i )
        {
            if ( k != 0 )
            {
                a = comp_idx_4(i >> 2, k-1);
            }
            else
            {
                a = comp_perm_4(i >> 2, i&3);
            }
            fft_4(s0, s1, s2, s3, i&3, a, a+j, a+2*j, a+3*j, wp);
            if ( k != 0 )
            {
                ++wp;
            }
        }
    }
}