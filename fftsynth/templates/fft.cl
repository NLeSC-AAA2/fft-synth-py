#include "transpose.cl"
#include "parity.cl"

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

__kernel void fft_{{ N }}(__global const float2 * restrict x, __global float2 * restrict y)
{
    {% for i in range(radix) %}
    float2 s{{ i }}[{{ L }}];
    {% endfor }
    float2 s{{ radix }}[{{ L }}];

    for ( int j = 0; j < {{ N }}; ++j )
    {
        int i = transpose_{{ radix }}(j);
        int p = parity_{{ radix }}(i);
        
        switch ( p )
        {
            {% for p in range(radix) %}
            case {{ p }}: s{{ p }}[i/{{ radix }}] = x[j]; break;
            {% endfor %}
        }
    }

    fft_1024_mc({% for i in range(radix) %} s{{ i }} {% endfor %});

    for ( int i = 0; i < {{ N }}; ++i )
    {
        int p = parity_{{ radix }}(i);
        
        switch ( p )
        {
            {% for p in range(radix) %}
            case {{ p }}: y[i] = s{{ p }}[i/{{ radix }}]; break;
            {% endfor %}
        }
    }
}
