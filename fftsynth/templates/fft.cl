void fft_{{ N }}_ps({% for i in range(radix) %} float2 * restrict s{{ i }}{%- if not loop.last %},{% endif %}{% endfor %})
{
    int wp = 0;

    for ( int k = 0; k < {{ depth }}; ++k )
    {
        int j = (k == 0 ? 0 : ipow(k - 1));

        for ( int i = 0; i < {{ M }}; ++i )
        {
        int a;
            if ( k != 0 )
            {
                a = comp_idx_4(DIVR(i), k-1);
            }
            else
            {
                a = comp_perm_4(DIVR(i), MODR(i));
            }
            fft_{{ radix }}({% for i in range(radix) %} s{{ i }},{% endfor %} MODR(i), {% for i in range(radix) %} a + {{ i }} * j,{% endfor %} wp);
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
    float2 s{{ i }}[{{ M }}];
    {% endfor }

    for ( int j = 0; j < {{ N }}; ++j )
    {
        int i = transpose_{{ radix }}(j);
        int p = parity_{{ radix }}(i);
        
        switch ( p )
        {
            {% for p in range(radix) %}
            case {{ p }}: s{{ p }}[DIVR(i)] = x[j]; break;
            {% endfor %}
        }
    }

    fft_{{ N }}_ps({% for i in range(radix) %} s{{ i }}{%- if not loop.last %},{% endif %}{% endfor %});

    for ( int i = 0; i < {{ N }}; ++i )
    {
        int p = parity_{{ radix }}(i);
        
        switch ( p )
        {
            {% for p in range(radix) %}
            case {{ p }}: y[i] = s{{ p }}[DIVR(i)]; break;
            {% endfor %}
        }
    }
}
