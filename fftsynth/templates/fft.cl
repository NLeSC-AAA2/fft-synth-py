void fft_{{ N }}_ps({% for i in range(radix) %} float2 * restrict s{{ i }}{%- if not loop.last %},{% endif %}{% endfor %})
{
    int wp = 0;

    #pragma unroll
    for ( {{ depth_type }} k = 0; k != {{ depth }}; ++k )
    {
        int j = (k == 0 ? 0 : ipow(k - 1));

        {% if fpga -%}
        #pragma ivdep
        {% endif -%}
        for ( {{ m_type }} i = 0; i != {{ M }}; ++i )
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

__kernel {%if fpga %}__attribute__((autorun)) __attribute__((max_global_work_dim(0))){% endif %}
void fft_{{ N }}({% if not fpga %}__global const float2 * restrict x, __global float2 * restrict y{% endif %})
{
    {% if fpga -%}
    while ( true )
    {
    {% endif -%}
    {%- for i in range(radix) %}
    float2 s{{ i }}[{{ M }}];
    {%- endfor %}

    for ( {{ n_type }} j = 0; j != {{ N }}; ++j )
    {
        int i = transpose_{{ radix }}(j);
        int p = parity_{{ radix }}(i);

        {% if fpga -%}
        float2 x = read_channel_intel(in_channel);
        {% endif -%}
        switch ( p )
        {
            {%- if fpga %}
            {%- for p in range(radix) %}
            case {{ p }}: s{{ p }}[DIVR(i)] = x; break;
            {%- endfor -%}
            {% else %}
            {%- for p in range(radix) %}
            case {{ p }}: s{{ p }}[DIVR(i)] = x[j]; break;
            {%- endfor -%}
            {%- endif %}
        }
    }

    fft_{{ N }}_ps({% for i in range(radix) %} s{{ i }}{%- if not loop.last %},{% endif %}{% endfor %});

    for ( {{ n_type }} i = 0; i != {{ N }}; ++i )
    {
        int p = parity_{{ radix }}(i);
        {% if fpga -%}
        float2 y;
        {% endif -%}
        
        switch ( p )
        {
            {%- if fpga -%}
            {%- for p in range(radix) %}
            case {{ p }}: y = s{{ p }}[DIVR(i)]; break;
            {%- endfor -%}
            {% else %}
            {%- for p in range(radix) %}
            case {{ p }}: y[i] = s{{ p }}[DIVR(i)]; break;
            {%- endfor -%}
            {%- endif %}
        }
        {%- if fpga %}
        write_channel_intel(out_channel, y);
        {%- endif %}
    }
    {%- if fpga %}
    }
    {%- endif %}
}
