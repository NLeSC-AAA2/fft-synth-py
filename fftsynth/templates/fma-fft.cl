/* ~\~ language=OpenCL filename=fftsynth/templates/fma-fft.cl */
/* ~\~ begin <<lit/code-generator.md|fftsynth/templates/fma-fft.cl>>[0] */
void fft_{{ N }}_ps({% for i in range(radix) %} float2 * restrict s{{ i }}{%- if not loop.last %},{% endif %}{% endfor %}{% if fpga %},{% for i in range(radix) %} float2 * restrict s{{ i }}_in,{% endfor %}{% for i in range(radix) %} float2 * restrict s{{ i }}_out{%- if not loop.last %},{% endif %}{% endfor %}{% endif %})
{
    int wp = 0;

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
                a = comp_idx_{{ radix }}(DIVR(i), k-1);
            }
            else
            {
                a = comp_perm_{{ radix }}(DIVR(i), MODR(i));
            }
            {% if fpga %}
            fft_{{ radix }}({% for i in range(radix) %} s{{ i }},{% endfor %}{% for i in range(radix) %} s{{ i }}_in,{% endfor %}{% for i in range(radix) %} s{{ i }}_out,{% endfor %} k == 0, k == {{ depth - 1 }}, MODR(i), {% for i in range(radix) %} a + {{ i }} * j,{% endfor %} wp);
            {% else %}
            fft_{{ radix }}({% for i in range(radix) %} s{{ i }},{% endfor %} MODR(i), {% for i in range(radix) %} a + {{ i }} * j,{% endfor %} wp);
            {% endif %}
            if ( k != 0 )
            {
                ++wp;
            }
        }
    }
}
/* ~\~ end */
/* ~\~ begin <<lit/code-generator.md|fftsynth/templates/fma-fft.cl>>[1] */
__kernel {%if fpga %}__attribute__((autorun)) __attribute__((max_global_work_dim(0))){% endif %}
void fft_{{ N }}({% if not fpga %}__global const float2 * restrict x, __global float2 * restrict y{% endif %})
{
    {% if fpga -%}
    while ( true )
    {
    {% endif -%}
    {%- for i in range(radix) %}
    float2 s{{ i }}[{{ M }}];
    {% if fpga -%}
    float2 s{{ i }}_in[{{ M }}], s{{ i }}_out[{{ M }}];
    {%- endif -%}
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
            case {{ p }}: s{{ p }}_in[DIVR(i)] = x; break;
            {%- endfor -%}
            {% else %}
            {%- for p in range(radix) %}
            case {{ p }}: s{{ p }}[DIVR(i)] = x[j]; break;
            {%- endfor -%}
            {%- endif %}
        }
    }

    {% if fpga %}
    fft_{{ N }}_ps({%- for i in range(radix) %} s{{ i }},{% endfor %} {%- for i in range(radix) %} s{{ i }}_in,{% endfor %} {%- for i in range(radix) %} s{{ i }}_out{%- if not loop.last %},{% endif %}{% endfor %});
    {% else %}
    fft_{{ N }}_ps({%- for i in range(radix) %} s{{ i }}{%- if not loop.last %},{% endif %}{% endfor %});
    {% endif %}

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
            case {{ p }}: y = s{{ p }}_out[DIVR(i)]; break;
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
/* ~\~ end */
