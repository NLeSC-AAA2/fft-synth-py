void fft_{{ N }}_ps({% for i in range(radix) %} float2 * restrict s{{ i }}{%- if not loop.last %},{% endif %}{% endfor %})
{
    int wp = 0;

    for ( int k = 0; k < {{ depth }}; ++k )
    {
        int j = (k == 0 ? 0 : ipow(k - 1));

        #ifdef OPENCL_FPGA
        #pragma ivdep
        #endif // OPENCL_FPGA
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

__kernel
#ifdef OPENCL_FPGA
__attribute__((autorun))
__attribute__((max_global_work_dim(0)))
void fft_{{ N }}()
#else
void fft_{{ N }}(__global const float2 * restrict x, __global float2 * restrict y)
#endif // OPENCL_FPGA
{
    {% for i in range(radix) %}
    float2 s{{ i }}[{{ M }}];
    {%- endfor %}

    #ifdef OPENCL_FPGA
    while ( true )
    {
    #endif // OPENCL_FPGA
    for ( int j = 0; j < {{ N }}; ++j )
    {
        int i = transpose_{{ radix }}(j);
        int p = parity_{{ radix }}(i);

        #ifdef OPENCL_FPGA
        float2 x = read_channel_intel(in_channel);
        #endif // OPENCL_FPGA
        switch ( p )
        {
            {% for p in range(radix) %}
            case {{ p }}: s{{ p }}[DIVR(i)] = x[j]; break;
            {%- endfor %}
        }
    }

    fft_{{ N }}_ps({% for i in range(radix) %} s{{ i }}{%- if not loop.last %},{% endif %}{% endfor %});

    for ( int i = 0; i < {{ N }}; ++i )
    {
        int p = parity_{{ radix }}(i);
        
        switch ( p )
        {
            #ifdef OPENCL_FPGA
            {% for p in range(radix) %}
            case {{ p }}: y = s{{ p }}[DIVR(i)]; break;
            {%- endfor %}
            #else
            {% for p in range(radix) %}
            case {{ p }}: y[i] = s{{ p }}[DIVR(i)]; break;
            {%- endfor %}
            #endif // OPENCL_FPGA
        }
        #ifdef OPENCL_FPGA
        write_channel_intel(out_channel, y);
        #endif // OPENCL_FPGA
    }
    #ifdef OPENCL_FPGA
    }
    #endif // OPENCL_FPGA
}
