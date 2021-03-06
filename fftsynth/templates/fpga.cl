/* ~\~ language=OpenCL filename=fftsynth/templates/fpga.cl */
/* ~\~ begin <<lit/code-generator.md|fftsynth/templates/fpga.cl>>[0] */
#ifdef TESTING
__kernel __attribute__((max_global_work_dim(0)))
void source(__global const volatile {{c_type}} * in, unsigned count)
{
    #pragma ii 1
    for ( unsigned i = 0; i < count; i++ )
    {
        write_channel_intel(in_channel, in[i]);
    }
}

__kernel __attribute__((max_global_work_dim(0)))
void sink(__global {{c_type}} *out, unsigned count)
{
    #pragma ii 1
    for ( unsigned i = 0; i < count; i++ )
    {
        out[i] = read_channel_intel(out_channel);
    }
}
#endif // TESTING
/* ~\~ end */
