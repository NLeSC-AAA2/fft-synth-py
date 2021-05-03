/* ~\~ language=OpenCL filename=fftsynth/templates/preprocessor.cl */
/* ~\~ begin <<lit/code-generator.md|fftsynth/templates/preprocessor.cl>>[0] */
{% if fpga -%}
#pragma OPENCL EXTENSION cl_intel_channels : enable
#include <ihc_apint.h>
channel {{c_type}} in_channel, out_channel;
#define SWAP(type, x, y) do { type temp = x; x = y, y = temp; } while ( false );
{% endif -%}
{%if radix == 2 %}
#define DIVR(x) ((x) >> 1)
#define MODR(x) ((x) & 1)
#define MULR(x) ((x) << 1)
{% elif radix == 4 %}
#define DIVR(x) ((x) >> 2)
#define MODR(x) ((x) & 3)
#define MULR(x) ((x) << 2)
{% else %}
#define DIVR(x) ((x) / {{ radix }})
#define MODR(x) ((x) % {{ radix }})
#define MULR(x) ((x) * {{ radix }})
{% endif %}
/* ~\~ end */
