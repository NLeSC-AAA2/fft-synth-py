#ifdef OPENCL_FPGA
#pragma OPENCL EXTENSION cl_intel_channels : enable
#include <ihc_apint.h>
channel float2 in_channel, out_channel;
#endif // OPENCL_FPGA
{% if radix == 4 %}
#define DIVR(x) ((x) >> 2)
#define MODR(x) ((x) & 3)
#define MULR(x) ((x) << 2)
{% else %}
#define DIVR(x) ((x) / {{ radix }})
#define MODR(x) ((x) % {{ radix }})
#define MULR(x) ((x) * {{ radix }})
{% endif %}
