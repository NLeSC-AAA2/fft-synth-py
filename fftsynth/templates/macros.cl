{% if radix == 4 %}
#define DIVR(x) ((x) >> 2)
#define MODR(x) ((x) & 3)
#define MULR(x) ((x) << 2)
{% else %}
#define DIVR(x) ((x) / {ps.radix})
#define MODR(x) ((x) % {ps.radix})
#define MULR(x) ((x) * {ps.radix})
{% endif %}
