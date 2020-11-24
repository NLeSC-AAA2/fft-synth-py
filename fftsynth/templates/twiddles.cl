__constant float2 W[{{ W.shape[0] }}][{{ radix - 1 }}] = {
{% for ws in W %}
{ {% for w in ws[1:] -%}
(float2)({{ w.real }}f, {{ w.imag }}f) {%- if not loop.last %}, {% endif %}
{%- endfor %} } {%- if not loop.last %},{% endif %}
{%- endfor %}
};