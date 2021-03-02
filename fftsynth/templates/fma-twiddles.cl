__constant float2 W[{{ W.shape[0] - radix }}][{{ radix // 2 }}] = {
{% for ws in W[radix:] %}
{ {% for w in ws[radix // 2:] -%}
(float2)({{ "%0.6f" | format(w.real) }}f, {{ "%0.6f" | format(w.imag) }}f) {%- if not loop.last %}, {% endif %}
{%- endfor %} } {%- if not loop.last %},{% endif %}
{%- endfor %}
};
