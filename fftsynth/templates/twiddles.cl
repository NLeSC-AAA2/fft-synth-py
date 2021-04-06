/* ~\~ language=OpenCL filename=fftsynth/templates/twiddles.cl */
/* ~\~ begin <<lit/code-generator.md|fftsynth/templates/twiddles.cl>>[0] */
__constant float2 W[{{ W.shape[0] - radix }}][{{ radix - 1 }}] = {
{% for ws in W[radix:] %}
{ {% for w in ws[1:] -%}
(float2)({{ "%0.6f" | format(w.real) }}f, {{ "%0.6f" | format(w.imag) }}f) {%- if not loop.last %}, {% endif %}
{%- endfor %} } {%- if not loop.last %},{% endif %}
{%- endfor %}
};
/* ~\~ end */
