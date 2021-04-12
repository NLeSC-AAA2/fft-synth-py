/* ~\~ language=OpenCL filename=fftsynth/templates/fma-twiddles.cl */
/* ~\~ begin <<lit/fma-codelets.md|fftsynth/templates/fma-twiddles.cl>>[0] */
__constant float2 W[{{ W.shape[0] - radix }}][{{ n_twiddles }}] = {
{% for ws in W[radix:] %}
{ {% for w in ws[1:n_twiddles+1] -%}
(float2)({{ "%0.6f" | format(w.real) }}f, {{ "%0.6f" | format(w.imag) }}f) {%- if not loop.last %}, {% endif %}
{%- endfor %} } {%- if not loop.last %},{% endif %}
{%- endfor %}
};

/* ~\~ end */
