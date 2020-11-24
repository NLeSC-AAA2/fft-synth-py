from jinja2 import Environment, FileSystemLoader

template_loader = FileSystemLoader("fftsynth/templates")
template_environment = Environment(loader=template_loader)


def generate_twiddle_array(parity_splitting, W):
    """
    Generate OpenCL constant array for twiddle factors
    """
    template = template_environment.get_template("twiddles.cl")

    return template.render(radix=parity_splitting.radix, W=W)


def generate_transpose_function(parity_splitting):
    """
    Generate inline OpenCL function to reverse the digits in base-n representation.
    """
    template = template_environment.get_template("transpose.cl")

    return template.render(radix=parity_splitting.radix, depth=parity_splitting.depth)


def generate_parity_function(parity_splitting):
    """
    Generate inline OpenCL function to compute the parity of the index.
    """
    template = template_environment.get_template("parity.cl")

    return template.render(radix=parity_splitting.radix, depth=parity_splitting.depth)

def generate_outer_loop_function(parity_splitting):
    """
    Generate outer loop for OpenCL FFT.
    """
    template = template_environment.get_template("fft.cl")

    return template.render(N=parity_splitting.N, depth=parity_splitting.depth, radix=parity_splitting.radix, L=parity_splitting.L)
