from jinja2 import Environment, FileSystemLoader
import numpy
from parity import ParitySplitting, comp_perm
from twiddle import make_twiddle

template_loader = FileSystemLoader("fftsynth/templates")
template_environment = Environment(loader=template_loader)


def generate_macros(parity_splitting: ParitySplitting):
    """
    Generate the preprocessor macros necessary for the FFT.
    """
    template = template_environment.get_template("macros.cl")

    return template.render(radix=parity_splitting.radix)


def generate_twiddle_array(parity_splitting: ParitySplitting):
    """
    Generate OpenCL constant array for twiddle factors
    """
    template = template_environment.get_template("twiddles.cl")
    W = numpy.ones(shape=[parity_splitting.radix, parity_splitting.radix])
    perm = numpy.array([comp_perm(parity_splitting.radix, i) for i in range(parity_splitting.M)])

    n = parity_splitting.radix
    for k in range(parity_splitting.depth - 1):
        w = make_twiddle(parity_splitting.radix, n).conj()
        w_r_x = (numpy.ones(shape=[parity_splitting.M // n, parity_splitting.radix, n]) * w) \
            .transpose([0, 2, 1]) \
            .reshape([-1, parity_splitting.radix])[perm]
        W = numpy.r_[W, w_r_x]
        n *= parity_splitting.radix

    return template.render(radix=parity_splitting.radix,
                           W=W)


def generate_transpose_function(parity_splitting: ParitySplitting):
    """
    Generate inline OpenCL function to reverse the digits in base-n representation.
    """
    template = template_environment.get_template("transpose.cl")

    return template.render(radix=parity_splitting.radix,
                           depth=parity_splitting.depth)


def generate_parity_function(parity_splitting: ParitySplitting):
    """
    Generate inline OpenCL function to compute the parity of the index.
    """
    template = template_environment.get_template("parity.cl")

    return template.render(radix=parity_splitting.radix,
                           depth=parity_splitting.depth)


def generate_ipow_function(parity_splitting: ParitySplitting):
    """
    Generate inline OpenCL function to compute the integer power of a radix.
    """
    template = template_environment.get_template("ipow.cl")

    return template.render(radix=parity_splitting.radix)


def generate_index_functions(parity_splitting: ParitySplitting):
    """
    Generate inline OpenCL function to compute indices and permutations of the indices.
    """
    template = template_environment.get_template("indices.cl")

    return template.render(radix=parity_splitting.radix)


def generate_fft_functions(parity_splitting: ParitySplitting):
    """
    Generate outer loop for OpenCL FFT.
    """
    template = template_environment.get_template("fft.cl")

    return template.render(N=parity_splitting.N,
                           depth=parity_splitting.depth,
                           radix=parity_splitting.radix,
                           M=parity_splitting.M)


def generate_fft(parity_splitting: ParitySplitting):
    """
    Generate and print the complete OpenCL FFT.
    """
    print(generate_macros(parity_splitting))
    print(generate_twiddle_array(parity_splitting))
    print(generate_parity_function(parity_splitting))
    print(generate_transpose_function(parity_splitting))
    print(generate_ipow_function(parity_splitting))
    print(generate_index_functions(parity_splitting))
    print(generate_fft_functions(parity_splitting))


if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser(description="Generate an OpenCL FFT kernel")
    parser.add_argument("--radix", type=int, default=4, help="FFT radix")
    parser.add_argument("--depth", type=int, default=3, help="FFT depth")
    args = parser.parse_args()
    print( "/* FFT")
    print(f" * command: python -m fftsynth.generator_templated {' '.join(sys.argv[1:])}")
    print( " */")
    N = args.radix**args.depth
    ps = ParitySplitting(N, args.radix)
    generate_fft(ps)
