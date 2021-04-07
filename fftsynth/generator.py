# ~\~ language=Python filename=fftsynth/generator.py
# ~\~ begin <<lit/code-generator.md|fftsynth/generator.py>>[0]
from jinja2 import Environment, FileSystemLoader
import numpy
from pkg_resources import resource_filename
from math import ceil

from .parity import ParitySplitting, comp_perm
from .twiddle import make_twiddle

template_loader = FileSystemLoader(resource_filename("fftsynth", "templates"))
template_environment = Environment(loader=template_loader)


def get_n_twiddles(radix):
    """Gives the width of the twiddle array as a function of radix."""
    return [0, 1, 1, 2, 2, 4][radix]


# ~\~ begin <<lit/code-generator.md|generate-preprocessor>>[0]
def generate_preprocessor(parity_splitting: ParitySplitting, fpga: bool):
    """
    Generate the preprocessor directives necessary for the FFT.
    """
    template = template_environment.get_template("preprocessor.cl")

    return template.render(radix=parity_splitting.radix, fpga=fpga)
# ~\~ end


# ~\~ begin <<lit/code-generator.md|generate-twiddle-array>>[0]
def generate_twiddle_array(parity_splitting: ParitySplitting):
    """
    Generate OpenCL constant array for twiddle factors
    """
    template = template_environment.get_template("twiddles.cl")
    twiddles = numpy.ones(shape=[parity_splitting.radix, parity_splitting.radix])
    perm = numpy.array([comp_perm(parity_splitting.radix, i) for i in range(parity_splitting.M)])

    n = parity_splitting.radix
    for k in range(parity_splitting.depth - 1):
        w = make_twiddle(parity_splitting.radix, n).conj()
        w_r_x = (numpy.ones(shape=[parity_splitting.M // n, parity_splitting.radix, n]) * w) \
            .transpose([0, 2, 1]) \
            .reshape([-1, parity_splitting.radix])[perm]
        twiddles = numpy.r_[twiddles, w_r_x]
        n *= parity_splitting.radix

    return template.render(radix=parity_splitting.radix,
                           W=twiddles)
# ~\~ end


# ~\~ begin <<lit/code-generator.md|generate-fpga-functions>>[0]
def generate_fpga_functions():
    """
    Generate OpenCL code for FPGA functions.
    """
    template = template_environment.get_template("fpga.cl")

    return template.render()
# ~\~ end


# ~\~ begin <<lit/code-generator.md|generate-transpose-function>>[0]
def generate_transpose_function(parity_splitting: ParitySplitting):
    """
    Generate inline OpenCL function to reverse the digits in base-n representation.
    """
    template = template_environment.get_template("transpose.cl")

    return template.render(radix=parity_splitting.radix,
                           depth=parity_splitting.depth)
# ~\~ end


# ~\~ begin <<lit/code-generator.md|generate-parity-function>>[0]
def generate_parity_function(parity_splitting: ParitySplitting):
    """
    Generate inline OpenCL function to compute the parity of the index.
    """
    template = template_environment.get_template("parity.cl")

    return template.render(radix=parity_splitting.radix,
                           depth=parity_splitting.depth)
# ~\~ end


# ~\~ begin <<lit/code-generator.md|generate-ipow-function>>[0]
def generate_ipow_function(parity_splitting: ParitySplitting):
    """
    Generate inline OpenCL function to compute the integer power of a radix.
    """
    template = template_environment.get_template("ipow.cl")

    return template.render(radix=parity_splitting.radix)
# ~\~ end


# ~\~ begin <<lit/code-generator.md|generate-index-functions>>[0]
def generate_index_functions(parity_splitting: ParitySplitting):
    """
    Generate inline OpenCL function to compute indices and permutations of the indices.
    """
    template = template_environment.get_template("indices.cl")

    return template.render(radix=parity_splitting.radix)
# ~\~ end


# ~\~ begin <<lit/code-generator.md|generate-fft-functions>>[0]
def generate_fft_functions(parity_splitting: ParitySplitting, fpga: bool):
    """
    Generate outer and inner loop for OpenCL FFT.
    """
    template = template_environment.get_template("fft.cl")
    depth_type = "unsigned int"
    m_type = "unsigned int"
    n_type = "unsigned int"
    if fpga:
        depth_type = "uint{}_t".format(int(numpy.ceil(numpy.log2(parity_splitting.depth + 0.5))))
        m_type = "uint{}_t".format(int(numpy.ceil(numpy.log2(parity_splitting.M + 0.5))))
        n_type = "uint{}_t".format(int(numpy.ceil(numpy.log2(parity_splitting.N + 0.5))))

    return template.render(N=parity_splitting.N,
                           depth=parity_splitting.depth,
                           radix=parity_splitting.radix,
                           M=parity_splitting.M,
                           fpga=fpga,
                           depth_type=depth_type,
                           m_type=m_type,
                           n_type=n_type)
# ~\~ end


# ~\~ begin <<lit/code-generator.md|generate-codelets>>[0]
def generate_codelets(parity_splitting: ParitySplitting, fpga: bool):
    """
    Generate OpenCL codelets for FFT.
    """
    template = template_environment.get_template("codelets.cl")

    return template.render(radix=parity_splitting.radix, fpga=fpga)
# ~\~ end


def generate_fft(parity_splitting: ParitySplitting, fpga: bool):
    """
    Generate and print the complete OpenCL FFT.
    """
    print(generate_preprocessor(parity_splitting, fpga))
    print("\n")
    print(generate_twiddle_array(parity_splitting))
    print("\n")
    if fpga:
        print(generate_fpga_functions())
        print("\n")
    print(generate_parity_function(parity_splitting))
    print("\n")
    print(generate_transpose_function(parity_splitting))
    print("\n")
    print(generate_ipow_function(parity_splitting))
    print("\n")
    print(generate_index_functions(parity_splitting))
    print("\n")
    print(generate_codelets(fpga))
    print("\n")
    print(generate_fft_functions(parity_splitting, fpga))


def generate_fma_twiddle_array(parity_splitting: ParitySplitting):
    """
    Generate OpenCL constant array for twiddle factors
    """
    template = template_environment.get_template("fma-twiddles.cl")
    twiddles = numpy.ones(shape=[parity_splitting.radix, parity_splitting.radix])
    perm = numpy.array([comp_perm(parity_splitting.radix, i) for i in range(parity_splitting.M)])

    n = parity_splitting.radix
    for k in range(parity_splitting.depth - 1):
        w = make_twiddle(parity_splitting.radix, n).conj()
        w_r_x = (numpy.ones(shape=[parity_splitting.M // n, parity_splitting.radix, n]) * w) \
            .transpose([0, 2, 1]) \
            .reshape([-1, parity_splitting.radix])[perm]
        twiddles = numpy.r_[twiddles, w_r_x]
        n *= parity_splitting.radix

    return template.render(radix=parity_splitting.radix,
                           n_twiddles=get_n_twiddles(parity_splitting.radix),
                           W=twiddles)


def generate_fma_codelets(fpga: bool):
    """
    Generate OpenCL codelets for FFT.
    """
    template = template_environment.get_template("fma-codelets.cl")

    return template.render(fpga=fpga)


def generate_fma_fft_functions(parity_splitting: ParitySplitting, fpga: bool):
    """
    Generate outer loop for OpenCL FFT.
    """
    template = template_environment.get_template("fma-fft.cl")
    depth_type = "unsigned int"
    m_type = "unsigned int"
    n_type = "unsigned int"
    if fpga:
        depth_type = "uint{}_t".format(int(numpy.ceil(numpy.log2(parity_splitting.depth + 0.5))))
        m_type = "uint{}_t".format(int(numpy.ceil(numpy.log2(parity_splitting.M + 0.5))))
        n_type = "uint{}_t".format(int(numpy.ceil(numpy.log2(parity_splitting.N + 0.5))))

    return template.render(N=parity_splitting.N,
                           depth=parity_splitting.depth,
                           radix=parity_splitting.radix,
                           n_twiddles=get_n_twiddles(parity_splitting.radix),
                           M=parity_splitting.M,
                           fpga=fpga,
                           depth_type=depth_type,
                           m_type=m_type,
                           n_type=n_type)


def generate_fma_fft(parity_splitting: ParitySplitting, fpga: bool):
    """
    Generate and print the complete OpenCL FFT (FMA version).
    """
    print(generate_preprocessor(parity_splitting, fpga))
    print("\n")
    print(generate_fma_twiddle_array(parity_splitting))
    print("\n")
    if fpga:
        print(generate_fpga_functions())
        print("\n")
    print(generate_parity_function(parity_splitting))
    print("\n")
    print(generate_transpose_function(parity_splitting))
    print("\n")
    print(generate_ipow_function(parity_splitting))
    print("\n")
    print(generate_index_functions(parity_splitting))
    print("\n")
    print(generate_fma_codelets(fpga))
    print("\n")
    print(generate_fma_fft_functions(parity_splitting, fpga))


if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser(description="Generate an OpenCL FFT kernel")
    parser.add_argument("--radix", type=int, default=4, help="FFT radix")
    parser.add_argument("--depth", type=int, default=3, help="FFT depth")
    parser.add_argument("--fpga", action="store_true")
    parser.add_argument("--fma", action="store_true")
    args = parser.parse_args()
    print("/* FFT")
    print(f" * command: python -m fftsynth.generator {' '.join(sys.argv[1:])}")
    print(" */")
    N = args.radix**args.depth
    ps = ParitySplitting(N, args.radix)
    if args.fma:
        generate_fma_fft(ps, args.fpga)
    else:
        generate_fft(ps, args.fpga)
# ~\~ end
