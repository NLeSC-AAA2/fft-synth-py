import pytest               # type: ignore
import numpy as np          # type: ignore
from kernel_tuner import run_kernel  # type: ignore

from fftsynth.parity import parity
from fftsynth.parity import ParitySplitting, comp_idx, comp_perm

N = 64
radix = 4
mc = ParitySplitting(N, radix)
kernel_file_content = open("test/fft64-fma.cl", "r").read()


def test_parity_4():
    x = np.arange(N, dtype=np.int32)
    y = np.zeros_like(x)
    y_ref = np.array([parity(radix, i) for i in range(N)])
    kernel_args = [x, y]

    results = run_kernel("test_parity_4", kernel_file_content, N, kernel_args, {}, compiler_options=["-DTESTING"])

    assert np.all(results[1] == y_ref)


def test_transpose_4():
    x = np.arange(N, dtype=np.int32)
    y = np.zeros_like(x)
    kernel_args = [x, y]

    results = run_kernel("test_transpose_4", kernel_file_content, N, kernel_args, {}, compiler_options=["-DTESTING"])
    y_ref = x.reshape(mc.factors).T.flatten()

    assert np.all(results[1] == y_ref)


def test_fft_1024():
    x = np.random.normal(size=(64, 2)).astype(np.float32)
    y = np.zeros_like(x)
    kernel_args = [x, y]

    results = run_kernel("fft_64", kernel_file_content, 1, kernel_args, {})

    y_Z = results[1][..., 0]+1j*results[1][..., 1]
    y_ref = np.fft.fft(x[..., 0]+1j*x[..., 1])

    assert abs(y_Z - y_ref).max() < 1e-3
