# ~\~ language=Python filename=test/test_radix_4_ps.py
# ~\~ begin <<lit/parity-splitting.md|test/test_radix_4_ps.py>>[0]
import pytest               # type: ignore
import numpy as np          # type: ignore
from kernel_tuner import run_kernel  # type: ignore

from fftsynth.parity import parity
from fftsynth.parity import ParitySplitting, comp_idx, comp_perm

N = 1024
radix = 4
mc = ParitySplitting(N, radix)
kernel_file_content = open("test/fft1024.cl", "r").read()


def test_parity_4():
    x = np.arange(N, dtype=np.int32)
    y = np.zeros_like(x)
    y_ref = np.array([parity(radix, i) for i in range(N)])
    kernel_args = [x, y]

    results = run_kernel("test_parity_4", kernel_file_content, N, kernel_args, {}, compiler_options=["-DTESTING"])

    assert np.all(results[1] == y_ref)


def test_comp_idx_4():
    x = np.arange(mc.L, dtype=np.int32)

    for k in range(1, 5):
        for j in range(4):
            y = np.zeros_like(x)
            kernel_args = [np.int32(k), np.int32(j), x, y]

            results = run_kernel("test_comp_idx_4", kernel_file_content, mc.L, kernel_args, {}, compiler_options=["-DTESTING"])
            y_ref = mc.index_loc \
                .reshape([-1, 4, 4**k]) \
                .transpose([0, 2, 1]) \
                .reshape([-1, 4])[::4, j]
            y_ref_2 = np.array([comp_idx(4, i, j, k-1) for i in range(mc.L)])

            assert np.all(results[3] == y_ref)
            assert np.all(results[3] == y_ref_2)


def test_comp_perm_4():
    x = np.arange(mc.L, dtype=np.int32)

    for r in range(4):
        y = np.zeros_like(x)
        kernel_args = [np.int32(r), x, y]

        results = run_kernel("test_comp_perm_4", kernel_file_content, mc.L, kernel_args, {}, compiler_options=["-DTESTING"])
        y_ref = np.array([comp_perm(radix, i*radix + r) for i in range(mc.L)])

        assert np.all(results[2] == y_ref)


def test_transpose_4():
    x = np.arange(N, dtype=np.int32)
    y = np.zeros_like(x)
    kernel_args = [x, y]

    results = run_kernel("test_transpose_4", kernel_file_content, N, kernel_args, {}, compiler_options=["-DTESTING"])
    y_ref = x.reshape(mc.factors).T.flatten()

    assert np.all(results[1] == y_ref)


def test_fft4_4():
    x = np.random.normal(size=(1024, 4, 2)).astype(np.float32)

    for cycle in range(4):
        y = np.zeros_like(x)
        kernel_args = [np.int32(cycle), x, y]
        y_ref = np.fft.fft(np.roll(x[..., 0]+1j*x[..., 1], -cycle, axis=1))

        results = run_kernel("test_fft_4", kernel_file_content, N, kernel_args, {}, compiler_options=["-DTESTING"])
        y_Z = np.roll(results[2][..., 0]+1j*results[2][..., 1], -cycle, axis=1)

        assert abs(y_Z - y_ref).max() < 1e-4


def test_fft_1024():
    x = np.random.normal(size=(1024, 2)).astype(np.float32)
    y = np.zeros_like(x)
    kernel_args = [x, y]

    results = run_kernel("fft_1024", kernel_file_content, 1, kernel_args, {})

    y_Z = results[1][..., 0]+1j*results[1][..., 1]
    y_ref = np.fft.fft(x[..., 0]+1j*x[..., 1])

    assert abs(y_Z - y_ref).max() < 1e-3


def test_fft_64(tmp_path):
    from contextlib import redirect_stdout
    from fftsynth.generator import (write_fft)
    code_path = tmp_path / "fft.cl"
    ps = ParitySplitting(1024, 4)
    with open(code_path, "w") as f:
        with redirect_stdout(f):
            write_fft(ps)
            print(
"""#ifdef TESTING
__kernel void test_fft_4(int cycle, __global const float2 *x, __global float2 *y)
{
    int i = get_global_id(0);
    float2 s[4][1];
    for (int k = 0; k < 4; ++k) s[k][0] = x[i*4+k];
    fft_4(s[0], s[1], s[2], s[3], cycle, 0, 0, 0, 0, 0);
    for (int k = 0; k < 4; ++k) y[i*4+k] = s[k][0];
}
__kernel void test_transpose_4(__global const int *x, __global int *y) {
    int i = get_global_id(0);
    y[i] = transpose_4(x[i]);
}

__kernel void test_parity_4(__global const int *x, __global int *y) {
    int i = get_global_id(0);
    y[i] = parity_4(x[i]);
}

__kernel void test_comp_idx_4(int k, int j, __global const int *x, __global int *y) {
    int i = get_global_id(0);
    y[i] = comp_idx_4(i, k-1) + j * (1 << 2*(k-1));
}

__kernel void test_comp_perm_4(int r, __global const int *x, __global int *y) {
    int i = get_global_id(0);
    y[i] = comp_perm_4(x[i], r);
}
#endif
""")
    kernel_file_content_64 = open(code_path, "r").read()

    x = np.arange(mc.L, dtype=np.int32)

    print("comp_idx")
    for k in range(1, 5):
        for j in range(4):
            y = np.zeros_like(x)
            kernel_args = [np.int32(k), np.int32(j), x, y]

            results = run_kernel("test_comp_idx_4", kernel_file_content_64, ps.L, kernel_args, {}, compiler_options=["-DTESTING"])
            y_ref = ps.index_loc \
                .reshape([-1, 4, 4**k]) \
                .transpose([0, 2, 1]) \
                .reshape([-1, 4])[::4, j]
            y_ref_2 = np.array([comp_idx(4, i, j, k-1) for i in range(ps.L)])

            assert np.all(results[3] == y_ref)
            assert np.all(results[3] == y_ref_2)


    print("comp_perm")
    x = np.arange(ps.L, dtype=np.int32)

    for r in range(4):
        y = np.zeros_like(x)
        kernel_args = [np.int32(r), x, y]

        results = run_kernel("test_comp_perm_4", kernel_file_content, ps.L, kernel_args, {}, compiler_options=["-DTESTING"])
        y_ref = np.array([comp_perm(radix, i*radix + r) for i in range(ps.L)])

        assert np.all(results[2] == y_ref)

    print("parity")
    x = np.arange(ps.N, dtype=np.int32)
    y = np.zeros_like(x)
    y_ref = np.array([parity(ps.radix, i) for i in range(ps.N)])
    kernel_args = [x, y]

    results = run_kernel("test_parity_4", kernel_file_content_64, ps.N, kernel_args, {}, compiler_options=["-DTESTING"])

    assert np.all(results[1] == y_ref)

    print("transpose")
    x = np.arange(ps.N, dtype=np.int32)
    y = np.zeros_like(x)
    kernel_args = [x, y]

    results = run_kernel("test_transpose_4", kernel_file_content_64, ps.N, kernel_args, {}, compiler_options=["-DTESTING"])
    y_ref = x.reshape(ps.factors).T.flatten()

    assert np.all(results[1] == y_ref)

    print("fft4")
    x = np.random.normal(size=(ps.N, 4, 2)).astype(np.float32)

    for cycle in range(4):
        y = np.zeros_like(x)
        kernel_args = [np.int32(cycle), x, y]
        y_ref = np.fft.fft(np.roll(x[..., 0]+1j*x[..., 1], -cycle, axis=1))

        results = run_kernel("test_fft_4", kernel_file_content_64, ps.N, kernel_args, {}, compiler_options=["-DTESTING"])
        y_Z = np.roll(results[2][..., 0]+1j*results[2][..., 1], -cycle, axis=1)

        assert abs(y_Z - y_ref).max() < 1e-4

    print("fft")
    x = np.random.normal(size=(ps.N, 2)).astype(np.float32)
    y = np.zeros_like(x)
    kernel_args = [x, y]

    results = run_kernel(f"fft_{ps.N}", kernel_file_content_64, 1, kernel_args, {})

    y_Z = results[1][..., 0]+1j*results[1][..., 1]
    y_ref = np.fft.fft(x[..., 0]+1j*x[..., 1])

    assert abs(y_Z - y_ref).max() < 1e-3
# ~\~ end
