#!/usr/bin/env python3
"""Simple tests for the implementation of the STI functionality."""

import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float
from yaslp.phantom import shepp_logan

from thesis import forward


@pytest.mark.parametrize(
    ("bdir", "k", "kernel_expected"),
    [
        (
            jnp.array([[1, 0, 0]]),
            jnp.array([0, 0, 0]),
            jnp.zeros(6).at[0].set(1 / 3),
        ),  # 11: (1/3 - 0), 12: 0, 13: 0
        (
            jnp.array([[1, 0, 0]]),
            jnp.array([1, 1, 1]),
            jnp.zeros(6).at[1:3].set(-1 / 3),
        ),  # 11: (1/3 - 1/3), 12: -1/3, 13: -1/3
        (
            jnp.array([[1, 0, 0]]),
            jnp.array([1, 1, 0]),
            jnp.zeros(6).at[0].set(-1 / 6).at[1].set(-1 / 2),
        ),  # 11: (1/3 - 1/2), 12: -1/2, 13: 0
        (
            jnp.array([[1, 0, 0]]),
            jnp.array([0, 1, 0]),
            jnp.zeros(6).at[0].set(1 / 3),
        ),  # 11: (1/3 - 0), 12: 0, 13: 0
        (
            jnp.array([[0, 1, 0]]),
            jnp.array([1, 0, 0]),
            jnp.zeros(6).at[3].set(1 / 3),
        ),  # 21: 0, 22: (1/3 - 0), 23: 0
        (
            jnp.array([[0, 0, 1]]),
            jnp.array([0, 1, 0]),
            jnp.zeros(6).at[5].set(1 / 3),
        ),  # 31: , 32: , 33: (1/3 - 0)
    ],
)
def test_kernels_per_k_space_point(
    bdir: Float[Array, " ndim"],
    k,
    kernel_expected: Float[Array, ""],
):
    """Test the way kernels are evaluated at b & k."""
    if jnp.linalg.norm(k) > 0:
        k = k / jnp.linalg.norm(k)
    kernel_actual = forward._simulate_unraveled_kernels_at_kspace_point(bdir, k)
    assert jnp.allclose(kernel_actual, kernel_expected, atol=1e-4, rtol=1e-4)


# @pytest.mark.parametrize(
#     ("bdir", "k", "field_ft"),
#     [
#         (
#             jnp.array([[1, 0, 0]]),
#             jnp.array([1, 1, 1]),
#             jnp.array(-2 / 3 - 1),
#         ),  #  (1/3 - 1/3) * F11(=1) -1/3 * F12(=2) -1/3 * F13(=3)
#         (
#             jnp.array([[1, 0, 0]]),
#             jnp.array([1, 1, 0]),
#             jnp.array(-1 / 6 - 1),
#         ),  # (1/3 - 1/2) * F11 -1/2 * F12 + 0 * F13
#         (
#             jnp.array([[1, 0, 0]]),
#             jnp.array([0, 1, 0]),
#             jnp.array(1 / 3),
#         ),  # (1/3 - 0) * F11
#         (
#             jnp.array([[0, 1, 0]]),
#             jnp.array([1, 0, 0]),
#             jnp.array(4 / 3),
#         ),  # (1/3 - 0) * F22
#         (
#             jnp.array([[0, 0, 1]]),
#             jnp.array([0, 1, 0]),
#             jnp.array(2),
#         ),  # (1/3 - 0) * F33
#     ],
# )
# def test_forward_per_k_space_point(
#     bdir: Float[Array, " ndim"],
#     k,
#     field_ft: Float[Array, ""],
# ):
#     chi_ft: Float[Array, " n_upper_diag"] = jnp.arange(1, 7)
#     field_ft_expected = _simulate_field_from_unraveled_tensor_at_kspace_point(
#         bdir, k, chi_ft
#     )
#     assert jnp.allclose(field_ft, field_ft_expected)


_VECTORS = [
    jnp.array([0, 1]),
    jnp.array([1, 0]),
    jnp.array([1, 1]),
    jnp.array([0, 0, 1]),
    jnp.array([0, 1, 0]),
    jnp.array([0, 1, 1]),
    jnp.array([1, 0, 0]),
    jnp.array([1, 0, 1]),
    jnp.array([1, 1, 0]),
    jnp.array([1, 1, 1]),
]
_VECTORS_STACKED = [
    jnp.array([[0, 1]]),
    jnp.array([[0, 1], [1, 0]]),
    # jnp.array([[0, 1], [1, 0], [1, 1]]),
    # jnp.array([[0, 0, 1], [0, 1, 0]]),
    # jnp.array([[0, 0, 1], [0, 1, 0], [1, 1, 1]]),
    # jnp.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1]]),
]


def _simulate_unraveled_kernels_at_kspace_point_explicitly(
    b: Float[Array, " ndim"],
    k: Float[Array, " ndim"],
) -> Float[Array, " n_triu"]:
    b = b / jnp.linalg.norm(b, axis=-1)
    k = k / jnp.linalg.norm(k)
    kb = b @ k  # skipping / (k@k) as it is == 1
    assert b.shape[-1] == k.size
    if k.size == 2:
        return jnp.stack(
            [
                1 / 3 * b[..., 0] * b[..., 0] - kb * b[..., 0] * k[0],
                2 / 3 * b[..., 0] * b[..., 1]
                - kb * (b[..., 0] * k[1] + b[..., 1] * k[0]),
                1 / 3 * b[..., 1] * b[..., 1] - kb * b[..., 1] * k[1],
            ],
            axis=-1,
        )
    elif k.size == 3:
        return jnp.stack(
            [
                1 / 3 * b[..., 0] * b[..., 0] - kb * b[..., 0] * k[0],
                2 / 3 * b[..., 0] * b[..., 1]
                - kb * (b[..., 0] * k[1] + b[..., 1] * k[0]),
                2 / 3 * b[..., 0] * b[..., 2]
                - kb * (b[..., 0] * k[2] + b[..., 2] * k[0]),
                1 / 3 * b[..., 1] * b[..., 1] - kb * b[..., 1] * k[1],
                2 / 3 * b[..., 1] * b[..., 2]
                - kb * (b[..., 1] * k[2] + b[..., 2] * k[1]),
                1 / 3 * b[..., 2] * b[..., 2] - kb * b[..., 2] * k[2],
            ],
            axis=-1,
        )
    else:
        raise ValueError(f"Unsupported dimensionality: {b.shape=}, {k.shape=}")


@pytest.mark.parametrize("k_norm", _VECTORS)
@pytest.mark.parametrize("bdir", _VECTORS + _VECTORS_STACKED)
def test_unraveled_kernels_at_kspace_point(
    bdir: Float[Array, " ndim"],
    k_norm: Float[Array, " ndim"],
):
    # bdir = bdir / jnp.linalg.norm(bdir)
    # k_norm = k_norm / jnp.linalg.norm(k_norm)
    if bdir.shape[-1] != k_norm.size:
        pytest.skip("Unconsistent dimensionality of test matrix inputs")
    kernels_actual = forward._simulate_unraveled_kernels_at_kspace_point(bdir, k_norm)
    print(f"{kernels_actual.shape=}")
    kernels_expected = _simulate_unraveled_kernels_at_kspace_point_explicitly(
        bdir, k_norm
    )
    print(f"{kernels_expected.shape=}")
    assert jnp.allclose(kernels_actual, kernels_expected)


@pytest.mark.parametrize(
    "b_dir",
    [
        jnp.array([0, 1]),
        jnp.array([0, 1, 0]),
        jnp.array([[0, 1], [1, 0]]),
        jnp.array([[0, 0, 1], [0, 1, 1]]),
    ],
)
def test_forward_sti_implementation_comparison(b_dir):
    ndim = b_dir.shape[-1]
    ncomp = int(ndim * (ndim + 1) / 2)
    N = 32
    grid_shape = (N,) * ndim
    phantom = shepp_logan(grid_shape)
    chi_comps = phantom[..., None] * jnp.arange(1, 1 + ncomp)
    fields_naively = forward.simulate_field_from_unraveled_tensor_and_bdir_naively(
        b_dir, chi_comps
    )
    # # if orient is at the last...
    # assert fields.shape == grid_shape + jnp.atleast_2d(b_dir).shape[:1]
    # # if orient end up being the leading dim
    assert fields_naively.shape == jnp.atleast_2d(b_dir).shape[:1] + grid_shape
    fields_leading = forward.simulate_field_from_unraveled_tensor_and_bdirs(
        b_dir, chi_comps
    )
    assert fields_leading.shape == fields_naively.shape
    assert jnp.allclose(fields_leading, fields_naively, atol=1e-4, rtol=1e-4)
    fields_trailing = forward.simulate_field_from_unraveled_tensor_and_bdirs(
        b_dir, chi_comps, orientation_axis="trailing"
    )
    fields_trailing = jnp.rollaxis(fields_trailing, -1)
    print(f"{fields_trailing.shape=}")
    assert fields_trailing.shape == fields_naively.shape
    assert jnp.allclose(fields_trailing, fields_naively, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "b_dir",
    [
        jnp.array([0, 1]),
        jnp.array([0, 1, 0]),
        jnp.array([[0, 1], [1, 0]]),
        jnp.array([[0, 0, 1], [0, 1, 1]]),
    ],
)
@pytest.mark.parametrize(
    "simulate_fn",
    [
        forward.simulate_field_from_unraveled_tensor_and_bdir_naively,
        forward.simulate_field_from_unraveled_tensor_and_bdirs,
        forward.simulate_field_from_unraveled_tensor_and_bdir_orientation_trails,
    ],
)
def test_forward_sti_benchmark(b_dir, simulate_fn, benchmark):
    """Meant to prove that one should not simulate tensor convolution niavely."""
    ndim = b_dir.shape[-1]
    ncomp = int(ndim * (ndim + 1) / 2)
    N = 32
    grid_shape = (N,) * ndim
    phantom = shepp_logan(grid_shape)
    chi_comps = phantom[..., None] * jnp.arange(1, 1 + ncomp)
    benchmark(simulate_fn, b_dir, chi_comps)
