#!/usr/bin/env python3

import jax.numpy as jnp
import jax.random
import pytest
from jaxtyping import Array, Float
from yaslp.utils import grid_basis

from thesis import forward


def simulate_field_from_scalar_and_bdir(
    b_dir: Float[Array, " ndim"],
    chi_iso: Float[Array, " *spatial"],
    norm="backward",
) -> Float[Array, " *spatial"]:
    assert b_dir.ndim == 1
    assert b_dir.size == chi_iso.ndim
    k_norm = grid_basis(
        chi_iso.shape, reciprocal=True, return_unit_vector=True, rfft=True
    )
    hk = k_norm @ b_dir
    dipole_kernel = ((1 / 3) - hk**2).at[(0,) * chi_iso.ndim].set(0.0)
    return forward.simulate_field_from_scalar_and_kernel(
        dipole_kernel, chi_iso, norm=norm
    )


@pytest.mark.parametrize(
    "b_dir",
    [
        jnp.array([0, 1]),
        jnp.array([0, 1, 0]),
        jnp.array([0, 1, 1]),
    ],
)
def test_csst_reduces_to_isotropic_when_msa_zero(b_dir):
    """
    If MSA = 0, the model should be exactly equivalent to a standard
    scalar forward simulation with chi = chi_perp.
    """
    b_dir = b_dir / jnp.linalg.norm(b_dir, axis=-1, keepdims=True)

    ndim = b_dir.size
    grid_shape = (16,) * ndim
    key = jax.random.PRNGKey(0)

    # Generate random inputs
    chi_perp = jax.random.normal(key, grid_shape)
    msa = jnp.zeros(grid_shape)  # MSA is zero

    # Random vectors (should be irrelevant when MSA=0)
    v1 = jax.random.normal(key, grid_shape + (ndim,))
    v1 = v1 / jnp.linalg.norm(v1, axis=-1, keepdims=True)

    # 1. Run CSST
    field_csst = forward.simulate_field_from_csst(b_dir, v1, chi_perp, msa)

    # 2. Run Reference Scalar Sim
    # Reconstruct scalar kernel manually for comparison
    field_scalar = simulate_field_from_scalar_and_bdir(b_dir, chi_perp)

    assert jnp.allclose(field_csst, field_scalar, atol=1e-5), (
        "MSA=0 should match scalar sim"
    )


@pytest.mark.parametrize(
    "b_dir",
    [
        jnp.array([0, 1]),
        jnp.array([0, 1, 0]),
        jnp.array([0, 1, 1]),
    ],
)
def test_csst_parallel_fiber_check(b_dir):
    """
    Physical Consistency Check:
    If fibers are everywhere parallel to B0 (v || H), then:
       (v.h)^2 = 1
       (v.h)v = v = h
    The terms simplifiy such that the total field is equivalent to
    a scalar simulation with susceptibility = (chi_perp + MSA).
    """
    b_dir = b_dir / jnp.linalg.norm(b_dir, axis=-1, keepdims=True)

    ndim = b_dir.size
    grid_shape = (16,) * ndim

    key = jax.random.PRNGKey(0)

    # Fibers perfectly aligned with B0 everywhere
    v1 = jnp.ones(grid_shape + (1,)) * b_dir

    chi_perp = jax.random.normal(key, grid_shape)
    msa = jax.random.normal(key, grid_shape)

    # 1. Run CSST
    field_csst = forward.simulate_field_from_csst(b_dir, v1, chi_perp, msa)
    # field_csst -= field_csst.mean()  # manually reference to 0

    # 2. Run Scalar Sim with Effective Susceptibility
    # Effective Chi = Chi_perp + MSA
    field_expected = simulate_field_from_scalar_and_bdir(b_dir, chi_perp + msa)
    assert jnp.allclose(field_csst, field_expected, atol=1e-2, rtol=1e-2), (
        "When v || B0, CSST should reduce to scalar sim of (chi_perp + MSA)"
    )


@pytest.mark.parametrize(
    "b_dir,v1",
    [
        (jnp.array([0, 1]), jnp.array([1, 0])),
        (jnp.array([0, 1, 0]), jnp.array([1, 0, 0])),
        (jnp.array([0, 1, 1]), jnp.array([1, 0, 0])),
    ],
)
def test_csst_perpendicular_fiber_check(b_dir, v1):
    """
    Physical Consistency Check:
    If fibers are everywhere perpendicular to B0 (v ⊥​ H), then:
       (v.h) = 0
    The terms simplifiy such that the total field is equivalent to
    a scalar simulation with susceptibility = chi_perp.
    """
    b_dir = b_dir / jnp.linalg.norm(b_dir, axis=-1, keepdims=True)

    ndim = b_dir.size
    grid_shape = (16,) * ndim

    key = jax.random.PRNGKey(0)

    # Fibers ought to be perpendicular to B0
    v1 = jnp.ones(grid_shape + (1,)) * v1

    chi_perp = jax.random.normal(key, grid_shape)
    msa = jax.random.normal(key, grid_shape)

    # 1. Run CSST
    field_csst = forward.simulate_field_from_csst(b_dir, v1, chi_perp, msa)
    # field_csst -= field_csst.mean()  # manually reference to 0

    # 2. Run Scalar Sim with Effective Susceptibility = chi_perp
    field_expected = simulate_field_from_scalar_and_bdir(b_dir, chi_perp)
    assert jnp.allclose(field_csst, field_expected, atol=1e-2, rtol=1e-2), (
        "When v || B0, CSST should reduce to scalar sim of chi_perp"
    )
