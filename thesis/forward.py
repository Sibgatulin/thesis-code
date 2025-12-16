from functools import partial
from typing import Literal

import jax.numpy as jnp
from jax import jit, vmap
from jax.core import Tracer
from jaxtyping import Array, Complex, Float, Shaped

from yaslp.utils import grid_basis

## SCALAR ##


@jit
def simulate_field_from_scalar_and_kernel(
    kernel_rft: Float[Array, " *rft"],
    chi_iso: Float[Array, " *spatial"],
    norm="backward",
) -> Float[Array, " *spatial"]:
    axes = tuple(range(-chi_iso.ndim, 0))
    return jnp.fft.irfftn(
        kernel_rft * jnp.fft.rfftn(chi_iso, norm=norm, axes=axes), axes=axes, norm=norm
    )


## TENSOR ##


### Option 1: By forming the dense tensor ###
def _populate_dense_symmetric_tensor(
    upper_tri_vec: Shaped[Array, "*batch m"], n: int
) -> Shaped[Array, "*batch n n"]:
    """Convert upper triangular vector to full symmetric matrix."""
    # Create empty matrix
    matrix = jnp.zeros(upper_tri_vec.shape[:-1] + (n, n), dtype=upper_tri_vec.dtype)
    # Get upper triangular indices
    i, j = jnp.triu_indices(n)

    # Fill upper triangle
    matrix = matrix.at[..., i, j].set(upper_tri_vec)
    # Fill lower triangle (transpose)
    matrix = matrix.at[..., j, i].set(upper_tri_vec)

    return matrix


def simulate_field_from_unraveled_tensor_and_bdir_naively(
    b_dir: Float[Array, " ndim"], chi_comps: Float[Array, " *spatial n_unravd_comp"]
) -> Float[Array, "norient *spatial"]:
    """Simulate field from compact tensor repr through the dense tesnor.

    As an intermediate step, the tensor is fully populated.
    Meant to serve as a reference.
    """

    @partial(jit, static_argnames="n_spat_dim")
    def _simulate_field_from_unraveled_tensor_and_bdir_naively(
        b_dir: Float[Array, " ndim"],
        chi_comps: Float[Array, " *spatial n_unravd_comp"],
        n_spat_dim: int,
    ) -> Float[Array, " *spatial"]:
        k_norm = grid_basis(
            chi_comps.shape[:-1], reciprocal=True, return_unit_vector=True, rfft=True
        )
        chi_comps_rft: Complex[Array, " *rft n_unravd_comp"] = jnp.fft.rfftn(
            chi_comps, axes=range(n_spat_dim)
        )
        chi_tensor_rft = _populate_dense_symmetric_tensor(chi_comps_rft, n_spat_dim)
        field_rft = 1 / 3 * jnp.einsum("i,...ij,j", b_dir, chi_tensor_rft, b_dir) - (
            k_norm @ b_dir
        ) * jnp.einsum("...i,...ij,j", k_norm, chi_tensor_rft, b_dir)
        return jnp.fft.irfftn(field_rft, axes=tuple(range(n_spat_dim)))

    n_spat_dim = b_dir.shape[-1]

    b_dir = jnp.atleast_2d(b_dir)
    b_dir = b_dir / jnp.linalg.norm(b_dir, axis=-1)[:, None]

    return vmap(
        _simulate_field_from_unraveled_tensor_and_bdir_naively, in_axes=(0, None, None)
    )(b_dir, chi_comps, n_spat_dim)


### Option 2: By working with the unraveled tensor ###
@partial(jnp.vectorize, signature="(n),(n)->(m)")
def _simulate_unraveled_kernels_at_kspace_point(
    b_dir: Float[Array, " ndim"], k_norm: Float[Array, " ndim"]
) -> Float[Array, " n_triu"]:
    """Compute a vector of 'kernel' components at given k-space point.

    Consider the forward model for multiple orientations in the fourier space:

    field(i, k) = 1/3 * b_i @ ST(k) @ b_i - k @ b_i * k @ ST(k) @ b_i / k²,

    where i indexes the orientation (field(i, k) is the fourier transform
    of the i-th orientation registered to the same array system; b_i – direction
    of the B0 field in the array system at the given orientation), and ST(k) is
    the fourier transform of the susceptibility tensor.
    Due to the symmetry ST == ST^T, it can be represented with its (e.g. upper)
    triangle components ST_tri that can be expressed as a vector (of 6 components
    in 3D space). Diligent handling of the equation above allows to rewrite it as
    an inner product between two vectors in k-space

    field(i, k) = A(i, k) @ ST_tri(k),

    or a matrix-vector product if all orientations are stacked together.
    This happens to be the way STI_suite [1] and COSMOS_STI [2] implement the forward
    model for the STI reconstruction.

    This function returns the components of the vector A(i, k) for a given orientation
    of B0 and a given (normalised) k-vector (i.e. k_norm = k / |k|).
    """
    assert b_dir.shape == k_norm.shape, (
        f"Inconsistent input shapes {b_dir.shape=} != {k_norm.shape=}"
    )
    n = k_norm.size
    # TODO: test how much it costs to do this actually. Conside dropping it
    assert isinstance(b_dir, Tracer) or (jnp.linalg.norm(b_dir) > 0)
    b_dir = b_dir / jnp.linalg.norm(b_dir)
    k_norm_ = jnp.linalg.norm(k_norm)
    k_norm = k_norm / jnp.where(k_norm_ == 0, jnp.inf, k_norm_)

    # Pre-compute the inner product
    inner_kb = b_dir @ k_norm

    # Get the indices of the upper triangle
    i, j = jnp.triu_indices(n)

    # Compute the elements directly using the indices
    result = 2 / 3 * b_dir[i] * b_dir[j] - inner_kb * (
        k_norm[i] * b_dir[j] + k_norm[j] * b_dir[i]
    )
    # in the result above the diagonal elements are counted twice
    diag_factor = jnp.where(i == j, 0.5, 1)

    return result * diag_factor


#### 2.1: k-space dims to the left, orientation trails ####
@jit
def simulate_field_from_unraveled_tensor_and_bdir_orientation_trails(
    b_dir: Float[Array, "#orient ndim"],
    chi_comps: Float[Array, " *spatial n_unravd_comp"],
) -> Float[Array, " *spatial orient"]:
    @partial(jnp.vectorize, signature="(o,d),(d),(m)->(o)")
    def _simulate_field_from_unraveled_tensor_at_kspace_point(
        b_dir: Float[Array, "orient ndim"],
        k_norm: Float[Array, " ndim"],
        chi_rft: Float[Array, " n_triu"],
    ) -> Float[Array, " orient"]:
        return _simulate_unraveled_kernels_at_kspace_point(b_dir, k_norm) @ chi_rft

    def _simulate_field_from_unraveled_tensor_and_bdir(
        b_dir: Float[Array, "orient ndim"],
        k_norm: Float[Array, "*rft ndim"],
        chi_comps: Float[Array, " *spatial n_unravd_comp"],
    ) -> Float[Array, " *spatial orient"]:
        axes_to_transform = range(chi_comps.ndim - 1)
        chi_comps_rft: Complex[Array, " *rft n_unravd_comp"] = jnp.fft.rfftn(
            chi_comps, axes=axes_to_transform
        )
        field_rft = _simulate_field_from_unraveled_tensor_at_kspace_point(
            b_dir, k_norm, chi_comps_rft
        )

        return jnp.fft.irfftn(field_rft, axes=axes_to_transform)

    k_norm = grid_basis(
        chi_comps.shape[:-1], reciprocal=True, return_unit_vector=True, rfft=True
    )
    return _simulate_field_from_unraveled_tensor_and_bdir(
        jnp.atleast_2d(b_dir), k_norm, chi_comps
    )


#### 2.2: orientation is the leading dim, k-space dims to the right ####
@jit
def simulate_field_from_unraveled_tensor_and_bdir_orientation_leads(
    b_dir: Float[Array, " ndim"],
    chi_comps: Float[Array, " *spatial n_unravd_comp"],
) -> Float[Array, " *spatial"]:
    @partial(jnp.vectorize, signature="(d),(d),(m)->()")
    def _simulate_field_from_unraveled_tensor_at_kspace_point(
        b_dir: Float[Array, " ndim"],
        k_norm: Float[Array, " ndim"],
        chi_rft: Float[Array, " n_triu"],
    ) -> Float[Array, ""]:
        return _simulate_unraveled_kernels_at_kspace_point(b_dir, k_norm) @ chi_rft

    def _simulate_field_from_unraveled_tensor_and_bdir(
        b_dir: Float[Array, " ndim"],
        k_norm: Float[Array, "*rft ndim"],
        chi_comps: Float[Array, " *spatial n_unravd_comp"],
    ) -> Float[Array, " *spatial"]:
        axes_to_transform = range(chi_comps.ndim - 1)
        chi_comps_rft: Complex[Array, " *rft n_unravd_comp"] = jnp.fft.rfftn(
            chi_comps, axes=axes_to_transform
        )
        field_rft = _simulate_field_from_unraveled_tensor_at_kspace_point(
            b_dir, k_norm, chi_comps_rft
        )

        return jnp.fft.irfftn(field_rft, axes=axes_to_transform)

    k_norm = grid_basis(
        chi_comps.shape[:-1], reciprocal=True, return_unit_vector=True, rfft=True
    )
    return _simulate_field_from_unraveled_tensor_and_bdir(b_dir, k_norm, chi_comps)


@partial(jit, static_argnames="orientation_axis")
def simulate_field_from_unraveled_tensor_and_bdirs(
    b_dir: Float[Array, " ndim"],
    chi_comps: Float[Array, " *spatial n_unravd_comp"],
    orientation_axis: Literal["leading", "trailing"] = "leading",
) -> Float[Array, " *spatial"]:
    if orientation_axis == "leading":
        return vmap(
            simulate_field_from_unraveled_tensor_and_bdir_orientation_leads,
            in_axes=(0, None),
        )(jnp.atleast_2d(b_dir), chi_comps)
    elif orientation_axis == "trailing":
        return simulate_field_from_unraveled_tensor_and_bdir_orientation_trails(
            b_dir, chi_comps
        )
    else:
        raise ValuerError(
            "orientation_axis must be 'leading' or 'trailing', got " + orientation_axis
        )
