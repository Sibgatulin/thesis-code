from functools import partial
from typing import Literal

import jax.numpy as jnp
from jax import jit, vmap
from jax.core import Tracer
from jaxtyping import Array, Complex, Float, Shaped

from yaslp.utils import grid_basis

## SCALAR ##


@partial(jit, static_argnames="norm")
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


## CSST


@jit
def simulate_field_from_csst(
    b_dir: Float[Array, "#orient ndim"],
    v1: Float[Array, " *spatial ndim"],
    mms: Float[Array, " *spatial"],
    msa: Float[Array, " *spatial"],
) -> Float[Array, "#orient *spatial"]:
    """Simple CSST forward problem for one or multiple B0 orientations.

    field(k) = (1/3 - (b·k)²) FT[chi_⊥] + 1/3 FT[MSA (b·v)²]
               - (b·k)(k, FT[v MSA (b·v)])

    where k is a *unit* vector in k-space (i.e. k/||k||), chi_⊥ = MMS - MSA/3
    field(k=0) := 0
    """
    if b_dir.ndim == 2:
        return vmap(_simulate_field_from_csst, in_axes=(0, None, None, None))(
            b_dir, v1, mms, msa
        )
    elif b_dir.ndim == 1:
        return _simulate_field_from_csst(b_dir, v1, mms, msa)
    else:
        raise ValueError(f"Unsopported {b_dir.shape=}")


@jit
def _simulate_field_from_csst(
    b_dir: Float[Array, " ndim"],
    v1: Float[Array, " *spatial ndim"],
    mms: Float[Array, " *spatial"],
    msa: Float[Array, " *spatial"],
) -> Float[Array, " *spatial"]:
    """Simple CSST forward problem for a single B0 orientation.

    field(k) = (1/3 - (b·k)²) FT[chi_⊥] + 1/3 FT[MSA (b·v)²]
               - (b·k)(k, FT[v MSA (b·v)])

    where k is a *unit* vector in k-space (i.e. k/||k||), chi_⊥ = MMS - MSA/3
    field(k=0) := 0
    """
    ndim = b_dir.shape[-1]
    assert b_dir.ndim == 1, f"Multiple orientations not supported, got {b_dir.shape=}"
    assert v1.shape[-1] == ndim, (
        f"Inconsistent dimensionality: {b_dir.shape[-1]=} != {v1.shape[-1]=}"
    )

    # 1. Precompute spatial terms to limit FFT calls
    # Projection of fibre direction onto B0: (H, v1)
    # b_dir is (3,), vectors is (..., 3). Result is (...)
    _bv: Float[Array, " *spatial"] = jnp.dot(v1, b_dir)

    # Term 2 spatial map: MSA * (H, v1)^2
    term2_spatial: Float[Array, " *spatial"] = 1 / 3 * msa * _bv**2

    # Term 3 vector field: MSA * (H, v1) * v1
    # We broaden (msa * hv_proj) to (..., 1) to broadcast against vectors (..., 3)
    term3_vector_field: Float[Array, " *spatial ndim"] = v1 * (msa * _bv)[..., None]

    # 2. Fourier Transforms
    # We use rfftn for efficiency as inputs are real.
    # Axes are the spatial dimensions (all except the last one for the vector field)
    grid_shape = mms.shape

    chi_perp = mms - msa / 3
    chi_perp_rft: Complex[Array, " *rft"] = jnp.fft.rfftn(chi_perp)
    term2_rft: Complex[Array, " *rft"] = jnp.fft.rfftn(term2_spatial)
    term3_rft: Complex[Array, "*rft ndim"] = jnp.fft.rfftn(
        term3_vector_field, axes=tuple(range(-ndim - 1, -1))
    )

    # 3. K-space Kernel Construction
    # Get unit k-vectors. shape: (..., ndim)
    k_norm = grid_basis(grid_shape, reciprocal=True, return_unit_vector=True, rfft=True)

    # Standard Dipole Kernel: 1/3 - (H.k)^2
    _bk: Float[Array, " *rft"] = k_norm @ b_dir
    kernel_rft = 1.0 / 3.0 - _bk**2  # .at[(0,) * ndim].set(0.0)

    # 4. Assemble the Field in K-space
    field_rft = kernel_rft * chi_perp_rft + term2_rft

    # Part C: Subtract non-local anisotropic term
    # We need (k . F{vector_field}). Since k_norm is unit vector,
    # and the formula has 1/k^2, the magnitudes cancel out (see reasoning above).
    # We just need (H.k_hat) * (k_hat . term3_k)
    k_dot_term3: Complex[Array, " *rft"] = jnp.sum(k_norm * term3_rft, axis=-1)
    field_rft = field_rft - (_bk * k_dot_term3)

    # 5. Inverse FFT
    # For the lack of better idea, null the DC on the whole field
    # (rather then on dipole kernel components)
    return jnp.fft.irfftn(field_rft.at[(0,) * ndim].set(0.0))
