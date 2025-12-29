#!/usr/bin/env python3
import jax.random as jr
from jaxtyping import Int, Array, Float
import jax.numpy as jnp
from yaslp.phantom import Ellipse, EllipsoidPhantom
from yaslp.utils import grid_basis

LUT = {
    # BG, FG, A, B, C, D
    "chi_iso": jnp.array([0.0, 0.05, -0.1, -0.1, -0.1, 0.1]),
    "chi_ani": jnp.array([0.0, 0.0, 0.018, 0.018, 0.018, 0.0]),
    "eigenvectors": jnp.array(
        [[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]
    ),
    "_primary_axis": jnp.array([0, 0, 0, 1, 2, 0]),
    # 0 is also used as a placeholder where the axis isn't defined (BG, FG, D)
}


def generate_pair_of_shepp_logans_in_2d(
    grid_shape: tuple[int, ...] = (64, 64),
) -> tuple[Int[Array, "n m"], Int[Array, "n m"]]:  # TODO: add mutliple dispatch
    """Return the pair of Shepp-Logan like phantom used in the thesis.

    One captures the 'true' and the other a simplified varsion of the ROIs.
    """
    ellipses_for_sim = [
        Ellipse(value=1, radius=(0.8, 0.9), center=(0, 0), phi=0),  # FG
        # The value of the following ellipses should be understood relative to the value
        # of the main, FG, ellipse defined above
        Ellipse(value=1, radius=(0.47, 0.16), center=(-0.3, 0.1), phi=1.9),  # A
        Ellipse(value=2, radius=(0.40, 0.12), center=(0.30, 0.1), phi=1.2),  # B
        Ellipse(value=3, radius=(0.09, 0.05), center=(-0.14, -0.6), phi=0),  # inside D
        Ellipse(value=4, radius=(0.05, 0.09), center=(0.14, -0.6), phi=0),  # inside D
        Ellipse(value=5, radius=(0.05, 0.05), center=(0.0, -0.6), phi=0),  # inside D
        Ellipse(value=6, radius=(0.09, 0.09), center=(0, 0.0), phi=0),  # tiny in center
        Ellipse(value=7, radius=(0.25, 0.25), center=(0, 0.5), phi=0),  # C
        Ellipse(value=8, radius=(0.08, 0.08), center=(0, 0.25), phi=0),  # tiny under C
        # want this to be the last to clip the map to 8 or 9...
    ]
    ellipses_for_reco = [
        Ellipse(value=1, radius=(0.8, 0.9), center=(0, 0), phi=0),  # FG
        # The value of the following ellipses should be understood relative to the value
        # of the main, FG, ellipse defined above
        Ellipse(value=1, radius=(0.47, 0.16), center=(-0.3, 0.1), phi=1.9),  # A
        Ellipse(value=2, radius=(0.40, 0.12), center=(0.30, 0.1), phi=1.2),  # B
        Ellipse(value=3, radius=(0.25, 0.18), center=(0, -0.6), phi=0),  # D
        Ellipse(value=4, radius=(0.25, 0.29), center=(0, 0.46), phi=0),  # C
    ]
    sim = EllipsoidPhantom(grid_shape, ellipses_for_sim)
    # the last two ellipses overlap, which means that in sim.label_map
    # at the place of overlap the label is the sum of their labels, which is off
    # I used to clip it at the last value + 1:
    # sim = sim.label_map.clip(max=sim.n + 1)
    # this labelled the intersection as the next consecutive ROI.
    # Eventually, I realised, I don't want it, and would simply clip sim.label_map
    # to the last value, merging the overlapping area into the tiny ROI.
    sim = sim.label_map.clip(max=sim.n)
    # Effectively, I don't make use of the ROIs overlapping in this label map, so I
    # can as well simply make them touch.

    # has only non-overlapping ROIs
    reco = EllipsoidPhantom(grid_shape, ellipses_for_reco).label_map
    return sim, reco


def generate_qsm_shepp_logan_in_2d(
    sim: Int[Array, "n m"], spread=False, units="ppb"
) -> tuple[Float[Array, "n m"], dict[str, Float[Array, " roi"]]]:
    """Given a structure, fill in from a LUT and rescale to mean(qsm) = 0."""
    if units not in ["ppb", "ppm"]:
        raise ValueError(f"units must be 'ppm' or 'ppb', got {units}")
    _factor = {"ppm": 1, "ppb": 1e3}[units]

    lut = _factor * jnp.array(
        [
            # loc, spread
            [0.0, 0.0],  # BG
            [jnp.nan, jnp.nan],  # FG
            [-0.07, 0.02],  # A
            [-0.08, 0.03],  # B
            [0.1, 0.01],  # inside D
            [0.13, 0.02],  # inside D
            [0.15, 0.02],  # inside D
            [-0.02, 0.01],  # tiny in center
            [0.0, 0.04],  # C
            [0.1, 0.01],  # tiny under C
        ]
    )
    chi = lut[sim, 0]  # prelim
    # hacky way to normalise total sum to 0
    total_wo_fg = jnp.nansum(chi)
    fg_size = (sim == 1).sum()
    target_fg_value = -total_wo_fg / fg_size
    lut = lut.at[1].set(jnp.array([target_fg_value, abs(target_fg_value)]))
    chi = lut[sim, 0]  # prelim
    assert jnp.isclose(chi.sum(), 0, atol=1e-3 * _factor, rtol=1e-3)
    # my inversion enforces zero-mean, so I would prefer the phantom
    # to be zero-mean too, so that I don't need to force / reference it additionally
    if spread:
        chi = chi + lut[sim, 1] * jr.normal(jr.PRNGKey(0), shape=sim.shape)

    luts = {
        "chi_loc": lut[:, 0],
        "chi_spread": lut[:, 1],
        "obs_scale": 0.3 * lut[:, 1].at[-1].set(lut[-1, 1] * 7),
    }
    return chi, luts


def generate_shepp_logan_in_2d(
    grid_shape: tuple[int, ...] = (64, 64), spread=False, units="ppb"
) -> tuple[
    Int[Array, "n m"],
    Int[Array, "n m"],
    Int[Array, "n m"],
    dict[str, Float[Array, " roi"]],
]:
    """Generate a pair of structures and a sample QSM with LUTs."""
    sim, reco = generate_pair_of_shepp_logans_in_2d(grid_shape)
    chi, luts = generate_qsm_shepp_logan_in_2d(sim, spread, units)
    return sim, reco, chi, luts


def generate_ast_shepp_logan_in_2d(
    sim: Int[Array, "n m"], units="ppb"
) -> tuple[Float[Array, "n m 3"], Float[Array, " roi 3"]]:
    """Generate an STI phantom and normalise its trace to 0."""
    if units not in ["ppb", "ppm"]:
        raise ValueError(f"units must be 'ppm' or 'ppb', got {units}")
    _factor = {"ppm": 1, "ppb": 1e3}[units]

    NROI = 10
    assert jnp.unique(sim).size == NROI and sim.max() == NROI - 1

    lut = jnp.array(
        [
            [0.0, 0.0, 0.0],  # BG
            [jnp.nan, jnp.nan, 0.0],  # FG
            [-0.07, -0.01, 0.001],  # A
            [-0.01, -0.08, 0.001],  # B
            [0.100, 0.100, 1.000],  # inside D
            [0.100, 0.100, -1.00],  # inside D
            [0.100, 0.100, 0.000],  # inside D
            [-0.10, -0.10, -0.75],  # tiny in center
            [0.100, -0.10, -0.75],  # C
            [-0.10, 0.100, 0.750],  # tiny under C
        ]
    )
    lut = lut.at[:, :2].set(lut[:, :2] * _factor)

    ast = lut[sim]
    assert ast.ndim == 3 and ast.shape[-1] == 3
    # hacky way to normalise total trace sum to 0
    total_trace_wo_fg = jnp.nansum(ast[..., :2])  # trace of the unraveled tensor
    fg_size = (sim == 1).sum()
    target_trace_fg_value = -total_trace_wo_fg / fg_size
    lut = lut.at[1, :2].set(target_trace_fg_value / 2)
    assert jnp.isfinite(lut).all()
    ast = lut[sim]  # prelim
    assert jnp.isclose(ast[..., :2].sum(), 0, atol=1e-3 * _factor, rtol=1e-3)

    return ast, lut


def generate_csst_shepp_logan_in_2d(
    sim: Int[Array, "n m"], units="ppb", flip_vector_at_centre=False, spread=False
) -> tuple[Float[Array, "n m"], Float[Array, "n m"], Float[Array, "n m 2"]]:
    """Generate a CSST phantom.

    MMS, MSA, vectors, + normalisation of MMS
    """

    if units not in ["ppb", "ppm"]:
        raise ValueError(f"units must be 'ppm' or 'ppb', got {units}")
    _factor = {"ppm": 1, "ppb": 1e3}[units]

    NROI = 10
    assert jnp.unique(sim).size == NROI and sim.max() == NROI - 1
    lut = _factor * jnp.array(
        [
            # MMS, MSA
            [0.0, 0.0],  # BG
            [jnp.nan, 0.0],  # FG
            [-0.07, 0.02],  # A
            [-0.08, 0.03],  # B
            [0.1, 0.01],  # inside D
            [0.13, 0.025],  # inside D
            [0.15, 0.015],  # inside D
            [-0.02, 0.04],  # tiny in center
            [0.0, 0.00],  # C
            [0.1, 0.005],  # tiny under C
        ]
    )

    mms = lut[sim, 0]  # prelim
    total_wo_fg = jnp.nansum(mms)
    fg_size = (sim == 1).sum()
    target_fg_value = -total_wo_fg / fg_size
    lut = lut.at[1, 0].set(target_fg_value)
    mms = lut[sim, 0]
    if spread:
        mms = mms + 0.1 * abs(lut[sim, 0]) * jr.normal(jr.PRNGKey(0), shape=sim.shape)
        # HACK: oh well, just can't rely on the LUT in this case too much
        mms -= mms.mean()
    else:
        assert jnp.isclose(mms.sum(), 0, atol=1e-3 * _factor, rtol=1e-3), (
            f"{mms.sum()=}"
        )
    msa = lut[sim, 1]
    if spread:
        msa = msa + 0.1 * abs(lut[sim, 1]) * jr.normal(jr.PRNGKey(0), shape=sim.shape)

    # define a perturbed vector field
    lut_eigenv = jnp.array(
        [
            # x, y, perurbation (> 0)
            [0.0, 0.0, 0.0],  # BG
            [0, 0, 0],  # FG
            [jnp.cos(1.9), jnp.sin(1.9), 7],  # A
            [jnp.cos(1.2 - jnp.pi / 2), jnp.sin(1.2 - jnp.pi / 2), 5],  # B
            [-1, -1, 4],  # inside D
            [1, -1, 6],  # inside D
            [0, -1, 5],  # inside D
            [1, 0, 8],  # tiny in center
            [0, 0, 0],  # C
            [0, 1, 9],  # tiny under C
        ]
    )

    grid_shape = sim.shape
    ndim = len(grid_shape)
    # Let's define a grid of `vec(r)` which indicates how far from 0 the poitn is.
    # To be consistent with `centroid`, I use domain="int"
    _grid: Float[Array, "n m"] = grid_basis(grid_shape)  # , domain="int")

    vectors = jnp.full(grid_shape + (ndim,), 0.0)
    for roi_label in range(1, NROI):  # skip the BG
        perturbation_strength = lut_eigenv[roi_label, -1]
        if perturbation_strength <= 0:
            # easy way to mark what should not have vectors
            continue
        # primary direction defined in the LUT
        eigenvector = lut_eigenv[roi_label, :-1]
        # normalise here for the convenience of LUT definition
        eigenvector /= jnp.linalg.norm(eigenvector)

        _roi_fg = sim == roi_label
        # This is an appoximation of `center` in `list[Ellipse]` but does not depend
        # on any assumptions about the overlap and summation of those ellipses
        roi_centre = (find_center(_roi_fg * 10) / jnp.array(grid_shape) - 0.5) * 2

        _grid_rel = _grid - roi_centre  # center the grid
        _grid_projected: Float[Array, "n m"] = (  # on the eigenvector direction
            _grid_rel / jnp.linalg.norm(_grid_rel, axis=-1, keepdims=True)
        ) @ eigenvector
        orthogonality_map = 1 - abs(_grid_projected)
        # perturb more in the direction perpendicular to the eigenvector
        # don't perturb the eigenvector direction itself
        perturbation_orthogonal_to_eigenvector = (
            _grid_rel * orthogonality_map[..., None]
        )
        # Assemble the resulting perturbed vector field
        _roi_vectors = perturbation_orthogonal_to_eigenvector * perturbation_strength
        if flip_vector_at_centre:
            # Increase the angle distribution: perturbations are be aplified where the eigv is flipped
            _roi_vectors += jnp.sign(_grid_projected)[..., None] * eigenvector
        else:
            # keep it straigtforward
            _roi_vectors += eigenvector

        # update where needed
        vectors = vectors.at[_roi_fg].set(_roi_vectors[_roi_fg])
    # normalise the vectors before returning them
    return (
        mms,
        msa,
        jnp.nan_to_num(vectors / jnp.linalg.norm(vectors, axis=-1, keepdims=True)),
    )


def find_center(binary_image):
    """Find center using average of all True/1 pixel positions."""
    # Get coordinates of all non-zero pixels
    coords = jnp.argwhere(binary_image)
    return coords.mean(axis=0)
