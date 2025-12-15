#!/usr/bin/env python3
import jax.numpy as jnp
from yaslp.phantom import Ellipse, EllipsoidPhantom

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


def generate_shepp_logan_in_2d(
    grid_shape: tuple[int, ...] = (64, 64), spread=False, units="ppb"
):
    if units not in ["ppb", "ppm"]:
        raise ValueError(f"units must be 'ppm' or 'ppb', got {units}")
    _factor = {"ppm": 1, "ppb": 1e3}[units]
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
        import jax.random as jr

        chi = chi + lut[sim, 1] * jr.normal(jr.PRNGKey(0), shape=grid_shape)

    luts = {
        "chi_loc": lut[:, 0],
        "chi_spread": lut[:, 1],
        "obs_scale": 0.3 * lut[:, 1].at[-1].set(lut[-1, 1] * 7),
    }
    return sim, reco, chi, luts
