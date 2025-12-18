"""Probabilistic models for Cylindrically Symmetric Susceptibility Tensor."""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jaxtyping import Array, Bool, Float
from numpyro import handlers

from thesis.forward import simulate_field_from_csst

DEFAULT_PRIOR = {
    "obs_var": {"concentration": 3, "rate": 100},
    "mms": {"loc": 0, "scale": 50},
    "msa": {"loc": 0, "scale": 50},  # TODO: fix: this is temporary!
}


def model_unpooled(
    b_dir: Float[Array, "orient ndim"],
    v1: Float[Array, "*spatial ndim"],
    mask_fg: Bool[Array, " *spatial"] = jnp.array(True),
    prior={},
    obs: Float[Array, " *spatial"] | None = None,
):
    """Simplest CSST model.

    mms(r) ~ N(0, σ_mms)
    msa(r) ~ N(0, σ_msa)
    σ²_obs ~ Γ^(-1)(α, β)
    obs ~ N(f(mms, msa), σ_obs)
    """
    b_dir = jnp.atleast_2d(b_dir)
    norient, ndim = b_dir.shape
    grid_shape = v1.shape[:-1]
    assert len(grid_shape) == ndim and v1.shape[-1] == ndim, (
        f"Inconsistent dimensionality {v1.shape=} & {b_dir.shape=}"
    )

    obs_var = numpyro.sample(
        "obs_var",
        dist.InverseGamma(**prior.get("obs_var", DEFAULT_PRIOR["obs_var"])),
    )
    with numpyro.plate_stack("spatial", grid_shape):
        mms = numpyro.sample(
            "mms", dist.Normal(**prior.get("mms", DEFAULT_PRIOR["mms"]))
        )
        msa = numpyro.sample(
            "msa",
            dist.Normal(**prior.get("msa", DEFAULT_PRIOR["msa"])),  # TODO: Fix!
        )
        obs_loc = numpyro.deterministic(
            "field", simulate_field_from_csst(b_dir, v1, mms, msa)
        )
        with numpyro.plate("orient", norient):
            with handlers.mask(mask=mask_fg):
                numpyro.sample("obs", dist.Normal(obs_loc, jnp.sqrt(obs_var)), obs=obs)
