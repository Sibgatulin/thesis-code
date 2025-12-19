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


def model_unpooled_marginalised_unconstrained_msa(
    b_dir: Float[Array, "orient ndim"],
    v1: Float[Array, "*spatial ndim"],
    mask_fg: Bool[Array, " *spatial"] = jnp.array(True),
    prior={},
    obs: Float[Array, " *spatial"] | None = None,
):
    """Simplest CSST model with the noise STD marginalised out.

    mms(r) ~ N(0, σ_mms)
    msa(r) ~ N(0, σ_msa) # WRONG AND TEMPORARY
    σ²_obs ~ Γ^(-1)(α, β)
    obs ~ StudentT(2α, f(mms, msa), √β/α)
    """
    b_dir = jnp.atleast_2d(b_dir)
    norient, ndim = b_dir.shape
    grid_shape = v1.shape[:-1]
    assert len(grid_shape) == ndim and v1.shape[-1] == ndim, (
        f"Inconsistent dimensionality {v1.shape=} & {b_dir.shape=}"
    )

    prior_obs_var = prior.get("obs_var", DEFAULT_PRIOR["obs_var"])
    obs_df = 2 * prior_obs_var["concentration"]
    obs_var = prior_obs_var["rate"] / prior_obs_var["concentration"]

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
                numpyro.sample(
                    "obs", dist.StudentT(obs_df, obs_loc, jnp.sqrt(obs_var)), obs=obs
                )
