"""Probabilistic models for Cylindrically Symmetric Susceptibility Tensor."""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jaxtyping import Array, Bool, Float, Int
from numpyro import handlers

from thesis.forward import simulate_field_from_csst

DEFAULT_PRIOR = {
    "obs_var": {"concentration": 3, "rate": 100},
    "mms": {"loc": 0, "scale": 50},
    "msa": {"loc": 0, "scale": 50},  # TODO: fix: this is temporary!
    "mms_loc": {"loc": 0, "scale": 50},
    "mms_var": {"concentration": 3, "rate": 100},
    "msa_loc": {"loc": 0, "scale": 50},
    "msa_var": {"concentration": 3, "rate": 100},
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
            # dist.HalfNormal(**prior.get("msa", DEFAULT_PRIOR["msa"])),  # TODO: Fix!
            # dist.TransformedDistribution(
            #     dist.Normal(**prior.get("msa", DEFAULT_PRIOR["msa"])),
            #     [dist.transforms.SoftplusTransform()],
            # ),
        )
        obs_loc = numpyro.deterministic(
            "field", simulate_field_from_csst(b_dir, v1, mms, msa)
        )
        with numpyro.plate("orient", norient):
            with handlers.mask(mask=mask_fg):
                numpyro.sample(
                    "obs", dist.StudentT(obs_df, obs_loc, jnp.sqrt(obs_var)), obs=obs
                )


def model_pooled_decentred_marginalised_unconstrained_msa(
    b_dir: Float[Array, "orient ndim"],
    v1: Float[Array, "*spatial ndim"],
    roi: Int[Array, " *spatial"],
    nroi: int,
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

    with numpyro.plate("roi", nroi):
        mms_loc = numpyro.sample(
            "mms_loc", dist.Normal(**prior.get("mms_loc", DEFAULT_PRIOR["mms_loc"]))
        )
        mms_var = numpyro.sample(
            "mms_var",
            dist.InverseGamma(**prior.get("mms_var", DEFAULT_PRIOR["mms_var"])),
        )
        msa_loc = numpyro.sample(
            "msa_loc", dist.Normal(**prior.get("msa_loc", DEFAULT_PRIOR["msa_loc"]))
        )
        msa_var = numpyro.sample(
            "msa_var",
            dist.InverseGamma(**prior.get("msa_var", DEFAULT_PRIOR["msa_var"])),
        )

    with numpyro.plate_stack("spatial", grid_shape):
        mms_eps = numpyro.sample("mms_eps", dist.Normal(0, 1))
        mms = numpyro.deterministic(
            "mms", mms_loc[roi] + jnp.sqrt(mms_var)[roi] * mms_eps
        )
        msa_eps = numpyro.sample("msa_eps", dist.Normal(0, 1))
        msa = numpyro.deterministic(
            "msa", msa_loc[roi] + jnp.sqrt(msa_var)[roi] * msa_eps
        )
        obs_loc = numpyro.deterministic(
            "field", simulate_field_from_csst(b_dir, v1, mms, msa)
        )
        with numpyro.plate("orient", norient):
            with handlers.mask(mask=mask_fg):
                numpyro.sample(
                    "obs", dist.StudentT(obs_df, obs_loc, jnp.sqrt(obs_var)), obs=obs
                )
