"""Simplest models from scratch for my sanity."""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import jit
from jaxtyping import Array, Bool, Float, Int
from numpyro import handlers

from thesis.forward import simulate_field_from_scalar_and_kernel


def model_unpooled(
    grid_shape: tuple[int, ...],
    kernel_rft: Float[Array, " *rft"],
    mask_fg: Bool[Array, " *spatial"] = jnp.array(True),
    prior={},
    obs: Float[Array, " *spatial"] | None = None,
):
    obs_scale = numpyro.sample(
        "obs_scale",
        dist.InverseGamma(
            **prior.get("obs_scale", {"concentration": 3, "rate": 5e-3}),
        ),
    )
    with numpyro.plate_stack("spatial", grid_shape):
        x = numpyro.sample(
            "chi", dist.Normal(**prior.get("chi", {"loc": 0, "scale": 1e-2}))
        )
        obs_loc = numpyro.deterministic(
            "field", simulate_field_from_scalar_and_kernel(kernel_rft, x)
        )
        with handlers.mask(mask=mask_fg):
            numpyro.sample("obs", dist.Normal(obs_loc, obs_scale), obs=obs)


def model_pooled_decentred(
    grid_shape: tuple[int, ...],
    kernel_rft: Float[Array, " *rft"],
    roi: Int[Array, " *spatial"],
    nroi: int,
    mask_fg: Bool[Array, " *spatial"] = jnp.array(True),
    prior={},
    obs: Float[Array, " *spatial"] | None = None,
):
    obs_scale = numpyro.sample(
        "obs_scale",
        dist.InverseGamma(
            **prior.get("obs_scale", {"concentration": 3, "rate": 5e-3}),
        ),
    )

    with numpyro.plate("roi", nroi):
        loc = numpyro.sample(
            "chi_loc", dist.Normal(**prior.get("chi_loc", {"loc": 0, "scale": 5e-2}))
        )
        scale = numpyro.sample(
            "chi_scale",
            dist.TruncatedNormal(
                **prior.get("chi_scale", {"loc": 2e-2, "scale": 5e-2}), low=0.0
            ),
        )

    with numpyro.plate_stack("spatial", grid_shape):
        x_eps = numpyro.sample("chi_eps", dist.Normal(0, 1))
        x = numpyro.deterministic("chi", loc[roi] + scale[roi] * x_eps)
        obs_loc = numpyro.deterministic(
            "field", simulate_field_from_scalar_and_kernel(kernel_rft, x)
        )
        with handlers.mask(mask=mask_fg):
            numpyro.sample("obs", dist.Normal(obs_loc, obs_scale), obs=obs)


def model_pooled_centred(
    grid_shape: tuple[int, ...],
    kernel_rft: Float[Array, " *rft"],
    roi: Int[Array, " *spatial"],
    nroi: int,
    mask_fg: Bool[Array, " *spatial"] = jnp.array(True),
    prior={},
    obs: Float[Array, " *spatial"] | None = None,
):
    obs_scale = numpyro.sample(
        "obs_scale",
        dist.InverseGamma(
            **prior.get("obs_scale", {"concentration": 3, "rate": 5e-3}),
        ),
    )

    with numpyro.plate("roi", nroi):
        loc = numpyro.sample(
            "chi_loc", dist.Normal(**prior.get("chi_loc", {"loc": 0, "scale": 5e-2}))
        )
        scale = numpyro.sample(
            "chi_scale",
            dist.TruncatedNormal(
                **prior.get("chi_scale", {"loc": 2e-2, "scale": 5e-2}), low=0.0
            ),
        )

    with numpyro.plate_stack("spatial", grid_shape):
        x = numpyro.sample("chi", dist.Normal(loc[roi], scale[roi]))
        obs_loc = numpyro.deterministic(
            "field", simulate_field_from_scalar_and_kernel(kernel_rft, x)
        )
        with handlers.mask(mask=mask_fg):
            numpyro.sample("obs", dist.Normal(obs_loc, obs_scale), obs=obs)
