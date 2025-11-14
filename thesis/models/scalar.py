"""Simplest models from scratch for my sanity."""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jaxtyping import Array, Float, Bool
from jax import jit
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
        dist.TruncatedNormal(
            **prior.get("obs_scale", {"loc": 5e-3, "scale": 5e-3}),
            low=0.0,
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
