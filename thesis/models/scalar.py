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
    """Simplest QSM model."""
    obs_var = numpyro.sample(
        "obs_var",
        dist.InverseGamma(
            **prior.get("obs_var", {"concentration": 3, "rate": 5e-3**2}),
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
            numpyro.sample("obs", dist.Normal(obs_loc, jnp.sqrt(obs_var)), obs=obs)


def model_unpooled_marginalised(
    grid_shape: tuple[int, ...],
    kernel_rft: Float[Array, " *rft"],
    mask_fg: Bool[Array, " *spatial"] = jnp.array(True),
    prior={},
    obs: Float[Array, " *spatial"] | None = None,
):
    """Simplest QSM model with the noise STD marginalised out."""

    prior_obs_var = prior.get("obs_var", {"concentration": 3, "rate": 5e-3**2})
    obs_df = 2 * prior_obs_var["concentration"]
    obs_var = prior_obs_var["rate"] / prior_obs_var["concentration"]

    with numpyro.plate_stack("spatial", grid_shape):
        x = numpyro.sample(
            "chi", dist.Normal(**prior.get("chi", {"loc": 0, "scale": 1e-2}))
        )
        obs_loc = numpyro.deterministic(
            "field", simulate_field_from_scalar_and_kernel(kernel_rft, x)
        )
        with handlers.mask(mask=mask_fg):
            numpyro.sample(
                "obs", dist.StudentT(obs_df, obs_loc, jnp.sqrt(obs_var)), obs=obs
            )


def model_pooled_decentred(
    grid_shape: tuple[int, ...],
    kernel_rft: Float[Array, " *rft"],
    roi: Int[Array, " *spatial"],
    nroi: int,
    mask_fg: Bool[Array, " *spatial"] = jnp.array(True),
    prior={},
    obs: Float[Array, " *spatial"] | None = None,
):
    obs_var = numpyro.sample(
        "obs_var",
        dist.InverseGamma(
            **prior.get("obs_var", {"concentration": 3, "rate": 5e-3**2}),
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
            numpyro.sample("obs", dist.Normal(obs_loc, jnp.sqrt(obs_var)), obs=obs)


def model_pooled_decentred_marginalised(
    grid_shape: tuple[int, ...],
    kernel_rft: Float[Array, " *rft"],
    roi: Int[Array, " *spatial"],
    nroi: int,
    mask_fg: Bool[Array, " *spatial"] = jnp.array(True),
    prior={},
    obs: Float[Array, " *spatial"] | None = None,
):
    prior_obs_var = prior.get("obs_var", {"concentration": 3, "rate": 5e-3**2})
    obs_df = 2 * prior_obs_var["concentration"]
    obs_var = prior_obs_var["rate"] / prior_obs_var["concentration"]

    with numpyro.plate("roi", nroi):
        loc = numpyro.sample(
            "chi_loc", dist.Normal(**prior.get("chi_loc", {"loc": 0, "scale": 5e-2}))
        )
        var = numpyro.sample(
            "chi_var",
            dist.InverseGamma(
                **prior.get("chi_var", {"concentration": 1, "rate": 5e-4})
            ),
        )

    with numpyro.plate_stack("spatial", grid_shape):
        x_eps = numpyro.sample("chi_eps", dist.Normal(0, 1))
        x = numpyro.deterministic("chi", loc[roi] + jnp.sqrt(var)[roi] * x_eps)
        obs_loc = numpyro.deterministic(
            "field", simulate_field_from_scalar_and_kernel(kernel_rft, x)
        )
        with handlers.mask(mask=mask_fg):
            numpyro.sample(
                "obs", dist.StudentT(obs_df, obs_loc, jnp.sqrt(obs_var)), obs=obs
            )


def model_pooled_fully_marginalised(
    grid_shape: tuple[int, ...],
    kernel_rft: Float[Array, " *rft"],
    roi: Int[Array, " *spatial"],
    nroi: int,
    mask_fg: Bool[Array, " *spatial"] = jnp.array(True),
    prior={},
    obs: Float[Array, " *spatial"] | None = None,
):
    """Dencentred parametrisation of a hierarchical model with ROI-variance marginalised."""
    prior_obs_var = prior.get("obs_var", {"concentration": 3, "rate": 5e-3**2})
    obs_df = 2 * prior_obs_var["concentration"]
    obs_var = prior_obs_var["rate"] / prior_obs_var["concentration"]

    prior_chi_var = prior.get("chi_var", {"concentration": 1, "rate": 5e-4})
    chi_df = 2 * prior_chi_var["concentration"]
    chi_var = prior_chi_var["rate"] / prior_chi_var["concentration"]

    with numpyro.plate("roi", nroi):
        loc = numpyro.sample(
            "chi_loc", dist.Normal(**prior.get("chi_loc", {"loc": 0, "scale": 5e-2}))
        )

    with numpyro.plate_stack("spatial", grid_shape):
        x_eps = numpyro.sample("chi_eps", dist.StudentT(chi_df, 0, 1))
        x = numpyro.deterministic("chi", loc[roi] + jnp.sqrt(chi_var) * x_eps)
        obs_loc = numpyro.deterministic(
            "field", simulate_field_from_scalar_and_kernel(kernel_rft, x)
        )
        with handlers.mask(mask=mask_fg):
            numpyro.sample(
                "obs", dist.StudentT(obs_df, obs_loc, jnp.sqrt(obs_var)), obs=obs
            )


def model_pooled_centred(
    grid_shape: tuple[int, ...],
    kernel_rft: Float[Array, " *rft"],
    roi: Int[Array, " *spatial"],
    nroi: int,
    mask_fg: Bool[Array, " *spatial"] = jnp.array(True),
    prior={},
    obs: Float[Array, " *spatial"] | None = None,
):
    obs_var = numpyro.sample(
        "obs_var",
        dist.InverseGamma(
            **prior.get("obs_var", {"concentration": 3, "rate": 5e-3**2}),
        ),
    )

    with numpyro.plate("roi", nroi):
        loc = numpyro.sample(
            "chi_loc", dist.Normal(**prior.get("chi_loc", {"loc": 0, "scale": 5e-2}))
        )
        var = numpyro.sample(
            "chi_scale",
            dist.InverseGamma(
                **prior.get("chi_var", {"concentration": 1, "rate": 5e-4})
            ),
        )

    with numpyro.plate_stack("spatial", grid_shape):
        x = numpyro.sample("chi", dist.Normal(loc[roi], jnp.sqrt(var)[roi]))
        obs_loc = numpyro.deterministic(
            "field", simulate_field_from_scalar_and_kernel(kernel_rft, x)
        )
        with handlers.mask(mask=mask_fg):
            numpyro.sample("obs", dist.Normal(obs_loc, jnp.sqrt(obs_var)), obs=obs)


def model_pooled_centred_marginalised(
    grid_shape: tuple[int, ...],
    kernel_rft: Float[Array, " *rft"],
    roi: Int[Array, " *spatial"],
    nroi: int,
    mask_fg: Bool[Array, " *spatial"] = jnp.array(True),
    prior={},
    obs: Float[Array, " *spatial"] | None = None,
):
    prior_obs_var = prior.get("obs_var", {"concentration": 3, "rate": 5e-3**2})
    obs_df = 2 * prior_obs_var["concentration"]
    obs_var = prior_obs_var["rate"] / prior_obs_var["concentration"]

    with numpyro.plate("roi", nroi):
        loc = numpyro.sample(
            "chi_loc", dist.Normal(**prior.get("chi_loc", {"loc": 0, "scale": 5e-2}))
        )
        var = numpyro.sample(
            "chi_scale",
            dist.InverseGamma(
                **prior.get("chi_var", {"concentration": 1, "rate": 5e-4})
            ),
        )

    with numpyro.plate_stack("spatial", grid_shape):
        x = numpyro.sample("chi", dist.Normal(loc[roi], jnp.sqrt(var)[roi]))
        obs_loc = numpyro.deterministic(
            "field", simulate_field_from_scalar_and_kernel(kernel_rft, x)
        )
        with handlers.mask(mask=mask_fg):
            numpyro.sample(
                "obs", dist.StudentT(obs_df, obs_loc, jnp.sqrt(obs_var)), obs=obs
            )


def model_rft_log_unpooled(
    grid_shape: tuple[int, ...],
    kernel_rft: Float[Array, " *rft"],
    mask_fg: Bool[Array, " *spatial"] = jnp.array(True),
    prior={},
    obs: Float[Array, " *spatial"] | None = None,
    ft_norm="backward",
):
    ndim = len(grid_shape)
    ft_axes = tuple(range(-ndim, 0))
    rft_shape = kernel_rft.shape

    @jit
    def _compute_all(x_ft_log_abs, x_ft_arg):
        x_ft = jnp.exp(x_ft_log_abs + 1j * x_ft_arg).at[(0,) * ndim].set(0.0)
        x = jnp.fft.irfftn(x_ft, axes=ft_axes, norm=ft_norm)
        field = jnp.fft.irfftn(kernel_rft * x_ft, axes=ft_axes, norm=ft_norm)
        return x_ft, x, field

    obs_var = numpyro.sample(
        "obs_var",
        dist.InverseGamma(
            **prior.get("obs_var", {"concentration": 3, "rate": 5e-3**2}),
        ),
    )

    with numpyro.plate_stack("rft", rft_shape):
        # Define the rest of the prior in the Fourier space
        _prior_ft_abs = prior.get("_ft_log_abs", {"loc": 0.0, "scale": 1.0})
        x_ft_log_abs = numpyro.sample(
            "_chi_ft_log_abs",
            dist.Normal(_prior_ft_abs["loc"], _prior_ft_abs["scale"]),
        )
        x_ft_arg = numpyro.sample("_ft_arg", dist.Uniform(low=-jnp.pi, high=jnp.pi))
        x_ft, x, field = _compute_all(x_ft_log_abs, x_ft_arg)
        numpyro.deterministic("chi_ft", x_ft)
    with numpyro.plate_stack("spatial", grid_shape):
        numpyro.deterministic("chi", x)
        obs_loc = numpyro.deterministic("field", field)
        with handlers.mask(mask=mask_fg):
            numpyro.sample("obs", dist.Normal(obs_loc, jnp.sqrt(obs_var)), obs=obs)
