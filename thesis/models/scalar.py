"""Simplest models from scratch for my sanity."""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import jit
from jaxtyping import Array, Bool, Float, Int
from numpyro import handlers

from thesis.forward import simulate_field_from_scalar_and_kernel

DEFAULT_PRIOR = {
    "obs_var": {"concentration": 3, "rate": 100},
    "chi": {"loc": 0, "scale": 50},
    "chi_loc": {"loc": 0, "scale": 50},
    "chi_scale": {"loc": 20, "scale": 50},
    "chi_var": {"loc": 3, "scale": 100},  # NOTE: not the same as before (1, 5e-4)
    "_ft_log_abs": {"loc": 0.0, "scale": 1.0},
}


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
        dist.InverseGamma(**prior.get("obs_var", DEFAULT_PRIOR["obs_var"])),
    )
    with numpyro.plate_stack("spatial", grid_shape):
        x = numpyro.sample("chi", dist.Normal(**prior.get("chi", DEFAULT_PRIOR["chi"])))
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

    prior_obs_var = prior.get("obs_var", DEFAULT_PRIOR["obs_var"])
    obs_df = 2 * prior_obs_var["concentration"]
    obs_var = prior_obs_var["rate"] / prior_obs_var["concentration"]

    with numpyro.plate_stack("spatial", grid_shape):
        x = numpyro.sample("chi", dist.Normal(**prior.get("chi", DEFAULT_PRIOR["chi"])))
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
            **prior.get("obs_var", DEFAULT_PRIOR["obs_var"]),
        ),
    )

    with numpyro.plate("roi", nroi):
        loc = numpyro.sample(
            "chi_loc", dist.Normal(**prior.get("chi_loc", DEFAULT_PRIOR["chi_loc"]))
        )
        scale = numpyro.sample(
            "chi_scale",
            dist.TruncatedNormal(  # TODO: consider switching over to InvGamma
                **prior.get("chi_scale", DEFAULT_PRIOR["chi_scale"]), low=0.0
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
    prior_obs_var = prior.get("obs_var", DEFAULT_PRIOR["obs_var"])
    obs_df = 2 * prior_obs_var["concentration"]
    obs_var = prior_obs_var["rate"] / prior_obs_var["concentration"]

    with numpyro.plate("roi", nroi):
        loc = numpyro.sample(
            "chi_loc", dist.Normal(**prior.get("chi_loc", DEFAULT_PRIOR["chi_loc"]))
        )
        var = numpyro.sample(
            "chi_var",
            dist.InverseGamma(**prior.get("chi_var", DEFAULT_PRIOR["chi_var"])),
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
    prior_obs_var = prior.get("obs_var", DEFAULT_PRIOR["obs_var"])
    obs_df = 2 * prior_obs_var["concentration"]
    obs_var = prior_obs_var["rate"] / prior_obs_var["concentration"]

    prior_chi_var = prior.get("chi_var", DEFAULT_PRIOR["chi_var"])
    chi_df = 2 * prior_chi_var["concentration"]
    chi_var = prior_chi_var["rate"] / prior_chi_var["concentration"]

    with numpyro.plate("roi", nroi):
        loc = numpyro.sample(
            "chi_loc", dist.Normal(**prior.get("chi_loc", DEFAULT_PRIOR["chi_loc"]))
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
            **prior.get("obs_var", DEFAULT_PRIOR["obs_var"]),
        ),
    )

    with numpyro.plate("roi", nroi):
        loc = numpyro.sample(
            "chi_loc", dist.Normal(**prior.get("chi_loc", DEFAULT_PRIOR["chi_loc"]))
        )
        var = numpyro.sample(
            "chi_var",
            dist.InverseGamma(**prior.get("chi_var", DEFAULT_PRIOR["chi_var"])),
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
    prior_obs_var = prior.get("obs_var", DEFAULT_PRIOR["obs_var"])
    obs_df = 2 * prior_obs_var["concentration"]
    obs_var = prior_obs_var["rate"] / prior_obs_var["concentration"]

    with numpyro.plate("roi", nroi):
        loc = numpyro.sample(
            "chi_loc", dist.Normal(**prior.get("chi_loc", DEFAULT_PRIOR["chi_loc"]))
        )
        var = numpyro.sample(
            "chi_scale",
            dist.InverseGamma(**prior.get("chi_var", DEFAULT_PRIOR["chi_var"])),
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
            **prior.get("obs_var", DEFAULT_PRIOR["obs_var"]),
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


def model_rft_log_pooled_decentred(
    grid_shape: tuple[int, ...],
    kernel_rft: Float[Array, " *rft"],
    roi: Int[Array, " *spatial"],
    nroi: int,
    mask_fg: Bool[Array, " *spatial"] = jnp.array(True),
    prior={},
    obs: Float[Array, " *spatial"] | None = None,
    ft_norm="backward",
):
    # This model is refactored to be decentred, and works VERY well
    # See https://gemini.google.com/share/6b03e6d86dbf
    ndim = len(grid_shape)
    ft_axes = tuple(range(-ndim, 0))
    rft_shape = kernel_rft.shape

    obs_var = numpyro.sample(
        "obs_var",
        dist.InverseGamma(**prior.get("obs_var", DEFAULT_PRIOR["obs_var"])),
    )

    with numpyro.plate("roi", nroi):
        loc = numpyro.sample(
            "chi_loc", dist.Normal(**prior.get("chi_loc", DEFAULT_PRIOR["chi_loc"]))
        )
        var = numpyro.sample(
            "chi_var",
            dist.InverseGamma(**prior.get("chi_var", DEFAULT_PRIOR["chi_var"])),
        )
    with numpyro.plate_stack("spatial", grid_shape):
        x_raw = numpyro.sample("chi_raw", dist.Normal(0.0, 1.0))

    # 2. Define `chi` deterministically using the non-centered trick.
    # This `chi` is the same as the original `x`.
    # This step breaks the funnel between `chi` and its hyperparameters.
    x = numpyro.deterministic("chi", loc[roi] + jnp.sqrt(var)[roi] * x_raw)

    # 3. Compute the Fourier transform of our new `chi`
    x_ft = numpyro.deterministic(
        "chi_ft", jnp.fft.rfftn(x, axes=ft_axes, norm=ft_norm).at[(0,) * ndim].set(0.0)
    )
    # 4. Compute the log-abs and angle from `chi_ft`
    # We add a small epsilon for numerical stability, to avoid log(0)
    eps = jnp.finfo(x_ft.dtype).eps
    x_ft_abs = jnp.abs(x_ft)
    x_ft_log_abs = jnp.log(x_ft_abs + eps)
    x_ft_arg = jnp.angle(x_ft)

    # 5. Apply the *original* Fourier priors as "factors"
    # Instead of sampling from them, we "observe" our computed values.
    # This now constrains `chi_raw` (our base latent) to follow
    # the Fourier-space statistics.
    with numpyro.plate_stack("rft", rft_shape):
        # Create a mask to handle the DC component
        # The original model zeroed the DC component *after* sampling,
        # so we will *not* apply the prior at the DC index.
        dc_mask = jnp.ones(rft_shape, dtype=bool).at[(0,) * ndim].set(False)

        with handlers.mask(mask=dc_mask):
            _prior_ft_abs = prior.get("_ft_log_abs", DEFAULT_PRIOR["_ft_log_abs"])
            numpyro.sample(
                "_chi_ft_log_abs",
                dist.Normal(_prior_ft_abs["loc"], _prior_ft_abs["scale"]),
                obs=x_ft_log_abs,
            )
            x_ft_arg = numpyro.sample(
                "_ft_arg",
                dist.Uniform(low=-jnp.pi, high=jnp.pi),
                obs=x_ft_arg,
            )

    with numpyro.plate_stack("spatial", grid_shape):
        # numpyro.deterministic("chi", jnp.fft.irfftn(x_ft, axes=ft_axes, norm=ft_norm))
        obs_loc = numpyro.deterministic(
            "field", jnp.fft.irfftn(kernel_rft * x_ft, axes=ft_axes, norm=ft_norm)
        )
        with handlers.mask(mask=mask_fg):
            numpyro.sample("obs", dist.Normal(obs_loc, jnp.sqrt(obs_var)), obs=obs)
