"""Probabilistic models for Cylindrically Symmetric Susceptibility Tensor."""

from loguru import logger
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


class ModelUnpooledMarginalisedUnconstrainedMSA:
    def __init__(
        self,
        b_dir: Float[Array, "orient ndim"],
        v1: Float[Array, "*spatial ndim"],
        prior={},
        obs: Float[Array, "orient *spatial"] | None = None,
        store_deterministic_sites: bool = True,
    ):
        """Simplest CSST model with the noise STD marginalised out.

        mms(r) ~ N(0, σ_mms)
        msa(r) ~ N(0, σ_msa) # WRONG AND TEMPORARY
        σ²_obs ~ Γ^(-1)(α, β)
        obs ~ StudentT(2α, f(mms, msa), √β/α)

        This implementation is a functor only because of its reliance on
        mask_{fg,wm}_size in __call__. These sizes (sums) cannot be computed in
        jit-ted __call__ because they act as array sizes and thus must be known
        at the compile time.
        Presumably, I could have marked them as constant args.
        But I obviously have a weak spot for classes.
        """
        self.b_dir = jnp.atleast_2d(b_dir)
        self.norient, self.ndim = b_dir.shape
        self.grid_shape = v1.shape[:-1]
        assert len(self.grid_shape) == self.ndim and v1.shape[-1] == self.ndim, (
            f"Inconsistent dimensionality {v1.shape=} & {self.b_dir.shape=}"
        )
        self.v1 = v1

        self.store_deterministic_sites = store_deterministic_sites
        self.prior = prior

        # Process the observations
        # 1. Identify the expected shape
        obs_shape = (self.norient,) + self.grid_shape
        # 2. If obs are given, validate the shape
        if obs is not None:
            assert obs.shape == obs_shape, (
                f"{obs.shape=} != ({self.norient=},) + {self.grid_shape=}"
            )
        # 3. Generate the mask for obs:
        if obs is None:
            self.mask_obs = jnp.ones(obs_shape, dtype=bool)
            # it could also be jnp.array(True), but I wish to have a predictable and fixed shape.
            # May be avoided

        else:
            # If obs is given, check where it is masked explicitly
            mask_finite = jnp.isfinite(obs)
            if not mask_finite.all():
                # Assume intentional masking
                logger.info(
                    f"Generating mask_obs from the field. Found {mask_finite.mean():.0%} finite values"
                )
            else:
                logger.warning(
                    "All observed values are finite. "
                    "Will be inverting the field in the entire FoV. Is it a simulation?"
                )
            self.mask_obs = mask_finite

            # This is a safety net for myself as I used to mask the field with zeros, not NaNs
            mask_nonzero = abs(obs) > 0
            frac_nonzero = mask_nonzero.mean()
            if frac_nonzero < 0.9:
                logger.warning(
                    f"Found {1 - frac_nonzero:.0%} zeros in obs. "
                    "Consider masking with NaNs or pass mask_obs"
                )
        self.obs = obs

        # Define domains for the unknown susceptibilities
        # 1. MSA currently assumed to be defined in the WM
        # 1.1 WM is currently assumed to be identified by the DTI eigenvectors
        self.mask_wm = jnp.linalg.norm(v1, axis=-1) > 0
        self.mask_wm_size: int = self.mask_wm.sum().item()
        # 2. MMS is assumed to be defined everywhere, where field is observed
        # Furthermore, I take all voxels which are observed in at least 1 orientation
        # (alternatively, I could limit to all voxels that are fully observed at all
        # orientations)
        self.mask_fg = jnp.any(self.mask_obs, axis=0)  # rely on mask_obs.ndim = 1+ndim
        self.mask_fg_size: int = self.mask_fg.sum().item()

    def __call__(self):
        """Simplest CSST model with the noise STD marginalised out.

        mms(r) ~ N(0, σ_mms)
        msa(r) ~ N(0, σ_msa) # WRONG AND TEMPORARY
        σ²_obs ~ Γ^(-1)(α, β)
        obs ~ StudentT(2α, f(mms, msa), √β/α)
        """
        prior_obs_var = self.prior.get("obs_var", DEFAULT_PRIOR["obs_var"])
        obs_df = 2 * prior_obs_var["concentration"]
        obs_var = prior_obs_var["rate"] / prior_obs_var["concentration"]

        # Sample as little as possible: for both unknown maps only within their
        # corresponding domains:
        # ...For MMS I would typically assume it to be the same as that of observed field
        with numpyro.plate("voxel_fg", self.mask_fg_size):
            _mms = numpyro.sample(
                "_mms", dist.Normal(**self.prior.get("mms", DEFAULT_PRIOR["mms"]))
            )
        # ...For MSA I *CURRENTLY* assume it to be limited to the WM mask
        with numpyro.plate("voxel_wm", self.mask_wm_size):
            _msa = numpyro.sample(
                "_msa",
                dist.Normal(
                    **self.prior.get("msa", DEFAULT_PRIOR["msa"])
                ),  # TODO: Fix!
            )

        # Attribute the sampled vectors of values to their respective maps
        msa = jnp.zeros(self.grid_shape).at[self.mask_wm].set(_msa)
        # I am free to choose the reference mean susceptibility
        mms = jnp.zeros(self.grid_shape).at[self.mask_fg].set(_mms - _mms.mean())

        with numpyro.plate_stack("spatial", self.grid_shape):
            # it's nice to have the correctly shaped and referenced samples available
            # out of the box but I want to be able to run MCMC when it gets tight
            if self.store_deterministic_sites:
                numpyro.deterministic("mms", mms)
                numpyro.deterministic("msa", msa)

            with numpyro.plate("orient", self.norient):
                obs_loc = numpyro.deterministic(
                    "field",
                    # NOTE: I wonder if using MMS and not chi_perp adds to the collinearity of the problem
                    # I mean to say, I wonder if chi_perp and MSA would be easier to sampler
                    # from than MMS and MSA
                    simulate_field_from_csst(self.b_dir, self.v1, mms, msa),
                )

                # Mask to avoid trying to fit the field outside of the (object unless desired)
                # see the docstring of `_mask_obs`
                with handlers.mask(mask=self.mask_obs):
                    numpyro.sample(
                        "obs",
                        # StudentT is the result of marginalisation of σ~Γ^{-1} and obs~N(..., σ)
                        dist.StudentT(obs_df, obs_loc, jnp.sqrt(obs_var)),
                        obs=self.obs,
                    )


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
    mask_wm = jnp.linalg.norm(v1, axis=-1) > 0

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
            "msa", (msa_loc[roi] + jnp.sqrt(msa_var)[roi] * msa_eps) * mask_wm
        )
        with numpyro.plate("orient", norient):
            obs_loc = numpyro.deterministic(
                "field", simulate_field_from_csst(b_dir, v1, mms, msa)
            )
            with handlers.mask(mask=mask_fg):
                numpyro.sample(
                    "obs", dist.StudentT(obs_df, obs_loc, jnp.sqrt(obs_var)), obs=obs
                )
