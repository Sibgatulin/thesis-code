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


class ModelUnpooled:
    def __init__(
        self,
        b_dir: Float[Array, "orient ndim"],
        v1: Float[Array, "*spatial ndim"],
        prior={},
        obs: Float[Array, "orient *spatial"] | None = None,
        store_deterministic_sites: bool = True,
        marginalise_obs_var: bool = True,
        msa_constr_fn=None,
    ):
        """Simplest CSST model.

        The prior on the local variables:

            mms(r) ~ N(0, σ_mms)
            msa(r) ~ N(0, σ_msa) # WRONG AND TEMPORARY

        if marginalise_obs_var == True:
            obs ~ StudentT(2α, f(mms, msa), √β/α)
        otherwise:
            σ²_obs ~ Γ^(-1)(α, β)
            obs ~ N(f(χ), σ_obs)

        This implementation is a functor only because of its reliance on
        mask_{fg,wm}_size in __call__. These sizes (sums) cannot be computed in
        jit-ted __call__ because they act as array sizes and thus must be known
        at the compile time.
        Presumably, I could have marked them as constant args.
        But I obviously have a weak spot for classes.
        """
        self.marginalise_obs_var = marginalise_obs_var
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
            frac_zero = jnp.isclose(obs[mask_finite], 0).mean()
            if frac_zero > 0.1:
                logger.warning(
                    f"Found {frac_zero:.0%} zeros in obs. "
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
        self.msa_constr_fn = msa_constr_fn

    def __call__(self):
        """Simplest CSST model.

        The prior on the local variables:

            mms(r) ~ N(0, σ_mms)
            msa(r) ~ N(0, σ_msa) # WRONG AND TEMPORARY

        if marginalise_obs_var == True:
            obs ~ StudentT(2α, f(mms, msa), √β/α)
        otherwise:
            σ²_obs ~ Γ^(-1)(α, β)
            obs ~ N(f(χ), σ_obs)
        """
        prior_obs_var = self.prior.get("obs_var", DEFAULT_PRIOR["obs_var"])
        if self.marginalise_obs_var:
            obs_df = 2 * prior_obs_var["concentration"]
            obs_var = prior_obs_var["rate"] / prior_obs_var["concentration"]

            # StudentT is the result of marginalisation of σ~Γ^{-1} and obs~N(..., σ)
            _likelihood_closure = lambda mu: dist.StudentT(
                obs_df, mu, jnp.sqrt(obs_var)
            )
        else:
            obs_var = numpyro.sample("obs_var", dist.InverseGamma(**prior_obs_var))
            _likelihood_closure = lambda mu: dist.Normal(mu, jnp.sqrt(obs_var))

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
        msa = (
            jnp.zeros(self.grid_shape)
            .at[self.mask_wm]
            .set(_msa if self.msa_constr_fn is None else self.msa_constr_fn(_msa))
        )
        # I am free to choose the reference mean susceptibility -> mean over the FG!
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
                    numpyro.sample("obs", _likelihood_closure(obs_loc), obs=self.obs)


class ModelPooled(ModelUnpooled):
    def __init__(
        self,
        b_dir: Float[Array, "orient ndim"],
        v1: Float[Array, "*spatial ndim"],
        roi: Int[Array, " *spatial"],
        prior={},
        obs: Float[Array, "orient *spatial"] | None = None,
        store_deterministic_sites: bool = True,
        marginalise_obs_var: bool = True,
        msa_constr_fn=None,
    ):
        super().__init__(
            b_dir=b_dir,
            v1=v1,
            prior=prior,
            obs=obs,
            store_deterministic_sites=store_deterministic_sites,
            marginalise_obs_var=marginalise_obs_var,
            msa_constr_fn=msa_constr_fn,
        )
        assert roi.shape == self.grid_shape, f"{roi.shape=}!={self.grid_shape=}"
        self.roi = roi

        labels = jnp.unique(roi)
        self.nroi = labels.size
        assert jnp.all(labels == jnp.arange(self.nroi)), (
            f"Provided labels run {labels}. Use a consecutive sequence instead"
        )

        # Additional complexity: some ROIs may not belong to WM.
        # I choose to exclude them from MSA estimation (i.e. null their global MSA parameters)
        # but this may well be an inapropriate assumption
        labels_wm = set(jnp.unique(roi[self.mask_wm]).tolist())
        labels_non_wm = set(jnp.unique(roi[~self.mask_wm]).tolist())
        labels_undecided = labels_wm.intersection(labels_non_wm)
        if labels_undecided:
            logger.warning(
                f"The following ROIs seem to be both in and outside of the WM: {labels_undecided}"
            )
        labels_non_msa = set(labels.tolist()).difference(labels_wm)
        if labels_non_msa:
            logger.info(
                f"MSA in the following ROIs will be suppressed: {labels_non_msa}"
            )
        self.labels_non_msa = jnp.array(list(labels_non_msa), dtype=int)
        self.labels_msa_mask = jnp.array(
            [i not in labels_non_msa for i in range(self.nroi)], dtype=bool
        )

    def __call__(self):
        """Pooled CSST model.

        The prior on the global variables per ROI:
            μ_{mms,msa} ~ N # WRONG AND TEMP FOR MSA
            σ_{mms,msa} ~ Γ^(-1)

        The prior on the local variables:
            mms(r) ~ μ_mms[roi] + N(0, 1) * σ_mms[roi]
            msa(r) ~ μ_msa[roi] + N(0, 1) * σ_msa[roi] # WRONG AND TEMPORARY

        if marginalise_obs_var == True:
            obs ~ StudentT(2α, f(mms, msa), √β/α)
        otherwise:
            σ²_obs ~ Γ^(-1)(α, β)
            obs ~ N(f(χ), σ_obs)
        """
        prior_obs_var = self.prior.get("obs_var", DEFAULT_PRIOR["obs_var"])
        if self.marginalise_obs_var:
            obs_df = 2 * prior_obs_var["concentration"]
            obs_var = prior_obs_var["rate"] / prior_obs_var["concentration"]

            # StudentT is the result of marginalisation of σ~Γ^{-1} and obs~N(..., σ)
            _likelihood_closure = lambda mu: dist.StudentT(
                obs_df, mu, jnp.sqrt(obs_var)
            )
        else:
            obs_var = numpyro.sample("obs_var", dist.InverseGamma(**prior_obs_var))
            _likelihood_closure = lambda mu: dist.Normal(mu, jnp.sqrt(obs_var))

        # Start with the ROI-level sites
        with numpyro.plate("roi", self.nroi):
            # somehow if I mask as done in the commented out code, the sampler fails pretty badly
            # with handlers.mask(mask=jnp.ones(self.nroi, dtype=bool).at[0].set(False)):
            mms_loc = numpyro.sample(
                "mms_loc",
                dist.Normal(**self.prior.get("mms_loc", DEFAULT_PRIOR["mms_loc"])),
            )
            mms_var = numpyro.sample(
                "mms_var",
                dist.InverseGamma(
                    **self.prior.get("mms_var", DEFAULT_PRIOR["mms_var"])
                ),
            )
            # Not 100% sure this is a good move
            msa_loc = numpyro.sample(
                "msa_loc",
                dist.Normal(**self.prior.get("msa_loc", DEFAULT_PRIOR["msa_loc"])),
            )
            msa_var = numpyro.sample(
                "msa_var",
                dist.InverseGamma(
                    **self.prior.get("msa_var", DEFAULT_PRIOR["msa_var"])
                ),
            )

        # Sample as little as possible: for both unknown maps only within their
        # corresponding domains:
        # ...For MMS I would typically assume it to be the same as that of observed field
        with numpyro.plate("voxel_fg", self.mask_fg_size):
            _mms_eps = numpyro.sample("_mms_eps", dist.Normal(0, 1))
        # ...For MSA I *CURRENTLY* assume it to be limited to the WM mask
        with numpyro.plate("voxel_wm", self.mask_wm_size):
            _msa_eps = numpyro.sample("_msa_eps", dist.Normal(0, 1))

        # Attribute the sampled vectors of values to their respective maps
        mms = mms_loc.at[0].set(0.0)[self.roi] + jnp.sqrt(mms_var).at[0].set(0.0)[
            self.roi
        ] * jnp.zeros(self.grid_shape).at[self.mask_fg].set(_mms_eps)
        # I am free to choose the reference mean susceptibility:
        # for consistency with the parent class, I'll compute the mean over the FG
        # and subtract it from the FG only, leaving all else 0.0
        mms = jnp.where(self.mask_fg, mms - mms[self.mask_fg].mean(), 0.0)
        # TODO: optimise. This above made the model noticeably slower.

        # With MSA also null those ROIs outside of WM
        msa = msa_loc.at[self.labels_non_msa].set(0.0)[self.roi] + jnp.sqrt(msa_var).at[
            self.labels_non_msa
        ].set(0.0)[self.roi] * jnp.zeros(self.grid_shape).at[self.mask_wm].set(_msa_eps)
        if self.msa_constr_fn is not None:
            msa = jnp.where(self.mask_wm, self.msa_constr_fn(msa), 0.0)

        # The rest is the same as in the parent class
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
                    numpyro.sample("obs", _likelihood_closure(obs_loc), obs=self.obs)


class ModelUnpooledMvNUnconstrainedMSA(ModelUnpooled):
    def __init__(
        self,
        b_dir: Float[Array, "orient ndim"],
        v1: Float[Array, "*spatial ndim"],
        prior={},
        obs: Float[Array, "orient *spatial"] | None = None,
        store_deterministic_sites: bool = True,
        marginalise_obs_var: bool = True,
    ):
        super().__init__(
            b_dir=b_dir,
            v1=v1,
            prior=prior,
            obs=obs,
            store_deterministic_sites=store_deterministic_sites,
            marginalise_obs_var=marginalise_obs_var,
        )

    def __call__(self):
        """Simplest CSST model with latent MvN

        The prior on the local variables:

            mms(r), msa(r) ~ MvN(0, Σ)

        if marginalise_obs_var == True:
            obs ~ StudentT(2α, f(mms, msa), √β/α)
        otherwise:
            σ²_obs ~ Γ^(-1)(α, β)
            obs ~ N(f(χ), σ_obs)
        """
        prior_obs_var = self.prior.get("obs_var", DEFAULT_PRIOR["obs_var"])
        if self.marginalise_obs_var:
            obs_df = 2 * prior_obs_var["concentration"]
            obs_var = prior_obs_var["rate"] / prior_obs_var["concentration"]

            # StudentT is the result of marginalisation of σ~Γ^{-1} and obs~N(..., σ)
            _likelihood_closure = lambda mu: dist.StudentT(
                obs_df, mu, jnp.sqrt(obs_var)
            )
        else:
            obs_var = numpyro.sample("obs_var", dist.InverseGamma(**prior_obs_var))
            _likelihood_closure = lambda mu: dist.Normal(mu, jnp.sqrt(obs_var))

        _latent_prior = self.prior.get("_mms_msa", DEFAULT_PRIOR["_mms_msa"])
        rho_tril = numpyro.sample(
            "_mms_msa_rho_tril",
            dist.LKJCholesky(2, **DEFAULT_PRIOR["_mms_msa_rho_tril"]),
        )
        cov_tril = _latent_prior["scale"][..., None] * rho_tril
        # Sample only within the FG
        with numpyro.plate("voxel_fg", self.mask_fg_size):
            _latent = numpyro.sample(
                "_mms_msa",
                dist.MultivariateNormal(_latent_prior["loc"], scale_tril=cov_tril),
            )

        # Attribute the sampled vectors of values to their respective maps
        msa = jnp.zeros(self.grid_shape).at[self.mask_fg].set(_latent[..., 1])
        # I am free to choose the reference mean susceptibility -> mean over the FG!
        mms = (
            jnp.zeros(self.grid_shape)
            .at[self.mask_fg]
            .set(_latent[..., 0] - _latent[..., 0].mean())
        )

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
                    numpyro.sample("obs", _likelihood_closure(obs_loc), obs=self.obs)
