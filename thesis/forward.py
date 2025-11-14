import jax.numpy as jnp
from jax import jit
from jaxtyping import Array, Float


@jit
def simulate_field_from_scalar_and_kernel(
    kernel_rft: Float[Array, " *rft"],
    chi_iso: Float[Array, " *spatial"],
    norm="backward",
) -> Float[Array, " *spatial"]:
    axes = tuple(range(-chi_iso.ndim, 0))
    return jnp.fft.irfftn(
        kernel_rft * jnp.fft.rfftn(chi_iso, norm=norm, axes=axes), axes=axes, norm=norm
    )
