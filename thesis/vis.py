import warnings
from typing import NamedTuple, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as onp
from matplotlib.colors import Normalize, TwoSlopeNorm


class Plot(NamedTuple):
    ax: plt.Axes
    im: mpl.image.AxesImage | None
    cbar: mpl.colorbar.Colorbar | None


class Plots(NamedTuple):
    axes: Iterable[plt.Axes]
    ims: Iterable[mpl.image.AxesImage] | None
    cbar: mpl.colorbar.Colorbar | None


def show_2d(
    x,
    ax=None,
    center: bool = True,
    cbar: bool = True,
    vlims: tuple[float, float] | None = None,
    cmap=None,
    norm=None,
    window_percentiles: tuple[float, float] = (0, 100),
    cbar_kws={},
    show_axis: bool = False,
    transpose_fn=lambda x: x.T,
    origin="lower",
    **kws,
):
    if ax is None:
        ax = plt.subplot()
    if vlims is None:
        vlims = onp.nanpercentile(x, window_percentiles)
        if onp.isnan(vlims).any():
            warnings.warn(
                f"{window_percentiles=} yield {vlims=}, aborting. "
                "Check your data or specify vlims manually"
            )
            if not show_axis:
                ax.axis("off")
            return Plot(ax, None, None)

    if norm is None:
        try:
            norm = TwoSlopeNorm(0, *vlims) if center else Normalize(*vlims)
        except ValueError:
            norm = Normalize(*vlims)

    if cmap is None:
        cmap = "RdBu_r" if isinstance(norm, TwoSlopeNorm) else "viridis"

    im = ax.imshow(
        transpose_fn(x),
        origin=origin,
        cmap=cmap,
        norm=norm,
        interpolation="none",
        **kws,
    )
    if cbar:
        cbar = plt.colorbar(im, ax=ax, **cbar_kws)
    else:
        cbar = None
    if not show_axis:
        ax.axis("off")
    return Plot(ax, im, cbar)


def show_projections(
    array,
    axes=None,
    loc: tuple[int, int, int] | None = None,
    b0=None,
    vlims: tuple[float, float] | None = None,
    window_percentiles: tuple[float, float] = (0, 100),
    cbar: bool = True,
    cbar_kws={},
    subplots_kws={},
):
    if not (array.ndim == 3 or (array.ndim == 4 and array.shape[-1] == 3)):
        raise NotImplementedError(
            "Currently can only show projections for 3D arrays (scalar or RGB)."
            f"Got {array.shape=}"
        )
    if axes is None:
        _, axes = plt.subplots(ncols=3, **subplots_kws)
    if loc is None:
        loc = tuple(int(sz / 2) for sz in array.shape)

    if vlims is None:
        vlims = onp.nanpercentile(array, window_percentiles)
        if onp.isnan(vlims).any():
            warnings.warn(
                f"{window_percentiles=} yield {vlims=}, aborting. "
                "Check your data or specify vlims manually"
            )
    ims = []
    for idx_ax, (ax, idx) in enumerate(zip(axes, loc)):
        _, _im, _ = show_2d(
            array.take(idx, axis=idx_ax),
            vlims=vlims,
            cbar=False,
            ax=ax,
            # Use leftmost axes instead of .T to support RGB arrays
            transpose_fn=lambda x: onp.swapaxes(x, 0, 1),
            origin="upper" if idx_ax == 2 else "lower",
        )
        assert _im is not None
        ims.append(_im)

    if cbar:
        cbar = plt.colorbar(_im, ax=axes, **cbar_kws)
    else:
        cbar = None

    # axes[0].imshow(array[loc[0]], origin="lower", **kwargs)
    # axes[1].imshow(np.swapaxes(array[:, loc[1]], 0, 1), **kwargs)
    # im = axes[2].imshow(
    #     np.swapaxes(array[:, :, loc[2]], 0, 1), origin="lower", **kwargs
    # )

    qvargs = dict(color="k", scale=7.5, width=0.02)
    if b0 is not None:
        axes[0].quiver(0.15, 0.15, b0[1], b0[2], transform=axes[0].transAxes, **qvargs)
        axes[1].quiver(0.15, 0.15, b0[0], b0[2], transform=axes[1].transAxes, **qvargs)
        axes[2].quiver(0.15, 0.15, b0[0], b0[1], transform=axes[2].transAxes, **qvargs)
    for ax, (x, y) in zip(axes, ["yz", "xy", "xz"]):
        ax.set_xlabel(x)
        ax.set_ylabel(y)
    for ax in axes.flat:
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )
    return Plots(axes, ims, cbar)


def show_2d_vector_field(
    vectors,
    ax=None,
    scale=20,
    pivot="mid",
    linewidth=3,
    transpose_fn=lambda x: x.T,
    show_every_nth: int | None = None,
):
    if ax is None:
        ax = plt.subplot()

    if vectors.shape[-1] != 2:
        raise ValueError("Can only handle a 2D vecotr field correctly")

    import jax.numpy as jnp  # onp leads to strange diagonals :shrug:

    show_at = (slice(None, None, show_every_nth),) * (vectors.ndim - 1)  # i.e. 2
    array = jnp.full_like(vectors, jnp.nan).at[show_at].set(vectors[show_at])
    mask = jnp.linalg.norm(vectors, axis=-1)
    array = jnp.where(mask[..., None], array, jnp.nan)

    ax.quiver(
        transpose_fn(array[..., 0]),
        transpose_fn(array[..., 1]),
        scale=scale,
        pivot=pivot,
        linewidth=linewidth,
    )
    ax.set_aspect(1)
