import warnings
from typing import NamedTuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as onp
from matplotlib.colors import Normalize, TwoSlopeNorm


class Plot(NamedTuple):
    ax: plt.Axes
    im: mpl.image.AxesImage | None
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
            return Plot(ax, None, None)

    if norm is None:
        try:
            norm = TwoSlopeNorm(0, *vlims) if center else Normalize(*vlims)
        except ValueError:
            norm = Normalize(*vlims)

    if cmap is None:
        cmap = "RdBu_r" if isinstance(norm, TwoSlopeNorm) else "viridis"

    im = ax.imshow(
        x.T, origin="lower", cmap=cmap, norm=norm, interpolation="none", **kws
    )
    if cbar:
        cbar = plt.colorbar(im, ax=ax, **cbar_kws)
    else:
        cbar = None
    if not show_axis:
        ax.axis("off")
    return Plot(ax, im, cbar)
