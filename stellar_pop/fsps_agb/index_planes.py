"""Figure helpers for the three 2-D spectral-index planes."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

PLANES = (("d4000", "hdelta_a"), ("d4000", "h_minus_bump"), ("hdelta_a", "h_minus_bump"))
AXIS_LABELS = {
    "d4000": "D4000",
    "hdelta_a": r"H$\delta_A$ [$\AA$]",
    "h_minus_bump": r"H$^-$ bump [mag]",
}
INVERTED_AXES = ("h_minus_bump",)


def new_plane_figure(n_rows=1):
    figure, axes = plt.subplots(n_rows, 3, figsize=(13.5, 4.2 * n_rows), squeeze=False)
    for row in axes:
        for axis, (x_key, y_key) in zip(row, PLANES, strict=True):
            axis.set_xlabel(AXIS_LABELS[x_key])
            axis.set_ylabel(AXIS_LABELS[y_key])
            if y_key in INVERTED_AXES and not axis.yaxis_inverted():
                axis.invert_yaxis()
    figure.tight_layout()
    return figure, axes


def plot_track(axes, indices, color_values, color_label, cmap="viridis"):
    norm = plt.Normalize(np.min(color_values), np.max(color_values))
    mappable = None
    for axis, (x_key, y_key) in zip(axes, PLANES, strict=True):
        points = np.column_stack([indices[x_key], indices[y_key]])
        segments = np.stack([points[:-1], points[1:]], axis=1)
        collection = LineCollection(segments, cmap=cmap, norm=norm, linewidths=1.8)
        collection.set_array(np.asarray(color_values)[:-1])
        axis.add_collection(collection)
        axis.autoscale_view()
        mappable = collection
    axes[-1].figure.colorbar(mappable, ax=list(axes), label=color_label, pad=0.02)


def plot_population(axes, indices, color_values, color_label, track_indices=None, cmap="plasma"):
    mappable = None
    for axis, (x_key, y_key) in zip(axes, PLANES, strict=True):
        mappable = axis.scatter(
            indices[x_key],
            indices[y_key],
            c=color_values,
            s=2,
            alpha=0.35,
            cmap=cmap,
            linewidths=0,
            rasterized=True,
        )
        if track_indices is not None:
            axis.plot(
                track_indices[x_key],
                track_indices[y_key],
                color="black",
                lw=1.2,
                label="fiducial track",
            )
    axes[-1].figure.colorbar(mappable, ax=list(axes), label=color_label, pad=0.02)
    if track_indices is not None:
        axes[0].legend(frameon=False, loc="best")
