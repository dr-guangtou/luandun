import matplotlib

matplotlib.use("Agg")
import numpy as np

from index_planes import PLANES, new_plane_figure, plot_population, plot_track


def test_planes_and_track_figure(tmp_path):
    indices = {
        "d4000": np.linspace(1.2, 1.9, 20),
        "hdelta_a": np.linspace(8, -1, 20),
        "h_minus_bump": np.linspace(-0.05, -0.1, 20),
    }
    figure, axes = new_plane_figure(n_rows=2)
    assert axes.shape == (2, 3)
    plot_track(axes[0], indices, np.arange(20), "time [Gyr]")
    plot_population(axes[1], indices, np.arange(20), "t_obs - t_q [Gyr]", track_indices=indices)
    assert axes[0, 1].yaxis_inverted()
    figure.savefig(tmp_path / "planes.png")
    assert (tmp_path / "planes.png").stat().st_size > 0
    assert len(PLANES) == 3
