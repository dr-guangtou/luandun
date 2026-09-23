import pytest

from cross_check_fsps_tabular import run_cross_check

pytestmark = pytest.mark.slow


def test_integrator_agrees_with_fsps_tabular_within_two_percent():
    report = run_cross_check(epochs_gyr=(1.0, 3.0, 3.5, 5.0, 13.0))
    for epoch, entry in report["epochs"].items():
        for window, value in entry["max_relative_flux_difference"].items():
            assert value < 0.02, (epoch, window, value)
