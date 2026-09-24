import pytest

from cross_check_fsps_families import run_cross_check, verify_sf_slope_sign

pytestmark = pytest.mark.slow


def test_sf_slope_negative_gives_a_declining_ramp():
    check = verify_sf_slope_sign(3.0, 0.3)
    assert check["declining"]
    assert check["ratio"] < 1.0


def test_linear_and_truncation_families_agree_with_fsps_within_two_percent(tmp_path):
    report = run_cross_check(
        epochs_gyr=(1.0, 3.0, 3.5, 5.0, 13.0), output_path=tmp_path / "cross_check.json"
    )
    assert report["sf_slope_sign_check"]["declining"]
    for family, family_report in report["families"].items():
        for epoch, entry in family_report["epochs"].items():
            for window, value in entry["max_relative_flux_difference"].items():
                assert value < 0.02, (family, epoch, window, value)
