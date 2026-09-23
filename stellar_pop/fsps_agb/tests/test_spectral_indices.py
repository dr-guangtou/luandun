import numpy as np

from spectral_indices import (
    H_MINUS_BANDS_A,
    HDELTA_A_BANDS_A,
    air_to_vacuum,
    band_mean,
    d4000,
    h_minus_bump,
    hdelta_a,
    measure_all,
)


def test_air_to_vacuum_matches_fsps_value():
    # FSPS (vacairconv.f90, Morton 1991): 5000 A air -> 5001.394 A vacuum
    assert np.isclose(air_to_vacuum(np.array([5000.0]))[0], 5001.394, atol=0.002)


def test_band_mean_of_linear_flux_is_midpoint_value():
    wave = np.linspace(4000.0, 4300.0, 301)
    flux = 2.0 + 0.01 * wave
    assert np.isclose(band_mean(wave, flux, 4050.0, 4250.0), 2.0 + 0.01 * 4150.0, rtol=1e-10)


def test_band_mean_uses_exact_band_limits():
    wave = np.linspace(4000.0, 4300.0, 31)  # 10 A pixels, band edges fall between pixels
    flux = np.ones_like(wave)
    assert np.isclose(band_mean(wave, flux, 4053.0, 4247.0), 1.0, rtol=1e-12)


def test_indices_vanish_on_linear_spectrum(linear_spectrum):
    wave_a, flux_lambda = linear_spectrum
    flux_nu = flux_lambda * wave_a**2
    assert abs(hdelta_a(wave_a, flux_nu)) < 1e-10
    assert abs(h_minus_bump(wave_a, flux_nu)) < 1e-10


def test_d4000_of_flat_flux_nu_is_one():
    wave_a = np.linspace(3300.0, 4500.0, 5000)
    assert np.isclose(d4000(wave_a, np.ones_like(wave_a)), 1.0, rtol=1e-12)


def test_d4000_is_ratio_of_mean_flux_nu():
    wave_a = np.linspace(3300.0, 4500.0, 12001)
    flux_nu = np.where(wave_a < 4000.0, 1.0, 2.0)
    assert np.isclose(d4000(wave_a, flux_nu), 2.0, rtol=1e-6)


def test_hdelta_a_gaussian_absorption_has_positive_equivalent_width():
    wave_a = np.linspace(4000.0, 4200.0, 20001)
    center = air_to_vacuum(np.array([4101.7]))[0]
    depth, sigma = 0.5, 3.0
    flux_lambda = 1.0 - depth * np.exp(-0.5 * ((wave_a - center) / sigma) ** 2)
    expected = depth * sigma * np.sqrt(2 * np.pi)
    measured = hdelta_a(wave_a, flux_lambda * wave_a**2)
    assert np.isclose(measured, expected, rtol=1e-3)


def test_h_minus_bump_is_negative_for_a_bump():
    wave_a = np.linspace(14000.0, 19000.0, 5001)
    flux_lambda = 1.0 + 0.1 * np.exp(-0.5 * ((wave_a - 16500.0) / 200.0) ** 2)
    value = h_minus_bump(wave_a, flux_lambda * wave_a**2)
    assert value < 0
    expected = -2.5 * np.log10(band_mean(wave_a, flux_lambda, *H_MINUS_BANDS_A["feature"]))
    assert np.isclose(value, expected, rtol=1e-3)


def test_measure_all_broadcasts_over_spectra(linear_spectrum):
    wave_a, flux_lambda = linear_spectrum
    flux_nu = np.stack([flux_lambda * wave_a**2, 2 * flux_lambda * wave_a**2])
    result = measure_all(wave_a, flux_nu)
    assert result["d4000"].shape == (2,)
    assert np.allclose(result["hdelta_a"], 0.0, atol=1e-10)
    assert np.allclose(result["h_minus_bump"], 0.0, atol=1e-10)


def test_band_definitions_are_vacuum_and_ordered():
    for bands in (HDELTA_A_BANDS_A, H_MINUS_BANDS_A):
        blue, feature, red = bands["blue"], bands["feature"], bands["red"]
        assert blue[0] < blue[1] <= feature[0] < feature[1] <= red[0] < red[1]
    assert np.isclose(HDELTA_A_BANDS_A["feature"][1], air_to_vacuum(np.array([4122.25]))[0])
