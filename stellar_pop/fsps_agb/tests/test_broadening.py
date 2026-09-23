import numpy as np

from broadening import (
    SPEED_OF_LIGHT_KM_S,
    added_sigma_km_s,
    gaussian_broaden,
    log_wavelength_grid,
    make_resolution_product,
    resample_flux,
    target_sigma_km_s,
)
from ssp_grid import SspGrid


def test_log_wavelength_grid_has_constant_velocity_step():
    wave = log_wavelength_grid(3400.0, 22000.0, 30.0)
    steps = np.diff(np.log(wave)) * SPEED_OF_LIGHT_KM_S
    assert np.allclose(steps, 30.0, rtol=1e-9)
    assert wave[0] >= 3400.0 - 1e-6 and wave[-1] <= 22000.0


def test_resample_flux_preserves_linear_spectrum():
    wave = np.linspace(3400.0, 22000.0, 3000)
    flux = np.stack([1.0 + 1e-4 * wave, 2.0 - 5e-5 * wave])
    target = log_wavelength_grid(3500.0, 21000.0)
    out = resample_flux(wave, flux, target)
    assert out.shape == (2, target.size)
    assert np.allclose(out[0], 1.0 + 1e-4 * target, rtol=1e-12)


def test_gaussian_broaden_recovers_quadrature_sum():
    wave = log_wavelength_grid(15000.0, 18000.0, 10.0)
    center = 16500.0
    sigma_line = 40.0
    velocity = SPEED_OF_LIGHT_KM_S * np.log(wave / center)
    flux_lambda = 1.0 - 0.5 * np.exp(-0.5 * (velocity / sigma_line) ** 2)
    flux_nu = flux_lambda * wave**2
    broadened = gaussian_broaden(wave, flux_nu[None, :], 300.0)[0] / wave**2
    depth = 1.0 - broadened
    window = np.abs(velocity) <= 3000.0
    second_moment = np.sum(depth[window] * velocity[window] ** 2) / np.sum(depth[window])
    recovered = np.sqrt(second_moment)
    expected = np.hypot(sigma_line, 300.0)
    assert abs(recovered / expected - 1.0) < 0.01


def test_gaussian_broaden_conserves_flux_per_log_wavelength():
    wave = log_wavelength_grid(15000.0, 18000.0, 10.0)
    flux_nu = np.exp(-0.5 * ((wave - 16500.0) / 100.0) ** 2)[None, :] / wave
    broadened = gaussian_broaden(wave, flux_nu, 500.0)
    assert np.isclose(np.sum(broadened[0] / wave), np.sum(flux_nu[0] / wave), rtol=1e-6)


def test_added_sigma_is_quadrature_difference():
    assert np.isclose(added_sigma_km_s(300.0, 42.4378), np.sqrt(300.0**2 - 42.4378**2))
    assert np.isclose(added_sigma_km_s(300.0, 254.6267), np.sqrt(300.0**2 - 254.6267**2))


def test_target_sigma_values():
    assert target_sigma_km_s("sigma300") == 300.0
    r100 = SPEED_OF_LIGHT_KM_S / (2.0 * np.sqrt(2.0 * np.log(2.0)) * 100.0)
    assert np.isclose(target_sigma_km_s("r100"), np.hypot(r100, 300.0))


def _toy_grid():
    wave = np.linspace(3400.0, 22000.0, 4000)
    flux = np.ones((4, 2, 107, wave.size)) * (1.0 + 1e-5 * wave)
    return SspGrid(
        wave_a=wave,
        log_age_yr=np.linspace(5.0, 10.3, 107),
        log_z_grid=np.array([-0.5, -0.25, 0.0, 0.25]),
        agb_weights=np.array([0.0, 1.0]),
        flux_nu=flux,
        native_sigma_km_s=np.where(wave < 10000.0, 42.4378, 254.6267),
        product="native",
    )


def test_make_resolution_product_shapes_and_ranges():
    grid = _toy_grid()
    sigma300 = make_resolution_product(grid, "sigma300")
    assert sigma300.product == "sigma300"
    assert sigma300.flux_nu.shape[:3] == (4, 2, 107)
    assert sigma300.wave_a[0] >= 3400.0 and sigma300.wave_a[-1] <= 22000.0
    r100 = make_resolution_product(grid, "r100")
    assert r100.product == "r100"
    assert r100.wave_a[0] >= 12500.0 and r100.wave_a[-1] <= 21000.0
    assert np.allclose(
        sigma300.flux_nu[0, 0, 0] / sigma300.wave_a**2,
        (1.0 + 1e-5 * sigma300.wave_a) / sigma300.wave_a**2 * 1.0,
        rtol=2e-3,
    )
