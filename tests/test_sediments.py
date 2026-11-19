"""Pruebas unitarias para ``sedmor.sediments`` (velocidad de caída y granulometría)."""

from __future__ import annotations

import math

import numpy as np
import pytest

import sedmor as sm
from sedmor.sediments import (
    blend_grain_size_curves,
    check_filter_compatibility,
    classify_grain_size_wentworth,
    compare_fall_velocity_models,
    dimensionless_grain_size,
    fall_velocity_ferguson_church,
    fall_velocity_rubey,
    fall_velocity_soulsby,
    fall_velocity_stokes,
    fall_velocity_van_rijn,
    flocculated_settling_velocity,
    grain_size_curve_statistics,
    lognormal_grain_size_curve,
    mm_to_phi,
    percentile_diameter,
    phi_to_mm,
    sieve_data_to_percent_passing,
    water_density,
    water_kinematic_viscosity,
)


def test_water_properties_and_dimensionless_grain_size() -> None:
    nu_20 = water_kinematic_viscosity(20.0)
    nu_5 = water_kinematic_viscosity(5.0)
    assert nu_20 == pytest.approx(1.792e-6 / (1.0 + 0.0337 * 20.0 + 0.000221 * 400.0), rel=1e-12)
    assert nu_5 > nu_20

    rho_fresh = water_density(20.0, salinity_psu=0.0)
    rho_sea = water_density(15.0, salinity_psu=35.0)
    assert rho_fresh == pytest.approx(998.2, abs=0.1)
    assert rho_sea == pytest.approx(1026.0, abs=0.2)

    d = 0.00025
    d_star = dimensionless_grain_size(d, rho_s=2650.0, rho_w=1000.0, nu=1e-6, gravity=9.81)
    assert d_star == pytest.approx(d * (1.65 * 9.81 / 1e-12) ** (1.0 / 3.0), rel=1e-12)


def test_fall_velocity_models_hindered_and_flocculation() -> None:
    # Partícula fina (20 um): Stokes, Soulsby y Ferguson-Church deben converger en régimen laminar
    d_fine = 20e-6
    st = fall_velocity_stokes(d_fine)
    fc_sphere = fall_velocity_ferguson_church(d_fine, shape_type="smooth_sphere")
    so = fall_velocity_soulsby(d_fine)
    assert st.regime == "stokes_laminar"
    assert fc_sphere.ws_clear_m_s == pytest.approx(st.ws_clear_m_s, rel=0.02)
    assert so.ws_clear_m_s == pytest.approx(st.ws_clear_m_s, rel=0.10)

    # Arena media (0.25 mm): coherencia entre van Rijn, Soulsby, Ferguson-Church y Rubey
    d_sand = 0.00025
    cmp = compare_fall_velocity_models(d_sand)
    for k in ("van_rijn_m_s", "soulsby_m_s", "ferguson_church_m_s", "rubey_m_s"):
        assert 0.025 < cmp[k] < 0.045

    # van Rijn debe coincidir exactamente con la implementación base en littoral.py
    vr = fall_velocity_van_rijn(np.array([0.00005, 0.00025, 0.002]))
    assert vr.ws_clear_m_s[0] == pytest.approx(
        1.65 * 9.81 * (0.00005**2) / (18.0 * 1e-6), rel=1e-12
    )
    assert vr.ws_clear_m_s[2] == pytest.approx(1.1 * math.sqrt(1.65 * 9.81 * 0.002), rel=1e-12)

    # Caída obstaculizada reduce ws al crecer c_vol; Rubey y Ferguson vectorizados
    fc_c = fall_velocity_ferguson_church(d_sand, c_vol=np.array([0.0, 0.1, 0.25]))
    assert np.all(np.diff(fc_c.ws_hindered_m_s) < 0.0)
    assert fc_c.ws_hindered_m_s[0] == pytest.approx(fc_c.ws_clear_m_s[0], rel=1e-12)
    assert fall_velocity_rubey(0.001).ws_clear_m_s > fall_velocity_rubey(0.0002).ws_clear_m_s

    # Floculación en fangos cohesivos: crece hasta c_floc_max y decrece al obstaculizarse
    ws_floc = flocculated_settling_velocity(np.array([0.5, 5.0, 10.0, 40.0, 80.0]))
    assert ws_floc[1] > ws_floc[0]
    assert ws_floc[2] > ws_floc[3] > ws_floc[4]


def test_phi_wentworth_sieve_and_folk_ward_lognormal() -> None:
    assert mm_to_phi(1.0) == pytest.approx(0.0)
    assert mm_to_phi(0.25) == pytest.approx(2.0)
    assert phi_to_mm(np.array([0.0, 1.0, 2.0])).tolist() == pytest.approx([1.0, 0.5, 0.25])
    assert "Arena fina" in classify_grain_size_wentworth(0.18)
    assert "Arcilla" in classify_grain_size_wentworth(0.002)

    # Curva log-normal perfecta (sigma_g = 1.6): Mz ~ phi(d50), kurtosis K_G ~ 1.0, Sk_I ~ 0
    ln_res = lognormal_grain_size_curve(d50_mm=0.35, sigma_g=1.6, n_points=121, n_sigma=3.5)
    assert ln_res.d50_mm == pytest.approx(0.35, rel=1e-3)
    assert ln_res.sigma_g_geometric == pytest.approx(1.6, rel=1e-2)
    assert ln_res.folk_ward_kurtosis == pytest.approx(1.0, abs=0.03)
    assert abs(ln_res.folk_ward_skewness) < 0.02
    assert ln_res.folk_ward_sorting_phi > 0.0
    assert ln_res.sand_pct == pytest.approx(100.0, abs=0.5)

    # Cola hacia los finos debe producir asimetría positiva (Sk_I > 0) en Folk & Ward
    d_skew = np.array([0.04, 0.08, 0.15, 0.25, 0.32, 0.40, 0.50])
    p_skew = np.array([2.0, 10.0, 22.0, 45.0, 75.0, 94.0, 99.5])
    sk_res = grain_size_curve_statistics(d_skew, p_skew)
    assert sk_res.folk_ward_skewness > 0.1

    # Conversión desde masas retenidas sobre cada tamiz de laboratorio (+ fondo pan_value)
    d_sieve = np.array([0.063, 0.125, 0.25, 0.50, 1.00, 2.00, 4.00])
    mass_ret = np.array([5.0, 15.0, 40.0, 25.0, 10.0, 5.0, 0.0])  # total=100 g (+0 fondo -> 100% pasa 4.00 mm)
    d_out, p_out = sieve_data_to_percent_passing(
        d_sieve, mass_ret, input_type="retained_mass", pan_value=0.0
    )
    assert p_out.tolist() == pytest.approx([0.0, 5.0, 20.0, 60.0, 85.0, 95.0, 100.0], rel=1e-12)
    assert percentile_diameter(d_out, p_out, 50.0) == pytest.approx(0.25 * (2.0**0.75), rel=1e-12)

    # Con masa recogida en cazoleta/fondo (pan_value > 0)
    _, p_pan = sieve_data_to_percent_passing([0.063, 0.125, 0.25], [10.0, 30.0, 40.0], input_type="retained_mass", pan_value=20.0)
    assert p_pan.tolist() == pytest.approx([20.0, 30.0, 60.0], rel=1e-12)


def test_blend_curves_filter_compatibility_and_exports() -> None:
    c1 = lognormal_grain_size_curve(d50_mm=0.20, sigma_g=1.4, n_points=51)
    c2 = lognormal_grain_size_curve(d50_mm=1.20, sigma_g=1.5, n_points=51)
    blended = blend_grain_size_curves(
        [(c1.diameters_mm, c1.percent_passing), (c2.diameters_mm, c2.percent_passing)],
        weights=[0.5, 0.5],
    )
    assert c1.d50_mm < blended.d50_mm < c2.d50_mm
    assert blended.cu_uniformity > c1.cu_uniformity

    # Filtro bien dimensionado frente a arena base fina (Terzaghi)
    base = lognormal_grain_size_curve(d50_mm=0.20, sigma_g=1.6, n_points=61)
    filt = lognormal_grain_size_curve(d50_mm=1.25, sigma_g=1.5, n_points=61)
    chk = check_filter_compatibility(
        base.diameters_mm, base.percent_passing, filt.diameters_mm, filt.percent_passing
    )
    assert chk["meets_retention"] is True
    assert chk["meets_permeability"] is True
    assert chk["is_compatible"] is True

    assert hasattr(sm, "sediments")
    assert sm.fall_velocity_soulsby is fall_velocity_soulsby
    assert sm.grain_size_curve_statistics is grain_size_curve_statistics


def test_sediments_validation() -> None:
    with pytest.raises(ValueError, match="temperature_c"):
        water_kinematic_viscosity(90.0)
    with pytest.raises(ValueError, match="rho_s"):
        dimensionless_grain_size(0.0002, rho_s=900.0, rho_w=1000.0)
    with pytest.raises(ValueError, match="c_vol"):
        fall_velocity_stokes(0.0001, c_vol=0.8)
    with pytest.raises(ValueError, match="c_gel"):
        flocculated_settling_velocity(5.0, c_floc_max=20.0, c_gel=10.0)
    with pytest.raises(ValueError, match="monótono"):
        sieve_data_to_percent_passing([0.1, 0.5, 1.0], [10.0, 60.0, 40.0])
    with pytest.raises(ValueError, match="sigma_g"):
        lognormal_grain_size_curve(0.3, 0.9)
