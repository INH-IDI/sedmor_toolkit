"""Pruebas unitarias para ``sedmor.propagation`` (refracción, difracción y corriente)."""

from __future__ import annotations

import math

import numpy as np
import pytest

import sedmor as sm
from sedmor.propagation import (
    diffraction_breakwater_gap,
    diffraction_random_directional,
    diffraction_semi_infinite_breakwater,
    dispersion_with_current,
    propagate_wave_profile,
    shoaling_refraction_snell,
)


def test_dispersion_with_current_and_wave_blocking() -> None:
    h, T = 12.0, 8.0
    # Sin corriente debe coincidir con waves.dispersion_relation exacto
    ref = sm.waves.dispersion_relation(h=h, T=T, method="exact")
    zero_u = dispersion_with_current(h=h, T=T, U=0.0)
    assert zero_u.is_blocked is False
    assert zero_u.L == pytest.approx(ref["L"], rel=1e-11)
    assert zero_u.k == pytest.approx(ref["k"], rel=1e-11)
    assert zero_u.sigma_intrinsic == pytest.approx(zero_u.omega_abs, rel=1e-12)

    # Corriente a favor alarga L; corriente en contra acorta L y satisface relación Doppler exacta
    follow = dispersion_with_current(h=h, T=T, U=1.5, current_angle_deg=0.0)
    oppose = dispersion_with_current(h=h, T=T, U=1.5, current_angle_deg=180.0)
    assert follow.L > zero_u.L > oppose.L
    assert follow.sigma_intrinsic < zero_u.omega_abs < oppose.sigma_intrinsic

    for res in (follow, oppose):
        lhs = (res.omega_abs - res.k * res.u_parallel) ** 2
        rhs = 9.81 * res.k * math.tanh(res.k * h)
        assert lhs == pytest.approx(rhs, rel=1e-11)
        assert res.cg_abs > 0.0

    # Bloqueo por corriente opuesta fuerte (U_block_deep = g / (4 * omega_a) ~ 3.12 m/s para T=8 s)
    vec = dispersion_with_current(
        h=100.0, T=8.0, U=np.array([0.0, 2.0, 4.0]), current_angle_deg=180.0
    )
    assert list(vec.is_blocked) == [False, False, True]
    assert vec.regime[2] == "blocked"
    assert math.isnan(vec.L[2])


def test_shoaling_refraction_snell_and_action_flux() -> None:
    H0, T, th0 = 2.0, 10.0, 35.0
    h = np.array([60.0, 20.0, 8.0, 3.0])

    # Sin corriente: invariancia k*sin(theta) y conservación de flujo de energía
    res = shoaling_refraction_snell(H0, T, th0, h, limit_breaking=False)
    omega = 2.0 * math.pi / T
    ky_expected = (omega**2 / 9.81) * math.sin(math.radians(th0))
    assert np.allclose(res.k * np.sin(np.radians(res.theta_deg)), ky_expected, rtol=1e-11)
    assert np.all(np.diff(res.theta_deg) < 0.0)
    assert np.allclose(res.Kc_current, 1.0, rtol=1e-12)

    flux = (res.H**2) * res.cg_abs_x / res.sigma_intrinsic
    flux0 = (H0**2) * (9.81 / (2.0 * omega)) * math.cos(math.radians(th0)) / omega
    assert np.allclose(flux, flux0, rtol=1e-11)

    # Corriente opuesta en x aumenta altura por flujo de acción; corriente transversal refracta
    curr_opp = shoaling_refraction_snell(H0, T, 25.0, 15.0, U=1.2, current_dir_deg=180.0)
    no_curr = shoaling_refraction_snell(H0, T, 25.0, 15.0, U=0.0)
    assert curr_opp.H > no_curr.H
    assert curr_opp.Kc_current > 1.0

    # Rotura límite en someras
    brk = shoaling_refraction_snell(H0=3.5, T=10.0, theta0_deg=10.0, h=2.0, gamma_b=0.78)
    assert brk.is_broken is True
    assert brk.H == pytest.approx(0.78 * 2.0, rel=1e-12)


def test_diffraction_semi_infinite_gap_and_random_goda() -> None:
    L = 50.0
    # Línea de sombra lejos de la punta tiende a 0.5; zona iluminada ~ 1.0; sombra profunda pequeña
    far_shadow = diffraction_semi_infinite_breakwater(x=0.0, y=50.0 * L, L=L, wave_angle_deg=90.0)
    assert far_shadow.Kd == pytest.approx(0.5, abs=0.02)

    lit = diffraction_semi_infinite_breakwater(x=-50.0 * L, y=2.0 * L, L=L, wave_angle_deg=90.0)
    deep_sh = diffraction_semi_infinite_breakwater(x=4.0 * L, y=1.0 * L, L=L, wave_angle_deg=90.0)
    assert lit.Kd == pytest.approx(1.0, abs=0.03)
    assert deep_sh.Kd < 0.25

    # Verificación directa de la EDP de Helmholtz (nabla^2 + k^2) phi = 0
    k = 2.0 * math.pi / L
    x0, y0, eps = 65.0, 95.0, 1e-3
    c = diffraction_semi_infinite_breakwater(x0, y0, L, wave_angle_deg=70.0).complex_amplitude
    px_p = diffraction_semi_infinite_breakwater(x0 + eps, y0, L, wave_angle_deg=70.0).complex_amplitude
    px_m = diffraction_semi_infinite_breakwater(x0 - eps, y0, L, wave_angle_deg=70.0).complex_amplitude
    py_p = diffraction_semi_infinite_breakwater(x0, y0 + eps, L, wave_angle_deg=70.0).complex_amplitude
    py_m = diffraction_semi_infinite_breakwater(x0, y0 - eps, L, wave_angle_deg=70.0).complex_amplitude
    lap = (px_p - 2.0 * c + px_m) / eps**2 + (py_p - 2.0 * c + py_m) / eps**2
    assert abs(lap + (k**2) * c) < 1e-6

    # Simetría en bocana bajo incidencia normal
    B = 2.5 * L
    g_sym = diffraction_breakwater_gap(
        x=np.array([-40.0, 40.0]), y=3.0 * L, L=L, gap_width=B, wave_angle_deg=90.0
    )
    assert g_sym.Kd[0] == pytest.approx(g_sym.Kd[1], rel=1e-12)

    # Difracción direccional aleatoria de Goda: viento amplio (s_max=10) penetra más en sombra que swell (s_max=75)
    kd_wind = diffraction_random_directional(x=3.5 * L, y=1.5 * L, L=L, wave_angle_deg=90.0, s_max=10.0)
    kd_swell = diffraction_random_directional(x=3.5 * L, y=1.5 * L, L=L, wave_angle_deg=90.0, s_max=75.0)
    assert kd_wind > kd_swell


def test_propagate_wave_profile_and_exports() -> None:
    x = np.linspace(0.0, 600.0, 61)
    depth = 15.0 - 0.022 * x  # de 15 m a 1.8 m

    tr_clean = propagate_wave_profile(x, depth, H0=2.8, T=9.0, theta0_deg=20.0, gamma_b=0.78)
    assert tr_clean.breaking_index is not None
    assert tr_clean.breaking_H is not None
    assert np.all(tr_clean.H[tr_clean.is_broken] <= 0.78 * depth[tr_clean.is_broken] + 1e-12)

    # Con fricción de fondo atenúa la altura no rota respecto al caso sin fricción
    tr_fric = propagate_wave_profile(
        x, depth, H0=2.8, T=9.0, theta0_deg=20.0, k_s=0.04, gamma_b=0.78
    )
    i_pre = max(tr_clean.breaking_index - 1, 1)
    assert tr_fric.H_unbroken[i_pre] < tr_clean.H_unbroken[i_pre]

    # Concordancia en fondo plano con sedmor.waves.wave_decay_friction (misma altura local inicial H=3.0 m)
    x_flat = np.arange(0, 1001, 50, dtype=float)
    h_flat = np.full_like(x_flat, 10.0)
    ks_init = shoaling_refraction_snell(3.0, 10.0, 0.0, 10.0).Ks
    tr_flat = propagate_wave_profile(x_flat, h_flat, H0=3.0 / ks_init, T=10.0, k_s=0.03)
    ref_fric = sm.waves.wave_decay_friction(
        h=10.0, H0=3.0, T=10.0, L=None, k_s=0.03, dx=50.0, n_steps=len(x_flat)
    )
    assert tr_flat.H[-1] == pytest.approx(ref_fric["H"][-1], rel=1e-3)

    assert hasattr(sm, "propagation")
    assert sm.dispersion_with_current is dispersion_with_current
    assert sm.diffraction_semi_infinite_breakwater is diffraction_semi_infinite_breakwater


def test_propagation_validation() -> None:
    with pytest.raises(ValueError, match="h"):
        dispersion_with_current(-2.0, 8.0)
    with pytest.raises(ValueError, match="theta0_deg"):
        shoaling_refraction_snell(2.0, 8.0, 95.0, 10.0)
    with pytest.raises(ValueError, match="wave_angle_deg"):
        diffraction_semi_infinite_breakwater(10.0, 20.0, 50.0, wave_angle_deg=180.0)
    with pytest.raises(ValueError, match=">= 0"):
        diffraction_breakwater_gap(0.0, -5.0, 50.0, 100.0)
    with pytest.raises(ValueError, match="creciente"):
        propagate_wave_profile([100.0, 50.0], [10.0, 5.0], 2.0, 8.0)
