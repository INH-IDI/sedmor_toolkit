"""Pruebas unitarias para ``sedmor.closure`` (profundidad de cierre y derivados)."""

from __future__ import annotations

import math

import numpy as np
import pytest

import sedmor as sm
from sedmor.closure import (
    beach_nourishment_advance,
    beach_nourishment_volume,
    bruun_rule_recession,
    closure_depth_from_wave_series,
    compare_closure_depth_methods,
    dean_scale_parameter,
    depth_of_closure_annual_stats,
    depth_of_closure_birkemeier,
    depth_of_closure_hallermeier,
    equilibrium_profile_dean,
    outer_closure_depth_hallermeier,
    shoreline_change_from_volume,
    structure_active_depth_ratio,
    volume_change_from_shoreline,
)


def test_hallermeier_and_birkemeier_closure_depths() -> None:
    He, Te = 4.2, 11.0
    g = 9.81
    dl_h = depth_of_closure_hallermeier(He, Te, gravity=g)
    dl_b = depth_of_closure_birkemeier(He, Te, gravity=g)
    dl_bl = depth_of_closure_birkemeier(He, linear=True)

    assert dl_h == pytest.approx(2.28 * He - 68.5 * (He**2) / (g * Te**2), rel=1e-12)
    assert dl_b == pytest.approx(1.75 * He - 57.9 * (He**2) / (g * Te**2), rel=1e-12)
    assert dl_bl == pytest.approx(1.57 * He, rel=1e-12)
    assert dl_h > dl_b

    # Vectorizado
    hs = np.array([2.0, 4.0, 6.0])
    vec = depth_of_closure_hallermeier(hs, 12.0)
    assert vec.shape == (3,)
    assert np.all(np.diff(vec) > 0.0)

    # Límite exterior Hallermeier (1981)
    di = outer_closure_depth_hallermeier(H_mean=1.2, sigma_H=0.55, T_mean=8.0, d50=0.00025)
    di_exp = (1.2 - 0.3 * 0.55) * 8.0 * math.sqrt(g / (5000.0 * 0.00025))
    assert di == pytest.approx(di_exp, rel=1e-12)
    assert di > dl_h


def test_annual_stats_and_wave_series_horizon() -> None:
    stats = depth_of_closure_annual_stats(
        H_mean=1.4, sigma_H=0.6, T_mean=8.5, d50=0.0003, houston_calibration="birkemeier"
    )
    assert stats.He_12h_m == pytest.approx(1.4 + 5.6 * 0.6, rel=1e-12)
    assert stats.hallermeier_dl_m == pytest.approx(2.0 * 1.4 + 11.0 * 0.6, rel=1e-12)
    assert stats.houston_dl_m == pytest.approx(6.75 * 1.4, rel=1e-12)

    stats_h = depth_of_closure_annual_stats(1.4, 0.6, houston_calibration="hallermeier")
    assert stats_h.houston_dl_m == pytest.approx(8.9 * 1.4, rel=1e-12)
    assert math.isnan(stats_h.hallermeier_di_outer_m)

    # Serie sintética de oleaje: mayor horizonte temporal produce mayor He y cierre más profundo
    rng = np.random.default_rng(2026)
    n_obs = 50000
    hs_ser = np.clip(rng.gamma(shape=2.5, scale=0.6, size=n_obs), 0.1, None)
    t_ser = 5.0 + 3.0 * np.sqrt(hs_ser)
    s_1y = closure_depth_from_wave_series(hs_ser, t_ser, years=1.0, d50=0.00025)
    s_20y = closure_depth_from_wave_series(hs_ser, t_ser, years=20.0, d50=0.00025)
    assert s_20y["He_m"] > s_1y["He_m"]
    assert s_20y["birkemeier_dl_m"] > s_1y["birkemeier_dl_m"]


def test_dean_equilibrium_profile_and_bruun_rule() -> None:
    a_val = dean_scale_parameter(d50=0.00025)
    prof = equilibrium_profile_dean(closure_depth=8.0, d50=0.00025, n_points=401)
    assert prof.A_param == pytest.approx(a_val, rel=1e-12)
    assert prof.active_width_m == pytest.approx((8.0 / a_val) ** 1.5, rel=1e-12)
    assert prof.x_m[-1] == pytest.approx(prof.active_width_m, rel=1e-12)
    assert prof.depth_m[-1] == pytest.approx(8.0, rel=1e-12)
    # Comprueba integración numérica de h(x) frente a 3/5 * dc * W*
    trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    vol_num = float(trap(prof.depth_m, prof.x_m))
    assert vol_num == pytest.approx(prof.profile_volume_m3_m, rel=1e-3)

    # Regla de Bruun clásica (1962) vs extensión de Rosati et al. (2013)
    slr, dc, B, W = 0.5, 8.0, 2.0, 400.0
    br = bruun_rule_recession(slr, dc, berm_height=B, active_width=W)
    assert br.recession_m == pytest.approx(0.5 * 400.0 / 10.0, rel=1e-12)

    br_ow = bruun_rule_recession(slr, dc, berm_height=B, d50=0.00025, overwash_volume=120.0)
    assert br_ow.active_width_m == pytest.approx(prof.active_width_m, rel=1e-12)
    assert br_ow.recession_m == pytest.approx(
        (slr * prof.active_width_m + 120.0) / (B + dc), rel=1e-12
    )
    assert br_ow.recession_m > bruun_rule_recession(slr, dc, berm_height=B, d50=0.00025).recession_m


def test_beach_nourishment_regimes_and_roundtrip() -> None:
    dc, B, A_N = 8.0, 2.0, 0.11

    # 1. Arena nativa (A_F == A_N)
    nat = beach_nourishment_volume(30.0, dc, B, A_N, A_fill=A_N)
    assert nat.profile_type == "native_match"
    assert nat.volume_m3_m == pytest.approx(30.0 * (B + dc), rel=1e-12)
    assert nat.overfill_ratio_to_native == pytest.approx(1.0, rel=1e-12)

    # 2. Arena más gruesa (intersecante -> menor volumen que nativo)
    coarse = beach_nourishment_volume(30.0, dc, B, A_N, A_fill=0.16)
    assert coarse.profile_type == "intersecting"
    assert coarse.intersection_depth_m is not None
    assert 0.0 < coarse.intersection_depth_m < dc
    assert coarse.overfill_ratio_to_native < 1.0
    coarse_inv = beach_nourishment_advance(coarse.volume_m3_m, dc, B, A_N, A_fill=0.16)
    assert coarse_inv.profile_type == "intersecting"
    assert coarse_inv.shoreline_advance_m == pytest.approx(30.0, rel=1e-6)

    # 3. Arena más gruesa pero avance grande -> no intersecante
    coarse_large = beach_nourishment_volume(400.0, dc, B, A_N, A_fill=0.16)
    assert coarse_large.profile_type == "non_intersecting"
    coarse_large_inv = beach_nourishment_advance(coarse_large.volume_m3_m, dc, B, A_N, A_fill=0.16)
    assert coarse_large_inv.shoreline_advance_m == pytest.approx(400.0, rel=1e-8)

    # 4. Arena más fina (no intersecante y posible perfil sumergido)
    fine = beach_nourishment_volume(30.0, dc, B, A_N, A_fill=0.08)
    assert fine.profile_type == "non_intersecting"
    assert fine.overfill_ratio_to_native > 1.0
    fine_inv = beach_nourishment_advance(fine.volume_m3_m, dc, B, A_N, A_fill=0.08)
    assert fine_inv.shoreline_advance_m == pytest.approx(30.0, rel=1e-8)

    sub = beach_nourishment_advance(0.5 * fine.critical_submerged_volume_m3_m, dc, B, A_N, A_fill=0.08)
    assert sub.profile_type == "submerged"
    assert sub.shoreline_advance_m == 0.0


def test_prism_exchange_structure_ratio_and_exports() -> None:
    dc, B = 7.5, 2.0
    dv = volume_change_from_shoreline(15.0, dc, B)
    assert dv == pytest.approx(15.0 * (dc + B), rel=1e-12)
    assert shoreline_change_from_volume(dv, dc, B) == pytest.approx(15.0, rel=1e-12)

    groin = structure_active_depth_ratio(dc, structure_length=120.0, A_param=0.12)
    h_tip = 0.12 * (120.0 ** (2.0 / 3.0))
    assert groin["depth_at_tip_m"] == pytest.approx(h_tip, rel=1e-12)
    assert groin["active_depth_ratio"] == pytest.approx(h_tip / dc, rel=1e-12)
    assert groin["active_width_ratio"] == pytest.approx((h_tip / dc) ** 1.5, rel=1e-12)

    cmp = compare_closure_depth_methods(4.0, 10.0, H_mean=1.2)
    assert cmp.houston_dl_m == pytest.approx(6.75 * 1.2, rel=1e-12)

    assert hasattr(sm, "closure")
    assert sm.bruun_rule_recession is bruun_rule_recession
    assert sm.beach_nourishment_volume is beach_nourishment_volume


def test_closure_validation() -> None:
    with pytest.raises(ValueError, match="linear=True"):
        depth_of_closure_birkemeier(3.0)
    with pytest.raises(ValueError, match="d50"):
        dean_scale_parameter()
    with pytest.raises(ValueError, match="d50"):
        dean_scale_parameter(d50=0.0002, ws=0.03)
    with pytest.raises(ValueError, match="exceedance_hours_per_year"):
        closure_depth_from_wave_series([1.0, 2.0], [6.0, 7.0], exceedance_hours_per_year=9000.0)
    with pytest.raises(ValueError, match="shoreline_advance"):
        beach_nourishment_volume(-5.0, 8.0, 2.0, 0.11)
    with pytest.raises(ValueError, match="A_param"):
        structure_active_depth_ratio(8.0, structure_length=100.0)
