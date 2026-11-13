"""Pruebas unitarias para ``sedmor.armor`` (peso de roca, escolleras y filtros)."""

from __future__ import annotations

import math

import numpy as np
import pytest

import sedmor as sm
from sedmor.armor import (
    HUDSON_KD_VALUES,
    classify_rock_grading,
    compare_armor_methods,
    filter_and_underlayer_design,
    hudson_armor_weight,
    isbash_stone_stability,
    rock_size_properties,
    van_der_meer_rock_armor,
    van_gent_shallow_rock_armor,
)


def test_rock_size_properties_bidirectional() -> None:
    rho_r, rho_w = 2650.0, 1025.0
    dn50_exp = 1.0
    p1 = rock_size_properties(dn50=dn50_exp, rho_r=rho_r, rho_w=rho_w)
    assert p1.mass_kg == pytest.approx(2650.0, rel=1e-12)
    assert p1.mass_tonnes == pytest.approx(2.65, rel=1e-12)
    assert p1.weight_kn == pytest.approx(2650.0 * 9.81 / 1000.0, rel=1e-12)
    assert p1.d50_sieve_m == pytest.approx(1.0 / 0.84, rel=1e-12)
    assert p1.layer_thickness_m == pytest.approx(2.0, rel=1e-12)
    assert p1.stones_per_m2 == pytest.approx(2.0 * 1.0 * (1.0 - 0.37), rel=1e-12)

    p2 = rock_size_properties(mass_kg=p1.mass_kg, rho_r=rho_r, rho_w=rho_w)
    assert p2.dn50_m == pytest.approx(dn50_exp, rel=1e-12)


def test_hudson_armor_weight() -> None:
    H, slope, KD = 4.0, 0.5, HUDSON_KD_VALUES["quarry_angular_rough_2_breaking_trunk"]
    rho_r, rho_w = 2650.0, 1025.0
    delta = rho_r / rho_w - 1.0
    cot_a = 1.0 / slope
    mass_exp = (rho_r * (H**3)) / (KD * (delta**3) * cot_a)
    dn_exp = (mass_exp / rho_r) ** (1.0 / 3.0)

    res = hudson_armor_weight(H, slope, KD=KD, rho_r=rho_r, rho_w=rho_w)
    assert res.cot_alpha == pytest.approx(2.0, rel=1e-12)
    assert res.mass_kg == pytest.approx(mass_exp, rel=1e-12)
    assert res.dn50_m == pytest.approx(dn_exp, rel=1e-12)
    assert res.stability_number == pytest.approx((KD * cot_a) ** (1.0 / 3.0), rel=1e-12)

    # Vectorizado en altura H escala cúbica en masa
    vec = hudson_armor_weight(np.array([2.0, 4.0]), slope=0.5, KD=2.0)
    assert vec.mass_kg[1] / vec.mass_kg[0] == pytest.approx(8.0, rel=1e-12)


def test_van_der_meer_plunging_surging_and_gentle_cap() -> None:
    Hs, Tm = 4.0, 8.0
    L0 = 9.81 * Tm**2 / (2.0 * math.pi)
    P, S, N = 0.4, 2.0, 3000.0
    rho_r, rho_w = 2650.0, 1025.0
    delta = rho_r / rho_w - 1.0

    # Talud suave-moderado 1:3 -> plunging
    s_plg = 1.0 / 3.0
    xi_plg = s_plg / math.sqrt(Hs / L0)
    res_plg = van_der_meer_rock_armor(Hs, Tm, s_plg, P=P, S=S, N=N, approach="mean")
    ns_exp = 6.2 * (P**0.18) * ((S / math.sqrt(N)) ** 0.2) * (xi_plg**-0.5)
    dn_exp = Hs / (delta * ns_exp)
    assert res_plg.regime == "plunging"
    assert res_plg.stability_number_Hs == pytest.approx(ns_exp, rel=1e-12)
    assert res_plg.dn50_m == pytest.approx(dn_exp, rel=1e-12)

    # Enfoque design da rocas más pesadas que mean
    res_des = van_der_meer_rock_armor(Hs, Tm, s_plg, P=P, S=S, N=N, approach="design")
    assert res_des.mass_kg > res_plg.mass_kg

    # Talud 1:1.5 con período largo -> surging
    s_srg = 1.0 / 1.5
    res_srg = van_der_meer_rock_armor(3.0, 13.0, s_srg, P=0.5, S=2.0, N=3000.0, approach="mean")
    L0_srg = 9.81 * 13.0**2 / (2.0 * math.pi)
    xi_srg = s_srg / math.sqrt(3.0 / L0_srg)
    ns_srg_exp = 1.0 * (0.5**-0.13) * ((2.0 / math.sqrt(3000.0)) ** 0.2) * math.sqrt(1.5) * (xi_srg**0.5)
    assert res_srg.regime == "surging"
    assert res_srg.stability_number_Hs == pytest.approx(ns_srg_exp, rel=1e-12)

    # Regla cot(alpha) >= 4 fuerza rama plunging aunque xi > xi_cr
    gentle_long = van_der_meer_rock_armor(1.5, 16.0, slope=0.20, P=0.4, S=2.0, N=3000.0)
    assert gentle_long.iribarren > gentle_long.xi_critical
    assert gentle_long.regime == "plunging"


def test_van_gent_shallow_and_isbash() -> None:
    # Van Gent con Rayleigh H2% = 1.4*Hs vs ola truncada en someras H2% = 1.2*Hs
    vg_deep = van_gent_shallow_rock_armor(3.0, 10.0, slope=0.5)
    vg_shal = van_gent_shallow_rock_armor(3.0, 10.0, slope=0.5, H2_percent=3.6)
    assert vg_shal.dn50_m / vg_deep.dn50_m == pytest.approx(3.6 / (1.4 * 3.0), rel=1e-12)
    assert vg_shal.mass_kg < vg_deep.mass_kg

    # Isbash sin pendiente vs con talud lateral
    V = 3.5
    rho_r, rho_w = 2650.0, 1025.0
    delta = rho_r / rho_w - 1.0
    isb_flat = isbash_stone_stability(V, regime="high_turbulence", rho_r=rho_r, rho_w=rho_w)
    d50_exp = V**2 / (2.0 * 9.81 * (0.86**2) * delta)
    assert isb_flat.slope_factor_ks == pytest.approx(1.0, rel=1e-12)
    assert isb_flat.d50_stone_m == pytest.approx(d50_exp, rel=1e-12)
    assert isb_flat.dn50_m == pytest.approx(0.84 * d50_exp, rel=1e-12)

    isb_slope = isbash_stone_stability(V, regime="high_turbulence", slope=1.0 / 3.0)
    assert isb_slope.slope_factor_ks < 1.0
    assert isb_slope.mass_kg > isb_flat.mass_kg


def test_grading_filter_compare_and_exports() -> None:
    cls_hma = classify_rock_grading(4200.0)
    assert cls_hma["grading_class"] == "HMA 3000-6000"
    assert classify_rock_grading(2.0)["status"] == "below_lma_5_40"
    assert classify_rock_grading(20000.0)["status"] == "above_hma_10000_15000"

    filt = filter_and_underlayer_design(
        4800.0,
        armor_to_underlayer_ratio=12.0,
        armor_to_core_ratio=500.0,
        d15_subsoil_m=0.015,
        d85_subsoil_m=0.05,
    )
    assert filt.mass_underlayer_kg == pytest.approx(400.0, rel=1e-12)
    assert filt.mass_core_kg == pytest.approx(9.6, rel=1e-12)
    assert filt.meets_terzaghi_retention is True
    assert filt.meets_terzaghi_permeability is True

    cmp = compare_armor_methods(Hs=3.5, T=9.0, slope=0.5)
    assert cmp["hudson_mass_kg"] > 0.0
    assert cmp["vdm_mass_kg"] > 0.0
    assert cmp["van_gent_mass_kg"] > 0.0

    assert hasattr(sm, "armor")
    assert sm.hudson_armor_weight is hudson_armor_weight
    assert sm.van_der_meer_rock_armor is van_der_meer_rock_armor


def test_armor_validation() -> None:
    with pytest.raises(ValueError, match="exactamente uno"):
        rock_size_properties()
    with pytest.raises(ValueError, match="exactamente uno"):
        rock_size_properties(mass_kg=100.0, dn50=0.4)
    with pytest.raises(ValueError, match="rho_r"):
        hudson_armor_weight(2.0, 0.5, rho_r=1000.0, rho_w=1025.0)
    with pytest.raises(ValueError, match="permeabilidad"):
        van_der_meer_rock_armor(2.0, 8.0, 0.5, P=0.8)
    with pytest.raises(ValueError, match="reposo"):
        isbash_stone_stability(3.0, slope=1.0, angle_of_repose_deg=35.0)
