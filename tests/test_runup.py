"""Pruebas unitarias para ``sedmor.runup`` (ascenso de oleaje y nivel total)."""

from __future__ import annotations

import math

import numpy as np
import pytest

import sedmor as sm
from sedmor.runup import (
    compare_runup_models,
    iribarren_number,
    runup_eurotop,
    runup_holman,
    runup_hunt,
    runup_mase,
    runup_nielsen_hanslow,
    runup_stockdon,
    total_water_level,
)


def test_iribarren_and_breaker_classification() -> None:
    T = 10.0
    L0_exp = 9.81 * T**2 / (2.0 * math.pi)
    res = iribarren_number(Hs0=2.0, T=T, slope=0.05)
    assert isinstance(res.iribarren, float)
    assert isinstance(res.breaker_type, str)
    assert res.L0 == pytest.approx(L0_exp, rel=1e-12)
    xi_exp = 0.05 / math.sqrt(2.0 / L0_exp)
    assert res.iribarren == pytest.approx(xi_exp, rel=1e-12)
    assert res.breaker_type == "spilling"  # xi ~ 0.442 < 0.5

    vec = iribarren_number(Hs0=2.0, T=10.0, slope=np.array([0.02, 0.12, 0.50]))
    assert list(vec.breaker_type) == ["spilling", "plunging", "collapsing_surging"]

    zero_wave = iribarren_number(Hs0=0.0, T=8.0, slope=0.05)
    assert zero_wave.iribarren == 0.0
    assert zero_wave.steepness == 0.0


def test_runup_stockdon_branches_and_decomposition() -> None:
    # Caso intermedio/reflejante (xi0 >= 0.3)
    Hs0, T, bf = 2.5, 10.0, 0.08
    L0 = 9.81 * T**2 / (2.0 * math.pi)
    root = math.sqrt(Hs0 * L0)
    res = runup_stockdon(Hs0=Hs0, T=T, slope=bf)
    assert res.is_dissipative is False
    assert res.setup == pytest.approx(0.35 * bf * root, rel=1e-12)
    assert res.swash_incident == pytest.approx(0.75 * bf * root, rel=1e-12)
    assert res.swash_infragravity == pytest.approx(0.06 * root, rel=1e-12)
    swash_tot = math.sqrt(Hs0 * L0 * (0.563 * bf**2 + 0.004))
    assert res.swash_total == pytest.approx(swash_tot, rel=1e-12)
    assert res.R2_percent == pytest.approx(1.1 * (res.setup + 0.5 * swash_tot), rel=1e-12)

    # Caso disipativo (xi0 < 0.3): H0=4.0, T=10, bf=0.02 -> xi0 ~ 0.125
    diss = runup_stockdon(Hs0=4.0, T=10.0, slope=0.02, use_dissipative_branch=True)
    assert diss.is_dissipative is True
    assert diss.iribarren < 0.3
    assert diss.R2_percent == pytest.approx(0.043 * math.sqrt(4.0 * L0), rel=1e-12)

    cont = runup_stockdon(Hs0=4.0, T=10.0, slope=0.02, use_dissipative_branch=False)
    assert cont.is_dissipative is False
    assert cont.R2_percent != pytest.approx(diss.R2_percent)


def test_holman_mase_hunt_and_nielsen_hanslow() -> None:
    Hs0, T, slope = 2.0, 9.0, 0.07
    L0 = 9.81 * T**2 / (2.0 * math.pi)
    xi = slope / math.sqrt(Hs0 / L0)

    ho = runup_holman(Hs0, T, slope)
    ma = runup_mase(Hs0, T, slope)
    hu = runup_hunt(Hs0, T, slope)
    hu_capped = runup_hunt(Hs0, T, 0.50, max_relative_runup=2.3)

    assert ho.R2_percent == pytest.approx(Hs0 * (0.83 * xi + 0.20), rel=1e-12)
    assert ma.R2_percent == pytest.approx(Hs0 * 1.86 * (xi**0.71), rel=1e-12)
    assert hu.R2_percent == pytest.approx(Hs0 * xi, rel=1e-12)
    assert hu_capped.relative_runup == pytest.approx(2.3, rel=1e-12)
    assert hu_capped.R2_percent == pytest.approx(2.3 * Hs0, rel=1e-12)

    # Nielsen & Hanslow (1991): vectorizado plano (<0.10) y fuerte (>=0.10)
    Hrms0 = Hs0 / math.sqrt(2.0)
    nh = runup_nielsen_hanslow(Hrms0, T, np.array([0.04, 0.12]), z100=0.15)
    assert list(nh.is_steep_branch) == [False, True]
    assert nh.R2_percent[0] == pytest.approx(1.98 * 0.06 * math.sqrt(Hrms0 * L0) + 0.15, rel=1e-12)
    assert nh.R2_percent[1] == pytest.approx(1.98 * 0.6 * 0.12 * math.sqrt(Hrms0 * L0) + 0.15, rel=1e-12)

    nh_orig = runup_nielsen_hanslow(Hrms0, T, 0.04, use_revised_flat=False)
    assert nh_orig.R2_percent == pytest.approx(1.98 * 0.05 * math.sqrt(Hrms0 * L0), rel=1e-12)


def test_eurotop_mean_design_obliqueness_and_cap() -> None:
    Hm0, Tm = 3.0, 8.0
    L0 = 9.81 * Tm**2 / (2.0 * math.pi)

    # Talud suave (rotura): no limitado por cap
    eu_gentle = runup_eurotop(Hm0, Tm, slope=1.0 / 6.0, gamma_f=0.8, angle_deg=45.0, method="mean")
    xi_g = (1.0 / 6.0) / math.sqrt(Hm0 / L0)
    gb_exp = 1.0 - 0.0022 * 45.0
    assert eu_gentle.gamma_beta == pytest.approx(gb_exp, rel=1e-12)
    assert eu_gentle.limited_by_cap is False
    assert eu_gentle.R2_percent == pytest.approx(Hm0 * 1.65 * 1.0 * 0.8 * gb_exp * xi_g, rel=1e-12)

    # Talud empinado (1:1.5) con ángulo extremo (>80 deg): limitado por cap y gamma_beta=0.824
    eu_steep = runup_eurotop(Hm0, Tm, slope=1.0 / 1.5, gamma_b=0.9, gamma_f=0.55, angle_deg=95.0, method="design")
    xi_s = (1.0 / 1.5) / math.sqrt(Hm0 / L0)
    assert eu_steep.gamma_beta == pytest.approx(0.824, rel=1e-12)
    assert eu_steep.limited_by_cap is True
    cap_exp = Hm0 * 0.55 * 0.824 * (4.3 - 1.6 / math.sqrt(0.9 * xi_s))
    assert eu_steep.R2_percent == pytest.approx(cap_exp, rel=1e-12)
    assert eu_steep.R2_percent < eu_steep.R2_breaking


def test_total_water_level_compare_and_exports() -> None:
    r2 = np.array([1.2, 2.0])
    twl = total_water_level(r2, swl=0.4, tide=1.1, storm_surge=0.8)
    assert np.allclose(twl.swl, [2.3, 2.3])
    assert np.allclose(twl.twl, [3.5, 4.3])

    comp = compare_runup_models(Hs0=2.5, T=9.0, slope=0.06, swl=1.5)
    assert comp["stockdon_TWL"] == pytest.approx(1.5 + comp["stockdon_R2"], rel=1e-12)
    assert comp["eurotop_R2"] > 0.0

    # Exportación desde raíz de sedmor
    assert hasattr(sm, "runup")
    assert sm.runup_stockdon(2.0, 8.0, 0.05).R2_percent == pytest.approx(
        sm.runup.runup_stockdon(2.0, 8.0, 0.05).R2_percent
    )


def test_runup_validation() -> None:
    with pytest.raises(ValueError, match="period"):
        runup_stockdon(2.0, 0.0, 0.05)
    with pytest.raises(ValueError, match="wave_height"):
        runup_holman(-1.0, 8.0, 0.05)
    with pytest.raises(ValueError, match="slope"):
        runup_mase(2.0, 8.0, -0.02)
    with pytest.raises(ValueError, match="method"):
        runup_eurotop(2.0, 8.0, 0.25, method="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="gamma_f"):
        runup_eurotop(2.0, 8.0, 0.25, gamma_f=0.0)
