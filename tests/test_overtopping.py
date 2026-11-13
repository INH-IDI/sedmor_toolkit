"""Pruebas unitarias para ``sedmor.overtopping`` (rebase de oleaje EurOtop 2018)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import gamma as gamma_func

import sedmor as sm
from sedmor.overtopping import (
    EUROTOP_ROUGHNESS_FACTORS,
    classify_overtopping_hazard,
    individual_overtopping_volumes,
    oblique_factor_overtopping,
    overtopping_rubble_mound,
    overtopping_slope,
    overtopping_vertical_wall,
    solve_required_freeboard,
)


def test_oblique_factor_and_slope_mean_design() -> None:
    assert oblique_factor_overtopping(0.0) == 1.0
    assert oblique_factor_overtopping(30.0) == pytest.approx(1.0 - 0.0033 * 30.0, rel=1e-12)
    assert oblique_factor_overtopping(-30.0) == pytest.approx(oblique_factor_overtopping(30.0))
    assert oblique_factor_overtopping(95.0) == pytest.approx(0.736, rel=1e-12)
    assert oblique_factor_overtopping(60.0, structure_type="vertical_long") == pytest.approx(
        0.721, rel=1e-12
    )

    # Talud suave en rotura (1:4)
    Hm0, Tm_10, slope, Rc = 2.5, 7.0, 0.25, 3.0
    L0 = 9.81 * Tm_10**2 / (2.0 * math.pi)
    xi = slope / math.sqrt(Hm0 / L0)
    res_m = overtopping_slope(Hm0, Tm_10, slope, Rc, method="mean")
    res_d = overtopping_slope(Hm0, Tm_10, slope, Rc, method="design")

    q_brk_exp = (0.023 / math.sqrt(slope)) * xi * math.exp(-((2.7 * Rc / (xi * Hm0)) ** 1.3))
    q_non_exp = 0.09 * math.exp(-((1.5 * Rc / Hm0) ** 1.3))
    assert res_m.is_breaking_limited is True
    assert res_m.q_breaking_dimless == pytest.approx(q_brk_exp, rel=1e-12)
    assert res_m.q_non_breaking_dimless == pytest.approx(q_non_exp, rel=1e-12)
    assert res_m.q_m3_s_m == pytest.approx(q_brk_exp * math.sqrt(9.81 * Hm0**3), rel=1e-12)
    assert res_m.q_l_s_m == pytest.approx(1000.0 * res_m.q_m3_s_m, rel=1e-12)
    assert res_d.q_l_s_m > res_m.q_l_s_m

    # Francobordo nulo y ola cero
    zero_rc = overtopping_slope(Hm0, Tm_10, 0.5, 0.0)
    assert zero_rc.q_dimless <= 0.09 + 1e-12
    zero_wave = overtopping_slope(0.0, Tm_10, 0.25, Rc)
    assert zero_wave.q_l_s_m == 0.0


def test_rubble_mound_and_roughness_reduction() -> None:
    gf = EUROTOP_ROUGHNESS_FACTORS["rock_2_layer_permeable"]
    assert gf == pytest.approx(0.40)

    smooth = overtopping_slope(3.0, 9.0, 0.5, 4.0, gamma_f=1.0, method="design")
    rubble = overtopping_rubble_mound(3.0, 9.0, 0.5, 4.0, gamma_f=gf, method="design")
    assert rubble.q_l_s_m < smooth.q_l_s_m

    # Vectorizado en Rc decrece estrictamente
    rcs = np.array([2.0, 3.0, 4.5, 6.0])
    vec = overtopping_rubble_mound(3.5, 10.0, 1.0 / 1.5, rcs, method="mean")
    assert np.all(np.diff(vec.q_l_s_m) < 0.0)


def test_vertical_wall_non_impulsive_and_impulsive() -> None:
    # No impulsivo (d* >= 0.23): h_toe = 12 m, Hm0 = 2 m, Tm-1,0 = 6 s
    Hm0, Tm10, h_deep, Rc = 2.0, 6.0, 12.0, 2.5
    L0 = 9.81 * Tm10**2 / (2.0 * math.pi)
    d_star_exp = h_deep**2 / (Hm0 * L0)
    assert d_star_exp > 0.23
    non_imp = overtopping_vertical_wall(Hm0, Tm10, h_deep, Rc, method="mean")
    assert non_imp.is_impulsive is False
    q_exp = 0.047 * math.exp(-((2.35 * Rc / Hm0) ** 1.3)) * math.sqrt(9.81 * Hm0**3)
    assert non_imp.q_m3_s_m == pytest.approx(q_exp, rel=1e-12)

    # Impulsivo (d* < 0.23): h_toe = 4 m, Hm0 = 2.5 m, Tm-1,0 = 10 s -> dos ramas Rc/Hm0
    h_sh, Hm0_sh, T_sh = 4.0, 2.5, 10.0
    L0_sh = 9.81 * T_sh**2 / (2.0 * math.pi)
    s_sh = Hm0_sh / L0_sh
    vec_imp = overtopping_vertical_wall(Hm0_sh, T_sh, h_sh, np.array([2.0, 4.5]), method="mean")
    assert list(vec_imp.is_impulsive) == [True, True]
    pref = math.sqrt(Hm0_sh / (h_sh * s_sh))
    q_low = 0.011 * pref * math.exp(-2.2 * (2.0 / Hm0_sh))
    q_high = 0.0014 * pref * ((4.5 / Hm0_sh) ** -3.0)
    assert vec_imp.q_dimless[0] == pytest.approx(q_low, rel=1e-12)
    assert vec_imp.q_dimless[1] == pytest.approx(q_high, rel=1e-12)

    # Desactivar foreshore_influencing fuerza régimen no impulsivo
    forced_non = overtopping_vertical_wall(
        Hm0_sh, T_sh, h_sh, 2.0, foreshore_influencing=False, method="design"
    )
    assert forced_non.is_impulsive is False


def test_individual_volumes_and_hazard_classification() -> None:
    Hm0, Tm_10, slope, Rc = 3.0, 9.0, 0.25, 3.0
    res_q = overtopping_slope(Hm0, Tm_10, slope, Rc, method="mean")
    ru = sm.runup.runup_eurotop(Hm0, Tm_10, slope, method="mean")

    vols = individual_overtopping_volumes(
        res_q.q_m3_s_m, Tm=7.5, Ru2_percent=ru.R2_percent, Rc=Rc, storm_duration_s=3600.0
    )
    p_exp = math.exp(math.log(0.02) * (Rc / ru.R2_percent) ** 2)
    n_ow_exp = p_exp * (3600.0 / 7.5)
    v_mean_exp = res_q.q_m3_s_m * 7.5 / p_exp
    a_exp = v_mean_exp / gamma_func(1.0 + 1.0 / 0.75)
    assert vols.P_ow == pytest.approx(p_exp, rel=1e-12)
    assert vols.N_overtopping == pytest.approx(n_ow_exp, rel=1e-12)
    assert vols.weibull_b == pytest.approx(0.75, rel=1e-12)
    assert vols.weibull_a == pytest.approx(a_exp, rel=1e-12)
    assert vols.V_max == pytest.approx(a_exp * (math.log(n_ow_exp) ** (1.0 / 0.75)), rel=1e-12)
    assert vols.V_1percent > vols.V_mean

    # Weibull b dinámico cuando se entregan Hm0 y Tm_10
    vols_dyn = individual_overtopping_volumes(
        res_q.q_m3_s_m, Tm=7.5, Ru2_percent=ru.R2_percent, Rc=Rc, Hm0=Hm0, Tm_10=Tm_10
    )
    b_dyn_exp = 0.73 + 55.0 * ((res_q.q_m3_s_m / (9.81 * Hm0 * Tm_10)) ** 0.8)
    assert vols_dyn.weibull_b == pytest.approx(b_dyn_exp, rel=1e-12)

    ok_haz = classify_overtopping_hazard(0.05, V_max_l_m=30.0, category="pedestrians_aware")
    bad_haz = classify_overtopping_hazard(2.0, V_max_l_m=800.0, category="pedestrians_aware")
    assert ok_haz.is_acceptable is True
    assert bad_haz.is_acceptable is False


def test_solve_required_freeboard_roundtrip_and_exports() -> None:
    targets = np.array([0.1, 1.0, 10.0, 50.0])
    rcs = solve_required_freeboard(
        targets, Hm0=3.2, Tm_10=9.5, slope=0.35, gamma_f=0.55, angle_deg=15.0, method="design"
    )
    check = overtopping_slope(
        Hm0=3.2, Tm_10=9.5, slope=0.35, Rc=rcs, gamma_f=0.55, angle_deg=15.0, method="design"
    )
    assert check.q_l_s_m == pytest.approx(targets, rel=1e-6)
    assert np.all(np.diff(rcs) < 0.0)

    # Caso q_allowable > q(Rc=0) retorna Rc = 0.0
    rc_zero = solve_required_freeboard(
        10000.0, Hm0=1.5, Tm_10=6.0, slope=0.5, method="mean", unit="l_s_m"
    )
    assert rc_zero == 0.0

    # Exportado en raíz de sedmor
    assert hasattr(sm, "overtopping")
    assert sm.overtopping_slope is overtopping_slope
    assert sm.overtopping_vertical_wall is overtopping_vertical_wall


def test_overtopping_validation() -> None:
    with pytest.raises(ValueError, match="Rc"):
        overtopping_slope(2.0, 8.0, 0.25, -0.5)
    with pytest.raises(ValueError, match="method"):
        overtopping_slope(2.0, 8.0, 0.25, 2.0, method="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="angle_deg"):
        oblique_factor_overtopping(130.0)
    with pytest.raises(ValueError, match="Categoría desconocida"):
        classify_overtopping_hazard(1.0, category="unknown")
    with pytest.raises(ValueError, match="q_allowable"):
        solve_required_freeboard(0.0, 2.0, 8.0, 0.25)
