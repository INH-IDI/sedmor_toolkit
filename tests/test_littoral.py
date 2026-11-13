"""Pruebas unitarias para ``sedmor.littoral`` frente a valores cacheados de Excel."""

from __future__ import annotations

import numpy as np
import pytest

from sedmor.littoral import (
    annual_longshore_transport,
    longshore_and_dune_erosion,
    wrap_wave_angle_to_normal,
)


def test_longshore_and_dune_erosion_volvol_cases():
    # Casos 1 y 15 de VOLVOL.xls ('Longsh tr. and dune erosion')
    res = longshore_and_dune_erosion(
        depth_offshore=30.0,
        d50=0.0025,
        tan_beta=0.02,
        gamma_br=0.6,
        wave_height_offshore=4.0,
        wave_period=15.0,
        swell_period=6.0,
        wave_angle_offshore=np.array([20.0, 0.0]),
        v_tide=-0.3,
        hwl=0.0,
        ssl=np.array([1.0, 8.0]),
        dune_height=10.0,
        duration_hours=np.array([3.0, 20.0]),
        rho_w=1025.0,
        rho_s=2650.0,
        rho_bulk=1600.0,
        p_mud=0.05,
        strict_excel=True,
    )

    assert res.d_eff[0] == pytest.approx(9.0, rel=1e-12)
    assert res.ws[0] == pytest.approx(0.216901194207825, rel=1e-11)
    assert res.L0[0] == pytest.approx(234.221725877671, rel=1e-11)
    assert res.c0[0] == pytest.approx(15.6147817251781, rel=1e-11)
    assert res.h_br[0] == pytest.approx(6.68979966651169, rel=1e-11)
    assert res.Hs_br[0] == pytest.approx(4.01387979990701, rel=1e-11)
    assert res.L_br[0] == pytest.approx(119.089724789055, rel=1e-11)
    assert res.theta_br_deg[0] == pytest.approx(10.0146323102009, rel=1e-11)
    assert res.sin_2theta_br[0] == pytest.approx(0.342438100941375, rel=1e-11)
    assert res.v_longshore[0] == pytest.approx(0.644213547587996, rel=1e-11)
    assert res.v_eff[0] == pytest.approx(0.344213547587996, rel=1e-11)
    assert res.k_grain[0] == pytest.approx(0.0894427190999916, rel=1e-11)
    assert res.k_slope[0] == pytest.approx(1.25, rel=1e-12)
    assert res.qt_kamphuis_kg_s[0] == pytest.approx(422.483311771179, rel=1e-10)
    assert res.qt_cerc_kg_s[0] == pytest.approx(1561.46812618278, rel=1e-10)
    assert res.qt_van_rijn_kg_s[0] == pytest.approx(49.5640479899725, rel=1e-10)
    assert res.kd_A_param[0] == pytest.approx(0.259435891246465, rel=1e-11)
    assert res.kd_surf_width[0] == pytest.approx(130.940677076513, rel=1e-11)
    assert res.kd_recession[0] == pytest.approx(8.08785036095049, rel=1e-11)
    assert res.kd_volume[0] == pytest.approx(80.8785036095049, rel=1e-11)
    assert res.hr_recession[0] == pytest.approx(0.203, rel=1e-11)
    assert res.hr_volume[0] == pytest.approx(2.03, rel=1e-11)
    assert res.vr_volume_5h[0] == pytest.approx(0.789541774573, rel=1e-11)
    assert res.vr_mean_recession_5h[0] == pytest.approx(0.0877268638414445, rel=1e-11)
    assert res.vr_volume_t[0] == pytest.approx(0.611576428811256, rel=1e-11)

    # Caso 15 (SSL >= 5 y duracion >= 5)
    assert res.qt_van_rijn_kg_s[1] == pytest.approx(-45.9688712398103, rel=1e-10)
    assert res.kd_recession[1] == pytest.approx(84.5637571217372, rel=1e-11)
    assert res.hr_volume[1] == pytest.approx(2810.4462893134, rel=1e-11)
    assert res.vr_volume_5h[1] == pytest.approx(7.40772029187894, rel=1e-11)
    assert res.vr_volume_t[1] == pytest.approx(9.77454552592715, rel=1e-11)


def test_longshore_and_dune_erosion_dutch_coast_case():
    res = longshore_and_dune_erosion(
        depth_offshore=30.0,
        d50=0.0002,
        tan_beta=0.01,
        gamma_br=0.6,
        wave_height_offshore=7.0,
        wave_period=15.0,
        swell_period=6.0,
        wave_angle_offshore=20.0,
        v_tide=0.0,
        hwl=1.0,
        ssl=4.0,
        dune_height=3.0,
        duration_hours=12.0,
        rho_w=1020.0,
        rho_s=2650.0,
        rho_bulk=1600.0,
        p_mud=0.05,
        strict_excel=True,
    )
    assert res.ws == pytest.approx(0.0250689878789935, rel=1e-11)
    assert res.qt_kamphuis_kg_s == pytest.approx(1306.96644238412, rel=1e-10)
    assert res.qt_cerc_kg_s == pytest.approx(5863.73331426104, rel=1e-10)
    assert res.qt_van_rijn_kg_s == pytest.approx(3897.35496721771, rel=1e-10)
    assert res.kd_volume == pytest.approx(1456.14926563596, rel=1e-11)
    assert res.hr_volume == pytest.approx(294.441738369064, rel=1e-11)
    assert res.vr_volume_5h == pytest.approx(137.16589598645, rel=1e-11)
    assert res.vr_volume_t == pytest.approx(163.413673542623, rel=1e-11)

    # Fisica consistente: la erosion de dunas es simetrica para +20 y -20 deg en modo por defecto
    res_pos = longshore_and_dune_erosion(wave_angle_offshore=20.0, ssl=4.0, strict_excel=False)
    res_neg = longshore_and_dune_erosion(wave_angle_offshore=-20.0, ssl=4.0, strict_excel=False)
    assert res_neg.vr_volume_5h == pytest.approx(res_pos.vr_volume_5h, rel=1e-12)
    assert res_neg.qt_kamphuis_kg_s == pytest.approx(-res_pos.qt_kamphuis_kg_s, rel=1e-12)
    assert res_neg.qt_cerc_kg_s == pytest.approx(-res_pos.qt_cerc_kg_s, rel=1e-12)


def test_wrap_wave_angle_and_annual_longshore_volvol():
    _, rel = wrap_wave_angle_to_normal(np.array([195.0, 245.0, 15.0, 345.0]), np.array([100.0, 270.0, 100.0, 100.0]))
    assert np.allclose(rel, [180.0, -25.0, -85.0, 180.0])

    days = np.array([15, 10, 7, 3, 15, 10, 56, 90, 30, 30, 30, 3, 15, 40, 7, 4], dtype=float)
    hs_o = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0])
    tp = np.array([5, 5, 5, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12], dtype=float)
    tswell = np.array([8, 8, 8, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14], dtype=float)

    ann = annual_longshore_transport(
        days=days,
        wave_height_offshore=hs_o,
        wave_period=tp,
        wave_angle_north=245.0,
        shore_normal_north=270.0,
        depth_offshore=20.0,
        d50=0.001,
        tan_beta=0.01,
        gamma_br=0.6,
        swell_period=tswell,
        v_tide_pos=0.3,
        v_tide_neg=-0.2,
        rho_bulk=1600.0,
        p_mud=0.0,
        k_cal=1.0,
        strict_excel=True,
    )
    assert ann.total_days == pytest.approx(365.0)
    assert ann.kamphuis_net_m3_yr == pytest.approx(-949258.612252475, rel=1e-9)
    assert ann.cerc_net_m3_yr == pytest.approx(-4877821.43794433, rel=1e-9)
    assert ann.van_rijn_net_m3_yr == pytest.approx(-797094.307898362, rel=1e-9)
    assert ann.kamphuis_gross_m3_yr == pytest.approx(949258.612252475, rel=1e-9)
    assert ann.cerc_gross_m3_yr == pytest.approx(4877821.43794433, rel=1e-9)
    assert ann.van_rijn_gross_m3_yr == pytest.approx(797094.307898362, rel=1e-9)
    assert ann.kamphuis_negative_m3_yr == pytest.approx(-949258.612252475, rel=1e-9)
    assert ann.kamphuis_positive_m3_yr == pytest.approx(0.0, abs=1e-6)


def test_annual_longshore_dutch_coast_bidirectional():
    # Clima representativo de dos sentidos de Littoral Dutch coast 2011.xls (filas 27, 98, 99, 103)
    days = np.array([9.71, 0.037, 0.037, 0.11])
    hs_o = np.array([0.75, 5.0, 5.0, 5.0])
    tp = np.array([5.0, 10.0, 10.0, 10.0])
    wave_ang = np.array([195.0, 15.0, 45.0, 165.0])
    ann = annual_longshore_transport(
        days=days,
        wave_height_offshore=hs_o,
        wave_period=tp,
        wave_angle_north=wave_ang,
        shore_normal_north=100.0,
        depth_offshore=20.0,
        d50=0.0002,
        tan_beta=0.01,
        gamma_br=0.6,
        swell_period=6.0,
        v_tide_pos=-0.3,
        v_tide_neg=0.2,
        rho_bulk=1600.0,
        p_mud=0.05,
        k_cal=1.0,
        strict_excel=True,
    )
    assert ann.angle_to_normal_deg[0] == 180.0
    assert ann.angle_to_normal_deg[1] == -85.0
    assert ann.angle_to_normal_deg[2] == -55.0
    assert ann.angle_to_normal_deg[3] == 65.0
    assert ann.kamphuis_gross_m3_yr > abs(ann.kamphuis_net_m3_yr)
    assert ann.van_rijn_positive_m3_yr > 0.0
    assert ann.van_rijn_negative_m3_yr < 0.0
