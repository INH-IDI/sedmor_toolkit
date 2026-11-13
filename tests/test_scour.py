"""Pruebas unitarias para ``sedmor.scour`` frente a valores cacheados de ``SCOUR.xls``."""

from __future__ import annotations

import numpy as np
import pytest

from sedmor.scour import (
    scour_bridge_pier,
    scour_downstream_structure,
    scour_horizontal_pipe,
    scour_river_groyne,
    scour_structure_tip_current_waves,
    scour_structure_tip_waves,
    scour_vertical_pile_waves,
    scour_wall_and_rubblemound_toe,
)


def test_scour_downstream_structure_sheet1():
    res = scour_downstream_structure(
        h0=5.0,
        u=np.array([0.4, 0.6]),
        duration_hours=np.array([100000.0, 2000.0]),
        d50=0.00025,
        d90=0.0005,
        turbulence_r=0.15,
        Hs=0.0,
        Tp=7.0,
        strict_excel=True,
    )
    assert res.d_star[0] == pytest.approx(6.24024700490621, rel=1e-12)
    assert res.ucr_c[0] == pytest.approx(0.38150468671021, rel=1e-12)
    assert res.ucr_w[0] == pytest.approx(0.180717750737475, rel=1e-12)
    assert res.qb[0] == pytest.approx(9.8917178745138e-06, rel=1e-11)
    assert res.qs[0] == pytest.approx(5.7354552253004e-05, rel=1e-11)
    assert res.time_scale_hours[0] == pytest.approx(935772.081444348, rel=1e-11)
    assert res.scour_depth_t[0] == pytest.approx(2.13759474149563, rel=1e-11)
    assert res.scour_depth_eq[0] == pytest.approx(2.60147935535804, rel=1e-11)
    assert res.scour_length[0] == pytest.approx(26.0147935535804, rel=1e-11)

    assert res.time_scale_hours[1] == pytest.approx(89447.1674502872, rel=1e-11)
    assert res.scour_depth_t[1] == pytest.approx(1.1796851096604, rel=1e-11)
    assert res.scour_depth_eq[1] == pytest.approx(6.40221903303706, rel=1e-11)


def test_scour_wall_and_rubblemound_toe_sheet2():
    res = scour_wall_and_rubblemound_toe(
        h0=np.array([5.0, 12.0]),
        d50=0.00025,
        d90=0.0005,
        Hs0=5.0,
        theta0_deg=30.0,
        Tp=7.0,
        gamma_br=0.6,
        strict_excel=True,
    )
    assert res.theta_toe_deg[0] == pytest.approx(17.3844308881848, rel=1e-11)
    assert res.H_toe[0] == pytest.approx(3.0, rel=1e-12)
    assert res.wall_spm[0] == pytest.approx(3.0, rel=1e-12)
    assert res.wall_fowler[0] == pytest.approx(5.0, rel=1e-12)
    assert res.wall_sato[0] == pytest.approx(0.9, rel=1e-12)
    assert res.wall_yokoyama[0] == pytest.approx(2.1, rel=1e-12)
    assert res.wall_xie[0] == pytest.approx(1.49831890949985, rel=1e-11)
    assert res.wall_average[0] == pytest.approx(2.38766378189997, rel=1e-11)
    assert res.wall_scour_length[0] == pytest.approx(23.8766378189997, rel=1e-11)
    assert res.rubblemound_average[0] == pytest.approx(1.54, rel=1e-12)
    assert res.rubblemound_scour_length[0] == pytest.approx(15.4, rel=1e-12)

    assert res.H_toe[1] == pytest.approx(4.44522317924047, rel=1e-11)
    assert res.wall_average[1] == pytest.approx(2.67935229518135, rel=1e-11)
    assert res.rubblemound_average[1] == pytest.approx(1.64023250185616, rel=1e-11)

    # Modo fisico: promedio real sobre los 5 metodos
    res_phys = scour_wall_and_rubblemound_toe(h0=5.0, strict_excel=False)
    expected_wall = (
        res_phys.wall_spm
        + res_phys.wall_fowler
        + res_phys.wall_sato
        + res_phys.wall_yokoyama
        + res_phys.wall_xie
    ) / 5.0
    assert res_phys.wall_average == pytest.approx(expected_wall, rel=1e-12)
    assert res_phys.rubblemound_average == pytest.approx(0.7 * expected_wall, rel=1e-12)


def test_scour_structure_tip_sheets3_and_4():
    res3 = scour_structure_tip_waves(strict_excel=True)
    assert res3.kc_number == pytest.approx(1.54752502306075, rel=1e-11)
    assert res3.wall_sumer_depth == pytest.approx(0.773762511530375, rel=1e-11)
    assert res3.wall_sumer_length == pytest.approx(7.73762511530375, rel=1e-11)
    assert res3.rubblemound_sumer_depth == pytest.approx(0.378192807969691, rel=1e-11)
    assert res3.rubblemound_sumer_length == pytest.approx(7.73762511530375, rel=1e-11)
    assert res3.rubblemound_field_depth == pytest.approx(1.803, rel=1e-12)
    assert res3.rubblemound_field_length == pytest.approx(18.03, rel=1e-12)

    res3_phys = scour_structure_tip_waves(strict_excel=False)
    assert res3_phys.rubblemound_sumer_length == pytest.approx(
        10.0 * res3_phys.rubblemound_sumer_depth, rel=1e-12
    )

    res4 = scour_structure_tip_current_waves(h0=np.array([8.0, 15.0]), strict_excel=True)
    assert res4.kc_number[0] == pytest.approx(0.736659733121956, rel=1e-11)
    assert res4.rubblemound_depth[0] == pytest.approx(17.6, rel=1e-12)
    assert res4.rubblemound_depth[1] == pytest.approx(14.0, rel=1e-12)


def test_scour_river_groyne_sheet5():
    res = scour_river_groyne(
        h0=5.0,
        groyne_length=np.array([30.0, 60.0]),
        d50=0.00025,
        d90=0.0005,
        mud_fraction=0.05,
        turbulence_r=0.15,
        Hs=np.array([1.0, 0.0]),
        Tp=7.0,
        Q=1000.0,
        B_river=200.0,
        strict_excel=True,
    )
    assert res.u_eff[0] == pytest.approx(1.30211449287422, rel=1e-11)
    assert res.ucr_cw[0] == pytest.approx(0.329104093798216, rel=1e-11)
    assert res.depth_F9_6_7a[0] == pytest.approx(1.70667919004131, rel=1e-11)
    assert res.depth_lacey[0] == pytest.approx(10.8980662662178, rel=1e-11)
    assert res.depth_rh[0] == pytest.approx(6.23948195471335, rel=1e-11)
    assert res.depth_clm[0] == pytest.approx(31.0247867268909, rel=1e-11)
    assert res.depth_F9_6_8_9[0] == pytest.approx(12.9423884385612, rel=1e-11)
    assert res.scour_length[0] == pytest.approx(129.423884385612, rel=1e-11)

    assert res.depth_F9_6_7a[1] == pytest.approx(2.41687649213306, rel=1e-11)
    assert res.depth_lacey[1] == pytest.approx(9.68351109661465, rel=1e-11)
    assert res.depth_rh[1] == pytest.approx(8.05951116128406, rel=1e-11)
    assert res.depth_clm[1] == pytest.approx(34.6410161513775, rel=1e-11)
    assert res.depth_F9_6_8_9[1] == pytest.approx(17.25, rel=1e-12)


def test_scour_bridge_pier_pile_and_pipe_sheets6_to_8():
    res6 = scour_bridge_pier(
        h0=5.0,
        pier_width=np.array([5.0, 2.0]),
        d50=np.array([0.0002, 0.00025]),
        alpha3=np.array([1.3, 1.0]),
        Tp=np.array([5.0, 0.0]),
        strict_excel=True,
    )
    assert res6.alpha2[0] == pytest.approx(1.52318831191153, rel=1e-11)
    assert res6.qt[0] == pytest.approx(18.1937157579114, rel=1e-11)
    assert res6.scour_depth[0] == pytest.approx(11.3858326315387, rel=1e-11)
    assert res6.scour_length[0] == pytest.approx(113.858326315387, rel=1e-11)
    assert res6.scour_depth[1] == pytest.approx(4.53842577149658, rel=1e-11)

    res7 = scour_vertical_pile_waves(
        u=0.3,
        h0=np.array([5.0, 20.0]),
        Hs=np.array([1.0, 8.0]),
        Tp=np.array([5.0, 10.0]),
        strict_excel=True,
    )
    assert res7.alpha2[0] == pytest.approx(1.11114140945028, rel=1e-11)
    assert res7.kc_number[0] == pytest.approx(4.05690096462541, rel=1e-11)
    assert res7.qt[0] == pytest.approx(0.0446785920724265, rel=1e-11)
    assert res7.scour_depth[0] == pytest.approx(0.270467439350045, rel=1e-11)
    assert res7.kc_number[1] == pytest.approx(23.4552077170033, rel=1e-11)
    assert res7.scour_depth[1] == pytest.approx(1.59870643019148, rel=1e-11)

    res8 = scour_horizontal_pipe(
        u=np.array([0.3, 1.0]),
        h0=np.array([5.0, 20.0]),
        Hs=np.array([1.0, 8.0]),
        Tp=np.array([5.0, 10.0]),
        strict_excel=True,
    )
    assert res8.scour_depth_current[0] == pytest.approx(0.686445389519347, rel=1e-11)
    assert res8.scour_depth_waves[0] == pytest.approx(0.402835001688057, rel=1e-11)
    assert res8.scour_depth_combined[0] == pytest.approx(0.447606651551574, rel=1e-11)
    assert res8.scour_length[0] == pytest.approx(6.86445389519347, rel=1e-11)
    assert res8.scour_depth_current[1] == pytest.approx(1.11111111111111, rel=1e-11)
    assert res8.scour_depth_combined[1] == pytest.approx(2.09259670313294, rel=1e-11)
