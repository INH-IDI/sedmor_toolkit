"""Tests para hidraulica fluvial, transporte en rios, perfiles medidos y transporte costero."""

from __future__ import annotations

import numpy as np
import pytest

from sedmor.coastal_transport import (
    DEFAULT_WAVE_CLASSES,
    annual_sand_transport,
    sand_transport_current_waves,
    wave_averaged_sand_transport,
)
from sedmor.river_transport import (
    compound_channel_transport,
    integrate_measured_suspended_transport,
)
from sedmor.rivers import (
    compound_channel_backwater,
    compound_channel_flow,
    free_overfall_brink_depth,
    solve_trapezoidal_depth,
    trapezoidal_channel_discharge,
)

COMPOUND_ZB = [4, 3, 3, 3, 3, 3, 1.5, 0, 0, 0, 0, 0, 1.5, 3, 3, 3, 3, 3, 4]
COMPOUND_BY = [2, 20, 20, 20, 20, 20, 5, 20, 20, 20, 20, 20, 5, 20, 20, 20, 20, 20, 2]
COMPOUND_ANG = [30, 0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 30]


def test_trapezoidal_channel_matches_excel():
    res1 = trapezoidal_channel_discharge(
        water_depth=1.0,
        bottom_width=20.0,
        side_angle_deg=45.0,
        slope=0.0001,
        ks=0.1,
        strict_excel=True,
    )
    assert res1.wetted_perimeter == pytest.approx(22.828421930064, rel=1e-11)
    assert res1.area == pytest.approx(21.0, rel=1e-12)
    assert res1.hydraulic_radius == pytest.approx(0.919905899073292, rel=1e-11)
    assert res1.shear_velocity == pytest.approx(0.0300404342011047, rel=1e-11)
    assert res1.chezy == pytest.approx(36.7726436984022, rel=1e-11)
    assert res1.discharge == pytest.approx(7.40654814813911, rel=1e-11)

    res2 = trapezoidal_channel_discharge(
        water_depth=4.65,
        bottom_width=20.0,
        side_angle_deg=45.0,
        slope=0.0001,
        ks=0.1,
        strict_excel=True,
    )
    assert res2.discharge == pytest.approx(100.434046141469, rel=1e-11)

    h_solved = solve_trapezoidal_depth(100.434046141469, bottom_width=20.0, side_angle_deg=45.0, slope=0.0001, ks=0.1)
    assert h_solved == pytest.approx(4.65, rel=1e-4)

    # Angulo != 45 deg demuestra la correccion geometrica en strict_excel=False
    non45_strict = trapezoidal_channel_discharge(2.0, 20.0, 30.0, 0.0001, 0.1, strict_excel=True)
    non45_phys = trapezoidal_channel_discharge(2.0, 20.0, 30.0, 0.0001, 0.1, strict_excel=False)
    assert non45_phys.area == pytest.approx(20.0 * 2.0 + 4.0 / np.tan(np.deg2rad(30.0)))
    assert non45_phys.area > non45_strict.area


def test_compound_channel_flow_transport_and_backwater():
    flow = compound_channel_flow(
        water_level=5.0,
        slope=3e-6,
        bed_levels=COMPOUND_ZB,
        widths=COMPOUND_BY,
        ks_values=[0.01] * 19,
        strict_excel=True,
    )
    assert flow.total_width == pytest.approx(314.0)
    assert flow.total_area == pytest.approx(939.0)
    assert flow.total_discharge == pytest.approx(199.089579068068, rel=1e-11)

    tr = compound_channel_transport(
        water_level=5.0,
        slope=3e-6,
        bed_levels=COMPOUND_ZB,
        widths=COMPOUND_BY,
        ks_values=[0.01] * 19,
        d50=0.0003,
        side_angles_deg=COMPOUND_ANG,
        strict_excel=True,
    )
    assert tr.total_qs_vol == pytest.approx(9.45416898063827e-06, rel=1e-10)
    assert tr.total_qs_mass == pytest.approx(0.0250535477986914, rel=1e-10)
    assert tr.mean_concentration == pytest.approx(0.000125840578477117, rel=1e-10)
    assert tr.annual_solid_volume_m3 == pytest.approx(298.146672973408, rel=1e-10)
    assert tr.bed_shear_stress[0] == pytest.approx(0.02943, rel=1e-12)
    assert tr.critical_shear_stress[0] == pytest.approx(0.0951815524187077, rel=1e-10)

    # Hoja Flow and transport capacity in river channel (H=6, I=0.0001, ks=0.05)
    tr2 = compound_channel_transport(
        water_level=6.0,
        slope=0.0001,
        bed_levels=COMPOUND_ZB,
        widths=COMPOUND_BY,
        ks_values=[0.05] * 19,
        d50=0.0003,
        strict_excel=True,
    )
    assert tr2.flow.total_area == pytest.approx(1253.0)
    assert tr2.flow.total_discharge == pytest.approx(1427.60818657907, rel=1e-11)
    assert tr2.total_qs_vol == pytest.approx(0.0744303766447343, rel=1e-10)
    assert tr2.total_qs_mass == pytest.approx(197.240498108546, rel=1e-10)
    assert tr2.mean_concentration == pytest.approx(0.138161506751504, rel=1e-10)

    # Curva de remanso
    bw = compound_channel_backwater(strict_excel=True)
    assert bw.dh_dx[0] == pytest.approx(-9.93421552191387e-05, rel=1e-10)
    assert bw.water_depth[1] == pytest.approx(4.02483553880478, rel=1e-11)
    assert bw.x[-1] == pytest.approx(-134500.0)
    assert bw.water_depth[-1] == pytest.approx(4.72830666484294, rel=1e-6)


def test_free_overfall_and_measured_suspended():
    ov = free_overfall_brink_depth(np.array([1.0, 2.0, 5.0]))
    assert ov.gamma_rectangular[0] == pytest.approx(0.666666666666667, rel=1e-12)
    assert ov.gamma_rectangular[2] == pytest.approx(0.980392156862745, rel=1e-12)
    g_h = ov.gamma_hager
    assert (3.0 * g_h[0] - g_h[0] ** 3) / (1.0 - g_h[0]) == pytest.approx(6.0, rel=1e-9)
    assert g_h[0] == pytest.approx(0.706, abs=1e-3)
    assert g_h[1] == pytest.approx(0.918, abs=1e-3)

    # Caso 1 Measured Suspended transport.xls
    z1 = [0.0, 0.01, 0.02, 0.03, 0.045, 0.07, 0.1, 0.13, 0.17, 0.255]
    u1 = [0.025, 0.05, 0.075, 0.12, 0.14, 0.17, 0.18, 0.2, 0.21, 0.21]
    c1 = [3.8, 3.8, 2.3, 1.8, 1.2, 0.75, 0.38, 0.22, 0.18, 0.0]
    m1 = integrate_measured_suspended_transport(z1, u1, c1)
    assert m1.total_suspended_transport == pytest.approx(0.01962075, rel=1e-12)
    assert m1.depth_averaged_velocity == pytest.approx(0.175686274509804, rel=1e-12)


def test_coastal_sand_transport_current_waves_and_annual():
    res = sand_transport_current_waves(
        depth=5.0,
        velocity=0.4,
        wave_height=1.0,
        wave_period=6.0,
        d50=0.00025,
        d90=0.0005,
        p_mud=0.05,
        ks=0.02,
        rho_w=1000.0,
        strict_excel=True,
    )
    assert res.d_star == pytest.approx(6.32392323526782, rel=1e-10)
    assert res.u_orb == pytest.approx(0.569174730119191, rel=1e-9)
    assert res.u_eff == pytest.approx(0.627669892047676, rel=1e-9)
    assert res.u_cr_weighted == pytest.approx(0.28083788224356, rel=1e-9)
    assert res.qb == pytest.approx(0.00663305773941775, rel=1e-8)
    assert res.qs == pytest.approx(0.0390159309796629, rel=1e-8)
    assert res.qt == pytest.approx(0.0456489887190807, rel=1e-8)

    # comp. wave-averaged transport con las 10 clases propias de la hoja y h=8, u=0.5, ks=0.03
    hs_wa = np.array([0.5, 0.75, 1.25, 1.75, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    tp_wa = np.array([4.0, 4.0, 5.0, 6.0, 6.5, 7.0, 7.0, 7.0, 7.0, 7.0])
    prob_wa = np.array([0.4432, 0.273, 0.163, 0.077, 0.0383, 0.0055, 0.0, 0.0, 0.0, 0.0])
    wa_04 = wave_averaged_sand_transport(
        8.0, 0.5, hs_wa, tp_wa, prob_wa, d50=0.00025, d90=0.0005, ks=0.03, strict_excel=True
    )
    assert wa_04.qb_total == pytest.approx(0.00997156675762369, rel=1e-7)
    assert wa_04.qs_total == pytest.approx(0.0635160629092836, rel=1e-7)
    assert wa_04.qt_total == pytest.approx(0.0734876296669072, rel=1e-7)

    # Fila 37: ola representativa unica (Hs=1, Tp=5, prob=1) con orbital_factor=0.5
    wa_05 = wave_averaged_sand_transport(
        8.0, 0.5, 1.0, 5.0, 1.0, d50=0.00025, d90=0.0005, ks=0.03, orbital_factor=0.5, strict_excel=True
    )
    assert wa_05.qb_total == pytest.approx(0.0132636911558274, rel=1e-7)
    assert wa_05.qs_total == pytest.approx(0.0844859651788169, rel=1e-7)
    assert wa_05.qt_total == pytest.approx(0.0977496563346444, rel=1e-7)

    # ANNUAL TRANSPORT
    ann_strict = annual_sand_transport(strict_excel=True)
    assert ann_strict.d_star == pytest.approx(4.99219760392497, rel=1e-10)
    assert ann_strict.flood_vol_incl_pores_m3_yr == pytest.approx(363.329471213296, rel=1e-7)
    assert ann_strict.flood_vol_solid_m3_yr == pytest.approx(217.997682727977, rel=1e-7)
    assert ann_strict.ebb_vol_incl_pores_m3_yr == pytest.approx(-218.919317511255, rel=1e-7)
    assert ann_strict.ebb_vol_solid_m3_yr == pytest.approx(-131.351590506753, rel=1e-7)
    assert ann_strict.net_vol_incl_pores_m3_yr == pytest.approx(144.41015370204, rel=1e-7)
    assert ann_strict.net_vol_solid_m3_yr == pytest.approx(86.6460922212243, rel=1e-7)

    # Fisica corregida: elimina el factor espurio 6x
    ann_phys = annual_sand_transport(strict_excel=False)
    assert ann_phys.net_vol_incl_pores_m3_yr == pytest.approx(ann_strict.net_vol_incl_pores_m3_yr / 6.0, rel=1e-3)


def test_measured_suspended_transport_calculation_xls():
    # 'susp sediment transport calcualtion.xls' (hoja 'Susp. transp')
    z = np.array([0.0, 0.01, 0.02, 0.03, 0.045, 0.07, 0.1, 0.13, 0.17, 0.255])
    u = np.array([0.025, 0.05, 0.075, 0.12, 0.14, 0.17, 0.18, 0.2, 0.21, 0.21])
    c = np.array([3.8, 3.8, 2.3, 1.8, 1.2, 0.75, 0.38, 0.22, 0.18, 0.0])

    res = integrate_measured_suspended_transport(z, u, c, bottom_sublayer_factor=0.75)
    assert res.total_suspended_transport == pytest.approx(0.01962075, rel=1e-12)
    assert res.depth_averaged_velocity == pytest.approx(0.175686274509804, rel=1e-12)
