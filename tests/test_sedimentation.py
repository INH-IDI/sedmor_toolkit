"""Pruebas unitarias para herramientas de sedimentación (``sedmor.sedimentation``)."""

from __future__ import annotations

import numpy as np
import pytest

from sedmor.sedimentation import (
    ReservoirSectionInput,
    channel_sedimentation,
    harbour_mud_sedimentation,
    harbour_sand_sedimentation,
    reservoir_sedimentation,
    stratified_channel_sedimentation,
    stratified_density_flow_profile,
    streamtube_sedimentation,
    streamtube_tidal_sedimentation,
)


def test_channel_sedimentation_all_five_workbooks() -> None:
    """Valida las 5 planillas SED PIT (*.xlsx) contra sus valores cacheados."""
    # 1. SED PIT.xlsx (canal base con arcilla, limo y arena)
    r_base = channel_sedimentation(
        length=1000.0,
        width=200.0,
        depth_in=20.0,
        depth_out=10.0,
        dur_h=[60.0] * 12,
        alpha0_deg=[90.0] * 12,
        b0=[1.0] * 12,
        dh_wl=[1.0] * 6 + [-1.0] * 6,
        u0=[1.0] * 12,
        hs=[0.0] * 12,
        tp=[7.0] * 12,
        c_clay=[0.1] * 12,
        c_silt=[0.05] * 12,
        d50=0.0002,
        d90=0.0003,
        p_sand=1.0,
        rho_w=1020.0,
        rho_s=2650.0,
        nu=1e-6,
        ws_clay=0.0001,
        ws_silt=0.001,
        ws_sand=0.02,
        ks_out=0.05,
        ks_in=0.05,
        cal_tr=0.67,
        cal_ucr=1.0,
        bulk_density_mode=0,
        bulk_density_const=1.2,
        trap_method=1,
        coef_van_rijn=0.25,
        strict_excel=True,
    )
    assert r_base.total_inflow_tons == pytest.approx(5112837.95983929, rel=1e-12)
    assert r_base.total_deposition_tons == pytest.approx(1010344.22383797, rel=1e-12)
    assert r_base.total_deposition_m3 == pytest.approx(666999.2253332, rel=1e-12)
    assert r_base.total_layer_thickness_m == pytest.approx(3.334996126666, rel=1e-12)

    # 2. SED PIT new.xlsx (arena pura)
    r_new = channel_sedimentation(
        length=500.0,
        width=300.0,
        depth_in=10.0,
        depth_out=6.0,
        dur_h=[60.0] * 12,
        alpha0_deg=[90.0] * 12,
        b0=[1.0] * 12,
        dh_wl=[1.0, 1.0, 1.5, 1.5, 1.0, 1.0, -1.0, -1.0, -1.5, -1.5, -1.0, -1.0],
        u0=[0.3, 0.6, 0.8, 0.8, 0.6, 0.3, 0.3, 0.6, 0.8, 0.8, 0.6, 0.3],
        hs=[0.0] * 12,
        tp=[7.0] * 12,
        c_clay=[0.0] * 12,
        c_silt=[0.0] * 12,
        ks_out=0.02,
        ks_in=0.02,
        cal_tr=1.0,
        bulk_density_mode=1,
        bulk_density_const=1.6,
        strict_excel=True,
    )
    assert r_new.total_inflow_tons == pytest.approx(118324.893841007, rel=1e-12)
    assert r_new.total_deposition_tons == pytest.approx(92290.5147348164, rel=1e-12)
    assert r_new.total_deposition_m3 == pytest.approx(57681.5717092603, rel=1e-12)
    assert r_new.total_layer_thickness_m == pytest.approx(0.384543811395068, rel=1e-12)

    # 3. SED PIT new Mar del plata.xlsx (incidencia oblicua a=50 deg, 15 años)
    r_mdp = channel_sedimentation(
        length=1000.0,
        width=300.0,
        depth_in=12.0,
        depth_out=6.0,
        dur_h=[10950.0] * 12,
        alpha0_deg=[50.0] * 12,
        b0=[1.0] * 12,
        dh_wl=[0.1, 0.3, 0.5, 0.5, 0.3, 0.1, -0.1, -0.3, -0.5, -0.5, -0.3, -0.1],
        u0=[0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.1, 0.15, 0.2, 0.2, 0.15, 0.1],
        hs=[1.3] * 12,
        tp=[9.0] * 12,
        c_clay=[0.0] * 12,
        c_silt=[0.0] * 12,
        strict_excel=True,
    )
    assert r_mdp.total_inflow_tons == pytest.approx(4975590.29572113, rel=1e-12)
    assert r_mdp.total_deposition_tons == pytest.approx(2612999.17953692, rel=1e-12)
    assert r_mdp.total_deposition_m3 == pytest.approx(1633124.48721057, rel=1e-12)
    assert r_mdp.total_layer_thickness_m == pytest.approx(5.44374829070191, rel=1e-12)

    # 4. SED PIT Gujarat India.xlsx (con min_depth=0.1 y update_depth=False en Excel)
    r_guj = channel_sedimentation(
        length=450.0,
        width=30.0,
        depth_in=2.3,
        depth_out=1.0,
        dur_h=[60.0] * 12,
        alpha0_deg=[90.0] * 12,
        b0=[1.0] * 12,
        dh_wl=[1.0, 2.0, 3.0, 3.0, 2.0, 1.0, -1.0, -2.0, -3.0, -3.0, -2.0, -1.0],
        u0=[1.0, 1.0, 1.5, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        hs=[0.0] * 12,
        tp=[7.0] * 12,
        c_clay=[0.2] * 6 + [0.05] * 6,
        c_silt=[0.05] * 6 + [0.01] * 6,
        d50=0.0001,
        d90=0.0002,
        p_sand=0.1,
        ws_clay=0.001,
        ws_silt=0.001,
        ws_sand=0.01,
        ks_out=0.01,
        ks_in=0.01,
        bulk_density_mode=0,
        bulk_density_const=1.2,
        min_depth=0.1,
        update_depth=False,
        strict_excel=True,
    )
    assert r_guj.total_inflow_tons == pytest.approx(586990.439048034, rel=1e-12)
    assert r_guj.total_deposition_tons == pytest.approx(22397.8237556759, rel=1e-12)
    assert r_guj.total_deposition_m3 == pytest.approx(20271.057345981, rel=1e-12)
    assert r_guj.total_layer_thickness_m == pytest.approx(1.501559803406, rel=1e-12)

    # 5. SED PIT trial trench Scheveningen 3.xlsx
    r_sch = channel_sedimentation(
        length=1000.0,
        width=20.0,
        depth_in=11.0,
        depth_out=8.0,
        dur_h=[96.0] * 12,
        alpha0_deg=[90.0] * 12,
        b0=[1.0] * 12,
        dh_wl=[0.5] * 6 + [-0.5] * 6,
        u0=[0.3, 0.3, 0.6, 0.6, 0.3, 0.3, 0.25, 0.25, 0.5, 0.5, 0.25, 0.25],
        hs=[1.0] * 12,
        tp=[7.0] * 12,
        c_clay=[0.01] * 12,
        c_silt=[0.01] * 12,
        p_sand=0.95,
        ws_sand=0.018,
        ks_out=0.05,
        ks_in=0.05,
        cal_tr=0.67,
        cal_ucr=0.5,
        bulk_density_mode=0,
        bulk_density_const=1.2,
        strict_excel=True,
    )
    assert r_sch.total_inflow_tons == pytest.approx(329861.422365491, rel=1e-12)
    assert r_sch.total_deposition_tons == pytest.approx(16231.3279088661, rel=1e-12)
    assert r_sch.total_deposition_m3 == pytest.approx(10636.2563268181, rel=1e-12)
    assert r_sch.total_layer_thickness_m == pytest.approx(0.531812816340904, rel=1e-12)

    # Verificar coherencia de modo estándar (strict_excel=False) y Eysink-Vermaas
    r_std = channel_sedimentation(
        length=1000.0,
        width=300.0,
        depth_in=12.0,
        depth_out=6.0,
        dur_h=[10950.0] * 12,
        alpha0_deg=[50.0] * 12,
        b0=[1.0] * 12,
        dh_wl=[0.1, 0.3, 0.5, 0.5, 0.3, 0.1, -0.1, -0.3, -0.5, -0.5, -0.3, -0.1],
        u0=[0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.1, 0.15, 0.2, 0.2, 0.15, 0.1],
        hs=[1.3] * 12,
        tp=[9.0] * 12,
        c_clay=[0.0] * 12,
        c_silt=[0.0] * 12,
        trap_method=2,
        strict_excel=False,
    )
    assert r_std.total_deposition_m3 > 0.0
    assert r_std.trap_efficiency_volume < 1.0


def test_stratified_parallel_flow_shipping_channel() -> None:
    """Sedimentation due to parallel stratied flow in tidal shipping channel.xlsx."""
    prof = stratified_density_flow_profile(
        depth=16.0,
        u_river=0.2,
        u_peak=1.0,
        d_rho_dx=0.0001,
        chezy=50.0,
        gamma=0.005,
        z_rel=[0.01, 0.1, 0.3, 0.5, 1.0],
        strict_excel=True,
    )
    assert prof.m_sal == pytest.approx(0.00260833333333333, rel=1e-12)
    assert prof.u_sal[0] == pytest.approx(-0.00163605795555556, rel=1e-12)
    assert prof.u_sal[2] == pytest.approx(-0.0230368, rel=1e-12)
    assert prof.u_river[-1] == pytest.approx(0.330208266666667, rel=1e-12)

    for eff, exp_vol, exp_lay in [
        (0.10, 388800.0, 0.1944),
        (0.25, 972000.0, 0.4860),
        (0.55, 2138400.0, 1.0692),
    ]:
        sed = stratified_channel_sedimentation(
            depth=20.0,
            u_mean=0.1,
            c_mean=0.75,
            rho_dry=300.0,
            width=400.0,
            length=5000.0,
            trapping_efficiency=eff,
            duration_days=30.0,
            safety_factor=1.5,
        )
        assert sed.sediment_flux_kg_s == pytest.approx(300.0)
        assert sed.deposited_volume_m3 == pytest.approx(exp_vol, rel=1e-12)
        assert sed.deposited_layer_m == pytest.approx(exp_lay, rel=1e-12)


def test_reservoir_sedimentation_all_three_sheets() -> None:
    """Valida SEDRES.xlsx en sus 3 hojas (VanRijn, Eysink Vermaas, Borland)."""
    sections = [
        ReservoirSectionInput(1050.0, 170.0, 8.2),
        ReservoirSectionInput(950.0, 300.0, 8.5),
        ReservoirSectionInput(550.0, 415.0, 4.5),
        ReservoirSectionInput(1350.0, 245.0, 10.0),
        ReservoirSectionInput(1000.0, 340.0, 11.7),
    ]
    common = dict(
        sections=sections,
        days=[180.0] * 12,
        discharge=[100.0] * 12,
        h0=[3.0] * 12,
        c_clay=[0.175] * 12,
        c_silt=[0.1] * 12,
        c_sand=[0.025] * 12,
        river_width=70.0,
        ws_clay=0.0001,
        ws_silt=0.001,
        ws_sand=0.01,
        chezy=70.0,
        corr_eq=0.0,
        strict_excel=True,
    )

    r_vr = reservoir_sedimentation(
        **common, coef_trap=0.25, bulk_density_mode=1, bulk_density_const=0.8, method="van_rijn"
    )
    assert r_vr.total_deposition_capped_m3 == pytest.approx(6500677.70170297, rel=1e-12)
    assert r_vr.total_deposition_uncapped_m3 == pytest.approx(6952454.8161539, rel=1e-12)
    assert r_vr.overall_trapping_efficiency == pytest.approx(0.928880558656688, rel=1e-12)

    r_ev = reservoir_sedimentation(
        **common, coef_trap=0.06, bulk_density_mode=0, bulk_density_const=1.2, method="eysink_vermaas"
    )
    assert r_ev.total_deposition_capped_m3 == pytest.approx(6004864.14890917, rel=1e-12)
    assert r_ev.total_deposition_uncapped_m3 == pytest.approx(6023990.89062973, rel=1e-12)
    assert r_ev.overall_trapping_efficiency == pytest.approx(0.981921825988948, rel=1e-12)

    r_bo = reservoir_sedimentation(
        **common, coef_trap=1.055, bulk_density_mode=0, bulk_density_const=1.2, method="borland"
    )
    assert r_bo.total_deposition_capped_m3 == pytest.approx(5259073.41144482, rel=1e-12)
    assert r_bo.total_deposition_uncapped_m3 == pytest.approx(5404396.4227523, rel=1e-12)
    assert r_bo.overall_trapping_efficiency == pytest.approx(0.859969324720511, rel=1e-12)

    # Con corr_eq > 0 se verifica también la diferencia de strict_excel=False en U0
    r_eq = reservoir_sedimentation(
        sections=sections,
        days=[180.0] * 12,
        discharge=[100.0] * 12,
        h0=[3.0] * 12,
        c_clay=[0.175] * 12,
        c_silt=[0.1] * 12,
        c_sand=[0.025] * 12,
        corr_eq=1.0,
        strict_excel=False,
    )
    assert r_eq.total_deposition_capped_m3 > 0.0


def test_harbour_sand_sedimentation() -> None:
    """Valida SED HARBOUR-SAND.xlsx en modo strict_excel=True y False."""
    kwargs = dict(
        surge=[0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        u=[0.1, 0.3, 0.5, 0.7, 0.8, 1.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        hs=[0.5, 0.75, 1.5, 2.25, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        tp=[5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        days=[70.0, 50.0, 25.0, 15.0, 4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        c_clay=[0.0] * 12,
        c_silt=[0.0, 0.0, 0.0, 0.01, 0.05, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    res_xls = harbour_sand_sedimentation(**kwargs, strict_excel=True)
    assert float(np.sum(res_xls.deposit_bed_m3)) == pytest.approx(8476.77358017049, rel=1e-11)
    assert float(np.sum(res_xls.deposit_silt_m3)) == pytest.approx(13879.35, rel=1e-11)
    assert float(np.sum(res_xls.deposit_sand_m3)) == pytest.approx(62632.8976037261, rel=1e-11)
    assert res_xls.total_deposition_m3 == pytest.approx(84989.0211838966, rel=1e-11)

    res_std = harbour_sand_sedimentation(**kwargs, strict_excel=False)
    # Corrigiendo la omisión de * days, las contribuciones con days>1 crecen sustancialmente
    assert res_std.total_deposition_m3 > res_xls.total_deposition_m3


def test_harbour_mud_sedimentation_both_workbooks() -> None:
    """Valida SED HARBOURS- MUD new.xlsx y Swinemunde."""
    r_mud = harbour_mud_sedimentation(strict_excel=True)
    assert r_mud.deposition_last_tide_kg == pytest.approx(411707.968330779, rel=1e-12)
    assert r_mud.deposition_per_year_kg == pytest.approx(292312657.514853, rel=1e-12)
    assert r_mud.deposition_per_year_m3 == pytest.approx(835179.021471009, rel=1e-12)
    assert r_mud.layer_thickness_per_year_m == pytest.approx(0.379626827941368, rel=1e-12)
    assert r_mud.exchange_total_m3 == pytest.approx(-17780.6395765176, rel=1e-12)
    assert r_mud.exchange_per_year_m3 == pytest.approx(-3156063.52483188, rel=1e-12)

    r_swin = harbour_mud_sedimentation(
        dt_s=300.0,
        depth_msl=10.0,
        entrance_width=300.0,
        basin_area=1_000_000.0,
        u_peak_out=0.3,
        tidal_amplitude=0.2,
        c_basin_init=0.01,
        c_out_mean=0.02,
        c_out_amp=0.01,
        delta_rho_max=0.0,
        rho_out=1025.0,
        ws=0.0001,
        rho_bulk_dry=500.0,
        tidal_period_s=43200.0,
        f_exchange_flood=0.025,
        f_exchange_ebb=0.0,
        hs_rel_harbour=0.05,
        q_freshwater=0.0,
        mixing_factor=2.0,
        freshwater_time_fraction=0.6,
        strict_excel=True,
    )
    assert r_swin.deposition_per_year_kg == pytest.approx(8322212.58300932, rel=1e-12)
    assert r_swin.exchange_per_year_m3 == pytest.approx(-15966.3899877667, rel=1e-11)

    # En strict_excel=False el último ciclo completo (144 pasos) casi duplica la media marea
    r_clean = harbour_mud_sedimentation(strict_excel=False)
    assert r_clean.deposition_last_tide_kg > 1.8 * r_mud.deposition_last_tide_kg


def test_streamtube_steady_all_four_workbooks() -> None:
    """Valida los 4 casos estacionarios/morfológicos de SED TUBE*.xlsx."""
    # 1. Case4 morphology (283 nodos, 50 pasos)
    x_c4 = np.arange(283, dtype=np.float64) * 5.0
    h_c4 = np.interp(x_c4, [0.0, 100.0, 350.0, 600.0, 1410.0], [10.0, 10.0, 5.0, 10.0, 10.0])
    r_c4 = streamtube_sedimentation(
        x=x_c4,
        width=np.ones(283),
        h_init=h_c4,
        hs=np.ones(283),
        tp=np.full(283, 6.0),
        discharge=10.0,
        bc_mode=1,
        bc_qsus=5.0,
        d50=0.0002,
        d90=0.0003,
        p_silt=0.05,
        rho_w=1020.0,
        rho_s=2650.0,
        nu=1e-6,
        ws=0.015,
        ks=0.03,
        porosity=0.4,
        cal_tr=0.67,
        cal_ucr=1.0,
        dt_hours=0.5,
        coef_adapt=0.1,
        n_steps=50,
        strict_excel=True,
    )
    assert r_c4.h_final_m[[0, 30, 50, 80, 120, 200, 282]] == pytest.approx(
        [10.0, 9.12621206218916, 7.52378489046653, 6.52110925209389, 9.42903065252572, 9.91766380587593, 9.98875348991119],
        abs=1e-12,
    )
    assert float(np.sum(r_c4.h_final_m)) == pytest.approx(2580.4561149615197, rel=1e-12)

    # 2. SED TUBE india.xlsx (un paso desde perfil uniforme h_tot = 4.3 m)
    x_in = np.arange(283, dtype=np.float64) * 2.0
    r_in = streamtube_sedimentation(
        x=x_in,
        width=np.full(283, 20.0),
        h_init=np.full(283, 4.3),
        hs=np.zeros(283),
        tp=np.full(283, 1.5),
        discharge=20.0,
        bc_mode=2,
        bc_qsus=30.0,
        d50=2e-5,
        d90=5e-5,
        p_silt=0.0,
        rho_w=1000.0,
        rho_s=2650.0,
        nu=1e-6,
        ws=0.0005,
        ks=0.01,
        porosity=0.85,
        cal_tr=1.0,
        cal_ucr=1.0,
        dt_hours=0.02,
        coef_adapt=0.1,
        n_steps=1,
        qbed_uses_cal_ucr=True,
        strict_excel=True,
    )
    assert float(np.sum(r_in.h_final_m)) == pytest.approx(1216.834637203731, rel=1e-12)

    # 3. SED TUBE lab1 havinga morphology.xlsx (255 pasos)
    x_lh = np.arange(283, dtype=np.float64) * 0.25
    h_lh = np.interp(
        x_lh,
        [0.0, 5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0, 7.5, 7.75, 8.0, 8.25, 8.75, 9.0, 9.25, 9.75, 10.0, 70.5],
        [0.42, 0.42, 0.45, 0.49, 0.52, 0.545, 0.585, 0.62, 0.625, 0.62, 0.55, 0.52, 0.495, 0.46, 0.43, 0.43, 0.44, 0.44, 0.445, 0.445],
    )
    r_lh = streamtube_sedimentation(
        x=x_lh,
        width=np.ones(283),
        h_init=h_lh,
        hs=np.full(283, 0.105),
        tp=np.full(283, 2.2),
        discharge=0.103,
        bc_mode=1,
        bc_qsus=5.0,
        d50=0.0001,
        d90=0.00013,
        p_silt=0.05,
        rho_w=1000.0,
        rho_s=2650.0,
        nu=1e-6,
        ws=0.006,
        ks=0.01,
        porosity=0.4,
        cal_tr=0.67,
        cal_ucr=2.2,
        dt_hours=0.1,
        coef_adapt=0.1,
        n_steps=255,
        strict_excel=True,
    )
    assert float(np.sum(r_lh.h_final_m)) == pytest.approx(125.59069751371209, rel=1e-12)

    # 4. SED TUBE mud.xlsx (3 pasos)
    x_md = np.arange(283, dtype=np.float64) * 5.0
    r_md = streamtube_sedimentation(
        x=x_md,
        width=np.ones(283),
        h_init=np.ones(283),
        hs=np.zeros(283),
        tp=np.full(283, 1.5),
        discharge=0.3,
        bc_mode=2,
        bc_qsus=15.0,
        d50=5e-5,
        d90=1.2e-5,
        p_silt=0.0,
        rho_w=1000.0,
        rho_s=2650.0,
        nu=1e-6,
        ws=0.0005,
        ks=0.001,
        porosity=0.85,
        cal_tr=1.0,
        cal_ucr=1.0,
        dt_hours=0.0001,
        coef_adapt=0.05,
        n_steps=3,
        qbed_uses_cal_ucr=True,
        strict_excel=True,
    )
    assert float(np.sum(r_md.h_final_m)) == pytest.approx(282.99227386416993, rel=1e-12)


@pytest.mark.parametrize(
    ("name", "dx", "bp_x", "bp_h", "params", "exp_samples", "exp_sum"),
    [
        (
            "Delfzijl",
            5.0,
            [0.0, 500.0, 600.0, 800.0, 900.0, 1410.0],
            [6.0, 6.0, 10.0, 10.0, 6.0, 6.0],
            dict(
                q_flood=6.0, eta_flood=1.5, hs_flood=0.0, tp_flood=7.0,
                q_ebb=3.6, eta_ebb=-1.5, hs_ebb=0.0, tp_ebb=7.0,
                d50=0.00015, d90=0.0003, p_silt=0.0, rho_w=1020.0, rho_s=2650.0,
                nu=1e-6, ws=0.015, ks=0.03, porosity=0.4, cal_tr=1.0, cal_ucr=1.0,
                slope_factor=1.0, dt_hours=2.0, coef_adapt=0.2, n_half_steps=1, n_tides=360,
            ),
            [6.14683373657738, 6.24396178628463, 6.36383438219406, 6.67506528915905, 8.2081277852226, 6.29168712923919, 6.02695518098828],
            1934.7190522332135,
        ),
        (
            "Mar del Plata",
            2.0,
            [0.0, 150.0, 174.0, 374.0, 398.0, 564.0],
            [6.0, 6.0, 12.0, 12.0, 6.0, 6.0],
            dict(
                q_flood=2.0, eta_flood=0.5, hs_flood=1.5, tp_flood=9.0,
                q_ebb=1.65, eta_ebb=-0.5, hs_ebb=1.5, tp_ebb=9.0,
                d50=0.0002, d90=0.0003, p_silt=0.0, rho_w=1020.0, rho_s=2650.0,
                nu=1e-6, ws=0.02, ks=0.03, porosity=0.4, cal_tr=1.5, cal_ucr=0.5,
                slope_factor=0.0, dt_hours=4.0, coef_adapt=0.1, n_half_steps=1, n_tides=7300,
            ),
            [7.85227195749409, 7.96445299625615, 8.11364214321534, 8.4826341979052, 9.3254815050131, 7.73727356926991, 7.29931620763637],
            2304.5654501961208,
        ),
        (
            "Scheveningen",
            2.0,
            [0.0, 150.0, 156.0, 160.0, 162.0, 164.0, 168.0, 172.0, 174.0, 176.0, 180.0, 184.0, 188.0, 190.0, 198.0, 200.0, 564.0],
            [8.0, 8.0, 8.9, 9.3, 10.1, 11.0, 11.5, 11.5, 11.0, 10.3, 9.1, 8.7, 8.5, 8.3, 8.1, 8.0, 8.0],
            dict(
                q_flood=4.0, eta_flood=0.5, hs_flood=0.75, tp_flood=6.0,
                q_ebb=3.8, eta_ebb=-0.5, hs_ebb=0.75, tp_ebb=6.0,
                d50=0.0002, d90=0.0003, p_silt=0.0, rho_w=1020.0, rho_s=2650.0,
                nu=1e-6, ws=0.015, ks=0.01, porosity=0.4, cal_tr=0.67, cal_ucr=0.5,
                slope_factor=0.0, dt_hours=2.0, coef_adapt=0.2, n_half_steps=1, n_tides=340,
            ),
            [8.02128307620066, 8.06957843632804, 8.15710482224384, 8.92770525458677, 8.07311530935255, 8.00110701691999, 8.00001427806243],
            2298.4779004576535,
        ),
        (
            "trench_noordzee",
            1.0,
            [0.0, 67.0, 72.0, 82.0, 87.0, 282.0],
            [28.0, 28.0, 28.5, 28.5, 28.0, 28.0],
            dict(
                q_flood=17.0, eta_flood=0.5, hs_flood=1.5, tp_flood=7.0,
                q_ebb=17.0, eta_ebb=-0.5, hs_ebb=1.5, tp_ebb=7.0,
                d50=0.0002, d90=0.0003, p_silt=0.0, rho_w=1020.0, rho_s=2650.0,
                nu=1e-6, ws=0.015, ks=0.03, porosity=0.4, cal_tr=1.0, cal_ucr=1.0,
                slope_factor=1.0, dt_hours=2.0, coef_adapt=0.2, n_half_steps=1, n_tides=340,
            ),
            [28.0059333565045, 28.007139950522, 28.0081159362509, 28.3348840489872, 28.0059191238484, 28.0033776852329, 28.0019567813578],
            7930.466926602424,
        ),
        (
            "westerschelde",
            100.0,
            [0.0, 5900.0, 6100.0, 10000.0, 10200.0, 28200.0],
            [15.0, 15.0, 14.0, 14.0, 15.0, 15.0],
            dict(
                q_flood=17.0, eta_flood=1.0, hs_flood=0.0, tp_flood=6.0,
                q_ebb=15.0, eta_ebb=-1.0, hs_ebb=0.0, tp_ebb=6.0,
                d50=0.0002, d90=0.0003, p_silt=0.0, rho_w=1020.0, rho_s=2650.0,
                nu=1e-6, ws=0.015, ks=0.05, porosity=0.4, cal_tr=1.0, cal_ucr=0.7,
                slope_factor=1.0, dt_hours=8.0, coef_adapt=0.05, n_half_steps=1, n_tides=365,
            ),
            [14.979275286133, 14.8610703088141, 14.6223356539263, 14.4053857001636, 14.8463181158674, 14.9996637556276, 14.9999998006818],
            4204.15240343265,
        ),
    ],
)
def test_streamtube_tidal_all_five_workbooks(
    name: str,
    dx: float,
    bp_x: list[float],
    bp_h: list[float],
    params: dict[str, float | int],
    exp_samples: list[float],
    exp_sum: float,
) -> None:
    """Valida las 5 planillas mareales SED TUBE (MorphodynamicRun)."""
    x = np.arange(283, dtype=np.float64) * dx
    h0 = np.interp(x, bp_x, bp_h)
    res = streamtube_tidal_sedimentation(
        x=x,
        width=np.ones(283),
        h_msl_init=h0,
        **params,
        strict_excel=True,
    )
    assert res.h_msl_final_m[[0, 30, 50, 80, 120, 200, 282]] == pytest.approx(exp_samples, abs=1e-12), name
    assert float(np.sum(res.h_msl_final_m)) == pytest.approx(exp_sum, rel=1e-12), name


def test_streamtube_tidal_clean_mode() -> None:
    """En strict_excel=False con olas activas corrige el factor (1 + Hs/h)^2 en todos los nodos."""
    x = np.arange(283, dtype=np.float64) * 5.0
    h0 = np.interp(x, [0.0, 500.0, 600.0, 800.0, 900.0, 1410.0], [6.0, 6.0, 10.0, 10.0, 6.0, 6.0])
    res_clean = streamtube_tidal_sedimentation(
        x=x,
        width=np.ones(283),
        h_msl_init=h0,
        q_flood=6.0, eta_flood=1.5, hs_flood=1.0, tp_flood=7.0,
        q_ebb=3.6, eta_ebb=-1.5, hs_ebb=1.0, tp_ebb=7.0,
        d50=0.00015, d90=0.0003, p_silt=0.0, rho_w=1020.0, rho_s=2650.0,
        nu=1e-6, ws=0.015, ks=0.03, porosity=0.4, cal_tr=1.0, cal_ucr=1.0,
        dt_hours=2.0, coef_adapt=0.2, n_half_steps=1, n_tides=20, strict_excel=False,
    )
    assert np.all(np.isfinite(res_clean.h_msl_final_m))
    assert float(np.max(res_clean.dh_total_m)) > 0.0
