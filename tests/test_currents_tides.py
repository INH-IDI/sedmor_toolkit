"""Tests de verificacion contra valores cacheados de Excel para corrientes y mareas."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sedmor import currents, tides


class TestCurrents:
    """Verifica las planillas de asimetria mareal, viento y circulacion estratificada."""

    def test_tidal_asymmetry_m2_m4_excel_parity(self) -> None:
        res = currents.tidal_asymmetry_m2_m4(
            u1=1.0, u2=0.3, beta_deg=360.0, period_hours=12.0, strict_excel=True
        )
        assert res.u_total[0] == pytest.approx(1.29999847807401, rel=1e-10)
        assert res.u_total[-1] == pytest.approx(-0.311880318744021, rel=1e-10)

        # Tabla de referencia: U2/U1 = 0.2, beta = 30 deg -> t_flood/T ~ 0.448, Ufl/Uebb ~ 1.36
        ref = currents.tidal_asymmetry_m2_m4(u1=1.0, u2=0.2, beta_deg=30.0)
        assert ref.flood_duration_fraction == pytest.approx(0.448, abs=0.005)
        assert ref.asymmetry_ratio == pytest.approx(1.36, abs=0.02)

    def test_wind_depth_averaged_excel_parity(self) -> None:
        res = currents.wind_induced_depth_averaged_velocity(
            wind_speed=[0.0, 10.0, 15.0, 20.0, 50.0],
            v_tide=-0.8,
            theta_deg=90.0,
            depth=15.0,
            k_s=0.01,
            wind_drag_k=5.0e-6,
            strict_excel=True,
        )
        assert res.chezy == pytest.approx(76.5949050918595, rel=1e-12)
        assert res.w_zero == pytest.approx(14.6298433189264, rel=1e-12)
        assert res.v_total_pos[1] == pytest.approx(0.969030580711373, rel=1e-12)
        assert res.v_no_tide[3] == pytest.approx(1.09365491148501, rel=1e-12)
        assert res.v_total_pos[3] == pytest.approx(1.35502068818719, rel=1e-12)
        assert res.v_total_neg[1] == pytest.approx(-0.583934699813412, rel=1e-12)
        assert res.v_total_neg[2] == pytest.approx(0.181095552944007, rel=1e-12)
        assert res.v_total_neg[4] == pytest.approx(2.6144801890329, rel=1e-12)

    def test_wind_velocity_profile_excel_parity_and_bugfix(self) -> None:
        z_eval = [0.01, 0.5, 3.0, 4.99]
        res_xls = currents.wind_induced_velocity_profile(z=z_eval, strict_excel=True)
        assert res_xls.coef_a == pytest.approx(0.265982313651102, rel=1e-12)
        assert res_xls.coef_b == pytest.approx(1.99994680353727, rel=1e-12)
        np.testing.assert_allclose(
            res_xls.u,
            [0.633794635131505, 1.44283628119702, 0.297707869384535, -9.97372671030654],
            rtol=1e-11,
        )

        # Con la integral corregida (strict_excel=False), q2 = -1 + (z_b/h)*ln((z_b+h)/z_b)
        res_fix = currents.wind_induced_velocity_profile(z=z_eval, strict_excel=False)
        assert res_fix.q2 == pytest.approx(-1.0 + (0.001 / 5.0) * math.log(5001.0), rel=1e-12)

    def test_stratified_density_flow_excel_parity(self) -> None:
        res = currents.stratified_density_flow_profile(
            z_rel=[0.01, 0.3, 1.0], strict_excel=True
        )
        assert res.m_sal == pytest.approx(0.00260833333333333, rel=1e-12)
        np.testing.assert_allclose(
            res.u_sal,
            [-0.00163605795555556, -0.0230368, 0.0556444444444445],
            rtol=1e-11,
        )
        np.testing.assert_allclose(
            res.u_river,
            [0.00537448266666667, 0.14302248, 0.330208266666667],
            rtol=1e-11,
        )

        res_std = currents.stratified_density_flow_profile(strict_excel=False)
        assert res_std.u_mean_analytical == pytest.approx(0.2, rel=1e-12)

    def test_stratified_channel_sedimentation_excel_parity(self) -> None:
        r365 = currents.stratified_channel_sedimentation(duration_days=365.0)
        assert r365.sediment_flux_kg_s == pytest.approx(300.0, rel=1e-12)
        assert r365.deposited_volume_m3 == pytest.approx(4730400.0, rel=1e-12)
        assert r365.deposited_layer_m == pytest.approx(2.3652, rel=1e-12)

        r30 = currents.stratified_channel_sedimentation(duration_days=30.0)
        assert r30.deposited_volume_m3 == pytest.approx(388800.0, rel=1e-12)
        assert r30.deposited_layer_m == pytest.approx(0.1944, rel=1e-12)


class TestTides:
    """Verifica las 11 hojas computacionales de Computation Tidal motion.xlsx."""

    def test_sheet1_analytical_no_reflection_and_energy(self) -> None:
        res = tides.tidal_analytical_no_reflection(strict_excel=True)
        p = res.params
        assert p.hydraulic_radius == pytest.approx(14.2241379310345, rel=1e-12)
        assert p.chezy == pytest.approx(63.5982695405747, rel=1e-12)
        assert p.linear_friction_m == pytest.approx(0.0001216384352212, rel=1e-12)
        assert p.c0 == pytest.approx(12.1305399714934, rel=1e-12)
        assert p.k == pytest.approx(4.7930762391574e-06, rel=1e-12)
        assert p.mu == pytest.approx(1.25355133146014e-05, rel=1e-12)
        assert p.phase_velocity == pytest.approx(30.3292422479651, rel=1e-12)
        assert p.phi_rad == pytest.approx(1.40811268533862, rel=1e-12)

        assert res.u_ebb_peak[0] == pytest.approx(0.835097152490497, rel=1e-12)
        assert res.eta_amplitude[-1] == pytest.approx(3.85437433555686, rel=1e-11)
        assert res.u_ebb_peak[-1] == pytest.approx(1.26226550284548, rel=1e-11)
        assert res.setup_z[-1] == pytest.approx(-0.0635412899281538, rel=1e-10)
        assert res.dh_energy[0] == pytest.approx(0.0247295115257492, rel=1e-12)
        assert res.dh_energy_alt[0] == pytest.approx(0.0234324507213334, rel=1e-12)

        # Iteracion automatica de Umax
        auto = tides.tidal_analytical_no_reflection(u_max=None)
        assert auto.u_max_used == pytest.approx(auto.u_ebb_peak[0], rel=1e-8)

    def test_sheet2_compound_channel(self) -> None:
        res = tides.tidal_compound_channel(strict_excel=True)
        assert res.a_total == pytest.approx(28000.0, rel=1e-12)
        assert res.h_effective == pytest.approx(6.0, rel=1e-12)
        assert res.params.hydraulic_radius == pytest.approx(6.95134061569017, rel=1e-12)
        assert res.params.c0 == pytest.approx(7.67202711152665, rel=1e-12)
        assert res.eta_amplitude[-1] == pytest.approx(2.48110127970787, rel=1e-11)
        assert res.u_ebb_peak[-1] == pytest.approx(1.52014022104294, rel=1e-11)
        assert res.setup_z[-1] == pytest.approx(0.511219769820975, rel=1e-10)

    def test_sheet4_reflection_prismatic(self) -> None:
        res = tides.tidal_reflection_prismatic(x_eval=0.0, strict_excel=True)
        assert res.params.k == pytest.approx(3.58163815868692e-05, rel=1e-12)
        assert res.params.mu == pytest.approx(2.91886347628903e-05, rel=1e-12)
        assert res.f_l == pytest.approx(95.6631286560132, rel=1e-12)
        assert res.time_hw_hours == pytest.approx(-0.31290685973533, rel=1e-11)
        assert res.eta_max_total == pytest.approx(2.09918677622829, rel=1e-11)
        assert res.u_max_total == pytest.approx(1.32085509254029, rel=1e-11)

    def test_sheet5_reflection_converging(self) -> None:
        res = tides.tidal_reflection_converging(x_eval=0.0, strict_excel=True)
        assert res.f_l == pytest.approx(4.49608165879266, rel=1e-12)
        assert res.eta_total[0] == pytest.approx(-0.554254649407799, rel=1e-12)
        assert res.u_total[0] == pytest.approx(-0.0814288886447085, rel=1e-12)
        assert res.eta_max_total == pytest.approx(0.649904663887887, rel=1e-12)
        assert res.u_max_total == pytest.approx(0.718004577905082, rel=1e-12)

    def test_sheet6_fourpole(self) -> None:
        res = tides.tidal_fourpole(strict_excel=True)
        assert res.eta_l.real == pytest.approx(2.22333141907099, rel=1e-11)
        assert res.eta_l.imag == pytest.approx(-0.123928372977478, rel=1e-11)
        assert res.eta_l_amp == pytest.approx(2.22678262088087, rel=1e-11)
        assert res.eta_l_phase_deg == pytest.approx(-3.19198049073096, rel=1e-11)
        assert res.q_l.real == pytest.approx(16903.4991526036, rel=1e-11)
        assert res.q_l.imag == pytest.approx(39079.1450893666, rel=1e-11)
        assert res.q_l_amp == pytest.approx(42578.2557712013, rel=1e-11)
        assert res.q_l_phase_deg == pytest.approx(66.6431180366469, rel=1e-11)
        assert res.k1_reflected.real == pytest.approx(0.579485118713177, rel=1e-11)
        assert res.k1_reflected.imag == pytest.approx(-0.454144888618293, rel=1e-11)
        assert res.k2_incoming.real == pytest.approx(1.62051488128682, rel=1e-11)
        assert res.k2_incoming.imag == pytest.approx(0.454144888618293, rel=1e-11)

    def test_sheet7_and_sheet8_lamb_and_progressive(self) -> None:
        lamb = tides.tidal_wave_nonlinear_lamb(strict_excel=True)
        assert lamb.wavelength == pytest.approx(435869.349004537, rel=1e-11)
        assert lamb.eta_x[0] == pytest.approx(0.00159265342146697, rel=1e-11)
        assert lamb.eta_t[0] == pytest.approx(1.33881891095093, rel=1e-11)

        prog = tides.tidal_progressive_wave(strict_excel=True)
        assert prog.u_max == pytest.approx(0.885889383614004, rel=1e-12)
        assert prog.water_level[0] == pytest.approx(1.598863760183, rel=1e-12)
        assert prog.surface_slope[0] == pytest.approx(-7.74312221467391e-06, rel=1e-11)
        assert prog.velocity[0] == pytest.approx(0.708208215495641, rel=1e-12)

    def test_sheet9_soulsby_coastal_flow(self) -> None:
        res = tides.soulsby_coastal_flow(strict_excel=True)
        assert res.h_star == pytest.approx(21.6737131952821, rel=1e-11)
        assert res.u_max == pytest.approx(0.528136191048793, rel=1e-11)
        assert res.u_max_no_friction == pytest.approx(0.687324840764331, rel=1e-11)
        assert res.alpha_deg == pytest.approx(45.5854004239223, rel=1e-11)
        assert res.phi_deg == pytest.approx(134.414599576078, rel=1e-11)

    def test_sheet10_steady_salinity_van_rijn(self) -> None:
        res = tides.salinity_van_rijn_steady(strict_excel=True)
        idx_1000 = 5  # x = 1000 m con dx = 200 m
        assert res.x[idx_1000] == pytest.approx(1000.0)
        assert res.ls_prismatic == pytest.approx(3333.33333333333, rel=1e-12)
        assert res.ls_converging == pytest.approx(3333.32777781196, rel=1e-12)
        assert res.salinity_prismatic[idx_1000] == pytest.approx(19.7093379077631, rel=1e-12)
        assert res.salinity_converging[idx_1000] == pytest.approx(19.7093350923563, rel=1e-12)

    def test_sheet11_kuijper_coupled_salinity(self) -> None:
        res = tides.tidal_salinity_kuijper(strict_excel=True)
        assert res.excursion_length == pytest.approx(11336.9734418613, rel=1e-11)
        assert res.d0_mouth == pytest.approx(48.9263004353018, rel=1e-11)
        assert res.ls_max_kuijper == pytest.approx(117444.077227345, rel=1e-11)
        assert res.salinity[1] == pytest.approx(29.8516837005192, rel=1e-11)
        assert res.u_salinity[0] == pytest.approx(-0.0137410420547948, rel=1e-11)
        assert res.u_stokes[0] == pytest.approx(-0.0221503974202731, rel=1e-11)
        assert res.u_salt_wedge[0] == pytest.approx(-0.0231097162967217, rel=1e-11)

    def test_sheet13_time_dependent_salinity(self) -> None:
        res = tides.salinity_time_dependent(strict_excel=True)
        assert res.ls_min_ippen == pytest.approx(11748.2300586718, rel=1e-11)
        assert res.ls_max_ippen == pytest.approx(20155.8733707737, rel=1e-11)
        assert res.ls_min_kuijper == pytest.approx(16681.0344827586, rel=1e-11)
        assert res.ls_max_kuijper == pytest.approx(25088.6777948605, rel=1e-11)
        assert res.salinity_ippen_harleman[0] == pytest.approx(21.8454378117099, rel=1e-11)
        assert res.salinity_kuijper_van_rijn[0] == pytest.approx(20.5640424375876, rel=1e-11)
