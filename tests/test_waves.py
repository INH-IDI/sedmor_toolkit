"""Pruebas unitarias para ``sedmor.waves`` frente a valores Excel cacheados y fisica."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sedmor import waves


def test_wavelength_nielsen_and_hunt():
    h = np.array([500.0, 100.0, 10.0, 5.0])
    niel = waves.wavelength_nielsen(h, 8.0, strict_excel=True)
    assert np.allclose(
        niel["L"],
        [15.2276076375187, 76.7209063525601, 70.9680963721219, 53.0978134746334],
        rtol=1e-11,
    )

    hunt_excel = waves.wavelength_hunt(h, 8.0, strict_excel=True)
    assert np.isnan(hunt_excel["L"][0])  # #NUM! en Excel por 0.000654^Y^6
    assert np.allclose(
        hunt_excel["L"][1:],
        [99.9530640595287, 70.8971808613235, 53.0812246127907],
        rtol=1e-11,
    )

    hunt_fixed = waves.wavelength_hunt(h, 8.0, strict_excel=False)
    assert np.all(np.isfinite(hunt_fixed["L"]))
    # En aguas profundas g*T^2/(2*pi) = 9.81*64/(2*pi) = 99.9235 m
    exact_deep = waves.dispersion_relation(500.0, 8.0, method="exact")
    assert math.isclose(hunt_fixed["L"][0], exact_deep["L"], rel_tol=1e-3)


def test_orbital_velocity_and_stokes_and_mass_flux():
    orb = waves.orbital_velocity_profile(np.array([0.0, -3.0, -4.0]), 10.0, 8.0, 3.0, 71.0, strict_excel=True)
    assert np.allclose(orb["Umax"], [1.66156311618513, 1.40427217960599, 1.34129787413848], rtol=1e-11)
    assert np.allclose(orb["Wmax"], [-1.1775, -0.773107117641872, -0.651762416914008], rtol=1e-11)
    assert np.allclose(orb["Amax"], [2.11664091233774, 1.78888175745986, 1.70865971227832], rtol=1e-11)

    orb_sheet = waves.orbital_velocity_profile(
        np.array([0.0, -0.5]), 1.0, 8.0, 1.0, use_orbital_sheet_hunt=True, strict_excel=True
    )
    assert np.allclose(orb_sheet["Umax"], [1.58264455063823, 1.54551300151947], rtol=1e-11)

    st = waves.stokes_second_order(np.array([0.0, 0.05]), 10.0, 8.0, 3.0, 70.9, 2.0, strict_excel=True)
    assert math.isclose(st["epsilon"], 0.232420595588598, rel_tol=1e-11)
    assert np.allclose(st["eta"], [-0.347435961157837, 0.182647224324341], rtol=1e-11)

    mf = waves.wave_mass_flux(np.array([0.0, -9.0]), 10.0, 8.0, 3.0, 71.0, strict_excel=True)
    assert np.allclose(mf["u_unbounded"], [0.233650605017849, 0.0786391773960788], rtol=1e-11)
    assert np.allclose(mf["u_bounded"], [0.109033371303965, -0.0459780563178056], rtol=1e-11)
    assert np.allclose(mf["u_longuet_higgins"], [0.0861509731712702, 0.115707711910161], rtol=1e-11)


def test_velocity_profile_and_jonswap():
    z = np.array([0.01, 0.02, 0.05, 0.08])
    vp = waves.wave_modified_velocity_profile(z, 0.5, 10.0, 0.13, 0.17, 0.67, 0.05)
    assert np.allclose(
        vp["u_no_wave"],
        [0.11635535016311, 0.161367744010467, 0.220870892055424, 0.251392531705182],
        rtol=1e-11,
    )
    assert np.allclose(
        vp["u_with_wave"],
        [0.0699149568827661, 0.0969617542205461, 0.132715675498082, 0.151055348896106],
        rtol=1e-11,
    )

    E = waves.jonswap_spectrum(np.array([0.06, 0.08, 0.16]), 3.6, 7.5)
    assert np.allclose(E, [1.67060945832452e-08, 0.113572801135299, 6.10899976009791], rtol=1e-11)


def test_wave_parameters_2_tools():
    tr = waves.wave_trains(np.array([0.25, 0.5, 1.0]), 2.0, 1.0, 25.0, 8.0, strict_excel=True)
    assert np.allclose(tr["eta_primary"], [0.0635534663158144, 0.126060043582689, 0.249337730807221], rtol=1e-11)
    assert np.allclose(tr["eta_harmonic"], [-0.991921913838489, -0.968217730823861, -0.875661391991811], rtol=1e-11)

    pr = waves.wave_pressure_profile(np.array([0.0, -1.5, -10.0]), 10.0, 3.0, 70.0, strict_excel=True)
    assert np.allclose(pr["p_total_crest"], [15000.0, 28688.5821444523, 110488.341913191], rtol=1e-11)
    assert np.allclose(pr["p_total_trough"], [-15000.0, 1311.41785554766, 89511.6580868092], rtol=1e-11)

    # Modo fisico por defecto usa g=9.81
    pr_phys = waves.wave_pressure_profile(0.0, 10.0, 3.0, 70.0, strict_excel=False)
    assert math.isclose(pr_phys["p_dyn_crest"], 1000.0 * 9.81 * 1.5, rel_tol=1e-12)

    regime = waves.classify_linear_wave_regime(H=0.5, h=20.0, T=8.0)
    assert not regime["is_broken"]
    assert regime["is_linear_ursell"]

    gv = waves.group_velocity_packet(np.array([0.0, 1.0, 4.0]), 6.0, 48.0, 0.05, 0.01, 1.0, 10.0, 3.0, strict_excel=True)
    assert math.isclose(gv["n"], 0.692162208464595, rel_tol=1e-11)
    assert math.isclose(gv["c_gr"], 5.53729766771676, rel_tol=1e-11)
    assert math.isclose(gv["L_computed"], 48.5793563394177, rel_tol=1e-11)
    assert np.allclose(gv["eta"], [-1.99437510681018, -1.97763385584158, -1.72810501382264], rtol=1e-11)

    sp_excel = waves.sine_power_integrals(8.0, 135.0, strict_excel=True)
    assert math.isclose(sp_excel["int_product"], -1.41386173051724, rel_tol=1e-11)
    assert math.isclose(sp_excel["analytic_product"], -1.99360209640304, rel_tol=1e-11)
    assert math.isclose(sp_excel["int_sin4"], 1.50028302469665, rel_tol=1e-11)
    assert math.isclose(sp_excel["int_abs_sin3"], 1.69797309495616, rel_tol=1e-11)

    sp_phys = waves.sine_power_integrals(8.0, 135.0, strict_excel=False)
    assert math.isclose(sp_phys["analytic_product"], 2.0 * math.cos(math.radians(135.0)), rel_tol=1e-12)


def test_boundary_layer_and_friction_decay():
    bl = waves.wave_boundary_layer(np.array([0.1, 0.8, 5.5]), 7.0, 1.0, 8.0, 1e-4, 0.25, strict_excel=True)
    assert math.isclose(bl["tau_max"], 8.86002257333468, rel_tol=1e-11)
    assert math.isclose(bl["delta"], 0.0159617376893524, rel_tol=1e-11)
    assert math.isclose(bl["uw_max"], -0.997608605584528, rel_tol=1e-11)
    assert np.allclose(bl["u_t"], [-0.00698589973063468, -0.259813807904173, -0.709054406670165], rtol=1e-11)
    assert np.allclose(bl["tau_t"], [-0.825757063429943, -2.86466143250866, 0.0254649651576946], rtol=1e-11)

    decay = waves.wave_decay_friction(strict_excel=True)
    assert math.isclose(decay["n"], 0.875107763095482, rel_tol=1e-11)
    assert math.isclose(decay["H"][43], 2.95181491259502, rel_tol=1e-11)
    assert math.isclose(decay["alpha"][43], 2.18778737286054e-05, rel_tol=1e-11)


def test_battjes_janssen_both_profiles():
    h_barred = np.array([
        9.3, 9.2, 9.1, 9.0, 8.9, 8.8, 8.7, 8.6, 8.5, 8.4, 8.3, 8.2, 8.1, 8.0, 7.9,
        7.8, 7.7, 7.6, 7.5, 7.4, 7.3, 7.1, 6.5, 5.5, 5.0, 4.7, 5.0, 5.3, 5.6, 5.9,
        5.9, 5.6, 5.3, 5.0, 4.7, 4.1, 3.7, 3.2, 2.6, 2.1, 2.2, 2.2, 2.1, 1.5, 1.2,
    ])
    bj1 = waves.battjes_janssen_profile(h_barred, 2.8, 30.0, 8.7, strict_excel=True)
    assert math.isclose(bj1["H_rms"][2], 2.74357513352774, rel_tol=1e-10)
    assert math.isclose(bj1["setup"][2], 0.00536102685404477, rel_tol=1e-10)
    assert math.isclose(bj1["v_longshore"][2], 0.378794820100397, rel_tol=1e-10)
    assert math.isclose(bj1["H_rms"][20], 2.28344537577646, rel_tol=1e-10)
    assert math.isclose(bj1["setup"][40], 0.290823681524038, rel_tol=1e-10)
    assert math.isclose(bj1["v_longshore"][40], 0.608164724132902, rel_tol=1e-10)

    h_plane = np.linspace(10.0, 1.2, 45)
    # En R78:R122 la hoja Excel heredo por error $R$19 (c_0 de la tabla 1) a partir de la fila 80;
    # verificamos consistencia fisica completa sobre el perfil plano.
    bj_plane_excel = waves.battjes_janssen_profile(h_plane, 2.8, 30.0, 8.7, strict_excel=True)
    assert math.isclose(bj_plane_excel["H_rms"][1], 2.7876484251068, rel_tol=1e-10)

    bj_phys = waves.battjes_janssen_profile(h_plane, 2.8, 30.0, 8.7, strict_excel=False)
    assert np.all(np.isfinite(bj_phys["H_rms"]))
    assert bj_phys["H_rms"][-1] < bj_phys["H_rms"][0]
    assert bj_phys["setup"][-1] > 0.25


def test_skewness_asymmetry_and_nielsen():
    ru = waves.wave_skewness_asymmetry(1.0, 1.5, 7.0, strict_excel=True)
    assert math.isclose(ru["SK"], 0.378648319588513, rel_tol=1e-11)
    assert math.isclose(ru["AS"], -0.652684414047311, rel_tol=1e-11)
    assert math.isclose(ru["Ur"], 2.7475768314026, rel_tol=1e-11)
    assert math.isclose(ru["U2_solution"], 0.537623479121118, rel_tol=1e-11)

    ab = waves.abreu_skewness_asymmetry(1.0, 1.5, 7.0, t=np.array([0.0, 0.05, 0.1, 0.15]), strict_excel=True)
    assert math.isclose(ab["r"], 0.638866842331673, rel_tol=1e-11)
    assert math.isclose(ab["H_par"], -0.208090303227821, rel_tol=1e-11)
    assert np.allclose(ab["u"], [-0.579063361632626, -0.46999964975936, -0.352530115177351, -0.226928618414674], rtol=1e-11)

    ih = waves.isobe_horikawa_skewness(t=np.array([0.0, 0.1, 0.2, 0.3]), strict_excel=True)
    assert math.isclose(ih["U_on"], 1.49005432032802, rel_tol=1e-11)
    assert math.isclose(ih["U_off"], 0.667310806652294, rel_tol=1e-11)
    assert math.isclose(ih["T_on"], 2.16522256160891, rel_tol=1e-11)
    assert np.allclose(ih["u"], [0.0, 0.215330665302913, 0.426140701986915, 0.628004387019725], rtol=1e-11)

    nl = waves.nielsen_acceleration_transport(strict_excel=True)
    assert math.isclose(nl["d_star"], 4.83469178942457, rel_tol=1e-11)
    assert math.isclose(nl["qb_nielsen"][0], 0.981393239204139, rel_tol=1e-11)
    assert math.isclose(nl["qb_vr_stokes"][1], 0.472272770161217, rel_tol=1e-11)
    assert math.isclose(nl["qb_vr_sine"][2], 1.14920145832847, rel_tol=1e-11)
    assert math.isclose(nl["net_qb_nielsen"], 0.31341126297674, rel_tol=1e-10)
    assert math.isclose(nl["net_qb_vr_stokes"], 0.160076640499326, rel_tol=1e-10)
    assert math.isclose(nl["net_qb_vr_sine"], 0.214477791510847, rel_tol=1e-10)


def test_undertow_and_streaming():
    u1 = waves.undertow_velocity(np.array([0.025, 0.03, 0.1, 0.2]), profile=1)
    assert np.allclose(u1, [0.05, 0.0278226258804805, -0.104115373412274, -0.15664806011841], rtol=1e-11)

    u2 = waves.undertow_velocity(np.array([0.0, 0.1, 0.5, 0.9]), profile=2, h=1.0, dh_dx=0.001, strict_excel=True)
    assert np.allclose(u2, [0.0, -0.058639, -0.173575, -0.097119], rtol=1e-11)

    sp = waves.eulerian_streaming_profile(np.array([0.0, 0.1, 0.5, 1.0]), strict_excel=True)
    assert np.allclose(sp, [0.0, -0.0398782165605096, -0.157414012738854, -0.209885350318471], rtol=1e-11)
    # Correccion dimensional mantiene coincidencia en U_max=1 pero corrige U_max=2
    assert math.isclose(waves.eulerian_streaming_profile(1.0, U_max=1.0, strict_excel=False), -0.2096999506561144, rel_tol=1e-3)
    assert waves.eulerian_streaming_profile(1.0, U_max=2.0, strict_excel=False) > 0.0

    kr = waves.eulerian_streaming(np.array([0.5, 1.0, 2.0, 0.15]), np.array([8.0, 8.0, 8.0, 1.5]), np.array([5.0, 5.0, 5.0, 0.45]), np.array([0.05, 0.05, 0.05, 0.0001]), strict_excel=True)
    assert np.allclose(kr["u_van_rijn"], [-0.00318256621709989, 0.00275436883982854, 0.0729560101922265, 0.0268057325572739], rtol=1e-10)
    assert np.allclose(kr["u_kranenburg_shape"], [-0.00271447809027608, -0.0138210125844913, -0.0616355947104884, 0.00651907616578167], rtol=1e-10)
    assert np.allclose(kr["u_kranenburg_no_shape"], [0.00669564848142772, 0.0238194937023239, 0.0889264304367723, 0.0124086504212085], rtol=1e-10)
