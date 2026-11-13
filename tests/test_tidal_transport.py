"""Tests para transporte mareal de arena, lodo y corrientes de densidad."""

from __future__ import annotations

import numpy as np
import pytest

from sedmor.tidal_transport import tidal_mud_sand_transport, tidal_sand_mud_transport


def test_tidal_sand_mud_matches_excel():
    res = tidal_sand_mud_transport(strict_excel=True)
    assert res.time.shape == (145,)
    assert res.z_profiles.shape == (20, 145)

    assert res.sand_formula_pos == pytest.approx(5354.66053527992, rel=1e-6)
    assert res.sand_formula_neg == pytest.approx(-2171.38552089536, rel=1e-6)
    assert res.sand_formula_net == pytest.approx(3183.27501438456, rel=1e-6)

    assert res.sand_profile_pos == pytest.approx(2971.98354318684, rel=1e-6)
    assert res.sand_profile_neg == pytest.approx(-893.977200495271, rel=1e-6)
    assert res.sand_profile_net == pytest.approx(2078.00634269157, rel=1e-6)

    assert res.mud_profile_pos == pytest.approx(1169.87670843562, rel=1e-6)
    assert res.mud_profile_neg == pytest.approx(-353.44696878461, rel=1e-6)
    assert res.mud_profile_net == pytest.approx(816.429739651011, rel=1e-6)

    assert res.ca_mud_mass[0] == pytest.approx(0.000128855608446842, rel=1e-6)

    # Modo fisico sin errores de indexado/constantes
    res_phys = tidal_sand_mud_transport(strict_excel=False)
    assert np.all(np.isfinite(res_phys.c_sand_profiles_mass))
    assert np.all(np.isfinite(res_phys.c_mud_profiles_mass))
    assert res_phys.sand_formula_net > 0.0


def test_tidal_mud_sand_density_matches_excel():
    res = tidal_mud_sand_transport(strict_excel=True)
    assert res.time.shape == (145,)
    assert res.z_profiles.shape == (50, 145)

    # Valores puntuales iniciales y en pico de flujo (AK = col index 35)
    assert res.density_driven_transport[0] == pytest.approx(3.78204235234094e-05, rel=1e-6)
    assert res.density_driven_transport[1] == pytest.approx(0.00019739772950228, rel=1e-6)
    assert res.suspended_mud_rate[1] == pytest.approx(2.33204375901638e-05, rel=1e-6)
    assert res.suspended_mud_rate[72] == pytest.approx(5.6208944138225e-08, rel=1e-6)
    assert res.suspended_mud_rate[73] == pytest.approx(-1.96868886479535e-05, rel=1e-6)

    assert res.mud_pos == pytest.approx(62763.9564678915, rel=1e-6)
    assert res.mud_neg == pytest.approx(-58838.0283761082, rel=1e-6)
    assert res.mud_net == pytest.approx(3925.92809178332, rel=1e-6)

    # Transporte de arena en columnas validas de LibreOffice Calc (AK=35, DF=108)
    assert res.suspended_sand_rate[35] == pytest.approx(1.19399170442574, rel=1e-5)
    assert res.suspended_sand_rate[108] == pytest.approx(-1.199154313216, rel=1e-5)

    res_phys = tidal_mud_sand_transport(strict_excel=False, ws_mud_mode=2, damping_coef=1.0)
    assert np.all(np.isfinite(res_phys.c_mud_profiles_mass))
    assert np.all(np.isfinite(res_phys.c_sand_profiles_mass))
    assert res_phys.density_net > 0.0
