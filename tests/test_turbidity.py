"""Pruebas unitarias para corrientes de turbidez (``sedmor.turbidity``)."""

from __future__ import annotations

import numpy as np
import pytest

from sedmor.turbidity import (
    _build_distance_grid,
    turbidity_current_mud,
    turbidity_current_sand,
)


def _grid_two_step(ds1: float, ds2: float, s_trans: float, n: int) -> np.ndarray:
    return _build_distance_grid(
        s=None,
        ds1=ds1,
        ds2=ds2,
        s_transition=s_trans,
        s_max=None,
        n_steps=n,
    )


def test_turbidity_mud_constant_slope() -> None:
    """Turbidity current_mud.xlsx (Constant slope, n=591, s=0..570.2 m)."""
    res = turbidity_current_mud(
        slope_deg=0.1,
        ds1=0.01,
        ds2=1.0,
        s_transition=0.2,
        n_steps=591,
        mode="constant",
        strict_excel=True,
    )
    assert res.s[0] == pytest.approx(0.0)
    assert res.s[20] == pytest.approx(0.2)
    assert res.s[21] == pytest.approx(1.2)
    assert res.s[-1] == pytest.approx(570.2)
    assert res.c2_mass[-1] == pytest.approx(86.4928121317, rel=1e-11)
    assert res.v2[-1] == pytest.approx(0.373342684851, rel=1e-11)
    assert res.h2[-1] == pytest.approx(2.30956805248, rel=1e-11)
    assert res.q[-1] == pytest.approx(0.862260337556, rel=1e-11)
    assert res.qs_mass[-1] == pytest.approx(74.5793213849, rel=1e-11)

    res_std = turbidity_current_mud(
        slope_deg=0.1,
        ds1=0.01,
        ds2=1.0,
        s_transition=0.2,
        n_steps=591,
        mode="constant",
        strict_excel=False,
    )
    assert res_std.h2[-1] == pytest.approx(res.h2[-1], rel=3e-2)


def test_turbidity_mud_varying_slope_and_bug_fix() -> None:
    """Turbidity current_mud.xlsx (Varying slope) y correccion de dh2 en strict_excel=False."""
    s_vec = _grid_two_step(0.1, 0.5, 2.0, 591)
    slope_fn = lambda x: 3.0 if x < 25.5 else 1.0  # noqa: E731

    res_xls = turbidity_current_mud(
        slope_deg=slope_fn,
        s=s_vec,
        ws1=1e-4,
        ws2=1e-4,
        c2max=0.3,
        eros=1e-4,
        tau_cr_dep=0.05,
        tau_cr_ero=0.1,
        h20=0.5,
        c20=100.0,
        fr20=1.1,
        c1=0.1,
        resi=1.0,
        rewi=0.1,
        mode="varying",
        strict_excel=True,
    )
    assert res_xls.s[-1] == pytest.approx(287.0)
    assert res_xls.c2_mass[-1] == pytest.approx(58.5902416544, rel=1e-11)
    assert res_xls.v2[-1] == pytest.approx(1.82941487876, rel=1e-11)
    assert res_xls.h2[-1] == pytest.approx(0.586396849495, rel=1e-11)
    assert res_xls.q[-1] == pytest.approx(1.07276312132, rel=1e-11)
    assert res_xls.qs_mass[-1] == pytest.approx(62.8534505161, rel=1e-11)

    # En strict_excel=False se corrige el paso dh2 en s >= 2.5 (usa ds=0.5 en vez de ds1=0.1)
    res_fixed = turbidity_current_mud(
        slope_deg=slope_fn,
        s=s_vec,
        ws1=1e-4,
        ws2=1e-4,
        c2max=0.3,
        eros=1e-4,
        tau_cr_dep=0.05,
        tau_cr_ero=0.1,
        h20=0.5,
        c20=100.0,
        fr20=1.1,
        c1=0.1,
        resi=1.0,
        rewi=0.1,
        mode="varying",
        strict_excel=False,
    )
    assert res_fixed.dh2[21] != pytest.approx(res_xls.dh2[21], rel=0.05)


@pytest.mark.parametrize(
    ("case_kwargs", "expected"),
    [
        # Turbidity current_sand.xlsx :: Constant slope (n=591, s=0..287)
        (
            {
                "slope_deg": 0.05,
                "s": _grid_two_step(0.1, 0.5, 2.0, 591),
                "d50": 5e-5,
                "wso": 0.008,
                "c2max": 0.6,
                "theta_cr": 0.1,
                "eros": 0.0025,
                "rho_w": 1000.0,
                "chezy": 70.0,
                "h20": 0.5,
                "c20": 100.0,
                "fr20": 1.2,
                "c1": 0.0,
                "resi": 0.1,
                "resb": 1.0,
                "rewi": 0.1,
                "rewb": 1.0,
                "viscosity_exponent": -3.0,
            },
            (287.0, 2.43896750827, 0.271015393887, 1.62171068637, 0.439508560436, 1.07194709851),
        ),
        # Turbidity current_sand.xlsx :: Varying slope (n=591, s=0..572)
        (
            {
                "slope_deg": lambda x: 12.0 if x < 451.0 else 2.4,
                "s": _grid_two_step(0.1, 1.0, 2.0, 591),
                "d50": 1e-4,
                "wso": 0.008,
                "c2max": 0.6,
                "theta_cr": 0.06,
                "eros": 0.0025,
                "rho_w": 1020.0,
                "chezy": 35.0,
                "h20": 0.04,
                "c20": 400.0,
                "fr20": 2.0,
                "c1": 0.0,
                "resi": 1.0,
                "resb": lambda x: 1.0 if x < 100.0 else 0.0,
                "rewi": 1.0,
                "rewb": 1.0,
                "viscosity_exponent": -1.5,
            },
            (572.0, 2.40363674065, 0.742383048156, 10.0325965575, 7.44802961327, 17.9023576239),
        ),
        # Turbidity current_sand minimum slope.xlsx :: Constant slope (n=593, s=0..288)
        (
            {
                "slope_deg": 1.5,
                "s": _grid_two_step(0.1, 0.5, 2.0, 593),
                "d50": 1e-4,
                "wso": 0.01,
                "c2max": 0.6,
                "theta_cr": 0.06,
                "eros": 0.0025,
                "rho_w": 1000.0,
                "chezy": 50.0,
                "h20": 0.05,
                "c20": 500.0,
                "fr20": 2.0,
                "c1": 0.0,
                "resi": 1.0,
                "resb": 1.0,
                "rewi": 1.0,
                "rewb": 1.0,
                "viscosity_exponent": -1.5,
            },
            (288.0, 0.000252516153772, 0.0745947016942, 7.62286849631, 0.568625601536, 0.000143587149836),
        ),
        # Turbidity current_sand minimum slope.xlsx :: Varying slope (n=593, s=0..574)
        (
            {
                "slope_deg": lambda x: 12.0 if x < 451.0 else 2.4,
                "s": _grid_two_step(0.1, 1.0, 2.0, 593),
                "d50": 1e-4,
                "wso": 0.008,
                "c2max": 0.6,
                "theta_cr": 0.06,
                "eros": 0.0025,
                "rho_w": 1020.0,
                "chezy": 35.0,
                "h20": 0.04,
                "c20": 400.0,
                "fr20": 2.0,
                "c1": 0.0,
                "resi": 1.0,
                "resb": lambda x: 1.0 if x < 100.0 else 0.0,
                "rewi": 1.0,
                "rewb": 1.0,
                "viscosity_exponent": -3.0,
            },
            (574.0, 2.39887426072, 0.740741301128, 10.070305696, 7.45949134405, 17.8943817833),
        ),
        # Turbidity current_sand Scripps.xlsx :: Constant slope (n=593, s=0..288)
        (
            {
                "slope_deg": 1.0,
                "s": _grid_two_step(0.1, 0.5, 2.0, 593),
                "d50": 1e-4,
                "wso": 0.01,
                "c2max": 0.6,
                "theta_cr": 0.06,
                "eros": 0.0025,
                "rho_w": 1000.0,
                "chezy": 50.0,
                "h20": 0.05,
                "c20": 500.0,
                "fr20": 2.0,
                "c1": 0.0,
                "resi": 1.0,
                "resb": 1.0,
                "rewi": 1.0,
                "rewb": 1.0,
                "viscosity_exponent": -3.0,
            },
            (288.0, 2.9614249155e-07, 0.0456644572015, 11.0593061411, 0.505017211957, 1.49557055424e-07),
        ),
        # Turbidity current_sand Scripps.xlsx :: Varying slope (n=593, s=0..574)
        (
            {
                "slope_deg": lambda x: 12.0 if x < 451.0 else 2.4,
                "s": _grid_two_step(0.1, 1.0, 2.0, 593),
                "d50": 1.1e-4,
                "wso": 0.008,
                "c2max": 0.6,
                "theta_cr": 0.06,
                "eros": 0.004,
                "rho_w": 1020.0,
                "chezy": 35.0,
                "h20": 0.04,
                "c20": 400.0,
                "fr20": 2.0,
                "c1": 0.01,
                "resi": 1.0,
                "resb": lambda x: 1.0 if x < 100.0 else 0.0,
                "rewi": 1.0,
                "rewb": 1.0,
                "viscosity_exponent": -1.5,
                "wi_c1_volumetric": False,
            },
            (574.0, 4.40860937229, 0.997758063176, 9.95128492586, 9.92897477373, 43.7729712448),
        ),
        # Turbidity current_sand Zaire case.xlsx :: Varying slope Zaire (n=593, s=0..288)
        (
            {
                "slope_deg": lambda x: 10.0 if x < 200.0 else 0.82,
                "s": _grid_two_step(0.1, 0.5, 2.0, 593),
                "d50": 1.5e-4,
                "wso": 0.015,
                "c2max": 0.6,
                "theta_cr": 0.06,
                "eros": 0.0025,
                "rho_w": 1020.0,
                "chezy": 50.0,
                "h20": 0.05,
                "c20": 200.0,
                "fr20": 2.0,
                "c1": 0.0,
                "resi": 0.2,
                "resb": 1.0,
                "rewi": 1.0,
                "rewb": 1.0,
                "viscosity_exponent": -3.0,
            },
            (288.0, 7.48791074574, 0.568196189889, 3.29731822366, 1.87352365153, 14.0287778827),
        ),
    ],
)
def test_turbidity_sand_workbooks(case_kwargs: dict, expected: tuple[float, ...]) -> None:
    """Verifica todos los libros de arena contra sus valores cacheados de Excel."""
    res = turbidity_current_sand(strict_excel=True, **case_kwargs)
    s_exp, c2_exp, v2_exp, h2_exp, q_exp, qs_exp = expected
    assert res.s[-1] == pytest.approx(s_exp)
    assert res.c2_mass[-1] == pytest.approx(c2_exp, rel=1e-10)
    assert res.v2[-1] == pytest.approx(v2_exp, rel=1e-10)
    assert res.h2[-1] == pytest.approx(h2_exp, rel=1e-10)
    assert res.q[-1] == pytest.approx(q_exp, rel=1e-10)
    assert res.qs_mass[-1] == pytest.approx(qs_exp, rel=1e-10)


def test_turbidity_sand_zaire_general_nonuniform_grid() -> None:
    """Turbidity current_sand Zaire case.xlsx (Varying slope general, paso ds=20 en s>=99)."""
    s_base = _grid_two_step(0.1, 1.0, 2.0, 118)
    assert s_base[-1] == pytest.approx(99.0)
    s_tail = 99.0 + 20.0 * np.arange(1, 593 - 118 + 1, dtype=float)
    s_vec = np.concatenate([s_base, s_tail])

    res = turbidity_current_sand(
        slope_deg=lambda x: 10.0 if x < 219.0 else 2.0,
        s=s_vec,
        d50=1e-4,
        wso=0.008,
        c2max=0.6,
        theta_cr=0.06,
        eros=0.0025,
        rho_w=1000.0,
        chezy=35.0,
        h20=0.05,
        c20=200.0,
        fr20=2.0,
        c1=0.0,
        resi=1.0,
        resb=lambda x: 1.0 if x < 100.0 else 0.0,
        rewi=0.5,
        rewb=1.0,
        viscosity_exponent=-1.5,
        strict_excel=True,
    )
    assert res.s[-1] == pytest.approx(9599.0)
    assert res.c2_mass[-1] == pytest.approx(0.149183991287, rel=1e-10)
    assert res.v2[-1] == pytest.approx(0.292941308377, rel=1e-10)
    assert res.h2[-1] == pytest.approx(35.0186394608, rel=1e-10)
    assert res.q[-1] == pytest.approx(10.2584060612, rel=1e-10)
    assert res.qs_mass[-1] == pytest.approx(1.53038996046, rel=1e-10)


def test_turbidity_validation() -> None:
    """Valida errores en malla y modo."""
    with pytest.raises(ValueError, match="strictly|creciente"):
        turbidity_current_mud(slope_deg=1.0, s=[0.0, 1.0, 0.5])
    with pytest.raises(ValueError, match="mode"):
        turbidity_current_mud(slope_deg=1.0, mode="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="longitud"):
        turbidity_current_sand(slope_deg=[1.0, 2.0], s=[0.0, 0.1, 0.2])
