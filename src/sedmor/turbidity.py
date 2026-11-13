"""Corrientes de turbidez submarinas de lodo y arena (``Turbidity current_*.xlsx``).

Implementa la integracion espacial 1D estacionaria por capas (modelo de dos
capas promediado en espesor) de Leo C. van Rijn para corrientes de turbidez
en taludes constantes o variables:

* ``turbidity_current_mud``: corrientes cohesivas/de lodo (``Turbidity current_mud.xlsx``,
  hojas ``Constant slope`` y ``Varying slope``), incluyendo transicion laminar-turbulenta
  en friccion interfacial/de fondo y viscosidad dependiente de la concentracion.
* ``turbidity_current_sand``: corrientes no cohesivas/de arena (``Turbidity current_sand.xlsx``,
  ``Turbidity current_sand minimum slope.xlsx``, ``Turbidity current_sand Scripps.xlsx`` y
  ``Turbidity current_sand Zaire case.xlsx``), incluyendo erosion limitada por lecho
  y atrapamiento de agua interfacial/del lecho.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from sedmor.constants import (
    GRAVITY,
    KINEMATIC_VISCOSITY_20C,
    PI_STANDARD,
    RHO_FRESH_WATER,
    RHO_SEA_WATER,
    RHO_SEDIMENT,
)

__all__ = [
    "TurbidityCurrentResult",
    "turbidity_current_mud",
    "turbidity_current_sand",
]

PI_TURBIDITY_EXCEL: float = 3.142
_EXCEL_RHOS_REF: float = 2650.0


@dataclass(frozen=True)
class TurbidityCurrentResult:
    """Resultado de la simulacion espacial 1D de una corriente de turbidez."""

    s: NDArray[np.float64]
    slope_deg: NDArray[np.float64]
    sin_beta: NDArray[np.float64]
    c2_mass: NDArray[np.float64]
    c2_vol: NDArray[np.float64]
    rho2: NDArray[np.float64]
    v2: NDArray[np.float64]
    h2: NDArray[np.float64]
    h2_equil: NDArray[np.float64]
    h2_crit: NDArray[np.float64]
    q: NDArray[np.float64]
    qs_vol: NDArray[np.float64]
    qs_mass: NDArray[np.float64]
    froude: NDArray[np.float64]
    richardson: NDArray[np.float64]
    nu: NDArray[np.float64]
    reynolds: NDArray[np.float64]
    tau_i: NDArray[np.float64]
    tau_b: NDArray[np.float64]
    wi: NDArray[np.float64]
    wb: NDArray[np.float64]
    si: NDArray[np.float64]
    sb: NDArray[np.float64]
    dh2: NDArray[np.float64]
    dq: NDArray[np.float64]
    dqs_vol: NDArray[np.float64]
    u20: float
    s_rel: float
    delta_rho: float
    cd: float
    cdi: float
    d_star: float | None = None


def _build_distance_grid(
    s: Sequence[float] | NDArray[np.floating] | None,
    ds1: float,
    ds2: float,
    s_transition: float,
    s_max: float | None,
    n_steps: int | None,
) -> NDArray[np.float64]:
    """Construye o valida la malla de distancias ``s`` a lo largo del talud."""
    if s is not None:
        arr = np.asarray(s, dtype=np.float64)
        if arr.ndim != 1 or arr.size < 2:
            raise ValueError("La malla `s` debe ser un vector 1D con al menos 2 puntos.")
        if np.any(np.diff(arr) <= 0.0):
            raise ValueError("La malla `s` debe ser estrictamente creciente.")
        return arr

    if ds1 <= 0.0 or ds2 <= 0.0:
        raise ValueError("Los pasos espaciales `ds1` y `ds2` deben ser positivos.")

    tol = 1e-12
    pts = [0.0]
    if n_steps is not None:
        if n_steps < 2:
            raise ValueError("`n_steps` debe ser al menos 2.")
        for _ in range(n_steps - 1):
            cur = pts[-1]
            step = ds1 if cur < s_transition - tol else ds2
            pts.append(cur + step)
    else:
        target = 100.0 if s_max is None else float(s_max)
        if target <= 0.0:
            raise ValueError("`s_max` debe ser positivo.")
        while pts[-1] < target - tol:
            cur = pts[-1]
            step = ds1 if cur < s_transition - tol else ds2
            pts.append(cur + step)
    return np.asarray(pts, dtype=np.float64)


def _resolve_profile(
    value: float | Sequence[float] | NDArray[np.floating] | Callable[[float], float],
    s: NDArray[np.float64],
    name: str,
) -> NDArray[np.float64]:
    """Convierte un parametro escalar, arreglo o funcion ``f(s)`` en arreglo 1D."""
    if callable(value):
        arr = np.asarray([float(value(float(x))) for x in s], dtype=np.float64)
    elif np.ndim(value) == 0:
        arr = np.full(s.shape, float(value), dtype=np.float64)
    else:
        arr = np.asarray(value, dtype=np.float64)
    if arr.shape != s.shape:
        raise ValueError(f"`{name}` debe ser escalar, funcion de `s` o tener longitud {s.size}.")
    return arr


def turbidity_current_mud(
    slope_deg: float | Sequence[float] | NDArray[np.floating] | Callable[[float], float] = 0.1,
    s: Sequence[float] | NDArray[np.floating] | None = None,
    *,
    ds1: float = 0.01,
    ds2: float = 1.0,
    s_transition: float = 0.2,
    s_max: float | None = None,
    n_steps: int | None = 581,
    ws1: float = 1.0e-4,
    ws2: float = 5.0e-4,
    rho_s: float = RHO_SEDIMENT,
    c2max: float = 0.2,
    sin_phi: float = 0.5,
    rho_w: float = RHO_FRESH_WATER,
    nu0: float = KINEMATIC_VISCOSITY_20C,
    g: float = GRAVITY,
    chezy: float = 100.0,
    porosity: float = 0.4,
    eros: float = 5.0e-6,
    tau_cr_dep: float = 0.1,
    tau_cr_ero: float = 0.1,
    h20: float = 1.0,
    c20: float = 100.0,
    fr20: float = 1.1,
    c1: float = 0.0,
    resi: float = 0.1,
    rewi: float = 0.1,
    mode: Literal["constant", "varying"] = "constant",
    si_divide_by_chezy: bool | None = None,
    strict_excel: bool = False,
) -> TurbidityCurrentResult:
    """Simula una corriente de turbidez de lodo a lo largo del talud.

    Reproduce las hojas ``Constant slope`` (``mode="constant"``) y
    ``Varying slope`` (``mode="varying"``) de ``Turbidity current_mud.xlsx``.

    Parameters
    ----------
    slope_deg:
        Angulo del talud (grados). Puede ser escalar, arreglo del mismo largo
        que ``s`` o funcion ``slope_deg(s)``.
    s:
        Arreglo opcional de distancias longitudinales (m). Si es ``None``,
        se construye usando ``ds1`` hasta ``s_transition`` y ``ds2`` en adelante.
    ds1, ds2, s_transition, s_max, n_steps:
        Parametros de generacion automatica de la malla ``s`` cuando ``s=None``.
    ws1, ws2:
        Velocidades de caida en la interfase y cerca del lecho (m/s).
    rho_s, rho_w:
        Densidades de solido y fluido (kg/m3).
    c2max:
        Concentracion volumetrica maxima en la capa inferior (-).
    sin_phi:
        Seno del angulo de reposo del sedimento (-).
    nu0:
        Viscosidad cinematica del agua limpia (m2/s).
    g, chezy, porosity:
        Gravedad (m/s2), coeficiente de Chezy (m0.5/s) y porosidad del lecho (-).
    eros, tau_cr_dep, tau_cr_ero:
        Coeficiente de erosion (m/s o -) y tensiones criticas de sedimentacion
        y erosion (N/m2).
    h20, c20, fr20, c1:
        Espesor inicial (m), concentracion inicial (kg/m3), Froude densimetrico
        inicial (-) y concentracion de la capa superior (kg/m3).
    resi, rewi:
        Factores multiplicativos para intercambio de sedimento (``Si``) y agua
        (``Wi``) en la interfase (-).
    mode:
        ``"constant"`` utiliza las formulaciones de ``Constant slope`` (viscosidad
        ``nu0*(1-c/c2max)**-3``, ``Si`` turbulento dividido por ``chezy`` y
        balance ``Vbe + Vbs``); ``"varying"`` utiliza las variantes de ``Varying slope``
        (viscosidad ``nu0*(1-c/c2max)**-5``, tope de ``0.9*c2max``, ``Wb`` con
        ``rewi``, ``Si`` turbulento sin dividir por ``chezy`` y tope en ``Sb``).
    si_divide_by_chezy:
        Sobreescribe si ``Si`` turbulento divide por ``chezy`` (por defecto
        ``True`` para ``mode="constant"`` y ``False`` para ``mode="varying"``).
    strict_excel:
        Si es ``True``, reproduce literalmente ``pi=3.142``, exponente ``0.33333333``,
        divisor ``2650.0`` y, en ``mode="varying"``, el error de hoja que fija
        ``ds = ds1`` en ``dh2`` despues de ``s_transition``. Si es ``False``
        (por defecto), usa ``math.pi``, ``1/3``, ``rho_s`` y ``s[i] - s[i-1]``.

    Returns
    -------
    TurbidityCurrentResult
        Perfiles espaciales completos a lo largo de ``s``.
    """
    if mode not in ("constant", "varying"):
        raise ValueError("`mode` debe ser 'constant' o 'varying'.")

    s_arr = _build_distance_grid(s, ds1, ds2, s_transition, s_max, n_steps)
    n = s_arr.size
    slope_arr = _resolve_profile(slope_deg, s_arr, "slope_deg")

    pi_val = PI_TURBIDITY_EXCEL if strict_excel else PI_STANDARD
    pow_third = 0.33333333 if strict_excel else (1.0 / 3.0)
    rhos_conv = _EXCEL_RHOS_REF if strict_excel else rho_s
    div_chezy = (mode == "constant") if si_divide_by_chezy is None else si_divide_by_chezy
    first_step = float(s_arr[1] - s_arr[0])

    s_rel = rho_s / rho_w
    delta_rho = rho_s - rho_w
    cd = g / (chezy * chezy)
    cdi = cd / 3.0

    c_vol_0 = c20 / rhos_conv
    rho2_0 = rho_s * c_vol_0 + (1.0 - c_vol_0) * rho_w
    u20 = fr20 * math.sqrt((delta_rho / rho2_0) * c_vol_0 * h20 * g)
    q_cur = u20 * h20
    qs_vol_cur = h20 * c_vol_0 * u20
    h2_cur = float(h20)

    sin_beta_arr = np.sin(slope_arr / 180.0 * pi_val)
    c2_mass = np.empty(n, dtype=np.float64)
    c2_vol = np.empty(n, dtype=np.float64)
    rho2_arr = np.empty(n, dtype=np.float64)
    v2_arr = np.empty(n, dtype=np.float64)
    h2_arr = np.empty(n, dtype=np.float64)
    h2_eq_arr = np.empty(n, dtype=np.float64)
    h2_cr_arr = np.empty(n, dtype=np.float64)
    q_arr = np.empty(n, dtype=np.float64)
    qs_vol_arr = np.empty(n, dtype=np.float64)
    qs_mass_arr = np.empty(n, dtype=np.float64)
    fr_arr = np.empty(n, dtype=np.float64)
    ri_arr = np.empty(n, dtype=np.float64)
    nu_arr = np.empty(n, dtype=np.float64)
    re_arr = np.empty(n, dtype=np.float64)
    tau_i_arr = np.empty(n, dtype=np.float64)
    tau_b_arr = np.empty(n, dtype=np.float64)
    wi_arr = np.empty(n, dtype=np.float64)
    wb_arr = np.empty(n, dtype=np.float64)
    si_arr = np.empty(n, dtype=np.float64)
    sb_arr = np.empty(n, dtype=np.float64)
    dh2_arr = np.empty(n, dtype=np.float64)
    dq_arr = np.empty(n, dtype=np.float64)
    dqs_arr = np.empty(n, dtype=np.float64)

    prev_c_vol = c_vol_0
    prev_vbe_diff = 0.0

    for i in range(n):
        sin_b = float(sin_beta_arr[i])
        if i == 0:
            c_vol = c_vol_0
        elif mode == "varying" and prev_c_vol >= 0.9 * c2max:
            c_vol = 0.9 * c2max
        else:
            c_vol = qs_vol_cur / q_cur

        c_mass = c_vol * rho_s
        rho2 = rho_s * c_vol + (1.0 - c_vol) * rho_w
        v2 = u20 if i == 0 else (q_cur / h2_cur)
        dc_ds = 0.0 if i == 0 else (c_vol - prev_c_vol) / float(s_arr[i] - s_arr[i - 1])

        fr = q_cur / (h2_cur * math.sqrt((delta_rho / rho2) * c_vol * h2_cur * g))
        ri = delta_rho * g * c_vol * h2_cur / (rho2 * v2 * v2)

        if mode == "constant":
            nu = 0.001 if c_vol > 0.9 * c2max else nu0 * (1.0 - c_vol / c2max) ** (-3.0)
        else:
            nu = 0.0001 if c_vol > 0.5 * c2max else nu0 * (1.0 - c_vol / c2max) ** (-5.0)

        re = v2 * h2_cur / nu
        if re > 5000.0:
            tau_i = rho2 * cdi * q_cur * q_cur / (h2_cur * h2_cur)
            tau_b = rho2 * cd * q_cur * q_cur / (h2_cur * h2_cur)
        else:
            tau_i = rho2 * nu * (v2 * h2_cur) / (h2_cur * h2_cur)
            tau_b = 3.0 * rho2 * nu * (v2 * h2_cur) / (h2_cur * h2_cur)

        efact = min(0.002 / ri, 0.02)
        if re < 5000.0:
            wi = -(1.0 - c_vol) * 1.0e-5 * rewi
        else:
            wi = rewi * efact * v2 * (1.0 - c1 / rhos_conv)

        wb = (1.0 - porosity) * 1.0e-5 * (rewi if mode == "varying" else 1.0)

        si_dep = (ws1 + wi) * (c1 / rho_s) * (1.0 - tau_i / tau_cr_dep) * resi
        si_turb = -0.025 * (math.sqrt(g) / chezy if div_chezy else math.sqrt(g)) * v2 * c_vol * resi
        si_lam = (ws1 if mode == "constant" else c2max) * c1 / rho_s
        if tau_b < tau_cr_dep:
            si = si_dep
        elif re > 5000.0:
            si = si_turb
        else:
            si = si_lam

        vbs = -ws2 * c_vol * (1.0 - c_vol) ** 4
        if mode == "constant":
            if tau_b < tau_cr_ero or (i >= 1 and prev_c_vol > c2max):
                vbe = 0.0
            else:
                vbe = eros * (tau_b - tau_cr_ero) / tau_cr_ero
            sb = vbe + vbs
        else:
            if i >= 1 and prev_vbe_diff > 0.1:
                vbe_diff = prev_vbe_diff
            else:
                vbe_diff = eros * ((tau_b - tau_cr_ero) / tau_cr_ero) - ws2 * ((1.0 - c_vol) ** 4) * c_vol
            prev_vbe_diff = vbe_diff
            sb = vbs if tau_b < tau_cr_ero else vbe_diff

        h2_eq = (((tau_i + tau_b) * h2_cur * h2_cur) / (delta_rho * c_vol * g * sin_b)) ** pow_third
        h2_cr = ((q_cur * q_cur * rho2) / (delta_rho * g * c_vol)) ** pow_third
        alpha1 = 2.0 * rho2 * v2 * v2 / (delta_rho * (1.0 - c_vol) * c_vol * h2_cur * g * sin_b)
        alpha2 = (
            2.0 * rho2 * v2 * v2
            + delta_rho * (1.0 - c_vol) * v2 * v2
            + 0.5 * delta_rho * (1.0 - c_vol) * h2_cur * g
        ) / (delta_rho * (1.0 - c_vol) * c_vol * h2_cur * g * sin_b)

        ds_step = first_step if i == 0 else float(s_arr[i] - s_arr[i - 1])
        ds_dh2 = first_step if (mode == "varying" and strict_excel) else ds_step
        dh2 = (
            ds_dh2
            * sin_b
            * (1.0 - (h2_eq / h2_cur) ** 3 - (alpha1 * (wi + wb) / v2) - alpha2 * h2_cur * dc_ds)
            / (1.0 - (h2_cr / h2_cur) ** 3)
        )
        dqs_vol = 0.0 if c_vol > c2max else (si + sb) * ds_step
        dq = dqs_vol + (wi + wb) * ds_step

        c2_mass[i] = c_mass
        c2_vol[i] = c_vol
        rho2_arr[i] = rho2
        v2_arr[i] = v2
        h2_arr[i] = h2_cur
        h2_eq_arr[i] = h2_eq
        h2_cr_arr[i] = h2_cr
        q_arr[i] = q_cur
        qs_vol_arr[i] = qs_vol_cur
        qs_mass_arr[i] = qs_vol_cur * rhos_conv
        fr_arr[i] = fr
        ri_arr[i] = ri
        nu_arr[i] = nu
        re_arr[i] = re
        tau_i_arr[i] = tau_i
        tau_b_arr[i] = tau_b
        wi_arr[i] = wi
        wb_arr[i] = wb
        si_arr[i] = si
        sb_arr[i] = sb
        dh2_arr[i] = dh2
        dq_arr[i] = dq
        dqs_arr[i] = dqs_vol

        prev_c_vol = c_vol
        h2_cur += dh2
        q_cur += dq
        qs_vol_cur += dqs_vol

    return TurbidityCurrentResult(
        s=s_arr,
        slope_deg=slope_arr,
        sin_beta=sin_beta_arr,
        c2_mass=c2_mass,
        c2_vol=c2_vol,
        rho2=rho2_arr,
        v2=v2_arr,
        h2=h2_arr,
        h2_equil=h2_eq_arr,
        h2_crit=h2_cr_arr,
        q=q_arr,
        qs_vol=qs_vol_arr,
        qs_mass=qs_mass_arr,
        froude=fr_arr,
        richardson=ri_arr,
        nu=nu_arr,
        reynolds=re_arr,
        tau_i=tau_i_arr,
        tau_b=tau_b_arr,
        wi=wi_arr,
        wb=wb_arr,
        si=si_arr,
        sb=sb_arr,
        dh2=dh2_arr,
        dq=dq_arr,
        dqs_vol=dqs_arr,
        u20=u20,
        s_rel=s_rel,
        delta_rho=delta_rho,
        cd=cd,
        cdi=cdi,
    )


def turbidity_current_sand(
    slope_deg: float | Sequence[float] | NDArray[np.floating] | Callable[[float], float] = 10.0,
    s: Sequence[float] | NDArray[np.floating] | None = None,
    *,
    ds1: float = 0.1,
    ds2: float = 0.5,
    s_transition: float = 2.0,
    s_max: float | None = None,
    n_steps: int | None = 581,
    d50: float = 1.0e-4,
    wso: float = 0.008,
    rho_s: float = RHO_SEDIMENT,
    c2max: float = 0.6,
    sin_phi: float = 0.5,
    theta_cr: float = 0.06,
    eros: float = 0.0025,
    rho_w: float = 1020.0,
    nu0: float = KINEMATIC_VISCOSITY_20C,
    g: float = GRAVITY,
    chezy: float = 35.0,
    porosity: float = 0.4,
    h20: float = 0.04,
    c20: float = 400.0,
    fr20: float = 2.0,
    c1: float = 0.0,
    resi: float = 1.0,
    resb: float | Sequence[float] | NDArray[np.floating] | Callable[[float], float] = 1.0,
    rewi: float = 1.0,
    rewb: float = 1.0,
    viscosity_exponent: float = -3.0,
    wi_c1_volumetric: bool = True,
    strict_excel: bool = False,
) -> TurbidityCurrentResult:
    """Simula una corriente de turbidez de arena a lo largo del talud.

    Reproduce todos los libros de arena de van Rijn (``Turbidity current_sand.xlsx``,
    ``minimum slope``, ``Scripps`` y ``Zaire case``), permitiendo taludes y
    factores ``resb(s)`` variables (p. ej. lecho no erodible en un tramo).

    Parameters
    ----------
    slope_deg:
        Angulo del talud (grados), escalar, arreglo o funcion ``f(s)``.
    s:
        Arreglo opcional de distancias longitudinales (m). Si es ``None``,
        se construye con ``ds1``, ``ds2``, ``s_transition`` y ``s_max`` / ``n_steps``.
    d50, wso:
        Diametro mediano del grano (m) y velocidad de caida de grano aislado (m/s).
    rho_s, rho_w:
        Densidad del sedimento y del agua ambiente (kg/m3).
    c2max, sin_phi, theta_cr, eros:
        Concentracion maxima (-), seno del angulo de reposo (-), parametro de
        Shields critico (-) y coeficiente de erosion (-).
    nu0, g, chezy, porosity:
        Viscosidad del agua limpia (m2/s), gravedad (m/s2), coeficiente de
        Chezy (m0.5/s) y porosidad del lecho (-).
    h20, c20, fr20, c1:
        Condiciones iniciales de espesor (m), concentracion (kg/m3), Froude (-)
        y concentracion de fondo en la capa superior (kg/m3).
    resi, resb, rewi, rewb:
        Factores de reduccion para intercambio de sedimento (``Si``, ``Sb``) y
        agua (``Wi``, ``Wb``). ``resb`` admite escalar, arreglo o ``f(s)``.
    viscosity_exponent:
        Exponente de hindered viscosity ``(1 - c_vol / c2max) ** p`` (``-3.0`` en
        ``Constant slope`` y Zaire; ``-1.5`` en algunas hojas de ``Varying slope``).
    wi_c1_volumetric:
        Si es ``True`` (defecto fisico), usa ``1 - c1 / rho_s`` en ``Wi``; si es
        ``False``, reproduce ``1 - c1`` de ciertas hojas cuando ``c1 > 0``.
    strict_excel:
        Si es ``True``, usa ``pi=3.142``, exponente ``0.33333333`` (``0.333333`` en
        ``D*``) y divisor ``2650.0``. Por defecto es ``False``.

    Returns
    -------
    TurbidityCurrentResult
        Perfiles espaciales completos a lo largo de ``s``.
    """
    s_arr = _build_distance_grid(s, ds1, ds2, s_transition, s_max, n_steps)
    n = s_arr.size
    slope_arr = _resolve_profile(slope_deg, s_arr, "slope_deg")
    resb_arr = _resolve_profile(resb, s_arr, "resb")

    pi_val = PI_TURBIDITY_EXCEL if strict_excel else PI_STANDARD
    pow_third = 0.33333333 if strict_excel else (1.0 / 3.0)
    pow_third_dstar = 0.333333 if strict_excel else (1.0 / 3.0)
    rhos_conv = _EXCEL_RHOS_REF if strict_excel else rho_s
    first_step = float(s_arr[1] - s_arr[0])

    s_rel = rho_s / rho_w
    delta_rho = rho_s - rho_w
    d_star = d50 * ((s_rel - 1.0) * g / (nu0 * nu0)) ** pow_third_dstar
    cd = g / (chezy * chezy)
    cdi = cd / 3.0

    c_vol_0 = c20 / rhos_conv
    rho2_0 = rho_s * c_vol_0 + (1.0 - c_vol_0) * rho_w
    u20 = fr20 * math.sqrt((delta_rho / rho2_0) * c_vol_0 * h20 * g)
    q_cur = u20 * h20
    qs_vol_cur = h20 * c_vol_0 * u20
    h2_cur = float(h20)

    sin_beta_arr = np.sin(slope_arr / 180.0 * pi_val)
    c2_mass = np.empty(n, dtype=np.float64)
    c2_vol = np.empty(n, dtype=np.float64)
    rho2_arr = np.empty(n, dtype=np.float64)
    v2_arr = np.empty(n, dtype=np.float64)
    h2_arr = np.empty(n, dtype=np.float64)
    h2_eq_arr = np.empty(n, dtype=np.float64)
    h2_cr_arr = np.empty(n, dtype=np.float64)
    q_arr = np.empty(n, dtype=np.float64)
    qs_vol_arr = np.empty(n, dtype=np.float64)
    qs_mass_arr = np.empty(n, dtype=np.float64)
    fr_arr = np.empty(n, dtype=np.float64)
    ri_arr = np.empty(n, dtype=np.float64)
    nu_arr = np.empty(n, dtype=np.float64)
    re_arr = np.empty(n, dtype=np.float64)
    tau_i_arr = np.empty(n, dtype=np.float64)
    tau_b_arr = np.empty(n, dtype=np.float64)
    wi_arr = np.empty(n, dtype=np.float64)
    wb_arr = np.empty(n, dtype=np.float64)
    si_arr = np.empty(n, dtype=np.float64)
    sb_arr = np.empty(n, dtype=np.float64)
    dh2_arr = np.empty(n, dtype=np.float64)
    dq_arr = np.empty(n, dtype=np.float64)
    dqs_arr = np.empty(n, dtype=np.float64)

    sqrt_g_d50 = math.sqrt((s_rel - 1.0) * g * d50)
    sqrt_g_over_c = math.sqrt(g) / chezy
    dstar_factor = d_star ** 0.3
    c1_term = (1.0 - c1 / rhos_conv) if wi_c1_volumetric else (1.0 - c1)
    prev_c_vol = c_vol_0

    for i in range(n):
        sin_b = float(sin_beta_arr[i])
        resb_i = float(resb_arr[i])
        c_vol = c_vol_0 if i == 0 else (qs_vol_cur / q_cur)
        c_mass = c_vol * rho_s
        rho2 = rho_s * c_vol + (1.0 - c_vol) * rho_w
        v2 = u20 if i == 0 else (q_cur / h2_cur)
        dc_ds = 0.0 if i == 0 else (c_vol - prev_c_vol) / float(s_arr[i] - s_arr[i - 1])

        fr = q_cur / (h2_cur * math.sqrt((delta_rho / rho2) * c_vol * h2_cur * g))
        ri = delta_rho * g * c_vol * h2_cur / (rho2 * v2 * v2)
        nu = nu0 * (1.0 - c_vol / c2max) ** viscosity_exponent
        re = v2 * h2_cur / nu
        tau_i = rho2 * cdi * q_cur * q_cur / (h2_cur * h2_cur)
        tau_b = rho2 * cd * q_cur * q_cur / (h2_cur * h2_cur)

        efact = min(0.002 / ri, 0.02)
        wi = rewi * efact * v2 * c1_term

        theta_diff = ((sqrt_g_over_c * v2) ** 2 / ((s_rel - 1.0) * g * d50)) - theta_cr
        theta_ex = theta_diff if theta_diff > 0.0 else 0.0
        vbee = eros * sqrt_g_d50 * math.sqrt(theta_ex) * dstar_factor / (1.0 - sin_b / sin_phi)
        vbeemax = 0.033 * (c2max / c_vol - 1.0) * sqrt_g_d50
        vbe = vbeemax if vbee > vbeemax else vbee
        vbs = wso * ((1.0 - 2.0 * c_vol) ** 4) * (2.0 * c_vol)

        sb = resb_i * ((1.0 - porosity) * vbe - vbs)
        wb = (rewb * porosity * vbe) if sb > 0.0 else 0.0
        si = resi * ((wso + wi) * (c1 / rhos_conv) - 0.05 * sqrt_g_over_c * v2 * 0.5 * c_vol)

        h2_eq = (((tau_i + tau_b) * h2_cur * h2_cur) / (delta_rho * c_vol * g * sin_b)) ** pow_third
        h2_cr = ((q_cur * q_cur * rho2) / (delta_rho * g * c_vol)) ** pow_third
        alpha1 = 2.0 * rho2 * v2 * v2 / (delta_rho * (1.0 - c_vol) * c_vol * h2_cur * g * sin_b)
        alpha2 = (
            2.0 * rho2 * v2 * v2
            + delta_rho * (1.0 - c_vol) * v2 * v2
            + 0.5 * delta_rho * (1.0 - c_vol) * h2_cur * g
        ) / (delta_rho * (1.0 - c_vol) * c_vol * h2_cur * g * sin_b)

        ds_step = first_step if i == 0 else float(s_arr[i] - s_arr[i - 1])
        dh2 = (
            ds_step
            * sin_b
            * (1.0 - (h2_eq / h2_cur) ** 3 - (alpha1 * (wi + wb) / v2) - alpha2 * h2_cur * dc_ds)
            / (1.0 - (h2_cr / h2_cur) ** 3)
        )
        dqs_vol = (sb + si) * ds_step
        dq = dqs_vol + (wi + wb) * ds_step

        c2_mass[i] = c_mass
        c2_vol[i] = c_vol
        rho2_arr[i] = rho2
        v2_arr[i] = v2
        h2_arr[i] = h2_cur
        h2_eq_arr[i] = h2_eq
        h2_cr_arr[i] = h2_cr
        q_arr[i] = q_cur
        qs_vol_arr[i] = qs_vol_cur
        qs_mass_arr[i] = qs_vol_cur * rhos_conv
        fr_arr[i] = fr
        ri_arr[i] = ri
        nu_arr[i] = nu
        re_arr[i] = re
        tau_i_arr[i] = tau_i
        tau_b_arr[i] = tau_b
        wi_arr[i] = wi
        wb_arr[i] = wb
        si_arr[i] = si
        sb_arr[i] = sb
        dh2_arr[i] = dh2
        dq_arr[i] = dq
        dqs_arr[i] = dqs_vol

        prev_c_vol = c_vol
        h2_cur += dh2
        q_cur += dq
        qs_vol_cur += dqs_vol

    return TurbidityCurrentResult(
        s=s_arr,
        slope_deg=slope_arr,
        sin_beta=sin_beta_arr,
        c2_mass=c2_mass,
        c2_vol=c2_vol,
        rho2=rho2_arr,
        v2=v2_arr,
        h2=h2_arr,
        h2_equil=h2_eq_arr,
        h2_crit=h2_cr_arr,
        q=q_arr,
        qs_vol=qs_vol_arr,
        qs_mass=qs_mass_arr,
        froude=fr_arr,
        richardson=ri_arr,
        nu=nu_arr,
        reynolds=re_arr,
        tau_i=tau_i_arr,
        tau_b=tau_b_arr,
        wi=wi_arr,
        wb=wb_arr,
        si=si_arr,
        sb=sb_arr,
        dh2=dh2_arr,
        dq=dq_arr,
        dqs_vol=dqs_arr,
        u20=u20,
        s_rel=s_rel,
        delta_rho=delta_rho,
        cd=cd,
        cdi=cdi,
        d_star=d_star,
    )
