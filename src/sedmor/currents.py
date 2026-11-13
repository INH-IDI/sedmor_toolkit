"""Corrientes mareales, inducidas por viento y estratificadas por densidad.

Este modulo implementa en Python las herramientas de calculo de:

1. Asimetria mareal M2 + M4 (``Computation Tidal asymmetry M2 and M4 tides.xlsx``).
2. Corrientes inducidas por viento promediadas en la vertical y perfil logaritmico
   bipartito viento-fondo (``Computation Wind induced velocity.xlsx``).
3. Flujos estratificados por gradiente longitudinal de densidad / salinidad y
   sedimentacion en canales paralelos estratificados
   (``Computation circulation flows density effect.xlsx`` y
   ``Sedimentation due to  parallel stratied flow in tidal shipping channel.xlsx``).

Convencion ``strict_excel``
--------------------------
* ``strict_excel=False`` (por defecto): utiliza ``math.pi``, ``1.0 / 3.0`` y corrige
  errores tipograficos documentados de las planillas originales (p. ej. la integral
  simetrica ``q2 = -1 + (z_b / h) * ln((z_b + h) / z_b)`` en el perfil logaritmico de
  viento).
* ``strict_excel=True``: reproduce 1:1 las constantes truncadas (``3.14``, ``0.3333``)
  y referencias de celda exactas de las planillas de Excel 2013.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sedmor.constants import (
    GRAVITY,
    PI_EXCEL,
    PI_STANDARD,
    TWO_PI_EXCEL,
)

__all__ = [
    "TidalAsymmetryResult",
    "WindDepthAveragedResult",
    "WindVelocityProfileResult",
    "StratifiedFlowResult",
    "StratifiedSedimentationResult",
    "TIDAL_ASYMMETRY_REFERENCE_TABLE",
    "tidal_asymmetry_m2_m4",
    "wind_induced_depth_averaged_velocity",
    "wind_induced_velocity_profile",
    "stratified_density_flow_profile",
    "stratified_channel_sedimentation",
]


@dataclass(frozen=True)
class TidalAsymmetryResult:
    """Resultado del analisis de asimetria mareal M2 + M4."""

    time_hours: NDArray[np.float64]
    u_m2: NDArray[np.float64]
    u_m4: NDArray[np.float64]
    u_total: NDArray[np.float64]
    u_flood_max: float
    u_ebb_max: float
    asymmetry_ratio: float
    flood_duration_fraction: float
    ebb_duration_fraction: float


@dataclass(frozen=True)
class WindDepthAveragedResult:
    """Velocidad promediada en la vertical inducida por viento y marea."""

    chezy: float
    wind_speed: NDArray[np.float64]
    w_zero: float
    v_no_tide: NDArray[np.float64]
    v_total_pos: NDArray[np.float64]
    v_total_neg: NDArray[np.float64]
    v_total: NDArray[np.float64]


@dataclass(frozen=True)
class WindVelocityProfileResult:
    """Perfil vertical de velocidad inducida por viento segun van Rijn."""

    z: NDArray[np.float64]
    u: NDArray[np.float64]
    coef_a: float
    coef_b: float
    p1: float
    p2: float
    q1: float
    q2: float
    u_mean_analytical: float


@dataclass(frozen=True)
class StratifiedFlowResult:
    """Perfil vertical de velocidad residual por gradiente de salinidad y caudal fluvial."""

    z_rel: NDArray[np.float64]
    z: NDArray[np.float64]
    m_sal: float
    u_sal: NDArray[np.float64]
    u_river: NDArray[np.float64]
    u_total: NDArray[np.float64]
    u_mean_analytical: float


@dataclass(frozen=True)
class StratifiedSedimentationResult:
    """Deposito de sedimentos finos por circulacion estratificada en canal."""

    sediment_flux_kg_s: float
    deposited_mass_kg: float
    deposited_volume_m3: float
    deposited_layer_m: float


# Tabla de referencia incluida en las columnas G..M de la hoja Skewness and asymmetry
TIDAL_ASYMMETRY_REFERENCE_TABLE: dict[float, dict[int, tuple[float, float]]] = {
    0.2: {
        0: (0.441, 1.50),
        30: (0.448, 1.36),
        60: (0.465, 1.17),
        90: (0.500, 1.00),
    },
    0.3: {
        0: (0.415, 1.82),
        30: (0.423, 1.448),
        60: (0.446, 1.20),
        90: (0.500, 1.00),
    },
}


def _pi_pair(strict_excel: bool) -> tuple[float, float]:
    if strict_excel:
        return PI_EXCEL, TWO_PI_EXCEL
    return PI_STANDARD, 2.0 * PI_STANDARD


def _crossings_duration_fraction(
    t_cycle: NDArray[np.float64],
    u_cycle: NDArray[np.float64],
    period_hours: float,
) -> float:
    """Calcula la fraccion del periodo con corriente positiva (flujo) mediante interpolacion lineal."""
    positive_duration = 0.0
    for i in range(len(t_cycle) - 1):
        u0 = float(u_cycle[i])
        u1 = float(u_cycle[i + 1])
        dt = float(t_cycle[i + 1] - t_cycle[i])
        if u0 >= 0.0 and u1 >= 0.0:
            positive_duration += dt
        elif u0 < 0.0 and u1 < 0.0:
            continue
        else:
            frac = abs(u0) / (abs(u0) + abs(u1))
            if u0 >= 0.0:
                positive_duration += frac * dt
            else:
                positive_duration += (1.0 - frac) * dt
    return float(positive_duration / period_hours)


def tidal_asymmetry_m2_m4(
    u1: float = 1.0,
    u2: float = 0.3,
    beta_deg: float = 360.0,
    period_hours: float = 12.0,
    time_hours: ArrayLike | None = None,
    dt_hours: float = 0.25,
    duration_hours: float = 45.0,
    *,
    strict_excel: bool = False,
) -> TidalAsymmetryResult:
    """Calcula la serie temporal e indicadores de asimetria mareal M2 + M4.

    Corresponde a ``Computation Tidal asymmetry M2 and M4 tides.xlsx``
    (hoja ``Skewness and asymmetry``):

    .. math::
        U(t) = U_1 \\cos\\!\\left(\\frac{2\\pi t}{T}\\right)
             + U_2 \\cos\\!\\left(\\frac{4\\pi t}{T} - \\beta\\right)

    Parametros
    ----------
    u1 : float
        Amplitud de velocidad de la componente principal M2 [m/s].
    u2 : float
        Amplitud de velocidad de la componente armónica M4 [m/s].
    beta_deg : float
        Desfase entre M2 y M4 en grados sexagesimales [grados].
    period_hours : float
        Periodo mareal M2 [horas] (tipicamente 12 o 12.42 h).
    time_hours : ArrayLike | None
        Vector opcional de instantes de tiempo [horas]. Si es ``None``, se genera
        ``0, dt_hours, ..., duration_hours``.
    dt_hours : float
        Paso temporal por defecto [horas].
    duration_hours : float
        Duracion total de la serie por defecto [horas].
    strict_excel : bool
        Si es ``True``, usa ``pi = 3.14`` y ``2*pi = 6.28``.
    """
    if u1 <= 0.0 or period_hours <= 0.0:
        raise ValueError("u1 y period_hours deben ser estrictamente positivos.")

    pi_val, two_pi = _pi_pair(strict_excel)

    if time_hours is None:
        n_steps = int(round(duration_hours / dt_hours)) + 1
        t = np.linspace(0.0, n_steps * dt_hours - dt_hours, n_steps, dtype=np.float64)
    else:
        t = np.asarray(time_hours, dtype=np.float64)

    beta_rad = (beta_deg / 360.0) * two_pi
    u_m2 = u1 * np.cos(two_pi * t / period_hours)
    u_m4 = u2 * np.cos(2.0 * two_pi * t / period_hours - beta_rad)
    u_total = u_m2 + u_m4

    # Evaluacion fina sobre un ciclo completo [0, T] para maximos y duraciones exactas
    t_fine = np.linspace(0.0, period_hours, 4001, dtype=np.float64)
    u_fine = u1 * np.cos(two_pi * t_fine / period_hours) + u2 * np.cos(
        2.0 * two_pi * t_fine / period_hours - beta_rad
    )
    u_flood_max = float(np.max(u_fine))
    u_ebb_max = float(abs(np.min(u_fine)))
    asymmetry_ratio = float(u_flood_max / u_ebb_max) if u_ebb_max > 0.0 else math.inf
    flood_frac = _crossings_duration_fraction(t_fine, u_fine, period_hours)

    return TidalAsymmetryResult(
        time_hours=t,
        u_m2=u_m2,
        u_m4=u_m4,
        u_total=u_total,
        u_flood_max=u_flood_max,
        u_ebb_max=u_ebb_max,
        asymmetry_ratio=asymmetry_ratio,
        flood_duration_fraction=flood_frac,
        ebb_duration_fraction=1.0 - flood_frac,
    )


def wind_induced_depth_averaged_velocity(
    wind_speed: ArrayLike = (
        0.0,
        2.0,
        5.0,
        7.0,
        10.0,
        12.0,
        15.0,
        20.0,
        25.0,
        30.0,
        40.0,
        50.0,
    ),
    v_tide: float = -0.8,
    theta_deg: float = 90.0,
    depth: float = 15.0,
    k_s: float = 0.01,
    wind_drag_k: float = 5.0e-6,
    gravity: float = GRAVITY,
    *,
    strict_excel: bool = False,
) -> WindDepthAveragedResult:
    """Calcula la velocidad promediada en la vertical inducida por viento y marea.

    Corresponde a ``Computation Wind induced velocity.xlsx``
    (hoja ``Depth-averaged velocity``):

    .. math::
        C = 18 \\log_{10}\\!\\left(\\frac{12 h}{k_s}\\right), \\qquad
        W_0 = \\sqrt{\\frac{g \\,|V_{\\mathrm{marea}}|^2}{C^2 K \\sin\\theta}}

    Parametros
    ----------
    wind_speed : ArrayLike
        Velocidad del viento :math:`W` a 10 m de altura [m/s].
    v_tide : float
        Velocidad mareal de fondo :math:`V_{\\mathrm{marea}}` [m/s]. El signo define
        el sentido respecto al componente longitudinal del viento en ``v_total``.
    theta_deg : float
        Angulo de incidencia del viento respecto a la costa/canal [grados].
    depth : float
        Profundidad de agua :math:`h` [m].
    k_s : float
        Rugosidad equivalente de Nikuradse :math:`k_s` [m].
    wind_drag_k : float
        Coeficiente adimensional de friccion eolica :math:`K` [-].
    gravity : float
        Aceleracion gravitacional [m/s^2].
    strict_excel : bool
        Si es ``True``, usa ``2*pi = 6.28`` en la conversion de grados.
    """
    if depth <= 0.0 or k_s <= 0.0:
        raise ValueError("depth y k_s deben ser estrictamente positivos.")

    _, two_pi = _pi_pair(strict_excel)
    chezy = 18.0 * math.log10(12.0 * depth / k_s)
    sin_theta = math.sin((theta_deg / 360.0) * two_pi)
    if sin_theta < 0.0 and not strict_excel:
        # El esfuerzo eolico proyectado debe tratarse con cuidado si theta > 180
        sin_eff = sin_theta
    else:
        sin_eff = sin_theta

    w_arr = np.asarray(wind_speed, dtype=np.float64)
    v2 = float(v_tide) ** 2
    denom = (chezy**2) * wind_drag_k * sin_eff
    w_zero = math.sqrt((gravity * v2) / denom) if denom > 0.0 else math.inf

    rest = (wind_drag_k * (chezy**2) * sin_eff / gravity) * (w_arr**2)
    v_no_tide = np.sqrt(np.maximum(rest, 0.0))
    v_total_pos = np.sqrt(np.maximum(v2 + rest, 0.0))
    v_total_neg = np.where(
        w_arr < w_zero,
        -np.sqrt(np.maximum(v2 - rest, 0.0)),
        np.sqrt(np.maximum(rest - v2, 0.0)),
    )
    v_total = np.where(v_tide >= 0.0, v_total_pos, v_total_neg)

    return WindDepthAveragedResult(
        chezy=float(chezy),
        wind_speed=w_arr,
        w_zero=float(w_zero),
        v_no_tide=v_no_tide,
        v_total_pos=v_total_pos,
        v_total_neg=v_total_neg,
        v_total=v_total,
    )


def wind_induced_velocity_profile(
    z: ArrayLike | None = None,
    depth: float = 5.0,
    z_s: float = 0.001,
    z_b: float = 0.001,
    u_star_wind: float = 1.0,
    lam: float = 0.5,
    *,
    strict_excel: bool = False,
) -> WindVelocityProfileResult:
    """Perfil vertical logaritmico bipartito de velocidad inducida por viento.

    Corresponde a ``Computation Wind induced velocity.xlsx``
    (hoja ``wind velocity profile``), donde ``z`` es la distancia bajo la superficie
    libre (``z=0`` en superficie, ``z=h`` en el fondo) o coordenada vertical segun
    el sistema de la planilla:

    .. math::
        u(z) = u_{*w} \\left[
            A \\ln\\!\\left(\\frac{z + z_s}{z_s}\\right)
            + B \\ln\\!\\left(\\frac{z_b + h - z}{z_b + h}\\right)
        \\right]

    donde :math:`A` y :math:`B` satisfacen continuidad de flujo de retorno neto nulo
    (``p1*A + p2*B = 0``, ``q1*A + q2*B = -u_mean/u_*`` segun la formulacion matricial
    de van Rijn).

    Nota sobre ``strict_excel``
    ---------------------------
    En la hoja original ``wind velocity profile``, la celda ``B14`` (coeficiente ``q2``)
    contiene un error de referencia ``-1 + ($B$4/$B$3)*LN(($B$5+$B$3)/$B$3)`` usando
    ``z_s/h`` y ``h`` en lugar de ``z_b/h`` y ``z_b``. Con ``strict_excel=False`` se
    utiliza la integral analitica exacta ``-1 + (z_b/h)*ln((z_b+h)/z_b)``; con
    ``strict_excel=True`` se reproduce exactamente la celda ``B14`` de Excel.
    """
    if depth <= 0.0 or z_s <= 0.0 or z_b <= 0.0:
        raise ValueError("depth, z_s y z_b deben ser estrictamente positivos.")

    if z is None:
        z_arr = np.concatenate(
            [
                np.array([0.0, 0.01, 0.05, 0.1, 0.2, 0.5]),
                np.linspace(1.0, max(1.0, depth - 0.5), 8),
                np.array([depth - 0.1, depth - 0.05, depth - 0.01, depth]),
            ]
        )
        z_arr = np.unique(np.clip(z_arr, 0.0, depth))
    else:
        z_arr = np.asarray(z, dtype=np.float64)

    p1 = lam * z_s / depth
    p2 = lam * z_s / z_b
    q1 = -1.0 + (1.0 + z_s / depth) * math.log((z_s + depth) / z_s)
    if strict_excel:
        q2 = -1.0 + (z_s / depth) * math.log((z_b + depth) / depth)
    else:
        q2 = -1.0 + (z_b / depth) * math.log((z_b + depth) / z_b)

    denom = p1 * q2 - q1 * p2
    if denom == 0.0:
        raise ZeroDivisionError("Determinante nulo en el sistema de coeficientes A, B.")

    coef_a = q2 / denom
    coef_b = -q1 / denom

    u = u_star_wind * (
        coef_a * np.log((z_arr + z_s) / z_s)
        + coef_b * np.log((z_b + depth - z_arr) / (z_b + depth))
    )

    q2_exact = -1.0 + (z_b / depth) * math.log((z_b + depth) / z_b)
    u_mean_analytical = u_star_wind * (coef_a * q1 + coef_b * q2_exact)

    return WindVelocityProfileResult(
        z=z_arr,
        u=u,
        coef_a=float(coef_a),
        coef_b=float(coef_b),
        p1=float(p1),
        p2=float(p2),
        q1=float(q1),
        q2=float(q2),
        u_mean_analytical=float(u_mean_analytical),
    )


def stratified_density_flow_profile(
    z_rel: ArrayLike | None = None,
    depth: float = 16.0,
    u_river: float = 0.2,
    u_peak: float = 1.0,
    d_rho_dx: float = 0.0001,
    chezy: float = 50.0,
    gamma: float = 0.005,
    *,
    strict_excel: bool = False,
) -> StratifiedFlowResult:
    """Calcula el perfil vertical de circulacion gravitacional por densidad y flujo fluvial.

    Corresponde a ``Computation circulation flows density effect.xlsx``
    (hoja ``Stratified flow``):

    .. math::
        M = \\frac{3.13\\,C}{\\gamma\\,h_0\\,(u_r + U_{\\mathrm{pico}})}
            \\left(\\frac{h_0}{1000}\\frac{\\partial \\rho}{\\partial x}\\right)

    .. math::
        u_{\\mathrm{sal}}(z/h_0) = M\\,h_0^2\\left(
            -\\frac{1}{6}\\left(\\frac{z}{h_0}\\right)^3
            + \\frac{1}{2}\\left(\\frac{z}{h_0}\\right)^2
            - \\frac{1}{4}\\left(\\frac{z}{h_0}\\right)
        \\right)

    Parametros
    ----------
    z_rel : ArrayLike | None
        Elevacion adimensional ``z / h_0`` desde el fondo (0) hasta superficie (1).
    depth : float
        Profundidad de referencia :math:`h_0` [m].
    u_river : float
        Velocidad media fluvial :math:`u_{\\mathrm{river}}` [m/s].
    u_peak : float
        Amplitud de velocidad mareal :math:`U_{\\mathrm{pico}}` [m/s].
    d_rho_dx : float
        Gradiente longitudinal de densidad :math:`\\partial\\rho / \\partial x` [kg/m^4].
    chezy : float
        Coeficiente de Chezy :math:`C` [m^(1/2)/s].
    gamma : float
        Parametro empirico :math:`\\gamma` [-].
    strict_excel : bool
        Si es ``True``, utiliza ``3.14`` para pi y ``0.3333`` para ``1/3``.
    """
    if depth <= 0.0 or u_river <= 0.0 or chezy <= 0.0 or gamma <= 0.0:
        raise ValueError("depth, u_river, chezy y gamma deben ser positivos.")

    if z_rel is None:
        zr = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    else:
        zr = np.asarray(z_rel, dtype=np.float64)

    pi_c = PI_EXCEL if strict_excel else PI_STANDARD
    third = 0.3333 if strict_excel else (1.0 / 3.0)

    m_sal = (
        (3.13 * chezy)
        / (gamma * depth * (u_river + u_peak))
        * (depth / 1000.0 * d_rho_dx)
    )
    u_sal = (
        m_sal
        * (depth**2)
        * (-(1.0 / 6.0) * (zr**3) + 0.5 * (zr**2) - 0.25 * zr)
    )
    u_riv = (
        (pi_c / (gamma * (1.0 + u_peak / u_river) * chezy))
        * (-0.5 * (zr**2) + third * zr)
        + 2.0 * zr
    ) * u_river
    u_total = u_sal + u_riv

    # Integral analitica exacta sobre [0, 1] para verificacion de caudal neto
    sal_int = -1.0 / 24.0 + 1.0 / 6.0 - 1.0 / 8.0  # = 0
    riv_int = (-1.0 / 6.0 + third / 2.0) * (
        pi_c / (gamma * (1.0 + u_peak / u_river) * chezy)
    ) + 1.0
    u_mean_analytical = float(m_sal * (depth**2) * sal_int + riv_int * u_river)

    return StratifiedFlowResult(
        z_rel=zr,
        z=zr * depth,
        m_sal=float(m_sal),
        u_sal=u_sal,
        u_river=u_riv,
        u_total=u_total,
        u_mean_analytical=u_mean_analytical,
    )


def stratified_channel_sedimentation(
    depth: float = 20.0,
    u_mean: float = 0.1,
    c_mean: float = 0.75,
    rho_dry: float = 300.0,
    width: float = 400.0,
    length: float = 5000.0,
    trapping_efficiency: float = 0.1,
    duration_days: float = 365.0,
    safety_factor: float = 1.5,
) -> StratifiedSedimentationResult:
    """Estima la tasa de sedimentacion en un canal de navegacion por flujo estratificado.

    Corresponde a la hoja ``Sedimentation stratified flow`` de
    ``Computation circulation flows density effect.xlsx`` (y la planilla gemela
    ``Sedimentation due to  parallel stratied flow in tidal shipping channel.xlsx``).
    """
    if depth <= 0.0 or rho_dry <= 0.0 or width <= 0.0 or length <= 0.0:
        raise ValueError("Dimensiones geometricas y rho_dry deben ser positivas.")

    flux_kg_s = 0.5 * depth * u_mean * c_mean * width
    deposited_mass = safety_factor * flux_kg_s * (duration_days * 86400.0) * trapping_efficiency
    deposited_vol = deposited_mass / rho_dry
    deposited_layer = deposited_vol / (width * length)

    return StratifiedSedimentationResult(
        sediment_flux_kg_s=float(flux_kg_s),
        deposited_mass_kg=float(deposited_mass),
        deposited_volume_m3=float(deposited_vol),
        deposited_layer_m=float(deposited_layer),
    )
