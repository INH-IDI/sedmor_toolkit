"""Transporte de arena en corrientes y olas costeras (formula de van Rijn).

Implementa las tres hojas del libro ``TRANSPOR SAND ANNUAL.xls``:
- ``Computation transport`` (corriente + oleaje regular/irregular)
- ``comp. wave-averaged transport`` (promediado por clima de olas)
- ``ANNUAL TRANSPORT`` (integracion horaria sobre un ciclo mareal y clases de oleaje)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sedmor.constants import (
    GRAVITY,
    KINEMATIC_VISCOSITY_20C,
    PI_STANDARD,
    RHO_SEA_WATER,
    RHO_SEDIMENT,
    TWO_PI_EXCEL,
)

__all__ = [
    "AnnualSandTransportResult",
    "DEFAULT_WAVE_CLASSES",
    "SandTransportCWResult",
    "WaveAveragedSandTransportResult",
    "annual_sand_transport",
    "sand_transport_current_waves",
    "wave_averaged_sand_transport",
]

# Clases de oleaje por defecto de ANNUAL TRANSPORT (H9:L18): (Vwind, Hs, Tp, porcentaje)
DEFAULT_WAVE_CLASSES: tuple[tuple[float, float, float, float], ...] = (
    (0.0, 0.25, 2.0, 50.0),
    (0.0, 0.75, 3.0, 20.0),
    (0.0, 1.25, 4.0, 10.0),
    (0.0, 1.75, 4.5, 7.0),
    (0.0, 2.25, 5.0, 5.0),
    (0.0, 2.75, 6.0, 4.0),
    (0.0, 3.25, 7.0, 2.0),
    (0.0, 3.75, 8.0, 1.25),
    (0.0, 4.75, 9.0, 0.5),
    (0.0, 5.50, 10.0, 0.25),
)

_TIDAL_SIN_WEIGHTS = np.array([0.26, 0.71, 0.965, 0.965, 0.71, 0.26], dtype=np.float64)


@dataclass(frozen=True)
class SandTransportCWResult:
    """Resultados de transporte de arena por corriente y oleaje (van Rijn 2007)."""

    d_star: float
    chezy: NDArray[np.float64] | float
    kh: NDArray[np.float64] | float
    wavelength: NDArray[np.float64] | float
    u_orb: NDArray[np.float64] | float
    u_eff: NDArray[np.float64] | float
    u_cr_current: NDArray[np.float64] | float
    u_cr_wave: NDArray[np.float64] | float
    u_cr_weighted: NDArray[np.float64] | float
    qb: NDArray[np.float64] | float
    qs: NDArray[np.float64] | float
    qt: NDArray[np.float64] | float


@dataclass(frozen=True)
class WaveAveragedSandTransportResult:
    """Transporte de arena promediado sobre clases de oleaje."""

    qb_per_class: NDArray[np.float64]
    qs_per_class: NDArray[np.float64]
    qt_per_class: NDArray[np.float64]
    qb_total: float
    qs_total: float
    qt_total: float


@dataclass(frozen=True)
class AnnualSandTransportResult:
    """Transporte neto anual de arena por ciclo mareal y clima de olas."""

    d_star: float
    flood_kg_m_s_per_class: NDArray[np.float64]
    ebb_kg_m_s_per_class: NDArray[np.float64]
    flood_kg_m_s_total: float
    ebb_kg_m_s_total: float
    flood_vol_incl_pores_m3_yr: float
    flood_vol_solid_m3_yr: float
    ebb_vol_incl_pores_m3_yr: float
    ebb_vol_solid_m3_yr: float
    net_vol_incl_pores_m3_yr: float
    net_vol_solid_m3_yr: float


def _as_out(arr: NDArray[np.float64], is_scalar: bool) -> NDArray[np.float64] | float:
    return float(arr) if is_scalar else arr


def _core_van_rijn_cw(
    depth: NDArray[np.float64],
    velocity: NDArray[np.float64],
    wave_height: NDArray[np.float64],
    wave_period: NDArray[np.float64],
    d50: float,
    d90: float,
    p_mud: float,
    ks: float,
    k_cr: float,
    k_cal: float,
    rho_w: float,
    rho_s: float,
    nu: float,
    gravity: float,
    orbital_factor: float,
    bedload_coef: float,
    bedload_exp: float,
    suspended_coef: float,
    use_mud_reduction: bool,
    force_magnitude_sign: bool,
    strict_excel: bool,
) -> tuple[
    float,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    s_rel = rho_s / rho_w - 1.0
    exp_dstar = 0.333333 if strict_excel else (1.0 / 3.0)
    d_star = float(d50 * (s_rel * gravity / nu**2) ** exp_dstar)

    chezy = 18.0 * np.log10(12.0 * depth / ks)

    x_w = np.where(wave_period > 0.0, 4.02 * depth / np.maximum(wave_period, 1e-12) ** 2, 0.0)
    kh = np.where(
        wave_period > 0.0,
        np.sqrt(x_w) * (1.0 + 0.166 * x_w + 0.031 * x_w**2),
        0.0,
    )
    two_pi = TWO_PI_EXCEL if strict_excel else 2.0 * PI_STANDARD
    pi_ub = 3.142 if strict_excel else PI_STANDARD
    wavelength = np.where(kh > 0.0, two_pi * depth / np.maximum(kh, 1e-30), 0.0)
    u_orb = np.where(
        (wave_period > 0.0) & (wave_height > 0.0),
        pi_ub * wave_height / (np.maximum(wave_period, 1e-12) * np.sinh(np.maximum(kh, 1e-12))),
        0.0,
    )

    u_abs = np.abs(velocity)
    u_eff = u_abs + orbital_factor * u_orb

    mud_cr = (1.0 + p_mud) ** 1.5
    if d50 < 0.0005:
        u_cr_c = 0.19 * mud_cr * d50**0.1 * np.log10(12.0 * depth / (3.0 * d90))
        u_cr_w = 0.24 * mud_cr * (s_rel * gravity) ** 0.66 * d50**0.33 * wave_period**0.33
    else:
        u_cr_c = 8.5 * mud_cr * d50**0.6 * np.log10(12.0 * depth / (3.0 * d90))
        g_prime = (rho_s - (rho_w / rho_w) * gravity) if strict_excel else (s_rel * gravity)
        u_cr_w = 0.95 * mud_cr * g_prime**0.57 * d50**0.43 * wave_period**0.14

    denom = u_abs + u_orb
    weight_c = np.where(denom > 0.0, u_abs / np.maximum(denom, 1e-30), 0.0)
    u_cr_cw = k_cr * (weight_c * u_cr_c + (1.0 - weight_c) * u_cr_w)

    excess = np.maximum(u_eff - u_cr_cw, 0.0)
    mud_factor = (1.0 - p_mud) if use_mud_reduction else 1.0

    # En la hoja ANNUAL TRANSPORT la columna S multiplica por u/|u| y T por -u/|u|
    # manteniendo el signo explicito de flujo (+) o reflujo (-).
    if force_magnitude_sign:
        u_term = np.where(velocity >= 0.0, u_abs, -u_abs)
    else:
        u_term = velocity

    bed_denom_exp = 0.75 if bedload_exp == 1.5 else 1.2
    qb = (
        k_cal
        * mud_factor
        * rho_s
        * bedload_coef
        * depth
        * u_term
        * (d50 / depth) ** 1.2
        * excess**bedload_exp
        / (s_rel * gravity * d50) ** bed_denom_exp
    )
    qs = (
        k_cal
        * mud_factor
        * rho_s
        * suspended_coef
        * depth
        * u_term
        * (d50 / depth)
        * excess**2.4
        / (s_rel * gravity * d50) ** 1.2
        * d_star ** (-0.6)
    )
    qb = np.where(u_eff > u_cr_cw, qb, 0.0)
    qs = np.where(u_eff > u_cr_cw, qs, 0.0)

    return d_star, chezy, kh, wavelength, u_orb, u_eff, u_cr_c, u_cr_w, u_cr_cw, qb, qs


def sand_transport_current_waves(
    depth: ArrayLike,
    velocity: ArrayLike,
    wave_height: ArrayLike,
    wave_period: ArrayLike,
    *,
    d50: float = 0.00025,
    d90: float = 0.0005,
    p_mud: float = 0.0,
    ks: float = 0.02,
    k_cr: float = 1.0,
    k_cal: float = 1.0,
    rho_w: float = RHO_SEA_WATER,
    rho_s: float = RHO_SEDIMENT,
    nu: float = KINEMATIC_VISCOSITY_20C,
    gravity: float = GRAVITY,
    strict_excel: bool = False,
) -> SandTransportCWResult:
    """Calcula el transporte de arena (fondo y suspension) bajo corrientes y olas.

    Implementa la hoja ``Computation transport`` de ``TRANSPOR SAND ANNUAL.xls``.
    """
    h_a, u_a, hs_a, tp_a = np.broadcast_arrays(
        np.asarray(depth, dtype=np.float64),
        np.asarray(velocity, dtype=np.float64),
        np.asarray(wave_height, dtype=np.float64),
        np.asarray(wave_period, dtype=np.float64),
    )
    is_scalar = h_a.ndim == 0

    (
        d_star,
        chezy,
        kh,
        wavelength,
        u_orb,
        u_eff,
        u_cr_c,
        u_cr_w,
        u_cr_cw,
        qb,
        qs,
    ) = _core_van_rijn_cw(
        h_a,
        u_a,
        hs_a,
        tp_a,
        d50=d50,
        d90=d90,
        p_mud=p_mud,
        ks=ks,
        k_cr=k_cr,
        k_cal=k_cal,
        rho_w=rho_w,
        rho_s=rho_s,
        nu=nu,
        gravity=gravity,
        orbital_factor=0.4,
        bedload_coef=0.015,
        bedload_exp=1.5,
        suspended_coef=0.008,
        use_mud_reduction=True,
        force_magnitude_sign=False,
        strict_excel=strict_excel,
    )
    qt = qb + qs
    return SandTransportCWResult(
        d_star=d_star,
        chezy=_as_out(chezy, is_scalar),
        kh=_as_out(kh, is_scalar),
        wavelength=_as_out(wavelength, is_scalar),
        u_orb=_as_out(u_orb, is_scalar),
        u_eff=_as_out(u_eff, is_scalar),
        u_cr_current=_as_out(u_cr_c, is_scalar),
        u_cr_wave=_as_out(u_cr_w, is_scalar),
        u_cr_weighted=_as_out(u_cr_cw, is_scalar),
        qb=_as_out(qb, is_scalar),
        qs=_as_out(qs, is_scalar),
        qt=_as_out(qt, is_scalar),
    )


def wave_averaged_sand_transport(
    depth: ArrayLike,
    velocity: ArrayLike,
    wave_height: ArrayLike,
    wave_period: ArrayLike,
    probabilities: ArrayLike,
    *,
    d50: float = 0.00025,
    d90: float = 0.0005,
    p_mud: float = 0.0,
    ks: float = 0.02,
    k_cr: float = 1.0,
    k_cal: float = 1.0,
    orbital_factor: float = 0.4,
    rho_w: float = RHO_SEA_WATER,
    rho_s: float = RHO_SEDIMENT,
    nu: float = KINEMATIC_VISCOSITY_20C,
    gravity: float = GRAVITY,
    strict_excel: bool = False,
) -> WaveAveragedSandTransportResult:
    """Calcula el transporte de arena promediado por probabilidad de ocurrencia de olas.

    Implementa la hoja ``comp. wave-averaged transport`` (usa coeficientes ``0.005`` para
    carga de fondo con exponente ``2.4`` y ``0.012`` para suspension).

    Parameters
    ----------
    probabilities : ArrayLike
        Fraccion de ocurrencia de cada clase de oleaje (debe sumar 1.0 en clima completo).
    orbital_factor : float, default=0.4
        Factor de velocidad orbital en ``Ueff = |u| + orbital_factor * Ub`` (``0.4`` en filas
        21-34 de la hoja o ``0.5`` en fila 37).
    """
    p_a = np.asarray(probabilities, dtype=np.float64)
    h_a, u_a, hs_a, tp_a, p_a = np.broadcast_arrays(
        np.asarray(depth, dtype=np.float64),
        np.asarray(velocity, dtype=np.float64),
        np.asarray(wave_height, dtype=np.float64),
        np.asarray(wave_period, dtype=np.float64),
        p_a,
    )

    _, _, _, _, _, _, _, _, _, qb_raw, qs_raw = _core_van_rijn_cw(
        h_a,
        u_a,
        hs_a,
        tp_a,
        d50=d50,
        d90=d90,
        p_mud=p_mud,
        ks=ks,
        k_cr=k_cr,
        k_cal=k_cal,
        rho_w=rho_w,
        rho_s=rho_s,
        nu=nu,
        gravity=gravity,
        orbital_factor=orbital_factor,
        bedload_coef=0.005,
        bedload_exp=2.4,
        suspended_coef=0.012,
        use_mud_reduction=False,
        force_magnitude_sign=False,
        strict_excel=strict_excel,
    )
    qb_cls = p_a * qb_raw
    qs_cls = p_a * qs_raw
    qt_cls = qb_cls + qs_cls

    return WaveAveragedSandTransportResult(
        qb_per_class=qb_cls,
        qs_per_class=qs_cls,
        qt_per_class=qt_cls,
        qb_total=float(np.sum(qb_cls)),
        qs_total=float(np.sum(qs_cls)),
        qt_total=float(np.sum(qt_cls)),
    )


def annual_sand_transport(
    depth_msl: float = 20.0,
    u_flood_peak: float = 0.5,
    u_ebb_peak: float = -0.4,
    eta_flood: float = 1.0,
    eta_ebb: float = -1.0,
    wave_classes: NDArray[np.float64] | tuple[tuple[float, float, float, float], ...] = DEFAULT_WAVE_CLASSES,
    *,
    d50: float = 0.0002,
    d90: float = 0.0004,
    p_mud: float = 0.0,
    k_cr: float = 1.0,
    k_cal: float = 1.0,
    porosity_factor: float = 0.6,
    tides_per_year: float = 730.0,
    rho_w: float = RHO_SEA_WATER,
    rho_s: float = RHO_SEDIMENT,
    nu: float = KINEMATIC_VISCOSITY_20C,
    gravity: float = GRAVITY,
    strict_excel: bool = False,
) -> AnnualSandTransportResult:
    """Calcula el transporte anual neto integrando 12 horas de marea y clases de oleaje.

    Implementa la hoja ``ANNUAL TRANSPORT`` de ``TRANSPOR SAND ANNUAL.xls``.

    Parameters
    ----------
    wave_classes : array-like of shape (n_classes, 4)
        Cada fila contiene ``(v_wind, Hs, Tp, porcentaje_ocurrencia)``.
    porosity_factor : float, default=0.6
        Fraccion solida ``(1 - p)`` usada para convertir masa a volumen incluyendo poros.
    strict_excel : bool, default=False
        Si es ``True``, reproduce exactamente el multiplicador ``730 * 6 * 3600`` aplicado sobre
        la suma de las 6 horas de flujo/reflujo y los signos forzados de las columnas ``S`` y ``T``.
        Si es ``False``, usa la integracion temporal fisica correcta ``730 * 3600`` sobre la suma
        de las 6 horas (1 h por paso).
    """
    wc = np.asarray(wave_classes, dtype=np.float64)
    if wc.ndim != 2 or wc.shape[1] != 4:
        raise ValueError("wave_classes debe tener forma (n_classes, 4): (Vwind, Hs, Tp, %).")

    n_cls = wc.shape[0]
    flood_cls = np.zeros(n_cls, dtype=np.float64)
    ebb_cls = np.zeros(n_cls, dtype=np.float64)

    h_flood = depth_msl + _TIDAL_SIN_WEIGHTS * eta_flood
    h_ebb = depth_msl + _TIDAL_SIN_WEIGHTS * eta_ebb

    exp_dstar = 0.333333 if strict_excel else (1.0 / 3.0)
    d_star = float(d50 * ((rho_s / rho_w - 1.0) * gravity / nu**2) ** exp_dstar)

    for idx in range(n_cls):
        v_wind, hs, tp, pct = wc[idx]
        prob = pct / 100.0
        u_f = _TIDAL_SIN_WEIGHTS * u_flood_peak + v_wind
        u_e = _TIDAL_SIN_WEIGHTS * u_ebb_peak + v_wind

        _, _, _, _, _, _, _, _, _, qbf, qsf = _core_van_rijn_cw(
            h_flood,
            u_f,
            np.full(6, hs, dtype=np.float64),
            np.full(6, tp, dtype=np.float64),
            d50=d50,
            d90=d90,
            p_mud=p_mud,
            ks=0.02,
            k_cr=k_cr,
            k_cal=k_cal,
            rho_w=rho_w,
            rho_s=rho_s,
            nu=nu,
            gravity=gravity,
            orbital_factor=0.4,
            bedload_coef=0.005,
            bedload_exp=2.4,
            suspended_coef=0.012,
            use_mud_reduction=False,
            force_magnitude_sign=strict_excel,
            strict_excel=strict_excel,
        )
        _, _, _, _, _, _, _, _, _, qbe, qse = _core_van_rijn_cw(
            h_ebb,
            u_e,
            np.full(6, hs, dtype=np.float64),
            np.full(6, tp, dtype=np.float64),
            d50=d50,
            d90=d90,
            p_mud=p_mud,
            ks=0.02,
            k_cr=k_cr,
            k_cal=k_cal,
            rho_w=rho_w,
            rho_s=rho_s,
            nu=nu,
            gravity=gravity,
            orbital_factor=0.4,
            bedload_coef=0.005,
            bedload_exp=2.4,
            suspended_coef=0.012,
            use_mud_reduction=False,
            force_magnitude_sign=strict_excel,
            strict_excel=strict_excel,
        )
        # En la hoja Excel, S41..46 usa +u/|u|*u y T47..52 usa -u/|u|*u.
        if strict_excel:
            q_flood = prob * (qbf + qsf)
            q_ebb = -prob * np.abs(qbe + qse)
            flood_cls[idx] = float(np.sum(q_flood))
            ebb_cls[idx] = float(np.sum(q_ebb))
        else:
            q_all = prob * np.concatenate([qbf + qsf, qbe + qse])
            flood_cls[idx] = float(np.sum(q_all[q_all > 0.0]))
            ebb_cls[idx] = float(np.sum(q_all[q_all < 0.0]))

    flood_sum = float(np.sum(flood_cls))
    ebb_sum = float(np.sum(ebb_cls))

    # Cada paso horario dura 3600 s y hay 730 mareas/ano; la hoja Excel multiplica
    # erroneamente por 6 adicional (730*6*3600) sobre una suma que ya incluye las 6 horas.
    sec_factor = tides_per_year * (6.0 * 3600.0 if strict_excel else 3600.0)
    flood_solid = sec_factor / rho_s * flood_sum
    ebb_solid = sec_factor / rho_s * ebb_sum
    flood_pores = flood_solid / porosity_factor
    ebb_pores = ebb_solid / porosity_factor

    return AnnualSandTransportResult(
        d_star=d_star,
        flood_kg_m_s_per_class=flood_cls,
        ebb_kg_m_s_per_class=ebb_cls,
        flood_kg_m_s_total=flood_sum,
        ebb_kg_m_s_total=ebb_sum,
        flood_vol_incl_pores_m3_yr=flood_pores,
        flood_vol_solid_m3_yr=flood_solid,
        ebb_vol_incl_pores_m3_yr=ebb_pores,
        ebb_vol_solid_m3_yr=ebb_solid,
        net_vol_incl_pores_m3_yr=flood_pores + ebb_pores,
        net_vol_solid_m3_yr=flood_solid + ebb_solid,
    )
