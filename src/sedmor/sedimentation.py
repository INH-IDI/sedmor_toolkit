"""Cálculos de sedimentación y morfodinámica en canales, embalses, puertos y tubos de corriente.

Implementa las planillas de cálculo de sedimentación del paquete original de
Leo C. van Rijn (2013):

1. ``SED PIT*.xlsx`` (5 planillas: caso base, ``new``, ``Gujarat India``,
   ``new Mar del plata``, ``trial trench Scheveningen 3``):
   sedimentación en canales de navegación y fosas de dragado
   (:func:`channel_sedimentation`).
2. ``Sedimentation due to parallel stratied flow in tidal shipping channel.xlsx``:
   perfil de circulación estratificada y depósito en canal mareal
   (reexporta :func:`stratified_density_flow_profile` y
   :func:`stratified_channel_sedimentation` desde :mod:`sedmor.currents`).
3. ``SEDRES.xlsx`` (3 hojas: ``Trap-VanRijn``, ``TRAP Eysink Vermaas``,
   ``TRAP Borland``): atrapamiento secuencial multi-tramo y consolidación en embalses
   (:func:`reservoir_sedimentation`).
4. ``SED HARBOUR-SAND.xlsx``: sedimentación de arenas/limos en dársenas y
   bocas portuarias (:func:`harbour_sand_sedimentation`).
5. ``SED HARBOURS- MUD new.xlsx`` y ``SED HARBOURS-MUD new swinemunde.xlsx``:
   sedimentación de fango por intercambio mareal, densidad y agua dulce en puertos
   (:func:`harbour_mud_sedimentation`).
6. ``SED TUBE*.xlsx`` (4 casos estacionarios/morfológicos y 5 casos mareales:
   ``Case4``, ``india``, ``lab1 havinga``, ``mud``, ``Delfzijl``,
   ``Mar del Plata``, ``Scheveningen``, ``trench noordzee``, ``westerschelde``):
   transporte en no equilibrio y evolución morfológica 1D en tubo de corriente
   (:func:`streamtube_sedimentation`, :func:`streamtube_tidal_sedimentation`).

Nota sobre ``strict_excel``
--------------------------
Por defecto (``strict_excel=False``) se emplea ``math.pi`` y se corrigen bugs o
inconsistencias detectados en las planillas originales (por ejemplo, omisión de
``days`` en tres columnas de ``SED HARBOUR-SAND``, desfase de índice en la
concentración de ``SED HARBOURS-MUD`` y suma de media marea en lugar de marea
completa, o ``(rho_s - g)`` sin paréntesis en ``Ucr,w`` para arena gruesa en
``SED TUBE``). Con ``strict_excel=True`` se reproducen celda a celda las
fórmulas cacheadas en Excel para validación exacta.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sedmor.constants import GRAVITY, KINEMATIC_VISCOSITY_20C, PI_STANDARD
from sedmor.currents import (
    StratifiedFlowResult,
    StratifiedSedimentationResult,
    stratified_channel_sedimentation,
    stratified_density_flow_profile,
)

__all__ = [
    "ChannelSedimentationResult",
    "HarbourMudSedimentationResult",
    "HarbourSandSedimentationResult",
    "ReservoirSectionInput",
    "ReservoirSectionResult",
    "ReservoirSedimentationResult",
    "StratifiedFlowResult",
    "StratifiedSedimentationResult",
    "StreamtubeSedimentationResult",
    "StreamtubeTidalResult",
    "channel_sedimentation",
    "harbour_mud_sedimentation",
    "harbour_sand_sedimentation",
    "reservoir_sedimentation",
    "stratified_channel_sedimentation",
    "stratified_density_flow_profile",
    "streamtube_sedimentation",
    "streamtube_tidal_sedimentation",
]


# ---------------------------------------------------------------------------
# Helpers físicos compartidos (sin duplicar lógica entre herramientas)
# ---------------------------------------------------------------------------


def _as_1d(values: ArrayLike, name: str) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"'{name}' debe ser un arreglo 1D no vacío.")
    return arr


def _d_star(
    d50: float,
    rho_s: float,
    rho_w: float,
    nu: float,
    *,
    strict_excel: bool,
    exp_strict: float = 0.33333,
) -> float:
    exp = exp_strict if strict_excel else (1.0 / 3.0)
    delta = (rho_s - rho_w) / rho_w
    return float(d50 * (delta * GRAVITY / (nu * nu)) ** exp)


def _wave_orbital(
    h: NDArray[np.float64],
    hs: NDArray[np.float64],
    tp: NDArray[np.float64],
    *,
    strict_excel: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    pi_w = 3.142 if strict_excel else PI_STANDARD
    kh = np.zeros_like(h, dtype=np.float64)
    ub = np.zeros_like(h, dtype=np.float64)
    mask = tp > 0.0
    if np.any(mask):
        x = 4.02 * h[mask] / (tp[mask] ** 2)
        kh_m = np.sqrt(x) * (1.0 + 0.166 * x + 0.031 * (x**2))
        kh[mask] = kh_m
        ub[mask] = pi_w * hs[mask] / (tp[mask] * np.sinh(kh_m))
    return kh, ub


def _van_rijn_critical_velocities(
    h: NDArray[np.float64] | float,
    d50: float,
    d90: float,
    tp: NDArray[np.float64] | float,
    p_fine_factor: float,
    rho_w: float,
    rho_s: float,
    *,
    slope_factor: float = 1.0,
    tube_coarse_bug: bool = False,
    strict_excel: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Velocidades críticas de corriente y oleaje según Van Rijn (2007)."""
    h_arr = np.asarray(h, dtype=np.float64)
    tp_arr = np.asarray(tp, dtype=np.float64)
    pf = p_fine_factor**1.5
    delta_g = ((rho_s - rho_w) / rho_w) * GRAVITY

    if d50 < 0.0005:
        u_cr_c = 0.19 * pf * (d50**0.1) * np.log10(12.0 * h_arr / (3.0 * d90))
        u_cr_w = (
            0.24
            * pf
            * (delta_g**0.66)
            * (d50**0.33)
            * np.power(np.maximum(tp_arr, 0.0), 0.33)
        )
    else:
        sf = slope_factor
        u_cr_c = sf * 8.5 * pf * (d50**0.6) * np.log10(12.0 * h_arr / (3.0 * d90))
        delta_w = (
            (rho_s - (rho_w / rho_w) * GRAVITY)
            if (strict_excel and tube_coarse_bug)
            else delta_g
        )
        u_cr_w = (
            0.95
            * pf
            * (delta_w**0.57)
            * (d50**0.43)
            * np.power(np.maximum(tp_arr, 0.0), 0.14)
        )
    u_cr_w = np.where(tp_arr > 0.0, u_cr_w, 0.0)
    return np.asarray(u_cr_c, dtype=np.float64), np.asarray(u_cr_w, dtype=np.float64)


def _combined_ucr(
    u: NDArray[np.float64],
    ub: NDArray[np.float64],
    u_cr_c: NDArray[np.float64],
    u_cr_w: NDArray[np.float64],
) -> NDArray[np.float64]:
    denom = u + ub
    w = np.where(denom > 0.0, u / np.where(denom > 0.0, denom, 1.0), 1.0)
    return w * u_cr_c + (1.0 - w) * u_cr_w


def _van_rijn_sand_rates_unit_width(
    h: NDArray[np.float64],
    u: NDArray[np.float64],
    u_eff: NDArray[np.float64],
    u_cr: NDArray[np.float64],
    d50: float,
    d_star: float,
    sand_factor: float,
    rho_w: float,
    rho_s: float,
    cal_tr: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Tasas de fondo y suspensión de arena por unidad de ancho (kg/s/m)."""
    delta_g = ((rho_s - rho_w) / rho_w) * GRAVITY
    excess = np.maximum(u_eff - u_cr, 0.0)
    qb = (
        sand_factor
        * rho_s
        * cal_tr
        * 0.015
        * h
        * u
        * ((d50 / h) ** 1.2)
        * (excess**1.5)
        / ((delta_g * d50) ** 0.75)
    )
    qs_sand = (
        sand_factor
        * rho_s
        * cal_tr
        * 0.012
        * u
        * h
        * (d50 / h)
        * (excess**2.4)
        / ((delta_g * d50) ** 1.2)
        * (d_star ** (-0.6))
    )
    return qb, qs_sand


def _consolidation_factor(total_days: float) -> float:
    years = total_days / 365.0
    if years <= 1.0:
        return 0.0
    return (years / (years - 1.0)) * math.log(years) - 1.0


def _van_rijn_bulk_density(
    mass_clay: float,
    mass_silt: float,
    mass_sand: float,
    total_mass: float,
    consolidation: float,
    mode: int,
    rho_const: float,
    *,
    zero_mass_rho: float = 0.0,
) -> float:
    if mode != 0:
        return float(rho_const)
    if total_mass <= 0.0:
        return float(zero_mass_rho)
    # En las planillas de Van Rijn, los términos de consolidación se suman sin
    # ponderar por fracción cuando el usuario elige consolidación variable (mode=0).
    return float(
        (mass_sand / total_mass) * 1.55
        + ((mass_clay / total_mass) * 0.415 + 0.11 * consolidation)
        + ((mass_silt / total_mass) * 1.12 + 0.04 * consolidation)
    )


# ---------------------------------------------------------------------------
# 1. SED PIT: Sedimentación en canales de navegación y fosas de dragado
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelSedimentationResult:
    """Resultado de sedimentación en canal de navegación o fosa (``SED PIT``)."""

    h_out: NDArray[np.float64]
    h_in: NDArray[np.float64]
    alpha_in_deg: NDArray[np.float64]
    u_in: NDArray[np.float64]
    qb_out: NDArray[np.float64]
    qs_clay_out: NDArray[np.float64]
    qs_silt_out: NDArray[np.float64]
    qs_sand_out: NDArray[np.float64]
    deposition_bed_tons: NDArray[np.float64]
    deposition_clay_tons: NDArray[np.float64]
    deposition_silt_tons: NDArray[np.float64]
    deposition_sand_tons: NDArray[np.float64]
    deposition_total_tons: NDArray[np.float64]
    bulk_density_t_m3: NDArray[np.float64]
    deposition_volume_m3: NDArray[np.float64]
    dh_m: NDArray[np.float64]
    cum_dh_m: NDArray[np.float64]
    total_hours: float
    total_days: float
    total_years: float
    consolidation_factor: float
    total_inflow_tons: float
    total_inflow_volume_m3: float
    total_deposition_tons: float
    total_deposition_m3: float
    total_layer_thickness_m: float
    pit_max_volume_m3: float
    trap_efficiency_tons: float
    trap_efficiency_volume: float
    sedimentation_per_year_m3: float
    filling_fraction: float


def channel_sedimentation(
    length: float,
    width: float,
    depth_in: float,
    depth_out: float,
    dur_h: ArrayLike,
    alpha0_deg: ArrayLike,
    b0: ArrayLike,
    dh_wl: ArrayLike,
    u0: ArrayLike,
    hs: ArrayLike,
    tp: ArrayLike,
    c_clay: ArrayLike,
    c_silt: ArrayLike,
    *,
    d50: float = 0.0002,
    d90: float = 0.0003,
    p_sand: float = 1.0,
    rho_w: float = 1020.0,
    rho_s: float = 2650.0,
    nu: float = KINEMATIC_VISCOSITY_20C,
    ws_clay: float = 0.0001,
    ws_silt: float = 0.001,
    ws_sand: float = 0.02,
    ks_out: float = 0.02,
    ks_in: float = 0.02,
    cal_tr: float = 1.0,
    cal_ucr: float = 1.0,
    bulk_density_mode: int = 1,
    bulk_density_const: float = 1.6,
    trap_method: int = 1,
    coef_van_rijn: float = 0.25,
    coef_eysink: float = 0.06,
    min_depth: float = 0.0,
    update_depth: bool = True,
    strict_excel: bool = False,
) -> ChannelSedimentationResult:
    """Calcula el depósito de arcilla, limo y arena en canales o fosas (``SED PIT``).

    Parameters
    ----------
    length, width : float
        Longitud paralela y ancho transversal del tramo o fosa (m).
    depth_in, depth_out : float
        Profundidades de referencia bajo nivel medio dentro y fuera de la fosa (m).
    dur_h, alpha0_deg, b0, dh_wl, u0, hs, tp, c_clay, c_silt : array_like
        Series de condiciones hidrodinámicas y concentraciones por periodo:
        duración (h), ángulo de incidencia (°), ancho del tubo incidente ``b0`` (m),
        sobrelevación del nivel de agua (m), velocidad incidente ``u0`` (m/s),
        altura significativa ``Hs`` (m), periodo pico ``Tp`` (s) y concentraciones
        de arcilla y limo (kg/m³).
    trap_method : {1, 2}, default=1
        ``1`` para Van Rijn (usa factor ``(h1 - h0) / h1^2`` y ``coef_van_rijn``);
        ``2`` para Eysink-Vermaas (usa ``1 / h1`` y ``coef_eysink``).
    min_depth : float, default=0.0
        Cota mínima de profundidad aplicada cuando ``min_depth > 0`` (0.1 en Gujarat).
    update_depth : bool, default=True
        Si ``True``, actualiza la profundidad interna restando el relleno acumulado
        del paso anterior (``Gujarat India`` omite esta actualización en Excel).
    strict_excel : bool, default=False
        Si ``True``, usa ``3.142`` en ``sin``, ``3.1416`` en ``atan`` y ``0.33333``
        en ``D*`` igual que las celdas de ``SED PIT*.xlsx``.
    """
    dur = _as_1d(dur_h, "dur_h")
    n = dur.size
    a0 = _as_1d(alpha0_deg, "alpha0_deg")
    b0_arr = _as_1d(b0, "b0")
    dh_arr = _as_1d(dh_wl, "dh_wl")
    u0_arr = _as_1d(u0, "u0")
    hs_arr = _as_1d(hs, "hs")
    tp_arr = _as_1d(tp, "tp")
    cc_arr = _as_1d(c_clay, "c_clay")
    cs_arr = _as_1d(c_silt, "c_silt")
    for arr, nm in (
        (a0, "alpha0_deg"),
        (b0_arr, "b0"),
        (dh_arr, "dh_wl"),
        (u0_arr, "u0"),
        (hs_arr, "hs"),
        (tp_arr, "tp"),
        (cc_arr, "c_clay"),
        (cs_arr, "c_silt"),
    ):
        if arr.size != n:
            raise ValueError(f"'{nm}' debe tener longitud {n}, recibido {arr.size}.")

    pi_trig = 3.142 if strict_excel else PI_STANDARD
    pi_atan = 3.1416 if strict_excel else PI_STANDARD
    d_star = _d_star(d50, rho_s, rho_w, nu, strict_excel=strict_excel, exp_strict=0.33333)

    h0 = depth_out + dh_arr
    if min_depth > 0.0:
        h0 = np.maximum(h0, min_depth)

    _, ub = _wave_orbital(h0, hs_arr, tp_arr, strict_excel=strict_excel)
    u_eff = u0_arr + cal_ucr * 0.4 * ub
    u_cr_c, u_cr_w = _van_rijn_critical_velocities(
        h0,
        d50,
        d90,
        tp_arr,
        2.0 - p_sand,
        rho_w,
        rho_s,
        strict_excel=strict_excel,
    )
    u_cr = _combined_ucr(u0_arr, ub, u_cr_c, u_cr_w)
    qb0, qs_sand0 = _van_rijn_sand_rates_unit_width(
        h0, u0_arr, u_eff, u_cr, d50, d_star, p_sand, rho_w, rho_s, cal_tr
    )
    qs_clay0 = h0 * u0_arr * cc_arr
    qs_silt0 = h0 * u0_arr * cs_arr

    sin_a0 = np.sin(a0 / 180.0 * pi_trig)
    inflow_scale = dur * 3600.0 / 1000.0 * length * sin_a0
    in_bed_t = qb0 * inflow_scale
    in_clay_t = qs_clay0 * inflow_scale
    in_silt_t = qs_silt0 * inflow_scale
    in_sand_t = qs_sand0 * inflow_scale

    total_hours = float(np.sum(dur))
    total_days = total_hours / 24.0
    total_years = total_days / 365.0
    cons = _consolidation_factor(total_days)

    h1 = np.empty(n, dtype=np.float64)
    alpha1 = np.empty(n, dtype=np.float64)
    u1 = np.empty(n, dtype=np.float64)
    ds_bed = np.empty(n, dtype=np.float64)
    ds_clay = np.empty(n, dtype=np.float64)
    ds_silt = np.empty(n, dtype=np.float64)
    ds_sand = np.empty(n, dtype=np.float64)
    ds_tot = np.empty(n, dtype=np.float64)
    rho_bulk = np.empty(n, dtype=np.float64)
    vol_m3 = np.empty(n, dtype=np.float64)
    dh_step = np.empty(n, dtype=np.float64)
    cum_dh = np.empty(n, dtype=np.float64)

    prev_cum = 0.0
    area = length * width
    for i in range(n):
        h1_i = depth_in + dh_arr[i] - (prev_cum if update_depth else 0.0)
        if min_depth > 0.0:
            h1_i = max(h1_i, min_depth)
        h1[i] = h1_i
        c1_i = 18.0 * math.log10(12.0 * h1_i / ks_in)
        a1_i = (
            90.0
            if a0[i] >= 89.0
            else math.atan((h0[i] / h1_i) * math.tan(a0[i] / 180.0 * pi_atan)) * 180.0 / pi_atan
        )
        alpha1[i] = a1_i
        sin_a1 = math.sin(a1_i / 180.0 * pi_trig)
        u1_i = min(u0_arr[i], u0_arr[i] * h0[i] * sin_a0[i] / (h1_i * sin_a1))
        u1[i] = u1_i
        b1_i = min(b0_arr[i], b0_arr[i] * h0[i] * u0_arr[i] / (h1_i * u1_i)) if u1_i > 0.0 else b0_arr[i]
        l_settle = width / sin_a1
        u_star1 = (GRAVITY**0.5) / c1_i * u1_i

        vel_ratio = (u1_i / u0_arr[i]) ** 3 if u0_arr[i] > 0.0 else 0.0
        q1_eq = [min(q0, vel_ratio * q0) for q0 in (qb0[i], qs_clay0[i], qs_silt0[i], qs_sand0[i])]

        e_frac = [1.0]
        for ws in (ws_clay, ws_silt, ws_sand):
            p_dim = (ws / u_star1) * (1.0 + 2.0 * ws / u_star1) if u_star1 > 0.0 else 0.0
            if trap_method == 1:
                e_val = 1.0 - math.exp(-coef_van_rijn * p_dim * l_settle * (h1_i - h0[i]) / (h1_i**2))
            else:
                e_val = 1.0 - math.exp(-coef_eysink * p_dim * l_settle / h1_i)
            e_frac.append(e_val)

        dep_scale = dur[i] * 3600.0 * length * sin_a1 / 1000.0
        q_out_list = (qb0[i], qs_clay0[i], qs_silt0[i], qs_sand0[i])
        ds_k = [
            e_frac[k] * (b0_arr[i] / b1_i * q_out_list[k] - q1_eq[k]) * dep_scale
            for k in range(4)
        ]
        ds_bed[i], ds_clay[i], ds_silt[i], ds_sand[i] = ds_k
        dtot = sum(ds_k)
        ds_tot[i] = dtot

        rb = _van_rijn_bulk_density(
            ds_clay[i],
            ds_silt[i],
            ds_bed[i] + ds_sand[i],
            dtot,
            cons,
            bulk_density_mode,
            bulk_density_const,
            zero_mass_rho=1.55,
        )
        rho_bulk[i] = rb
        vol_i = dtot / rb if dtot > 0.0 else 0.0
        vol_m3[i] = vol_i
        max_vol_i = area * (h1_i - h0[i]) if h1_i > h0[i] else 0.0
        dh_i = (vol_i / area) if vol_i < max_vol_i else (max_vol_i / area)
        dh_step[i] = dh_i
        prev_cum += dh_i
        cum_dh[i] = prev_cum

    in_tot_t = float(np.sum(in_bed_t) + np.sum(in_clay_t) + np.sum(in_silt_t) + np.sum(in_sand_t))
    rho_in = _van_rijn_bulk_density(
        float(np.sum(in_clay_t)),
        float(np.sum(in_silt_t)),
        float(np.sum(in_bed_t) + np.sum(in_sand_t)),
        in_tot_t,
        cons,
        bulk_density_mode,
        bulk_density_const,
        zero_mass_rho=bulk_density_const,
    )
    in_tot_m3 = in_tot_t / rho_in if rho_in > 0.0 else 0.0
    dep_tot_t = float(np.sum(ds_tot))
    dep_tot_m3 = float(np.sum(vol_m3))
    pit_max_vol = float(area * max(h1[0] - h0[0], 0.0))

    return ChannelSedimentationResult(
        h_out=h0,
        h_in=h1,
        alpha_in_deg=alpha1,
        u_in=u1,
        qb_out=qb0,
        qs_clay_out=qs_clay0,
        qs_silt_out=qs_silt0,
        qs_sand_out=qs_sand0,
        deposition_bed_tons=ds_bed,
        deposition_clay_tons=ds_clay,
        deposition_silt_tons=ds_silt,
        deposition_sand_tons=ds_sand,
        deposition_total_tons=ds_tot,
        bulk_density_t_m3=rho_bulk,
        deposition_volume_m3=vol_m3,
        dh_m=dh_step,
        cum_dh_m=cum_dh,
        total_hours=total_hours,
        total_days=total_days,
        total_years=total_years,
        consolidation_factor=cons,
        total_inflow_tons=in_tot_t,
        total_inflow_volume_m3=in_tot_m3,
        total_deposition_tons=dep_tot_t,
        total_deposition_m3=dep_tot_m3,
        total_layer_thickness_m=float(np.sum(dh_step)),
        pit_max_volume_m3=pit_max_vol,
        trap_efficiency_tons=dep_tot_t / in_tot_t if in_tot_t > 0.0 else 0.0,
        trap_efficiency_volume=dep_tot_m3 / in_tot_m3 if in_tot_m3 > 0.0 else 0.0,
        sedimentation_per_year_m3=dep_tot_m3 / total_years if total_years > 0.0 else 0.0,
        filling_fraction=dep_tot_m3 / pit_max_vol if pit_max_vol > 0.0 else 0.0,
    )


# ---------------------------------------------------------------------------
# 2. SEDRES: Sedimentación multi-tramo en embalses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReservoirSectionInput:
    """Geometría de un tramo del embalse en ``SEDRES.xlsx``."""

    length_m: float
    width_m: float
    depth_drop_m: float


@dataclass(frozen=True)
class ReservoirSectionResult:
    """Resultados por periodo y totales en un tramo de embalse."""

    h_m: NDArray[np.float64]
    u_m_s: NDArray[np.float64]
    ds_tons: NDArray[np.float64]
    ds_volume_m3: NDArray[np.float64]
    dh_m: NDArray[np.float64]
    total_dh_m: float
    uncapped_volume_m3: float
    capped_volume_m3: float


@dataclass(frozen=True)
class ReservoirSedimentationResult:
    """Resultado completo de sedimentación multi-tramo en embalse (``SEDRES``)."""

    sections: tuple[ReservoirSectionResult, ...]
    outflow_clay_kg_s: NDArray[np.float64]
    outflow_silt_kg_s: NDArray[np.float64]
    outflow_sand_kg_s: NDArray[np.float64]
    total_days: float
    total_years: float
    consolidation_factor: float
    inflow_tons_total: float
    inflow_volume_m3: float
    total_deposition_capped_m3: float
    total_deposition_uncapped_m3: float
    overall_trapping_efficiency: float


def reservoir_sedimentation(
    sections: Sequence[ReservoirSectionInput],
    days: ArrayLike,
    discharge: ArrayLike,
    h0: ArrayLike,
    c_clay: ArrayLike,
    c_silt: ArrayLike,
    c_sand: ArrayLike,
    *,
    river_width: float = 70.0,
    ws_clay: float = 0.0001,
    ws_silt: float = 0.001,
    ws_sand: float = 0.015,
    chezy: float = 60.0,
    coef_trap: float = 0.25,
    method: str = "van_rijn",
    corr_eq: float = 0.0,
    bulk_density_mode: int = 1,
    bulk_density_const: float = 0.8,
    strict_excel: bool = False,
) -> ReservoirSedimentationResult:
    """Calcula el atrapamiento secuencial multi-tramo en un embalse (``SEDRES.xlsx``).

    Parameters
    ----------
    sections : sequence of ReservoirSectionInput
        Tramos consecutivos aguas abajo (A, B, C, D, E en ``SEDRES.xlsx``).
    days, discharge, h0, c_clay, c_silt, c_sand : array_like
        Duración (días), caudal fluvial (m³/s), calado en la entrada ``x=0`` (m)
        y concentraciones entrantes por periodo (kg/m³).
    method : {"van_rijn", "eysink_vermaas", "borland"}, default="van_rijn"
        Fórmula de eficiencia de atrapamiento correspondiente a las tres hojas de
        ``SEDRES.xlsx``.
    strict_excel : bool, default=False
        Si ``True``, reproduce la referencia de celda ``E45 = days / (h0 * W0)``
        (en lugar de ``Q / (h0 * W0)``) presente en las columnas de entrada de
        ``SEDRES.xlsx``.
    """
    method_norm = method.lower().strip()
    if method_norm not in {"van_rijn", "eysink_vermaas", "borland"}:
        raise ValueError(f"Método desconocido '{method}'. Use 'van_rijn', 'eysink_vermaas' o 'borland'.")

    days_arr = _as_1d(days, "days")
    n = days_arr.size
    q_arr = _as_1d(discharge, "discharge")
    h0_arr = _as_1d(h0, "h0")
    cc_arr = _as_1d(c_clay, "c_clay")
    cs_arr = _as_1d(c_silt, "c_silt")
    csa_arr = _as_1d(c_sand, "c_sand")
    for arr, nm in (
        (q_arr, "discharge"),
        (h0_arr, "h0"),
        (cc_arr, "c_clay"),
        (cs_arr, "c_silt"),
        (csa_arr, "c_sand"),
    ):
        if arr.size != n:
            raise ValueError(f"'{nm}' debe tener longitud {n}.")

    total_days = float(np.sum(days_arr))
    total_years = total_days / 365.0
    cons = _consolidation_factor(total_days)

    u0 = (days_arr if strict_excel else q_arr) / (h0_arr * river_width)
    qs0_fractions = (q_arr * cc_arr, q_arr * cs_arr, q_arr * csa_arr)
    inflow_tons_k = [float(np.sum(qf * days_arr * 86400.0 / 1000.0)) for qf in qs0_fractions]
    inflow_tons_total = sum(inflow_tons_k)
    rho_in = _van_rijn_bulk_density(
        inflow_tons_k[0],
        inflow_tons_k[1],
        inflow_tons_k[2],
        inflow_tons_total,
        cons,
        bulk_density_mode,
        bulk_density_const,
    )
    inflow_vol_m3 = inflow_tons_total / rho_in if rho_in > 0.0 else 0.0

    qs_cur = [qf.copy() for qf in qs0_fractions]
    ws_tuple = (ws_clay, ws_silt, ws_sand)
    section_results: list[ReservoirSectionResult] = []

    for sec in sections:
        area = sec.length_m * sec.width_m
        h_s = np.empty(n, dtype=np.float64)
        u_s = np.empty(n, dtype=np.float64)
        ds_tot = np.empty(n, dtype=np.float64)
        vol_s = np.empty(n, dtype=np.float64)
        dh_s = np.empty(n, dtype=np.float64)
        qs_next = [np.empty(n, dtype=np.float64) for _ in range(3)]

        for t in range(n):
            h_t = (sec.depth_drop_m + h0_arr[t]) if t == 0 else max(h0_arr[t], h_s[t - 1] - dh_s[t - 1])
            h_s[t] = h_t
            u_t = q_arr[t] / (h_t * sec.width_m)
            u_s[t] = u_t
            u_star = (GRAVITY**0.5) / chezy * u_t

            vel3 = ((u_t / u0[t]) ** 3) * corr_eq if (u0[t] > 0.0 and corr_eq != 0.0) else 0.0
            ds_k = [0.0, 0.0, 0.0]
            for k, ws in enumerate(ws_tuple):
                if h_t == h0_arr[t]:
                    e_k = 0.0
                elif method_norm == "van_rijn":
                    e_k = 1.0 - math.exp(
                        -(sec.length_m / h_t)
                        * ((h_t - h0_arr[t]) / h_t)
                        * coef_trap
                        * (ws / u_star)
                        * (1.0 + 2.0 * ws / u_star)
                    )
                elif method_norm == "eysink_vermaas":
                    e_k = 1.0 - math.exp(
                        -(sec.length_m / h_t) * coef_trap * (ws / u_star) * (1.0 + 2.0 * ws / u_star)
                    )
                else:
                    e_k = 1.0 - math.exp(-(sec.length_m / h_t) * coef_trap * (ws / u_t))

                q_in = qs_cur[k][t]
                q_eq = vel3 * qs0_fractions[k][t]
                if q_in > q_eq:
                    ds_k[k] = e_k * (q_in - q_eq) * days_arr[t] * 86400.0 / 1000.0
                    qs_next[k][t] = q_in - e_k * (q_in - q_eq)
                else:
                    ds_k[k] = 0.0
                    qs_next[k][t] = min(q_in, q_eq)

            dtot = sum(ds_k)
            ds_tot[t] = dtot
            rb = _van_rijn_bulk_density(
                ds_k[0],
                ds_k[1],
                ds_k[2],
                dtot,
                cons,
                bulk_density_mode,
                bulk_density_const,
            )
            vol_t = dtot / rb if dtot > 0.0 else 0.0
            vol_s[t] = vol_t
            max_vol = area * (h_t - h0_arr[t])
            dh_s[t] = (vol_t / area) if vol_t < max_vol else (max_vol / area)

        qs_cur = qs_next
        tot_dh = float(np.sum(dh_s))
        section_results.append(
            ReservoirSectionResult(
                h_m=h_s,
                u_m_s=u_s,
                ds_tons=ds_tot,
                ds_volume_m3=vol_s,
                dh_m=dh_s,
                total_dh_m=tot_dh,
                uncapped_volume_m3=float(np.sum(vol_s)),
                capped_volume_m3=tot_dh * area,
            )
        )

    tot_capped = sum(s.capped_volume_m3 for s in section_results)
    tot_uncapped = sum(s.uncapped_volume_m3 for s in section_results)
    return ReservoirSedimentationResult(
        sections=tuple(section_results),
        outflow_clay_kg_s=qs_cur[0],
        outflow_silt_kg_s=qs_cur[1],
        outflow_sand_kg_s=qs_cur[2],
        total_days=total_days,
        total_years=total_years,
        consolidation_factor=cons,
        inflow_tons_total=inflow_tons_total,
        inflow_volume_m3=inflow_vol_m3,
        total_deposition_capped_m3=tot_capped,
        total_deposition_uncapped_m3=tot_uncapped,
        overall_trapping_efficiency=tot_capped / inflow_vol_m3 if inflow_vol_m3 > 0.0 else 0.0,
    )


# ---------------------------------------------------------------------------
# 3. SED HARBOUR-SAND: Sedimentación de arenas/limos en puertos
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HarbourSandSedimentationResult:
    """Resultado de sedimentación de arenas/limos en dársenas (``SED HARBOUR-SAND``)."""

    h_m: NDArray[np.float64]
    u_eff_m_s: NDArray[np.float64]
    u_cr_m_s: NDArray[np.float64]
    inflow_bed_tons: NDArray[np.float64]
    inflow_clay_tons: NDArray[np.float64]
    inflow_silt_tons: NDArray[np.float64]
    inflow_sand_tons: NDArray[np.float64]
    deposit_bed_m3: NDArray[np.float64]
    deposit_clay_m3: NDArray[np.float64]
    deposit_silt_m3: NDArray[np.float64]
    deposit_sand_m3: NDArray[np.float64]
    total_days: float
    total_inflow_tons: float
    total_deposition_m3: float


def harbour_sand_sedimentation(
    surge: ArrayLike,
    u: ArrayLike,
    hs: ArrayLike,
    tp: ArrayLike,
    days: ArrayLike,
    c_clay: ArrayLike,
    c_silt: ArrayLike,
    *,
    q_in: float = 0.0,
    entrance_width: float = 300.0,
    depth_mean: float = 10.0,
    basin_area: float = 1_000_000.0,
    tidal_range: float = 0.2,
    tidal_period_h: float = 12.0,
    d50: float = 0.0002,
    d90: float = 0.0003,
    p_sand: float = 1.0,
    rho_w: float = 1020.0,
    rho_s: float = 2650.0,
    nu: float = KINEMATIC_VISCOSITY_20C,
    ks: float = 0.02,
    cal_tr: float = 1.0,
    cal_ucr: float = 1.0,
    rho_bulk_dry: float = 1.6,
    trap_efficiency: float = 1.0,
    strict_excel: bool = False,
) -> HarbourSandSedimentationResult:
    """Calcula el ingreso y depósito de arena, limo y arcilla en puertos (``SED HARBOUR-SAND``).

    Parameters
    ----------
    strict_excel : bool, default=False
        Si ``True``, reproduce la omisión de ``* days`` en las columnas de entrada
        de arcilla, limo y arena en suspensión (``Z, AA, AB`` de ``SED HARBOUR-SAND.xlsx``)
        así como ``3.142`` y ``0.33333``.
    """
    surge_arr = _as_1d(surge, "surge")
    n = surge_arr.size
    u_arr = _as_1d(u, "u")
    hs_arr = _as_1d(hs, "hs")
    tp_arr = _as_1d(tp, "tp")
    days_arr = _as_1d(days, "days")
    cc_arr = _as_1d(c_clay, "c_clay")
    cs_arr = _as_1d(c_silt, "c_silt")
    for arr, nm in (
        (u_arr, "u"),
        (hs_arr, "hs"),
        (tp_arr, "tp"),
        (days_arr, "days"),
        (cc_arr, "c_clay"),
        (cs_arr, "c_silt"),
    ):
        if arr.size != n:
            raise ValueError(f"'{nm}' debe tener longitud {n}.")

    d_star = _d_star(d50, rho_s, rho_w, nu, strict_excel=strict_excel, exp_strict=0.33333)
    h = depth_mean + surge_arr
    _, ub = _wave_orbital(h, hs_arr, tp_arr, strict_excel=strict_excel)
    u_tide = (0.75 * basin_area * tidal_range) / (entrance_width * h * tidal_period_h * 0.5 * 3600.0)
    u_riv = q_in / (h * entrance_width)
    u_eff = u_arr + u_tide + u_riv + cal_ucr * 0.4 * ub

    u_cr_c, u_cr_w = _van_rijn_critical_velocities(
        h, d50, d90, tp_arr, 2.0 - p_sand, rho_w, rho_s, strict_excel=strict_excel
    )
    u_cr = _combined_ucr(u_arr, ub, u_cr_c, u_cr_w)
    qb, qs_sand = _van_rijn_sand_rates_unit_width(
        h, u_arr, u_eff, u_cr, d50, d_star, p_sand, rho_w, rho_s, cal_tr
    )
    qs_clay = h * u_arr * cc_arr
    qs_silt = h * u_arr * cs_arr

    half_b = 0.5 * entrance_width
    sec_per_day = 86400.0 / 1000.0
    in_bed = qb * half_b * days_arr * sec_per_day
    days_susp = np.ones_like(days_arr) if strict_excel else days_arr
    in_clay = qs_clay * half_b * days_susp * sec_per_day
    in_silt = qs_silt * half_b * days_susp * sec_per_day
    in_sand = qs_sand * half_b * days_susp * sec_per_day

    fac_m3 = trap_efficiency / rho_bulk_dry
    dep_bed = fac_m3 * in_bed
    dep_clay = fac_m3 * in_clay
    dep_silt = fac_m3 * in_silt
    dep_sand = fac_m3 * in_sand

    return HarbourSandSedimentationResult(
        h_m=h,
        u_eff_m_s=u_eff,
        u_cr_m_s=u_cr,
        inflow_bed_tons=in_bed,
        inflow_clay_tons=in_clay,
        inflow_silt_tons=in_silt,
        inflow_sand_tons=in_sand,
        deposit_bed_m3=dep_bed,
        deposit_clay_m3=dep_clay,
        deposit_silt_m3=dep_silt,
        deposit_sand_m3=dep_sand,
        total_days=float(np.sum(days_arr)),
        total_inflow_tons=float(np.sum(in_bed + in_clay + in_silt + in_sand)),
        total_deposition_m3=float(np.sum(dep_bed + dep_clay + dep_silt + dep_sand)),
    )


# ---------------------------------------------------------------------------
# 4. SED HARBOURS-MUD: Sedimentación de fango en puertos mareales
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HarbourMudSedimentationResult:
    """Resultado de sedimentación de fango en puertos (``SED HARBOURS-MUD``)."""

    t_s: NDArray[np.float64]
    water_level_m: NDArray[np.float64]
    h_m: NDArray[np.float64]
    u_out_m_s: NDArray[np.float64]
    c_out_kg_m3: NDArray[np.float64]
    c_basin_kg_m3: NDArray[np.float64]
    dc_basin_kg_m3: NDArray[np.float64]
    ws_eff_m_s: float
    deposition_step_kg: NDArray[np.float64]
    net_exchange_step_kg: NDArray[np.float64]
    deposition_last_tide_kg: float
    deposition_per_year_kg: float
    deposition_per_year_m3: float
    layer_thickness_per_year_m: float
    exchange_total_m3: float
    exchange_per_year_m3: float


def harbour_mud_sedimentation(
    *,
    dt_s: float = 300.0,
    depth_msl: float = 10.7,
    entrance_width: float = 220.0,
    basin_area: float = 2_200_000.0,
    u_peak_out: float = 0.8,
    tidal_amplitude: float = 1.5,
    c_basin_init: float = 0.01,
    c_out_mean: float = 0.15,
    c_out_amp: float = 0.1,
    delta_rho_max: float = 3.0,
    rho_out: float = 1025.0,
    ws: float = 0.0001,
    rho_bulk_dry: float = 350.0,
    tidal_period_s: float = 43200.0,
    f_exchange_flood: float = 0.025,
    f_exchange_ebb: float = 0.0,
    hs_rel_harbour: float = 0.0,
    q_freshwater: float = 20.0,
    mixing_factor: float = 2.0,
    freshwater_time_fraction: float = 0.6,
    n_tides_sim: int = 4,
    n_tides_year: float = 710.0,
    strict_excel: bool = False,
) -> HarbourMudSedimentationResult:
    """Simula la sedimentación de fango en una dársena portuaria (``SED HARBOURS-MUD``).

    Integra paso a paso el balance de sedimentos finos en la dársena considerando
    prisma mareal, intercambio horizontal, corriente de densidad salina y pluma
    de agua dulce.

    Parameters
    ----------
    strict_excel : bool, default=False
        Si ``True``, reproduce los cuatro rasgos peculiares de
        ``SED HARBOURS- MUD new.xlsx`` y ``swinemunde.xlsx``:
        ``3.141`` como π, velocidad de caída sin amortiguación en ``t=0``,
        fórmula ``P{r} = P{r-2} + Q{r-2}`` desde la primera bajamar (fila 104),
        suma de depósito anual únicamente sobre la última media marea (72 pasos)
        en lugar de la última marea completa (144 pasos), y el cambio de fase en
        la columna ``V`` de intercambio neto.
    """
    steps_per_tide = int(round(tidal_period_s / dt_s))
    n_steps = n_tides_sim * steps_per_tide + 1
    half_steps = steps_per_tide // 2
    quarter_steps = steps_per_tide // 4

    pi_m = 3.141 if strict_excel else PI_STANDARD
    omega = 2.0 * pi_m / tidal_period_s

    if hs_rel_harbour > 0.05:
        ws_eff = ((-2.2 * hs_rel_harbour + 1.11) * ws) if hs_rel_harbour < 0.5 else (0.001 * ws)
    else:
        ws_eff = ws

    t_arr = np.arange(n_steps, dtype=np.float64) * dt_s
    eta = -tidal_amplitude * np.cos(omega * t_arr)
    h = depth_msl + eta
    u_out = -u_peak_out * np.cos(omega * t_arr)
    c_out = c_out_mean - c_out_amp * np.cos(omega * t_arr)
    d_rho = -delta_rho_max * np.cos(omega * t_arr)
    dndt = tidal_amplitude * omega * np.sin(omega * t_arr)

    q_tide = basin_area * dndt
    q_dens = 0.15 * entrance_width * h * np.sqrt((np.abs(d_rho) / rho_out) * GRAVITY * h)
    q_riv = freshwater_time_fraction * (mixing_factor - 1.0) * q_freshwater

    c_basin = np.empty(n_steps, dtype=np.float64)
    dc_basin = np.empty(n_steps, dtype=np.float64)
    dep_step = np.empty(n_steps, dtype=np.float64)
    exch_step = np.empty(n_steps, dtype=np.float64)
    q_hor_arr = np.empty(n_steps, dtype=np.float64)

    for k in range(n_steps):
        if k == 0:
            c_k = c_basin_init
        elif strict_excel and k >= half_steps + 1:
            c_k = c_basin[k - 2] + dc_basin[k - 2]
        else:
            c_k = c_basin[k - 1] + dc_basin[k - 1]
        c_basin[k] = c_k

        is_ebb = (
            (k >= 1 and ((k - 1) % steps_per_tide) >= half_steps)
            if strict_excel
            else (dndt[k] < 0.0)
        )
        if is_ebb:
            qh = f_exchange_ebb * abs(u_out[k]) * entrance_width * h[k]
            dc_k = (
                (
                    (-abs(q_tide[k]) - abs(q_dens[k]) - abs(qh) - abs(q_riv)) * c_k
                    + (abs(q_dens[k]) + abs(qh) + abs(q_riv)) * c_out[k]
                )
                / (basin_area * h[k])
                - ws_eff * c_k / h[k]
                - dndt[k] * c_k / h[k]
            ) * dt_s
        else:
            qh = f_exchange_flood * abs(u_out[k]) * entrance_width * h[k]
            if strict_excel and k >= steps_per_tide + 1:
                # Desde R176 Excel omite ABS() alrededor de los flujos de llenante.
                in_q = q_tide[k] + qh + q_dens[k] + q_riv
                out_q = qh + q_dens[k] + q_riv
            else:
                in_q = abs(q_tide[k] + qh + q_dens[k] + q_riv)
                out_q = abs(qh + q_dens[k] + q_riv)
            dc_k = (
                in_q * c_out[k] / (basin_area * h[k])
                - out_q * c_k / (basin_area * h[k])
                - ws_eff * c_k / h[k]
                - dndt[k] * c_k / h[k]
            ) * dt_s
        q_hor_arr[k] = qh
        dc_basin[k] = dc_k

        ws_step = ws if (strict_excel and k == 0) else ws_eff
        dep_step[k] = basin_area * ws_step * c_k * dt_s

    for k in range(n_steps):
        if strict_excel:
            # En Excel: k=0..36 y mod 109..143 usan fila k con fórmula de vaciante;
            # mod 37..108 usan fila k-1 con fórmula de llenante.
            mod = k % steps_per_tide
            if mod <= quarter_steps or mod > 3 * quarter_steps:
                use_ebb_form = True
                idx = k
            else:
                use_ebb_form = False
                idx = k - 1
            if use_ebb_form:
                exch_step[k] = (
                    (abs(q_tide[idx]) + q_hor_arr[idx] + q_dens[idx]) * c_basin[idx]
                    - (q_hor_arr[idx] + q_dens[idx]) * c_out[idx]
                ) * dt_s
            else:
                exch_step[k] = (
                    (q_hor_arr[idx] + q_dens[idx]) * c_basin[idx]
                    - (abs(q_tide[idx]) + q_hor_arr[idx] + q_dens[idx]) * c_out[idx]
                ) * dt_s
        else:
            q_mix = q_hor_arr[k] + q_dens[k] + q_riv
            if dndt[k] >= 0.0:
                exch_step[k] = (q_mix * c_basin[k] - (abs(q_tide[k]) + q_mix) * c_out[k]) * dt_s
            else:
                exch_step[k] = ((abs(q_tide[k]) + q_mix) * c_basin[k] - q_mix * c_out[k]) * dt_s

    # R536:R607 (últimos half_steps) en Excel vs últimos steps_per_tide (1 marea completa)
    start_idx = (n_steps - half_steps) if strict_excel else (n_steps - steps_per_tide)
    dep_last_tide = float(np.sum(dep_step[start_idx:]))
    dep_year_kg = dep_last_tide * n_tides_year
    dep_year_m3 = dep_year_kg / rho_bulk_dry
    layer_year_m = dep_year_m3 / basin_area

    exch_tot_m3 = float(np.sum(exch_step)) / rho_bulk_dry
    exch_year_m3 = exch_tot_m3 / n_tides_sim * n_tides_year

    return HarbourMudSedimentationResult(
        t_s=t_arr,
        water_level_m=eta,
        h_m=h,
        u_out_m_s=u_out,
        c_out_kg_m3=c_out,
        c_basin_kg_m3=c_basin,
        dc_basin_kg_m3=dc_basin,
        ws_eff_m_s=ws_eff,
        deposition_step_kg=dep_step,
        net_exchange_step_kg=exch_step,
        deposition_last_tide_kg=dep_last_tide,
        deposition_per_year_kg=dep_year_kg,
        deposition_per_year_m3=dep_year_m3,
        layer_thickness_per_year_m=layer_year_m,
        exchange_total_m3=exch_tot_m3,
        exchange_per_year_m3=exch_year_m3,
    )


# ---------------------------------------------------------------------------
# 5. SED TUBE: Modelo morfológico 1D en tubo de corriente (estacionario y mareal)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StreamtubeSedimentationResult:
    """Resultado de evolución morfológica 1D en tubo de corriente (``SED TUBE``)."""

    x_m: NDArray[np.float64]
    width_m: NDArray[np.float64]
    h_initial_m: NDArray[np.float64]
    h_final_m: NDArray[np.float64]
    u_m_s: NDArray[np.float64]
    qbed_kg_s: NDArray[np.float64]
    qeq_sus_kg_s: NDArray[np.float64]
    qsus_kg_s: NDArray[np.float64]
    qtot_kg_s: NDArray[np.float64]
    dh_last_step_m: NDArray[np.float64]


@dataclass(frozen=True)
class StreamtubeTidalResult:
    """Resultado morfológico 1D en tubo de corriente bajo marea bidireccional."""

    x_m: NDArray[np.float64]
    width_m: NDArray[np.float64]
    h_msl_initial_m: NDArray[np.float64]
    h_msl_final_m: NDArray[np.float64]
    bed_level_initial_m: NDArray[np.float64]
    bed_level_final_m: NDArray[np.float64]
    dh_total_m: NDArray[np.float64]


def _streamtube_step(
    x: NDArray[np.float64],
    b: NDArray[np.float64],
    h_tot: NDArray[np.float64],
    hs: NDArray[np.float64],
    tp: NDArray[np.float64],
    *,
    discharge: float,
    bc_mode: int,
    bc_qsus: float,
    d50: float,
    d90: float,
    p_silt: float,
    rho_w: float,
    rho_s: float,
    d_star: float,
    ws: float,
    ks: float,
    porosity: float,
    cal_tr: float,
    cal_ucr: float,
    slope_factor: float,
    dt_hours: float,
    coef_adapt: float,
    qbed_uses_cal_ucr: bool,
    tidal_wave_fac_bug: bool,
    strict_excel: bool,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    u = discharge / (b * h_tot)
    chezy = 18.0 * np.log10(12.0 * h_tot / ks)
    _, ub = _wave_orbital(h_tot, hs, tp, strict_excel=strict_excel)
    u_eff = u + cal_ucr * 0.4 * ub

    u_cr_c, u_cr_w = _van_rijn_critical_velocities(
        h_tot,
        d50,
        d90,
        tp,
        1.0 + p_silt,
        rho_w,
        rho_s,
        slope_factor=slope_factor,
        tube_coarse_bug=True,
        strict_excel=strict_excel,
    )
    u_cr = _combined_ucr(u, ub, u_cr_c, u_cr_w)
    u_star = u_eff * (GRAVITY**0.5) / chezy

    if tidal_wave_fac_bug and strict_excel:
        wave_fac = np.ones_like(h_tot, dtype=np.float64)
        wave_fac[0] = (1.0 + hs[0] / h_tot[0]) ** 2
    else:
        wave_fac = (1.0 + hs / h_tot) ** 2

    t_adap = (coef_adapt / h_tot) * (ws / u_star) * (1.0 + 2.0 * ws / u_star) * wave_fac

    qb_unit, qeq_unit = _van_rijn_sand_rates_unit_width(
        h_tot, u, u_eff, u_cr, d50, d_star, 1.0 - p_silt, rho_w, rho_s, cal_tr
    )
    qbed = b * qb_unit * (cal_ucr if qbed_uses_cal_ucr else 1.0)
    qeq = b * qeq_unit

    n = x.size
    qsus = np.empty(n, dtype=np.float64)
    qsus[0] = qeq[0] if bc_mode == 1 else bc_qsus
    for i in range(1, n):
        dx = x[i] - x[i - 1]
        qsus[i] = (qsus[i - 1] + t_adap[i] * qeq[i] * dx) / (1.0 + t_adap[i] * dx)

    qtot = qsus + qbed
    qvol = qtot / ((1.0 - porosity) * rho_s)

    dh = np.zeros(n, dtype=np.float64)
    dh[1:] = -((qvol[1:] - qvol[:-1]) / (x[1:] - x[:-1])) * dt_hours * 3600.0 / b[:-1]
    return dh, u, qbed, qeq, qsus, qtot


def streamtube_sedimentation(
    x: ArrayLike,
    width: ArrayLike,
    h_init: ArrayLike,
    hs: ArrayLike,
    tp: ArrayLike,
    *,
    discharge: float = 10.0,
    eta: float = 0.0,
    bc_mode: int = 1,
    bc_qsus: float = 0.0,
    d50: float = 0.0002,
    d90: float = 0.0003,
    p_silt: float = 0.05,
    rho_w: float = 1020.0,
    rho_s: float = 2650.0,
    nu: float = KINEMATIC_VISCOSITY_20C,
    ws: float = 0.015,
    ks: float = 0.03,
    porosity: float = 0.4,
    cal_tr: float = 0.67,
    cal_ucr: float = 1.0,
    slope_factor: float = 1.0,
    dt_hours: float = 0.5,
    coef_adapt: float = 0.1,
    n_steps: int = 1,
    qbed_uses_cal_ucr: bool = False,
    strict_excel: bool = False,
) -> StreamtubeSedimentationResult:
    """Resuelve transporte en no equilibrio y evolución morfológica 1D (``SED TUBE``).

    Reproduce la macro estacionaria de ``SED TUBE Case4 morphology``,
    ``SED TUBE india``, ``SED TUBE lab1 havinga morphology`` y ``SED TUBE mud``.

    Parameters
    ----------
    bc_mode : {0, 1}, default=1
        ``1`` impone condición de equilibrio aguas arriba (``qsus[0] = qeq[0]``);
        ``0`` fija la tasa entrante ``bc_qsus`` (kg/s).
    n_steps : int, default=1
        Número de iteraciones morfológicas (``h <- h - dh`` en cada paso).
    """
    x_arr = _as_1d(x, "x")
    n = x_arr.size
    b_arr = _as_1d(width, "width")
    h0_arr = _as_1d(h_init, "h_init")
    hs_arr = _as_1d(hs, "hs")
    tp_arr = _as_1d(tp, "tp")
    for arr, nm in ((b_arr, "width"), (h0_arr, "h_init"), (hs_arr, "hs"), (tp_arr, "tp")):
        if arr.size != n:
            raise ValueError(f"'{nm}' debe tener longitud {n}.")
    if n_steps < 1:
        raise ValueError("n_steps debe ser >= 1.")

    d_star = _d_star(d50, rho_s, rho_w, nu, strict_excel=strict_excel, exp_strict=0.333333)
    h_cur = h0_arr + eta
    for _ in range(n_steps):
        dh, u, qbed, qeq, qsus, qtot = _streamtube_step(
            x_arr,
            b_arr,
            h_cur,
            hs_arr,
            tp_arr,
            discharge=discharge,
            bc_mode=bc_mode,
            bc_qsus=bc_qsus,
            d50=d50,
            d90=d90,
            p_silt=p_silt,
            rho_w=rho_w,
            rho_s=rho_s,
            d_star=d_star,
            ws=ws,
            ks=ks,
            porosity=porosity,
            cal_tr=cal_tr,
            cal_ucr=cal_ucr,
            slope_factor=slope_factor,
            dt_hours=dt_hours,
            coef_adapt=coef_adapt,
            qbed_uses_cal_ucr=qbed_uses_cal_ucr,
            tidal_wave_fac_bug=False,
            strict_excel=strict_excel,
        )
        h_cur = h_cur - dh

    return StreamtubeSedimentationResult(
        x_m=x_arr,
        width_m=b_arr,
        h_initial_m=h0_arr + eta,
        h_final_m=h_cur,
        u_m_s=u,
        qbed_kg_s=qbed,
        qeq_sus_kg_s=qeq,
        qsus_kg_s=qsus,
        qtot_kg_s=qtot,
        dh_last_step_m=dh,
    )


def streamtube_tidal_sedimentation(
    x: ArrayLike,
    width: ArrayLike,
    h_msl_init: ArrayLike,
    *,
    q_flood: float,
    eta_flood: float,
    hs_flood: float,
    tp_flood: float,
    q_ebb: float,
    eta_ebb: float,
    hs_ebb: float,
    tp_ebb: float,
    d50: float,
    d90: float,
    p_silt: float,
    rho_w: float,
    rho_s: float,
    nu: float,
    ws: float,
    ks: float,
    porosity: float,
    cal_tr: float,
    cal_ucr: float,
    slope_factor: float = 1.0,
    dt_hours: float = 0.05,
    coef_adapt: float = 0.1,
    n_half_steps: int = 120,
    n_tides: int = 360,
    strict_excel: bool = False,
) -> StreamtubeTidalResult:
    """Ejecuta la evolución morfológica mareal bidireccional (``MorphodynamicRun``).

    En cada marea ejecuta ``n_half_steps`` pasos en llenante con inversión del eje
    ``x`` antes y después de ejecutar ``n_half_steps`` pasos en vaciante, replicando
    exactamente la macro de ``SED TUBE including tide Delfzijl``, ``Mar del Plata``,
    ``Scheveningen``, ``trench noordzee`` y ``westerschelde tide``.

    Parameters
    ----------
    strict_excel : bool, default=False
        Si ``True``, mantiene el factor de onda ``(1 + Hs/h)^2`` únicamente en el
        primer nodo y ``1.0`` en los nodos restantes (error de arrastre en columna ``Y``
        de las hojas mareales), además de ``3.142`` y ``D*`` con exponente ``0.333333``.
    """
    x_arr = _as_1d(x, "x")
    n = x_arr.size
    b_flood = _as_1d(width, "width")
    h_msl_0 = _as_1d(h_msl_init, "h_msl_init")
    if b_flood.size != n or h_msl_0.size != n:
        raise ValueError(f"'width' y 'h_msl_init' deben tener longitud {n}.")
    if n_half_steps < 1 or n_tides < 1:
        raise ValueError("n_half_steps y n_tides deben ser >= 1.")

    b_ebb = b_flood[::-1].copy()
    x_rev_asc = x_arr[-1] - x_arr[::-1]  # Coordenadas crecientes al invertir el eje
    hs_f = np.full(n, float(hs_flood), dtype=np.float64)
    tp_f = np.full(n, float(tp_flood), dtype=np.float64)
    hs_e = np.full(n, float(hs_ebb), dtype=np.float64)
    tp_e = np.full(n, float(tp_ebb), dtype=np.float64)
    d_star = _d_star(d50, rho_s, rho_w, nu, strict_excel=strict_excel, exp_strict=0.333333)

    h_cur = h_msl_0.copy()
    common_kw = dict(
        bc_mode=1,
        bc_qsus=0.0,
        d50=d50,
        d90=d90,
        p_silt=p_silt,
        rho_w=rho_w,
        rho_s=rho_s,
        d_star=d_star,
        ws=ws,
        ks=ks,
        porosity=porosity,
        cal_tr=cal_tr,
        cal_ucr=cal_ucr,
        slope_factor=slope_factor,
        dt_hours=dt_hours,
        coef_adapt=coef_adapt,
        qbed_uses_cal_ucr=False,
        tidal_wave_fac_bug=True,
        strict_excel=strict_excel,
    )

    for _ in range(n_tides):
        for _ in range(n_half_steps):
            dh, *_ = _streamtube_step(
                x_arr,
                b_flood,
                h_cur + eta_flood,
                hs_f,
                tp_f,
                discharge=q_flood,
                **common_kw,
            )
            h_cur = h_cur - dh

        # Inversión de perfil espacial para vaciante; dx permanece positivo.
        h_rev = h_cur[::-1].copy()
        for _ in range(n_half_steps):
            dh, *_ = _streamtube_step(
                x_rev_asc,
                b_ebb,
                h_rev + eta_ebb,
                hs_e,
                tp_e,
                discharge=q_ebb,
                **common_kw,
            )
            h_rev = h_rev - dh

        h_cur = h_rev[::-1].copy()

    return StreamtubeTidalResult(
        x_m=x_arr,
        width_m=b_flood,
        h_msl_initial_m=h_msl_0,
        h_msl_final_m=h_cur,
        bed_level_initial_m=-h_msl_0,
        bed_level_final_m=-h_cur,
        dh_total_m=h_msl_0 - h_cur,
    )
