"""Hidraulica fluvial en canales trapezoidales, compuestos y vertederos libres.

Implementa las herramientas Excel:
- ``Computation Discharge trapezoidal channels.xls``
- ``Computation Discharge and surface slope of compound river channels.xls``
  (modulo de flujo y curva de remanso)
- ``free overfall.xls``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

from sedmor.constants import GRAVITY, PI_STANDARD

__all__ = [
    "BackwaterProfileResult",
    "CompoundChannelFlowResult",
    "FreeOverfallResult",
    "TrapezoidalChannelResult",
    "compound_channel_backwater",
    "compound_channel_flow",
    "free_overfall_brink_depth",
    "solve_trapezoidal_depth",
    "trapezoidal_channel_discharge",
]


@dataclass(frozen=True)
class TrapezoidalChannelResult:
    """Resultados hidraulicos para una seccion trapezoidal uniforme."""

    water_depth: NDArray[np.float64] | float
    side_length: NDArray[np.float64] | float
    wetted_perimeter: NDArray[np.float64] | float
    area: NDArray[np.float64] | float
    hydraulic_radius: NDArray[np.float64] | float
    shear_velocity: NDArray[np.float64] | float
    chezy: NDArray[np.float64] | float
    velocity: NDArray[np.float64] | float
    discharge: NDArray[np.float64] | float


@dataclass(frozen=True)
class CompoundChannelFlowResult:
    """Resultados hidraulicos por subseccion para un canal fluvial compuesto."""

    water_level: float
    slope: float
    depth: NDArray[np.float64]
    width: NDArray[np.float64]
    area: NDArray[np.float64]
    chezy: NDArray[np.float64]
    strickler: NDArray[np.float64]
    velocity: NDArray[np.float64]
    subsection_discharge: NDArray[np.float64]
    total_width: float
    total_area: float
    total_discharge: float
    mean_velocity: float


@dataclass(frozen=True)
class BackwaterProfileResult:
    """Perfil longitudinal de remanso en un canal compuesto con llanura de inundacion."""

    x: NDArray[np.float64]
    water_depth: NDArray[np.float64]
    floodplain_depth: NDArray[np.float64]
    dh_dx: NDArray[np.float64]
    dh: NDArray[np.float64]
    surface_width: NDArray[np.float64]
    area: NDArray[np.float64]
    wetted_perimeter: NDArray[np.float64]
    hydraulic_radius: NDArray[np.float64]
    velocity: NDArray[np.float64]
    ks_effective: NDArray[np.float64]
    chezy_mean: NDArray[np.float64]
    friction_slope: NDArray[np.float64]


@dataclass(frozen=True)
class FreeOverfallResult:
    """Relacion de calado en caida libre (brink depth) en canal rectangular y ecuacion de Hager."""

    froude: NDArray[np.float64] | float
    gamma_rectangular: NDArray[np.float64] | float
    gamma_hager: NDArray[np.float64] | float
    brink_depth_rectangular: NDArray[np.float64] | float | None = None
    brink_depth_hager: NDArray[np.float64] | float | None = None


def _to_scalar_or_array(arr: NDArray[np.float64], is_scalar: bool) -> NDArray[np.float64] | float:
    return float(arr) if is_scalar else arr


def trapezoidal_channel_discharge(
    water_depth: ArrayLike,
    bottom_width: float = 20.0,
    side_angle_deg: float = 45.0,
    slope: float = 0.0001,
    ks: float = 0.1,
    *,
    gravity: float = GRAVITY,
    strict_excel: bool = False,
) -> TrapezoidalChannelResult:
    """Calcula las propiedades hidraulicas y el caudal de un canal trapezoidal.

    Parameters
    ----------
    water_depth : ArrayLike
        Tirante o profundidad de flujo ``h`` (m). Puede ser escalar o vector.
    bottom_width : float, default=20.0
        Ancho de fondo ``b`` (m).
    side_angle_deg : float, default=45.0
        Angulo de inclinacion de los taludes respecto a la horizontal (grados).
    slope : float, default=0.0001
        Pendiente longitudinal del fondo/superficie libre ``I`` (-).
    ks : float, default=0.1
        Rugosidad equivalente de Nikuradse ``k_s`` (m).
    gravity : float, default=9.81
        Aceleracion de la gravedad (m/s2).
    strict_excel : bool, default=False
        Si es ``True``, reproduce exactamente las formulas de la hoja Excel
        (usa ``pi = 3.1416`` y ``A = b*h + h^2`` valido unicamente para 45 grados).
        Si es ``False``, usa ``math.pi`` y ``A = b*h + h^2 / tan(alpha)``.
    """
    h_arr = np.asarray(water_depth, dtype=np.float64)
    is_scalar = h_arr.ndim == 0
    if np.any(h_arr <= 0.0):
        raise ValueError("El tirante water_depth debe ser estrictamente positivo.")
    if bottom_width <= 0.0 or slope <= 0.0 or ks <= 0.0:
        raise ValueError("bottom_width, slope y ks deben ser positivos.")
    if not (0.0 < side_angle_deg <= 90.0):
        raise ValueError("side_angle_deg debe estar en el rango (0, 90].")

    pi_val = 3.1416 if strict_excel else PI_STANDARD
    alpha_rad = float(side_angle_deg) / 180.0 * pi_val
    sin_a = float(np.sin(alpha_rad))
    tan_a = float(np.tan(alpha_rad))

    side_length = h_arr / sin_a
    wetted_perimeter = bottom_width + 2.0 * side_length
    if strict_excel:
        area = bottom_width * h_arr + h_arr**2
    else:
        area = bottom_width * h_arr + h_arr**2 / tan_a

    hydraulic_radius = area / wetted_perimeter
    shear_velocity = np.sqrt(gravity * hydraulic_radius * slope)
    chezy = 18.0 * np.log10(12.0 * hydraulic_radius / ks)
    velocity = chezy * np.sqrt(hydraulic_radius * slope)
    discharge = area * velocity

    return TrapezoidalChannelResult(
        water_depth=_to_scalar_or_array(h_arr, is_scalar),
        side_length=_to_scalar_or_array(side_length, is_scalar),
        wetted_perimeter=_to_scalar_or_array(wetted_perimeter, is_scalar),
        area=_to_scalar_or_array(area, is_scalar),
        hydraulic_radius=_to_scalar_or_array(hydraulic_radius, is_scalar),
        shear_velocity=_to_scalar_or_array(shear_velocity, is_scalar),
        chezy=_to_scalar_or_array(chezy, is_scalar),
        velocity=_to_scalar_or_array(velocity, is_scalar),
        discharge=_to_scalar_or_array(discharge, is_scalar),
    )


def solve_trapezoidal_depth(
    target_discharge: float,
    bottom_width: float = 20.0,
    side_angle_deg: float = 45.0,
    slope: float = 0.0001,
    ks: float = 0.1,
    *,
    h_min: float = 0.05,
    h_max: float = 100.0,
    gravity: float = GRAVITY,
    strict_excel: bool = False,
) -> float:
    """Resuelve el tirante ``h`` correspondiente a un caudal objetivo en canal trapezoidal."""
    if target_discharge <= 0.0:
        raise ValueError("target_discharge debe ser positivo.")

    def residual(h_val: float) -> float:
        res = trapezoidal_channel_discharge(
            h_val,
            bottom_width=bottom_width,
            side_angle_deg=side_angle_deg,
            slope=slope,
            ks=ks,
            gravity=gravity,
            strict_excel=strict_excel,
        )
        return float(res.discharge) - target_discharge

    low = max(h_min, ks / 10.0 + 1e-4)
    high = h_max
    while residual(high) < 0.0 and high < 1e5:
        high *= 2.0

    return float(brentq(residual, low, high, xtol=1e-10))


def compound_channel_flow(
    water_level: float,
    slope: float,
    bed_levels: Sequence[float] | ArrayLike,
    widths: Sequence[float] | ArrayLike,
    ks_values: Sequence[float] | ArrayLike,
    *,
    strict_excel: bool = False,
) -> CompoundChannelFlowResult:
    """Calcula velocidad, Chezy, Strickler y caudal por subsecciones en un canal compuesto.

    Divide la seccion transversal en ``N`` franjas verticales con cota de fondo ``z_b``,
    ancho ``b_y`` y rugosidad ``k_{s,y}`` independientes.
    """
    if slope < 0.0:
        raise ValueError("slope debe ser no negativo.")

    zb = np.asarray(bed_levels, dtype=np.float64)
    by = np.asarray(widths, dtype=np.float64)
    ks = np.asarray(ks_values, dtype=np.float64)
    if zb.ndim != 1 or by.shape != zb.shape or ks.shape != zb.shape:
        raise ValueError("bed_levels, widths y ks_values deben ser vectores unidimensionales de igual longitud.")
    if np.any(by < 0.0) or np.any(ks <= 0.0):
        raise ValueError("Los anchos deben ser >= 0 y las rugosidades > 0.")

    hy = np.maximum(float(water_level) - zb, 0.0)
    area = hy * by
    wet = hy > 0.0

    chezy = np.zeros_like(hy)
    chezy[wet] = 18.0 * np.log10(12.0 * hy[wet] / ks[wet])

    exponent = 0.166666666 if strict_excel else (1.0 / 6.0)
    strickler = np.zeros_like(hy)
    strickler[by > 0.0] = np.where(wet[by > 0.0], 25.0 * (hy[by > 0.0] / ks[by > 0.0]) ** exponent, 0.0)

    velocity = chezy * np.sqrt(hy * slope)
    dq = velocity * area

    total_area = float(np.sum(area))
    total_q = float(np.sum(dq))
    mean_vel = total_q / total_area if total_area > 0.0 else 0.0

    return CompoundChannelFlowResult(
        water_level=float(water_level),
        slope=float(slope),
        depth=hy,
        width=by,
        area=area,
        chezy=chezy,
        strickler=strickler,
        velocity=velocity,
        subsection_discharge=dq,
        total_width=float(np.sum(by)),
        total_area=total_area,
        total_discharge=total_q,
        mean_velocity=mean_vel,
    )


def _section_hydraulics(
    h_curr: float,
    h_fp: float,
    h_stage_for_b3: float,
    q: float,
    slope: float,
    b_vals: tuple[float, float, float, float, float],
    ks_vals: tuple[float, float, float, float, float],
    tan_a: tuple[float, float, float],
    h_main: float,
    gravity: float,
) -> tuple[float, float, float, float, float, float, float, float, float]:
    b2 = b_vals[1]
    b3 = b_vals[2]
    b4 = b_vals[3]
    tan_a1, tan_a2, tan_a3 = tan_a

    overbank = h_curr > h_main
    b1_s = h_fp / tan_a1 if overbank else 0.0
    b2_s = b2 if overbank else 0.0
    b3_s = b3 if h_stage_for_b3 > h_main else h_curr / tan_a2
    b4_s = b4
    b5_s = h_curr / tan_a3

    bs = b1_s + b2_s + b3_s + b4_s + b5_s
    if overbank:
        area = 0.5 * b1_s * h_fp + h_fp * b2_s + 0.5 * (h_curr + h_fp) * b3_s + h_curr * b4_s + 0.5 * h_curr * b5_s
        perim = (
            np.hypot(h_fp, b1_s)
            + b2_s
            + np.hypot(h_main, b3_s)
            + b4_s
            + np.hypot(h_curr, b5_s)
        )
    else:
        area = 0.5 * h_curr * b3_s + h_curr * b4_s + 0.5 * h_curr * b5_s
        perim = np.hypot(h_curr, b3_s) + b4_s + np.hypot(h_curr, b5_s)

    hyd_rad = area / perim
    ks_eff = (
        ks_vals[0] * b1_s
        + ks_vals[1] * b2_s
        + ks_vals[2] * b3_s
        + ks_vals[3] * b4_s
        + ks_vals[4] * b5_s
    ) / bs
    chezy = 18.0 * np.log10(12.0 * hyd_rad / ks_eff)
    vel = q / area
    sf = q**2 / (chezy**2 * area**2 * hyd_rad)
    froude_term = bs * q**2 / (gravity * area**3)
    dh_dx = (slope - sf) / (1.0 - froude_term)
    return bs, area, float(perim), hyd_rad, vel, ks_eff, chezy, sf, dh_dx


def compound_channel_backwater(
    h0: float = 4.0,
    slope: float = 0.0001,
    discharge: float = 1000.0,
    widths: tuple[float, float, float, float, float] = (100.0, 100.0, 100.0, 100.0, 100.0),
    ks_values: tuple[float, float, float, float, float] = (0.01, 0.01, 0.01, 0.01, 0.01),
    h_floodplain: float = 2.0,
    h_main_channel: float = 5.0,
    dx: float = 250.0,
    n_steps: int = 538,
    *,
    gravity: float = GRAVITY,
    strict_excel: bool = False,
) -> BackwaterProfileResult:
    """Integra la curva de remanso aguas arriba en un canal compuesto trapezoidal-llanura.

    Reproduce la hoja ``Computation Water surface slope`` del libro
    ``Computation Discharge and surface slope of compound river channels.xls``.

    Parameters
    ----------
    strict_excel : bool, default=False
        Si es ``True``, reproduce exactamente el desfase de una fila presente en las
        formulas arrastradas de la fila 23 de la hoja Excel original (``B22`` evaluada
        dos veces consecutivas en el inicio de la integracion). Si es ``False``, evalua
        cada punto ``i`` con su propio tirante ``h[i]``.
    """
    if len(widths) != 5 or len(ks_values) != 5:
        raise ValueError("widths y ks_values deben tener exactamente 5 elementos (b1..b5 y ks1..ks5).")
    if n_steps < 1 or dx <= 0.0 or h0 <= 0.0 or discharge <= 0.0:
        raise ValueError("Parametros geometricos e hidraulicos deben ser positivos.")

    tan_a = (
        h_floodplain / widths[0],
        h_main_channel / widths[2],
        (h_floodplain + h_main_channel) / widths[4],
    )

    n_pts = n_steps + 1
    x = -float(dx) * np.arange(n_pts, dtype=np.float64)
    h = np.zeros(n_pts, dtype=np.float64)
    h_fp = np.zeros(n_pts, dtype=np.float64)
    dh_dx = np.zeros(n_pts, dtype=np.float64)
    dh = np.zeros(n_pts, dtype=np.float64)
    bs = np.zeros(n_pts, dtype=np.float64)
    area = np.zeros(n_pts, dtype=np.float64)
    perim = np.zeros(n_pts, dtype=np.float64)
    hyd_rad = np.zeros(n_pts, dtype=np.float64)
    vel = np.zeros(n_pts, dtype=np.float64)
    ks_eff = np.zeros(n_pts, dtype=np.float64)
    chezy = np.zeros(n_pts, dtype=np.float64)
    sf = np.zeros(n_pts, dtype=np.float64)

    h[0] = float(h0)
    h_fp[0] = max(h0 - h_main_channel, 0.0)

    for i in range(n_pts):
        if strict_excel and i >= 1:
            # En la hoja Excel, la fila 23 apunta aun a la fila 22 salvo la celda O (b3).
            h_eval = h[i - 1]
            h_fp_eval = h_fp[i - 1]
            h_b3_eval = h[i]
        else:
            h_eval = h[i]
            h_fp_eval = max(h_eval - h_main_channel, 0.0)
            h_fp[i] = h_fp_eval
            h_b3_eval = h_eval

        res = _section_hydraulics(
            h_eval,
            h_fp_eval,
            h_b3_eval,
            discharge,
            slope,
            widths,
            ks_values,
            tan_a,
            h_main_channel,
            gravity,
        )
        bs[i], area[i], perim[i], hyd_rad[i], vel[i], ks_eff[i], chezy[i], sf[i], dh_dx[i] = res

        if i < n_pts - 1:
            dh[i] = dh_dx[i] * (-float(dx))
            h[i + 1] = h[i] + dh[i]
            if strict_excel:
                h_fp[i + 1] = (h_fp[i] + dh[i]) if h[i] > h_main_channel else 0.0

    return BackwaterProfileResult(
        x=x,
        water_depth=h,
        floodplain_depth=h_fp,
        dh_dx=dh_dx,
        dh=dh,
        surface_width=bs,
        area=area,
        wetted_perimeter=perim,
        hydraulic_radius=hyd_rad,
        velocity=vel,
        ks_effective=ks_eff,
        chezy_mean=chezy,
        friction_slope=sf,
    )


def _solve_single_hager_gamma(froude_val: float) -> float:
    target = 6.0 * froude_val**2

    def residual(g_val: float) -> float:
        return (3.0 * g_val - g_val**3) / (1.0 - g_val) - target

    return float(brentq(residual, 1e-6, 0.99999999, xtol=1e-12))


def free_overfall_brink_depth(
    froude: ArrayLike,
    critical_depth: ArrayLike | None = None,
) -> FreeOverfallResult:
    """Calcula la razon de calado en caida libre (brink depth / critical depth).

    Implementa ``free overfall.xls`` combinando:
    - Formulacion rectangular: ``gamma = Fr^2 / (0.5 + Fr^2)``
    - Ecuacion implicita de Hager (1983): ``(3*gamma - gamma^3)/(1 - gamma) = 6*Fr^2``
    """
    fr_arr = np.asarray(froude, dtype=np.float64)
    is_scalar = fr_arr.ndim == 0
    if np.any(fr_arr <= 0.0):
        raise ValueError("El numero de Froude debe ser estrictamente positivo.")

    gamma_rect = fr_arr**2 / (0.5 + fr_arr**2)
    flat_fr = fr_arr.ravel()
    gamma_hager = np.array([_solve_single_hager_gamma(float(v)) for v in flat_fr], dtype=np.float64).reshape(
        fr_arr.shape
    )

    if critical_depth is None:
        return FreeOverfallResult(
            froude=_to_scalar_or_array(fr_arr, is_scalar),
            gamma_rectangular=_to_scalar_or_array(gamma_rect, is_scalar),
            gamma_hager=_to_scalar_or_array(gamma_hager, is_scalar),
        )

    hc_arr = np.asarray(critical_depth, dtype=np.float64)
    if np.any(hc_arr <= 0.0):
        raise ValueError("critical_depth debe ser positivo.")
    yb_rect = hc_arr * gamma_rect
    yb_hager = hc_arr * gamma_hager
    out_scalar = is_scalar and hc_arr.ndim == 0

    return FreeOverfallResult(
        froude=_to_scalar_or_array(fr_arr, out_scalar),
        gamma_rectangular=_to_scalar_or_array(gamma_rect, out_scalar),
        gamma_hager=_to_scalar_or_array(gamma_hager, out_scalar),
        brink_depth_rectangular=_to_scalar_or_array(yb_rect, out_scalar),
        brink_depth_hager=_to_scalar_or_array(yb_hager, out_scalar),
    )
