"""Calculo de socavacion local junto a estructuras hidraulicas y costeras (``SCOUR.xls``).

Implementa las ocho hojas del libro ``SCOUR.xls`` de Leo C. van Rijn:

1. ``scour_downstream_structure`` (``scour1``): socavacion aguas abajo de una estructura
   por corrientes mas oleaje (incluye escala temporal ``T_s``, evolucion temporal y equilibrio).
2. ``scour_wall_and_rubblemound_toe`` (``2``): socavacion al pie de muros verticales y
   diques en talud por oleaje (SPM 1984, Fowler 1992, Sato 1968, Yokoyama 2002, Xie 1981).
3. ``scour_structure_tip_waves`` (``3``): socavacion en el morro/extremo por oleaje con
   corrientes debiles (< 0.3 m/s) segun Sumer (1997) y datos de campo de Japon/EE.UU.
4. ``scour_structure_tip_current_waves`` (``4``): socavacion en el morro por corriente
   fuerte (> 0.3 m/s) mas oleaje (diques en talud).
5. ``scour_river_groyne`` (``5``): socavacion en la punta de espigones fluviales o de
   marea (F9.6.7a, Lacey, Richardson-Hancu, Copeland/Lim/Melville y van Rijn F9.6.8/9).
6. ``scour_bridge_pier`` (``6``): socavacion en pilas de puente verticales en corrientes
   relativamente fuertes (con o sin oleaje).
7. ``scour_vertical_pile_waves`` (``7``): socavacion en pilotes/cilindros verticales en
   oleaje con corrientes debiles (incluyendo factor de grupo).
8. ``scour_horizontal_pipe`` (``8``): socavacion bajo tuberias horizontales apoyadas en
   el lecho por corriente, oleaje y accion combinada.
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
    "BridgePierScourResult",
    "DownstreamScourResult",
    "HorizontalPipeScourResult",
    "RiverGroyneScourResult",
    "StructureTipCurrentWavesResult",
    "StructureTipWavesResult",
    "VerticalPileWavesResult",
    "WallRubbleToeScourResult",
    "scour_bridge_pier",
    "scour_downstream_structure",
    "scour_horizontal_pipe",
    "scour_river_groyne",
    "scour_structure_tip_current_waves",
    "scour_structure_tip_waves",
    "scour_vertical_pile_waves",
    "scour_wall_and_rubblemound_toe",
]


def _as_out(arr: NDArray[np.float64], is_scalar: bool) -> NDArray[np.float64] | float:
    return float(arr) if is_scalar else arr


def _compute_wave_and_critical_velocities(
    h0: NDArray[np.float64],
    d50: NDArray[np.float64],
    d90: NDArray[np.float64],
    u: NDArray[np.float64],
    Hs: NDArray[np.float64],
    Tp: NDArray[np.float64],
    *,
    rho_w: float,
    rho_s: float,
    nu: float,
    gravity: float,
    corr_ucr: float,
    apply_corr_to_ucr_c: bool,
    apply_corr_to_ucr_cw: bool,
    p_mud: NDArray[np.float64] | None = None,
    strict_excel: bool = False,
) -> tuple[
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
    """Calcula ``D*``, parametros de onda locales, ``Ub``, ``Ucr`` y ``Ueff``."""
    s_rel = rho_s / rho_w - 1.0
    d_star_exp = 0.333333 if strict_excel else (1.0 / 3.0)
    d_star = d50 * ((s_rel * gravity) / (nu**2)) ** d_star_exp

    # Dispersion aproximada y velocidad orbital maxima en el fondo
    two_pi = TWO_PI_EXCEL if strict_excel else (2.0 * PI_STANDARD)
    pi_orb = 3.142 if strict_excel else PI_STANDARD
    coef_k = 4.02 if strict_excel else ((2.0 * PI_STANDARD) ** 2 / gravity)
    coef_L0 = 1.56 if strict_excel else (gravity / (2.0 * PI_STANDARD))

    pos_tp = Tp > 0.0
    tp_safe = np.where(pos_tp, Tp, 1.0)
    x_disp = np.where(pos_tp, coef_k * h0 / (tp_safe**2), 0.0)
    kh = np.where(
        pos_tp,
        np.sqrt(np.maximum(x_disp, 0.0)) * (1.0 + 0.166 * x_disp + 0.031 * (x_disp**2)),
        0.0,
    )
    wavelength = np.where(pos_tp & (kh > 0.0), two_pi * h0 / np.maximum(kh, 1e-30), 0.0)
    wavelength_0 = coef_L0 * (Tp**2)
    sinh_kh = np.where(pos_tp & (kh > 0.0), np.sinh(kh), 1.0)
    u_b = np.where(pos_tp & (kh > 0.0), pi_orb * Hs / (tp_safe * sinh_kh), 0.0)
    u_eff = u + 0.5 * u_b

    mud_factor = (1.0 + p_mud) ** 1.5 if p_mud is not None else 1.0
    rough_ratio = np.maximum(12.0 * h0 / np.maximum(3.0 * d90, 1e-30), 1e-30)
    log_term = np.log10(rough_ratio)

    c_corr = corr_ucr if apply_corr_to_ucr_c else 1.0
    ucr_c_fine = c_corr * 0.19 * mud_factor * (d50**0.1) * log_term
    if p_mud is not None and strict_excel:
        # La hoja 5 escribe E25^((1+G25)^1.5)*0.6 en la rama gruesa
        ucr_c_coarse = c_corr * 8.5 * (d50**mud_factor) * 0.6 * log_term
    else:
        ucr_c_coarse = c_corr * 8.5 * mud_factor * (d50**0.6) * log_term
    ucr_c = np.where(d50 < 0.0005, ucr_c_fine, ucr_c_coarse)

    # Velocidad critica de oleaje: las hojas tienen parentesis omitidos en la rama gruesa
    # ($E$9 - $E$8/$E$8*9.81) cuando strict_excel=True.
    ucr_w_fine = 0.24 * mud_factor * (s_rel * gravity) ** 0.66 * (d50**0.33) * (Tp**0.33)
    if strict_excel:
        g_prime_bug = rho_s - (rho_w / rho_w) * gravity
        ucr_w_coarse = (
            0.95 * mud_factor * (max(g_prime_bug, 0.0) ** 0.57) * (d50**0.43) * (Tp**0.14)
        )
    else:
        ucr_w_coarse = 0.95 * mud_factor * ((s_rel * gravity) ** 0.57) * (d50**0.43) * (Tp**0.14)
    ucr_w = np.where(d50 < 0.0005, ucr_w_fine, ucr_w_coarse)

    denom_u = u + u_b
    w_curr = np.where(denom_u > 0.0, u / np.maximum(denom_u, 1e-30), 0.0)
    cw_corr = corr_ucr if apply_corr_to_ucr_cw else 1.0
    ucr_cw = cw_corr * (w_curr * ucr_c + (1.0 - w_curr) * ucr_w)

    return d_star, kh, wavelength, wavelength_0, u_b, u_eff, ucr_c, ucr_w, ucr_cw, np.asarray(
        np.full_like(h0, s_rel), dtype=np.float64
    )


def _compute_bed_and_suspended_transport(
    h0: NDArray[np.float64],
    u: NDArray[np.float64],
    d50: NDArray[np.float64],
    d_star: NDArray[np.float64],
    u_eff: NDArray[np.float64],
    ucr_cw: NDArray[np.float64],
    *,
    rho_w: float,
    rho_s: float,
    gravity: float,
    cal_transport: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Calcula ``qb``, ``qs`` y ``qt`` usados en las hojas ``scour1``, ``6``, ``7`` y ``8``."""
    s_rel = rho_s / rho_w - 1.0
    excess = np.maximum(u_eff - ucr_cw, 0.0)
    denom = np.maximum(s_rel * gravity * d50, 1e-30) ** 1.2
    qb = np.where(
        u_eff > ucr_cw,
        cal_transport
        * rho_s
        * 0.005
        * h0
        * u
        * (d50 / np.maximum(h0, 1e-30)) ** 1.2
        * (excess**2.4)
        / denom,
        0.0,
    )
    qs = np.where(
        u_eff > ucr_cw,
        cal_transport
        * rho_s
        * 0.012
        * h0
        * u
        * (d50 / np.maximum(h0, 1e-30))
        * (excess**2.4)
        / denom
        * (np.maximum(d_star, 1e-30) ** (-0.6)),
        0.0,
    )
    return qb, qs, qb + qs


@dataclass(frozen=True)
class DownstreamScourResult:
    """Resultados de socavacion aguas abajo de una estructura (hoja ``scour1``)."""

    d_star: NDArray[np.float64] | float
    u_eff: NDArray[np.float64] | float
    ucr_c: NDArray[np.float64] | float
    ucr_w: NDArray[np.float64] | float
    ucr_cw: NDArray[np.float64] | float
    kh: NDArray[np.float64] | float
    wavelength: NDArray[np.float64] | float
    u_b: NDArray[np.float64] | float
    qb: NDArray[np.float64] | float
    qs: NDArray[np.float64] | float
    qt: NDArray[np.float64] | float
    time_scale_hours: NDArray[np.float64] | float
    scour_depth_t: NDArray[np.float64] | float
    scour_depth_eq: NDArray[np.float64] | float
    scour_length: NDArray[np.float64] | float


@dataclass(frozen=True)
class WallRubbleToeScourResult:
    """Resultados de socavacion al pie de muros verticales y diques en talud (hoja ``2``)."""

    d_star: NDArray[np.float64] | float
    ucr_c: NDArray[np.float64] | float
    ucr_w: NDArray[np.float64] | float
    kh: NDArray[np.float64] | float
    wavelength: NDArray[np.float64] | float
    wavelength_0: NDArray[np.float64] | float
    u_b: NDArray[np.float64] | float
    theta_toe_deg: NDArray[np.float64] | float
    H_toe: NDArray[np.float64] | float
    wall_spm: NDArray[np.float64] | float
    wall_fowler: NDArray[np.float64] | float
    wall_sato: NDArray[np.float64] | float
    wall_yokoyama: NDArray[np.float64] | float
    wall_xie: NDArray[np.float64] | float
    wall_average: NDArray[np.float64] | float
    wall_scour_length: NDArray[np.float64] | float
    rubblemound_spm: NDArray[np.float64] | float
    rubblemound_fowler: NDArray[np.float64] | float
    rubblemound_sato: NDArray[np.float64] | float
    rubblemound_yokoyama: NDArray[np.float64] | float
    rubblemound_xie: NDArray[np.float64] | float
    rubblemound_average: NDArray[np.float64] | float
    rubblemound_scour_length: NDArray[np.float64] | float


@dataclass(frozen=True)
class StructureTipWavesResult:
    """Resultados de socavacion en morro por oleaje con corriente debil (hoja ``3``)."""

    d_star: NDArray[np.float64] | float
    ucr_c: NDArray[np.float64] | float
    ucr_w: NDArray[np.float64] | float
    kh: NDArray[np.float64] | float
    wavelength: NDArray[np.float64] | float
    wavelength_0: NDArray[np.float64] | float
    u_b: NDArray[np.float64] | float
    theta_toe_deg: NDArray[np.float64] | float
    H_toe: NDArray[np.float64] | float
    kc_number: NDArray[np.float64] | float
    wall_sumer_depth: NDArray[np.float64] | float
    wall_sumer_length: NDArray[np.float64] | float
    rubblemound_sumer_depth: NDArray[np.float64] | float
    rubblemound_sumer_length: NDArray[np.float64] | float
    rubblemound_field_depth: NDArray[np.float64] | float
    rubblemound_field_length: NDArray[np.float64] | float


@dataclass(frozen=True)
class StructureTipCurrentWavesResult:
    """Resultados de socavacion en morro con corriente fuerte mas oleaje (hoja ``4``)."""

    d_star: NDArray[np.float64] | float
    ucr_c: NDArray[np.float64] | float
    ucr_w: NDArray[np.float64] | float
    kh: NDArray[np.float64] | float
    wavelength: NDArray[np.float64] | float
    wavelength_0: NDArray[np.float64] | float
    u_b: NDArray[np.float64] | float
    theta_toe_deg: NDArray[np.float64] | float
    H_toe: NDArray[np.float64] | float
    kc_number: NDArray[np.float64] | float
    rubblemound_depth: NDArray[np.float64] | float
    rubblemound_length: NDArray[np.float64] | float


@dataclass(frozen=True)
class RiverGroyneScourResult:
    """Resultados de socavacion junto al extremo de un espigon fluvial (hoja ``5``)."""

    u: NDArray[np.float64] | float
    blocking_coeff: NDArray[np.float64] | float
    d_star: NDArray[np.float64] | float
    u_eff: NDArray[np.float64] | float
    ucr_c: NDArray[np.float64] | float
    ucr_w: NDArray[np.float64] | float
    ucr_cw: NDArray[np.float64] | float
    flow_intensity: NDArray[np.float64] | float
    kh: NDArray[np.float64] | float
    wavelength: NDArray[np.float64] | float
    u_b: NDArray[np.float64] | float
    depth_F9_6_7a: NDArray[np.float64] | float
    depth_lacey: NDArray[np.float64] | float
    depth_rh: NDArray[np.float64] | float
    depth_clm: NDArray[np.float64] | float
    depth_F9_6_8_9: NDArray[np.float64] | float
    scour_length: NDArray[np.float64] | float


@dataclass(frozen=True)
class BridgePierScourResult:
    """Resultados de socavacion junto a pila de puente vertical (hoja ``6``)."""

    u: NDArray[np.float64] | float
    alpha1: NDArray[np.float64] | float
    alpha2: NDArray[np.float64] | float
    alpha5: NDArray[np.float64] | float
    alpha6: NDArray[np.float64] | float
    d_star: NDArray[np.float64] | float
    u_eff: NDArray[np.float64] | float
    ucr_c: NDArray[np.float64] | float
    ucr_w: NDArray[np.float64] | float
    ucr_cw: NDArray[np.float64] | float
    kh: NDArray[np.float64] | float
    wavelength: NDArray[np.float64] | float
    u_b: NDArray[np.float64] | float
    qb: NDArray[np.float64] | float
    qs: NDArray[np.float64] | float
    qt: NDArray[np.float64] | float
    scour_depth: NDArray[np.float64] | float
    scour_length: NDArray[np.float64] | float


@dataclass(frozen=True)
class VerticalPileWavesResult:
    """Resultados de socavacion en pilote/pila vertical por oleaje (hoja ``7``)."""

    alpha2: NDArray[np.float64] | float
    d_star: NDArray[np.float64] | float
    u_eff: NDArray[np.float64] | float
    ucr_c: NDArray[np.float64] | float
    ucr_w: NDArray[np.float64] | float
    ucr_cw: NDArray[np.float64] | float
    kh: NDArray[np.float64] | float
    wavelength: NDArray[np.float64] | float
    u_b: NDArray[np.float64] | float
    kc_number: NDArray[np.float64] | float
    qb: NDArray[np.float64] | float
    qs: NDArray[np.float64] | float
    qt: NDArray[np.float64] | float
    scour_depth: NDArray[np.float64] | float
    scour_length: NDArray[np.float64] | float


@dataclass(frozen=True)
class HorizontalPipeScourResult:
    """Resultados de socavacion bajo tuberia horizontal sobre el lecho (hoja ``8``)."""

    alpha: NDArray[np.float64] | float
    d_star: NDArray[np.float64] | float
    u_eff: NDArray[np.float64] | float
    ucr_c: NDArray[np.float64] | float
    ucr_w: NDArray[np.float64] | float
    ucr_cw: NDArray[np.float64] | float
    kh: NDArray[np.float64] | float
    wavelength: NDArray[np.float64] | float
    u_b: NDArray[np.float64] | float
    kc_number: NDArray[np.float64] | float
    qb: NDArray[np.float64] | float
    qs: NDArray[np.float64] | float
    qt: NDArray[np.float64] | float
    scour_depth_current: NDArray[np.float64] | float
    scour_depth_waves: NDArray[np.float64] | float
    scour_depth_combined: NDArray[np.float64] | float
    scour_length: NDArray[np.float64] | float


def scour_downstream_structure(
    h0: ArrayLike = 5.0,
    u: ArrayLike = 0.4,
    duration_hours: ArrayLike = 100000.0,
    d50: ArrayLike = 0.00025,
    d90: ArrayLike = 0.0005,
    turbulence_r: ArrayLike = 0.15,
    Hs: ArrayLike = 0.0,
    Tp: ArrayLike = 7.0,
    *,
    rho_w: float = RHO_SEA_WATER,
    rho_s: float = RHO_SEDIMENT,
    nu: float = KINEMATIC_VISCOSITY_20C,
    cal_transport: float = 1.0,
    corr_ucr: float = 1.0,
    safety_factor: float = 1.0,
    gravity: float = GRAVITY,
    strict_excel: bool = False,
) -> DownstreamScourResult:
    """Socavacion aguas abajo de una estructura por corriente y oleaje (hoja ``scour1``)."""
    h0_a, u_a, t_a, d50_a, d90_a, r_a, Hs_a, Tp_a = np.broadcast_arrays(
        np.asarray(h0, dtype=np.float64),
        np.asarray(u, dtype=np.float64),
        np.asarray(duration_hours, dtype=np.float64),
        np.asarray(d50, dtype=np.float64),
        np.asarray(d90, dtype=np.float64),
        np.asarray(turbulence_r, dtype=np.float64),
        np.asarray(Hs, dtype=np.float64),
        np.asarray(Tp, dtype=np.float64),
    )
    is_scalar = h0_a.ndim == 0

    d_star, kh, wavelength, _, u_b, u_eff, ucr_c, ucr_w, ucr_cw, s_arr = (
        _compute_wave_and_critical_velocities(
            h0_a,
            d50_a,
            d90_a,
            u_a,
            Hs_a,
            Tp_a,
            rho_w=rho_w,
            rho_s=rho_s,
            nu=nu,
            gravity=gravity,
            corr_ucr=corr_ucr,
            apply_corr_to_ucr_c=True,
            apply_corr_to_ucr_cw=True,
            strict_excel=strict_excel,
        )
    )
    qb, qs, qt = _compute_bed_and_suspended_transport(
        h0_a,
        u_a,
        d50_a,
        d_star,
        u_eff,
        ucr_cw,
        rho_w=rho_w,
        rho_s=rho_s,
        gravity=gravity,
        cal_transport=cal_transport,
    )

    u_turb = (1.0 + 3.0 * r_a) * u_eff
    t_scale = np.where(
        u_turb > ucr_w,
        330.0 * (s_arr**1.7) * (h0_a**2) / np.maximum(u_turb - ucr_w, 1e-30) ** 4.3,
        0.0,
    )
    ds_t = np.where(t_scale > 0.0, safety_factor * (t_a / np.maximum(t_scale, 1e-30)) ** 0.38 * h0_a, 0.0)
    ds_eq = np.where(
        u_turb > ucr_cw,
        safety_factor * h0_a * (u_turb - ucr_cw) / np.maximum(ucr_cw, 1e-30),
        0.0,
    )
    ls = 10.0 * ds_eq

    return DownstreamScourResult(
        d_star=_as_out(d_star, is_scalar),
        u_eff=_as_out(u_eff, is_scalar),
        ucr_c=_as_out(ucr_c, is_scalar),
        ucr_w=_as_out(ucr_w, is_scalar),
        ucr_cw=_as_out(ucr_cw, is_scalar),
        kh=_as_out(kh, is_scalar),
        wavelength=_as_out(wavelength, is_scalar),
        u_b=_as_out(u_b, is_scalar),
        qb=_as_out(qb, is_scalar),
        qs=_as_out(qs, is_scalar),
        qt=_as_out(qt, is_scalar),
        time_scale_hours=_as_out(t_scale, is_scalar),
        scour_depth_t=_as_out(ds_t, is_scalar),
        scour_depth_eq=_as_out(ds_eq, is_scalar),
        scour_length=_as_out(ls, is_scalar),
    )


def _toe_wave_transformation(
    h0: NDArray[np.float64],
    Hs0: NDArray[np.float64],
    theta0_deg: NDArray[np.float64],
    gamma_br: NDArray[np.float64],
    kh: NDArray[np.float64],
    wavelength: NDArray[np.float64],
    wavelength_0: NDArray[np.float64],
    *,
    strict_excel: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Transformacion por refraccion y someramiento al pie de una estructura (hojas 2-4)."""
    pi_ang = 3.1416 if strict_excel else PI_STANDARD
    theta0_rad = theta0_deg / 180.0 * pi_ang
    sin_arg = np.where(
        wavelength_0 > 0.0,
        np.clip(wavelength / np.maximum(wavelength_0, 1e-30) * np.sin(theta0_rad), -1.0, 1.0),
        0.0,
    )
    theta_toe_deg = np.arcsin(sin_arg) * 180.0 / pi_ang
    theta_toe_rad = theta_toe_deg / 180.0 * pi_ang

    two_kh = 2.0 * kh
    n_ratio = np.where(two_kh > 0.0, 0.5 + kh / np.maximum(np.sinh(two_kh), 1e-30), 1.0)
    ks = np.where(
        (wavelength > 0.0) & (wavelength_0 > 0.0),
        np.sqrt(0.5 * wavelength_0 / np.maximum(n_ratio * wavelength, 1e-30)),
        1.0,
    )
    kr = np.sqrt(np.maximum(np.cos(theta0_rad) / np.maximum(np.cos(theta_toe_rad), 1e-30), 0.0))
    h_toe = np.minimum(ks * kr * Hs0, gamma_br * h0)
    return theta_toe_deg, h_toe


def scour_wall_and_rubblemound_toe(
    h0: ArrayLike = 5.0,
    d50: ArrayLike = 0.00025,
    d90: ArrayLike = 0.0005,
    Hs0: ArrayLike = 5.0,
    theta0_deg: ArrayLike = 30.0,
    Tp: ArrayLike = 7.0,
    gamma_br: ArrayLike = 0.6,
    *,
    rho_w: float = RHO_SEA_WATER,
    rho_s: float = RHO_SEDIMENT,
    nu: float = KINEMATIC_VISCOSITY_20C,
    safety_factor: float = 1.0,
    gravity: float = GRAVITY,
    strict_excel: bool = False,
) -> WallRubbleToeScourResult:
    """Socavacion al pie de muro vertical o dique en talud por oleaje (hoja ``2``)."""
    h0_a, d50_a, d90_a, Hs0_a, th0_a, Tp_a, gam_a = np.broadcast_arrays(
        np.asarray(h0, dtype=np.float64),
        np.asarray(d50, dtype=np.float64),
        np.asarray(d90, dtype=np.float64),
        np.asarray(Hs0, dtype=np.float64),
        np.asarray(theta0_deg, dtype=np.float64),
        np.asarray(Tp, dtype=np.float64),
        np.asarray(gamma_br, dtype=np.float64),
    )
    is_scalar = h0_a.ndim == 0

    zero_u = np.zeros_like(h0_a)
    d_star, kh, wavelength, wavelength_0, u_b, _, ucr_c, ucr_w, _, _ = (
        _compute_wave_and_critical_velocities(
            h0_a,
            d50_a,
            d90_a,
            zero_u,
            Hs0_a,
            Tp_a,
            rho_w=rho_w,
            rho_s=rho_s,
            nu=nu,
            gravity=gravity,
            corr_ucr=1.0,
            apply_corr_to_ucr_c=False,
            apply_corr_to_ucr_cw=False,
            strict_excel=strict_excel,
        )
    )
    theta_toe_deg, h_toe = _toe_wave_transformation(
        h0_a, Hs0_a, th0_a, gam_a, kh, wavelength, wavelength_0, strict_excel=strict_excel
    )

    h_ratio = h_toe / np.maximum(h0_a, 1e-30)
    h_l0 = h0_a / np.maximum(wavelength_0, 1e-30)
    h_lw = h0_a / np.maximum(wavelength, 1e-30)

    wall_spm = safety_factor * h_toe
    wall_fowler = np.where(
        h_l0 < 0.06,
        safety_factor * (11.4 * h_l0 + 0.54) * Hs0_a,
        safety_factor * Hs0_a,
    )
    wall_sato = np.where(
        h0_a < 9.0,
        safety_factor * (-0.2 * h0_a + 1.9),
        0.1 * safety_factor * h0_a,
    )
    wall_yokoyama = np.where(
        h_ratio > 0.25,
        safety_factor * (2.0 * h_ratio - 0.5) * h_toe,
        0.1 * safety_factor * h_toe,
    )
    wall_xie = np.where(
        h_lw < 0.15,
        safety_factor * (-6.4 * h_lw + 1.2) * h_toe,
        0.25 * safety_factor * h_toe,
    )

    rm_spm = safety_factor * 0.7 * wall_spm
    rm_fowler = safety_factor * 0.7 * wall_fowler
    rm_sato = safety_factor * 0.7 * wall_sato
    rm_yokoyama = safety_factor * 0.7 * wall_yokoyama
    rm_xie = safety_factor * 0.7 * wall_xie

    if strict_excel:
        # En la hoja 2: AW omite el 5to termino (AU123 en lugar de AU23) y AI usa AW en vez de AE
        rm_avg = (rm_spm + rm_fowler + rm_sato + rm_yokoyama) / 5.0
        wall_avg = (wall_spm + wall_fowler + wall_sato + rm_avg + wall_xie) / 5.0
    else:
        rm_avg = (rm_spm + rm_fowler + rm_sato + rm_yokoyama + rm_xie) / 5.0
        wall_avg = (wall_spm + wall_fowler + wall_sato + wall_yokoyama + wall_xie) / 5.0

    wall_len = np.maximum(0.25 * wavelength, 10.0 * wall_avg)
    rm_len = np.maximum(0.25 * wavelength, 10.0 * rm_avg)

    return WallRubbleToeScourResult(
        d_star=_as_out(d_star, is_scalar),
        ucr_c=_as_out(ucr_c, is_scalar),
        ucr_w=_as_out(ucr_w, is_scalar),
        kh=_as_out(kh, is_scalar),
        wavelength=_as_out(wavelength, is_scalar),
        wavelength_0=_as_out(wavelength_0, is_scalar),
        u_b=_as_out(u_b, is_scalar),
        theta_toe_deg=_as_out(theta_toe_deg, is_scalar),
        H_toe=_as_out(h_toe, is_scalar),
        wall_spm=_as_out(wall_spm, is_scalar),
        wall_fowler=_as_out(wall_fowler, is_scalar),
        wall_sato=_as_out(wall_sato, is_scalar),
        wall_yokoyama=_as_out(wall_yokoyama, is_scalar),
        wall_xie=_as_out(wall_xie, is_scalar),
        wall_average=_as_out(wall_avg, is_scalar),
        wall_scour_length=_as_out(wall_len, is_scalar),
        rubblemound_spm=_as_out(rm_spm, is_scalar),
        rubblemound_fowler=_as_out(rm_fowler, is_scalar),
        rubblemound_sato=_as_out(rm_sato, is_scalar),
        rubblemound_yokoyama=_as_out(rm_yokoyama, is_scalar),
        rubblemound_xie=_as_out(rm_xie, is_scalar),
        rubblemound_average=_as_out(rm_avg, is_scalar),
        rubblemound_scour_length=_as_out(rm_len, is_scalar),
    )


def scour_structure_tip_waves(
    h0: ArrayLike = 3.0,
    d50: ArrayLike = 0.00025,
    d90: ArrayLike = 0.0005,
    structure_width: ArrayLike = 10.0,
    Hs0: ArrayLike = 2.0,
    theta0_deg: ArrayLike = 30.0,
    Tp: ArrayLike = 9.0,
    gamma_br: ArrayLike = 0.6,
    *,
    rho_w: float = RHO_SEA_WATER,
    rho_s: float = RHO_SEDIMENT,
    nu: float = KINEMATIC_VISCOSITY_20C,
    safety_factor: float = 1.0,
    gravity: float = GRAVITY,
    strict_excel: bool = False,
) -> StructureTipWavesResult:
    """Socavacion en el morro de estructura por oleaje y corriente debil <0.3 m/s (hoja ``3``)."""
    h0_a, d50_a, d90_a, b_a, Hs0_a, th0_a, Tp_a, gam_a = np.broadcast_arrays(
        np.asarray(h0, dtype=np.float64),
        np.asarray(d50, dtype=np.float64),
        np.asarray(d90, dtype=np.float64),
        np.asarray(structure_width, dtype=np.float64),
        np.asarray(Hs0, dtype=np.float64),
        np.asarray(theta0_deg, dtype=np.float64),
        np.asarray(Tp, dtype=np.float64),
        np.asarray(gamma_br, dtype=np.float64),
    )
    is_scalar = h0_a.ndim == 0

    zero_u = np.zeros_like(h0_a)
    d_star, kh, wavelength, wavelength_0, u_b, _, ucr_c, ucr_w, _, _ = (
        _compute_wave_and_critical_velocities(
            h0_a,
            d50_a,
            d90_a,
            zero_u,
            Hs0_a,
            Tp_a,
            rho_w=rho_w,
            rho_s=rho_s,
            nu=nu,
            gravity=gravity,
            corr_ucr=1.0,
            apply_corr_to_ucr_c=False,
            apply_corr_to_ucr_cw=False,
            strict_excel=strict_excel,
        )
    )
    theta_toe_deg, h_toe = _toe_wave_transformation(
        h0_a, Hs0_a, th0_a, gam_a, kh, wavelength, wavelength_0, strict_excel=strict_excel
    )

    kc = u_b * Tp_a / np.maximum(b_a, 1e-30)
    wall_depth = np.maximum(
        np.where(kc < 10.0, 0.05 * safety_factor * kc * b_a, 0.5 * safety_factor * b_a),
        0.2 * h_toe,
    )
    wall_length = np.maximum(
        np.where(kc < 10.0, 0.5 * safety_factor * kc * b_a, 5.0 * safety_factor * b_a),
        10.0 * wall_depth,
    )

    sumer_param = Tp_a * np.sqrt(np.maximum(gravity * h_toe, 0.0)) / np.maximum(h0_a, 1e-30)
    rm_sumer_depth = np.maximum(
        np.where(sumer_param < 15.0, 0.03 * safety_factor * sumer_param, 0.5 * safety_factor * h_toe),
        0.15 * h_toe,
    )
    # En la hoja 3 original, AI refiere por error a AC (longitud del muro vertical)
    rm_sumer_length = wall_length if strict_excel else 10.0 * rm_sumer_depth

    rm_field_depth = np.where(
        h0_a < 15.0, safety_factor * (-0.033 * h0_a + 0.7) * h0_a, 3.0 * safety_factor
    )
    rm_field_length = 10.0 * rm_field_depth

    return StructureTipWavesResult(
        d_star=_as_out(d_star, is_scalar),
        ucr_c=_as_out(ucr_c, is_scalar),
        ucr_w=_as_out(ucr_w, is_scalar),
        kh=_as_out(kh, is_scalar),
        wavelength=_as_out(wavelength, is_scalar),
        wavelength_0=_as_out(wavelength_0, is_scalar),
        u_b=_as_out(u_b, is_scalar),
        theta_toe_deg=_as_out(theta_toe_deg, is_scalar),
        H_toe=_as_out(h_toe, is_scalar),
        kc_number=_as_out(kc, is_scalar),
        wall_sumer_depth=_as_out(wall_depth, is_scalar),
        wall_sumer_length=_as_out(wall_length, is_scalar),
        rubblemound_sumer_depth=_as_out(rm_sumer_depth, is_scalar),
        rubblemound_sumer_length=_as_out(rm_sumer_length, is_scalar),
        rubblemound_field_depth=_as_out(rm_field_depth, is_scalar),
        rubblemound_field_length=_as_out(rm_field_length, is_scalar),
    )


def scour_structure_tip_current_waves(
    h0: ArrayLike = 8.0,
    d50: ArrayLike = 0.0002,
    d90: ArrayLike = 0.0005,
    structure_length: ArrayLike = 100.0,
    structure_width: ArrayLike = 10.0,
    Hs0: ArrayLike = 2.0,
    theta0_deg: ArrayLike = 30.0,
    Tp: ArrayLike = 8.0,
    gamma_br: ArrayLike = 0.6,
    *,
    rho_w: float = RHO_SEA_WATER,
    rho_s: float = RHO_SEDIMENT,
    nu: float = KINEMATIC_VISCOSITY_20C,
    safety_factor: float = 1.0,
    gravity: float = GRAVITY,
    strict_excel: bool = False,
) -> StructureTipCurrentWavesResult:
    """Socavacion en morro con corriente fuerte >0.3 m/s mas oleaje (hoja ``4``)."""
    _ = structure_length  # Columna de entrada informativa en la hoja 4
    h0_a, d50_a, d90_a, b_a, Hs0_a, th0_a, Tp_a, gam_a = np.broadcast_arrays(
        np.asarray(h0, dtype=np.float64),
        np.asarray(d50, dtype=np.float64),
        np.asarray(d90, dtype=np.float64),
        np.asarray(structure_width, dtype=np.float64),
        np.asarray(Hs0, dtype=np.float64),
        np.asarray(theta0_deg, dtype=np.float64),
        np.asarray(Tp, dtype=np.float64),
        np.asarray(gamma_br, dtype=np.float64),
    )
    is_scalar = h0_a.ndim == 0

    zero_u = np.zeros_like(h0_a)
    d_star, kh, wavelength, wavelength_0, u_b, _, ucr_c, ucr_w, _, _ = (
        _compute_wave_and_critical_velocities(
            h0_a,
            d50_a,
            d90_a,
            zero_u,
            Hs0_a,
            Tp_a,
            rho_w=rho_w,
            rho_s=rho_s,
            nu=nu,
            gravity=gravity,
            corr_ucr=1.0,
            apply_corr_to_ucr_c=False,
            apply_corr_to_ucr_cw=False,
            strict_excel=strict_excel,
        )
    )
    theta_toe_deg, h_toe = _toe_wave_transformation(
        h0_a, Hs0_a, th0_a, gam_a, kh, wavelength, wavelength_0, strict_excel=strict_excel
    )

    kc = u_b * Tp_a / np.maximum(b_a, 1e-30)
    rm_depth = np.where(h0_a < 14.0, safety_factor * (-0.2 * h0_a + 3.8) * h0_a, safety_factor * 14.0)
    rm_length = 10.0 * rm_depth

    return StructureTipCurrentWavesResult(
        d_star=_as_out(d_star, is_scalar),
        ucr_c=_as_out(ucr_c, is_scalar),
        ucr_w=_as_out(ucr_w, is_scalar),
        kh=_as_out(kh, is_scalar),
        wavelength=_as_out(wavelength, is_scalar),
        wavelength_0=_as_out(wavelength_0, is_scalar),
        u_b=_as_out(u_b, is_scalar),
        theta_toe_deg=_as_out(theta_toe_deg, is_scalar),
        H_toe=_as_out(h_toe, is_scalar),
        kc_number=_as_out(kc, is_scalar),
        rubblemound_depth=_as_out(rm_depth, is_scalar),
        rubblemound_length=_as_out(rm_length, is_scalar),
    )


def scour_river_groyne(
    h0: ArrayLike = 5.0,
    groyne_length: ArrayLike = 30.0,
    d50: ArrayLike = 0.00025,
    d90: ArrayLike = 0.0005,
    mud_fraction: ArrayLike = 0.05,
    turbulence_r: ArrayLike = 0.15,
    Hs: ArrayLike = 1.0,
    Tp: ArrayLike = 7.0,
    *,
    Q: ArrayLike = 1000.0,
    B_river: ArrayLike = 200.0,
    rho_w: float = RHO_SEA_WATER,
    rho_s: float = RHO_SEDIMENT,
    nu: float = KINEMATIC_VISCOSITY_20C,
    safety_factor: float = 1.0,
    gravity: float = GRAVITY,
    strict_excel: bool = False,
) -> RiverGroyneScourResult:
    """Socavacion en el extremo de un espigon en rio o rio mareal (hoja ``5``)."""
    h0_a, lg_a, d50_a, d90_a, mud_a, r_a, Hs_a, Tp_a, q_a, b_a = np.broadcast_arrays(
        np.asarray(h0, dtype=np.float64),
        np.asarray(groyne_length, dtype=np.float64),
        np.asarray(d50, dtype=np.float64),
        np.asarray(d90, dtype=np.float64),
        np.asarray(mud_fraction, dtype=np.float64),
        np.asarray(turbulence_r, dtype=np.float64),
        np.asarray(Hs, dtype=np.float64),
        np.asarray(Tp, dtype=np.float64),
        np.asarray(Q, dtype=np.float64),
        np.asarray(B_river, dtype=np.float64),
    )
    is_scalar = h0_a.ndim == 0

    u_a = q_a / np.maximum(b_a * h0_a, 1e-30)
    blocking = lg_a / np.maximum(b_a, 1e-30)

    d_star, kh, wavelength, _, u_b, u_eff, ucr_c, ucr_w, ucr_cw, _ = (
        _compute_wave_and_critical_velocities(
            h0_a,
            d50_a,
            d90_a,
            u_a,
            Hs_a,
            Tp_a,
            rho_w=rho_w,
            rho_s=rho_s,
            nu=nu,
            gravity=gravity,
            corr_ucr=1.0,
            apply_corr_to_ucr_c=False,
            apply_corr_to_ucr_cw=False,
            p_mud=mud_a,
            strict_excel=strict_excel,
        )
    )

    flow_int = u_a / np.maximum(ucr_cw, 1e-30)
    s_vel = np.minimum(1.0, flow_int)
    wave_corr1 = (1.0 + u_b / np.maximum(u_a, 1e-30)) ** 0.25
    wave_corr2 = (1.0 + u_b / np.maximum(u_a, 1e-30)) ** 0.5

    exp_23 = 0.6666 if strict_excel else (2.0 / 3.0)
    exp_13_b = 0.33333 if strict_excel else (1.0 / 3.0)
    exp_13_c = 0.3333 if strict_excel else (1.0 / 3.0)

    lac_core = q_a / (56.0 * (d50_a**0.5) * (h0_a**3))
    depth_7a = np.maximum(
        0.0,
        (2.0 * (q_a / np.maximum(b_a - lg_a, 1e-30)) ** exp_23 - h0_a) * safety_factor * wave_corr1,
    )
    depth_lacey = np.maximum(
        0.0,
        (0.47 * 3.0 * h0_a * (lac_core**exp_13_b) - h0_a) * safety_factor * wave_corr1,
    )
    depth_rh = np.maximum(
        0.0,
        (
            0.47
            * h0_a
            * ((1.0 + 1.5 * lg_a / np.maximum(h0_a, 1e-30)) ** exp_13_b)
            * (lac_core**exp_13_c)
            - h0_a
        )
        * safety_factor
        * wave_corr1,
    )
    depth_clm = np.where(
        (h0_a / np.maximum(lg_a, 1e-30)) < 0.04,
        safety_factor * 10.0 * h0_a * flow_int * wave_corr1,
        safety_factor * 2.0 * np.sqrt(np.maximum(h0_a * lg_a, 0.0)) * s_vel * wave_corr2,
    )
    depth_89 = np.where(
        (lg_a / np.maximum(h0_a, 1e-30)) < 11.0,
        safety_factor * (1.0 + r_a) * (0.2 * (lg_a / np.maximum(h0_a, 1e-30)) + 0.8) * h0_a * wave_corr1,
        3.0 * safety_factor * h0_a * (1.0 + r_a) * wave_corr1,
    )
    scour_len = 10.0 * depth_89

    return RiverGroyneScourResult(
        u=_as_out(u_a, is_scalar),
        blocking_coeff=_as_out(blocking, is_scalar),
        d_star=_as_out(d_star, is_scalar),
        u_eff=_as_out(u_eff, is_scalar),
        ucr_c=_as_out(ucr_c, is_scalar),
        ucr_w=_as_out(ucr_w, is_scalar),
        ucr_cw=_as_out(ucr_cw, is_scalar),
        flow_intensity=_as_out(flow_int, is_scalar),
        kh=_as_out(kh, is_scalar),
        wavelength=_as_out(wavelength, is_scalar),
        u_b=_as_out(u_b, is_scalar),
        depth_F9_6_7a=_as_out(depth_7a, is_scalar),
        depth_lacey=_as_out(depth_lacey, is_scalar),
        depth_rh=_as_out(depth_rh, is_scalar),
        depth_clm=_as_out(depth_clm, is_scalar),
        depth_F9_6_8_9=_as_out(depth_89, is_scalar),
        scour_length=_as_out(scour_len, is_scalar),
    )


def scour_bridge_pier(
    h0: ArrayLike = 5.0,
    pier_width: ArrayLike = 5.0,
    d50: ArrayLike = 0.0002,
    d90: ArrayLike = 0.0005,
    alpha3: ArrayLike = 1.3,
    alpha4: ArrayLike = 1.0,
    turbulence_r: ArrayLike = 0.15,
    Hs: ArrayLike = 0.0,
    Tp: ArrayLike = 5.0,
    *,
    Q: ArrayLike = 10000.0,
    B_river: ArrayLike = 1000.0,
    u: ArrayLike | None = None,
    rho_w: float = RHO_SEA_WATER,
    rho_s: float = RHO_SEDIMENT,
    nu: float = KINEMATIC_VISCOSITY_20C,
    cal_transport: float = 1.0,
    corr_ucr: float = 1.0,
    safety_factor: float = 1.0,
    gravity: float = GRAVITY,
    strict_excel: bool = False,
) -> BridgePierScourResult:
    """Socavacion junto a pila de puente vertical con o sin oleaje (hoja ``6``).

    Si se omite ``u``, la velocidad media se obtiene de ``Q / (B_river * h0)``.
    """
    if u is None:
        h0_a, D_a, d50_a, d90_a, a3_a, a4_a, r_a, Hs_a, Tp_a, q_a, b_a = np.broadcast_arrays(
            np.asarray(h0, dtype=np.float64),
            np.asarray(pier_width, dtype=np.float64),
            np.asarray(d50, dtype=np.float64),
            np.asarray(d90, dtype=np.float64),
            np.asarray(alpha3, dtype=np.float64),
            np.asarray(alpha4, dtype=np.float64),
            np.asarray(turbulence_r, dtype=np.float64),
            np.asarray(Hs, dtype=np.float64),
            np.asarray(Tp, dtype=np.float64),
            np.asarray(Q, dtype=np.float64),
            np.asarray(B_river, dtype=np.float64),
        )
        u_a = q_a / np.maximum(b_a * h0_a, 1e-30)
    else:
        h0_a, u_a, D_a, d50_a, d90_a, a3_a, a4_a, r_a, Hs_a, Tp_a = np.broadcast_arrays(
            np.asarray(h0, dtype=np.float64),
            np.asarray(u, dtype=np.float64),
            np.asarray(pier_width, dtype=np.float64),
            np.asarray(d50, dtype=np.float64),
            np.asarray(d90, dtype=np.float64),
            np.asarray(alpha3, dtype=np.float64),
            np.asarray(alpha4, dtype=np.float64),
            np.asarray(turbulence_r, dtype=np.float64),
            np.asarray(Hs, dtype=np.float64),
            np.asarray(Tp, dtype=np.float64),
        )
    is_scalar = h0_a.ndim == 0

    d_star, kh, wavelength, _, u_b, u_eff, ucr_c, ucr_w, ucr_cw, _ = (
        _compute_wave_and_critical_velocities(
            h0_a,
            d50_a,
            d90_a,
            u_a,
            Hs_a,
            Tp_a,
            rho_w=rho_w,
            rho_s=rho_s,
            nu=nu,
            gravity=gravity,
            corr_ucr=corr_ucr,
            apply_corr_to_ucr_c=True,
            apply_corr_to_ucr_cw=True,
            strict_excel=strict_excel,
        )
    )
    qb, qs, qt = _compute_bed_and_suspended_transport(
        h0_a,
        u_a,
        d50_a,
        d_star,
        u_eff,
        ucr_cw,
        rho_w=rho_w,
        rho_s=rho_s,
        gravity=gravity,
        cal_transport=cal_transport,
    )

    vel_ratio = u_eff / np.maximum(ucr_cw, 1e-30)
    alpha1 = np.minimum(np.where(vel_ratio > 0.5, 2.0 * (vel_ratio - 0.5), 0.0), 1.0)
    alpha2 = np.maximum(2.0 * np.tanh(h0_a / np.maximum(D_a, 1e-30)), 1.5)
    alpha5 = 1.0 + r_a
    alpha6 = np.maximum(np.sqrt(u_eff / np.maximum(u_a, 1e-30)), 1.0)

    ds = safety_factor * alpha1 * alpha2 * a4_a * a3_a * alpha5 * alpha6 * D_a
    ls = 10.0 * ds

    return BridgePierScourResult(
        u=_as_out(u_a, is_scalar),
        alpha1=_as_out(alpha1, is_scalar),
        alpha2=_as_out(alpha2, is_scalar),
        alpha5=_as_out(alpha5, is_scalar),
        alpha6=_as_out(alpha6, is_scalar),
        d_star=_as_out(d_star, is_scalar),
        u_eff=_as_out(u_eff, is_scalar),
        ucr_c=_as_out(ucr_c, is_scalar),
        ucr_w=_as_out(ucr_w, is_scalar),
        ucr_cw=_as_out(ucr_cw, is_scalar),
        kh=_as_out(kh, is_scalar),
        wavelength=_as_out(wavelength, is_scalar),
        u_b=_as_out(u_b, is_scalar),
        qb=_as_out(qb, is_scalar),
        qs=_as_out(qs, is_scalar),
        qt=_as_out(qt, is_scalar),
        scour_depth=_as_out(ds, is_scalar),
        scour_length=_as_out(ls, is_scalar),
    )


def scour_vertical_pile_waves(
    u: ArrayLike = 0.3,
    h0: ArrayLike = 5.0,
    diameter: ArrayLike = 1.0,
    d50: ArrayLike = 0.00025,
    d90: ArrayLike = 0.0005,
    alpha1: ArrayLike = 1.0,
    group_factor: ArrayLike = 2.0,
    Hs: ArrayLike = 1.0,
    Tp: ArrayLike = 5.0,
    *,
    rho_w: float = RHO_SEA_WATER,
    rho_s: float = RHO_SEDIMENT,
    nu: float = KINEMATIC_VISCOSITY_20C,
    cal_transport: float = 1.0,
    corr_ucr: float = 1.0,
    safety_factor: float = 2.0,
    gravity: float = GRAVITY,
    strict_excel: bool = False,
) -> VerticalPileWavesResult:
    """Socavacion junto a pilotes o pilas verticales en oleaje con corriente debil (hoja ``7``)."""
    u_a, h0_a, D_a, d50_a, d90_a, a1_a, gf_a, Hs_a, Tp_a = np.broadcast_arrays(
        np.asarray(u, dtype=np.float64),
        np.asarray(h0, dtype=np.float64),
        np.asarray(diameter, dtype=np.float64),
        np.asarray(d50, dtype=np.float64),
        np.asarray(d90, dtype=np.float64),
        np.asarray(alpha1, dtype=np.float64),
        np.asarray(group_factor, dtype=np.float64),
        np.asarray(Hs, dtype=np.float64),
        np.asarray(Tp, dtype=np.float64),
    )
    is_scalar = h0_a.ndim == 0

    d_star, kh, wavelength, _, u_b, u_eff, ucr_c, ucr_w, ucr_cw, _ = (
        _compute_wave_and_critical_velocities(
            h0_a,
            d50_a,
            d90_a,
            u_a,
            Hs_a,
            Tp_a,
            rho_w=rho_w,
            rho_s=rho_s,
            nu=nu,
            gravity=gravity,
            corr_ucr=corr_ucr,
            apply_corr_to_ucr_c=True,
            apply_corr_to_ucr_cw=True,
            strict_excel=strict_excel,
        )
    )
    qb, qs, qt = _compute_bed_and_suspended_transport(
        h0_a,
        u_a,
        d50_a,
        d_star,
        u_eff,
        ucr_cw,
        rho_w=rho_w,
        rho_s=rho_s,
        gravity=gravity,
        cal_transport=cal_transport,
    )

    kc = (u_b + u_a) * Tp_a / np.maximum(D_a, 1e-30)
    alpha2 = np.where(u_a > ucr_cw, np.sqrt(u_a / np.maximum(ucr_cw, 1e-30)), 1.0)
    ds = np.maximum(
        np.where(kc < 100.0, 0.015 * safety_factor * kc * a1_a * gf_a * alpha2 * D_a, 1.5 * D_a),
        0.1 * D_a,
    )
    ls = 10.0 * ds

    return VerticalPileWavesResult(
        alpha2=_as_out(alpha2, is_scalar),
        d_star=_as_out(d_star, is_scalar),
        u_eff=_as_out(u_eff, is_scalar),
        ucr_c=_as_out(ucr_c, is_scalar),
        ucr_w=_as_out(ucr_w, is_scalar),
        ucr_cw=_as_out(ucr_cw, is_scalar),
        kh=_as_out(kh, is_scalar),
        wavelength=_as_out(wavelength, is_scalar),
        u_b=_as_out(u_b, is_scalar),
        kc_number=_as_out(kc, is_scalar),
        qb=_as_out(qb, is_scalar),
        qs=_as_out(qs, is_scalar),
        qt=_as_out(qt, is_scalar),
        scour_depth=_as_out(ds, is_scalar),
        scour_length=_as_out(ls, is_scalar),
    )


def scour_horizontal_pipe(
    u: ArrayLike = 0.3,
    h0: ArrayLike = 5.0,
    diameter: ArrayLike = 1.0,
    d50: ArrayLike = 0.00025,
    d90: ArrayLike = 0.0005,
    Hs: ArrayLike = 1.0,
    Tp: ArrayLike = 5.0,
    *,
    rho_w: float = RHO_SEA_WATER,
    rho_s: float = RHO_SEDIMENT,
    nu: float = KINEMATIC_VISCOSITY_20C,
    cal_transport: float = 1.0,
    corr_ucr: float = 1.0,
    safety_factor: float = 2.0,
    gravity: float = GRAVITY,
    strict_excel: bool = False,
) -> HorizontalPipeScourResult:
    """Socavacion bajo tuberia horizontal apoyada sobre el lecho (hoja ``8``)."""
    u_a, h0_a, D_a, d50_a, d90_a, Hs_a, Tp_a = np.broadcast_arrays(
        np.asarray(u, dtype=np.float64),
        np.asarray(h0, dtype=np.float64),
        np.asarray(diameter, dtype=np.float64),
        np.asarray(d50, dtype=np.float64),
        np.asarray(d90, dtype=np.float64),
        np.asarray(Hs, dtype=np.float64),
        np.asarray(Tp, dtype=np.float64),
    )
    is_scalar = h0_a.ndim == 0

    d_star, kh, wavelength, _, u_b, u_eff, ucr_c, ucr_w, ucr_cw, _ = (
        _compute_wave_and_critical_velocities(
            h0_a,
            d50_a,
            d90_a,
            u_a,
            Hs_a,
            Tp_a,
            rho_w=rho_w,
            rho_s=rho_s,
            nu=nu,
            gravity=gravity,
            corr_ucr=corr_ucr,
            apply_corr_to_ucr_c=True,
            apply_corr_to_ucr_cw=True,
            strict_excel=strict_excel,
        )
    )
    qb, qs, qt = _compute_bed_and_suspended_transport(
        h0_a,
        u_a,
        d50_a,
        d_star,
        u_eff,
        ucr_cw,
        rho_w=rho_w,
        rho_s=rho_s,
        gravity=gravity,
        cal_transport=cal_transport,
    )

    kc = (u_b + u_a) * Tp_a / np.maximum(D_a, 1e-30)
    alpha = np.where(u_a > ucr_cw, np.sqrt(u_a / np.maximum(ucr_cw, 1e-30)), 1.0)
    ds_curr = np.maximum(safety_factor * 1.0 * (D_a**0.8) * (u_a**0.4) / 1.8, 0.25 * D_a)
    ds_waves = 0.1 * safety_factor * np.sqrt(np.maximum(kc, 0.0)) * D_a
    ds_comb = alpha * ds_waves
    ls = 10.0 * ds_curr

    return HorizontalPipeScourResult(
        alpha=_as_out(alpha, is_scalar),
        d_star=_as_out(d_star, is_scalar),
        u_eff=_as_out(u_eff, is_scalar),
        ucr_c=_as_out(ucr_c, is_scalar),
        ucr_w=_as_out(ucr_w, is_scalar),
        ucr_cw=_as_out(ucr_cw, is_scalar),
        kh=_as_out(kh, is_scalar),
        wavelength=_as_out(wavelength, is_scalar),
        u_b=_as_out(u_b, is_scalar),
        kc_number=_as_out(kc, is_scalar),
        qb=_as_out(qb, is_scalar),
        qs=_as_out(qs, is_scalar),
        qt=_as_out(qt, is_scalar),
        scour_depth_current=_as_out(ds_curr, is_scalar),
        scour_depth_waves=_as_out(ds_waves, is_scalar),
        scour_depth_combined=_as_out(ds_comb, is_scalar),
        scour_length=_as_out(ls, is_scalar),
    )
