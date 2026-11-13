"""Transporte litoral longitudinal y erosion de dunas (``VOLVOL.xls`` y ``Littoral*.xls``).

Implementa las dos hojas principales de la familia litoral del paquete:
- ``Longsh tr. and dune erosion``: transporte longitudinal instantaneo (Kamphuis, CERC y
  van Rijn) y retroceso/volumen de erosion de dunas (Kriebel-Dean, Hallermeier-Rhodes y
  van Rijn tras 5 horas y tras ``t`` horas).
- ``annual longshore transport``: integracion anual (neto, bruto y por sentido positivo y
  negativo) sobre un clima de oleaje definido respecto al Norte.
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
    "AnnualLongshoreResult",
    "LongshoreDuneResult",
    "annual_longshore_transport",
    "dune_erosion_hallermeier_rhodes",
    "dune_erosion_kriebel_dean",
    "dune_erosion_van_rijn",
    "longshore_and_dune_erosion",
    "wrap_wave_angle_to_normal",
]


@dataclass(frozen=True)
class LongshoreDuneResult:
    """Resultados de transporte longitudinal instantaneo y erosion de dunas."""

    d_eff: NDArray[np.float64] | float
    ws: NDArray[np.float64] | float
    L0: NDArray[np.float64] | float
    c0: NDArray[np.float64] | float
    h_br: NDArray[np.float64] | float
    Hs_br: NDArray[np.float64] | float
    L_br: NDArray[np.float64] | float
    theta_br_deg: NDArray[np.float64] | float
    sin_2theta_br: NDArray[np.float64] | float
    v_longshore: NDArray[np.float64] | float
    v_eff: NDArray[np.float64] | float
    k_swell: NDArray[np.float64] | float
    k_grain: NDArray[np.float64] | float
    k_slope: NDArray[np.float64] | float
    qt_kamphuis_kg_s: NDArray[np.float64] | float
    qt_cerc_kg_s: NDArray[np.float64] | float
    qt_van_rijn_kg_s: NDArray[np.float64] | float
    kd_A_param: NDArray[np.float64] | float
    kd_surf_width: NDArray[np.float64] | float
    kd_recession: NDArray[np.float64] | float
    kd_volume: NDArray[np.float64] | float
    hr_recession: NDArray[np.float64] | float
    hr_volume: NDArray[np.float64] | float
    vr_volume_5h: NDArray[np.float64] | float
    vr_mean_recession_5h: NDArray[np.float64] | float
    vr_max_recession_5h: NDArray[np.float64] | float
    vr_volume_t: NDArray[np.float64] | float
    vr_mean_recession_t: NDArray[np.float64] | float
    vr_max_recession_t: NDArray[np.float64] | float


@dataclass(frozen=True)
class AnnualLongshoreResult:
    """Resultados de transporte longitudinal anual acumulado por sector/clase de oleaje."""

    angle_diff_deg: NDArray[np.float64]
    angle_to_normal_deg: NDArray[np.float64]
    onshore_mask: NDArray[np.bool_]
    L0: NDArray[np.float64]
    c0: NDArray[np.float64]
    h_br: NDArray[np.float64]
    Hs_br: NDArray[np.float64]
    L_br: NDArray[np.float64]
    theta_br_deg: NDArray[np.float64]
    v_longshore: NDArray[np.float64]
    v_eff: NDArray[np.float64]
    kamphuis_signed_m3: NDArray[np.float64]
    cerc_signed_m3: NDArray[np.float64]
    van_rijn_signed_m3: NDArray[np.float64]
    kamphuis_abs_m3: NDArray[np.float64]
    cerc_abs_m3: NDArray[np.float64]
    van_rijn_abs_m3: NDArray[np.float64]
    total_days: float
    kamphuis_net_m3_yr: float
    cerc_net_m3_yr: float
    van_rijn_net_m3_yr: float
    kamphuis_gross_m3_yr: float
    cerc_gross_m3_yr: float
    van_rijn_gross_m3_yr: float
    kamphuis_negative_m3_yr: float
    cerc_negative_m3_yr: float
    van_rijn_negative_m3_yr: float
    kamphuis_positive_m3_yr: float
    cerc_positive_m3_yr: float
    van_rijn_positive_m3_yr: float


def _as_out(arr: NDArray[np.float64], is_scalar: bool) -> NDArray[np.float64] | float:
    return float(arr) if is_scalar else arr


def _hunt_wavelength(
    depth: NDArray[np.float64],
    period: NDArray[np.float64],
    two_pi: float,
) -> NDArray[np.float64]:
    """Aproximacion de Hunt de las planillas litorales (``x = 4.02*h/T^2``)."""
    x = np.where((depth > 0.0) & (period > 0.0), 4.02 * depth / np.maximum(period, 1e-12) ** 2, 0.0)
    poly = (
        1.0
        + 0.666 * x
        + 0.355 * x**2
        + 0.161 * x**3
        + 0.0632 * x**4
        + 0.0218 * x**5
        + 0.00654 * x**6
    )
    kh = np.sqrt(x**2 + x / poly)
    return np.where((depth > 0.0) & (kh > 0.0), two_pi * depth / np.maximum(kh, 1e-30), 0.0)


def _settling_velocity_van_rijn(
    d50: NDArray[np.float64],
    s_rel: float,
    gravity: float,
    nu: float,
) -> NDArray[np.float64]:
    """Velocidad de caida de arena/grava (van Rijn 1993) usada en la columna O."""
    return np.where(
        d50 < 0.0001,
        s_rel * gravity * d50**2 / (18.0 * nu),
        np.where(
            d50 > 0.001,
            1.1 * np.sqrt(np.maximum(s_rel * gravity * d50, 0.0)),
            (10.0 * nu / d50) * (np.sqrt(1.0 + 0.01 * s_rel * gravity * d50**3 / nu**2) - 1.0),
        ),
    )


def wrap_wave_angle_to_normal(
    wave_angle_north: ArrayLike,
    shore_normal_north: ArrayLike,
) -> tuple[NDArray[np.float64] | float, NDArray[np.float64] | float]:
    """Convierte el rumbo de propagacion respecto al Norte a angulo respecto a la normal.

    Las olas con incidencia mayor o igual a 89 grados respecto a la normal costera se
    marcan con ``180.0`` (propagacion hacia mar abierto, sin aporte longitudinal en la costa).
    """
    wa = np.asarray(wave_angle_north, dtype=np.float64)
    sn = np.asarray(shore_normal_north, dtype=np.float64)
    wa_b, sn_b = np.broadcast_arrays(wa, sn)
    is_scalar = wa_b.ndim == 0

    diff = wa_b - sn_b
    m = np.where(diff >= 89.0, np.where(diff <= 271.0, 180.0, -360.0 + diff), diff)
    theta_rel = np.where(m <= -89.0, np.where(m >= -271.0, 180.0, 360.0 + m), m)
    return _as_out(diff, is_scalar), _as_out(theta_rel, is_scalar)


def _breaker_and_longshore_core(
    depth_offshore: NDArray[np.float64],
    d50: NDArray[np.float64],
    tan_beta: NDArray[np.float64],
    gamma_br: NDArray[np.float64],
    hs_o: NDArray[np.float64],
    period: NDArray[np.float64],
    swell_period: NDArray[np.float64],
    theta_rel_deg: NDArray[np.float64],
    v_tide: NDArray[np.float64],
    *,
    rho_bulk: float,
    p_mud: float,
    k_cal: float,
    gravity: float,
    strict_excel: bool,
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
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    two_pi = TWO_PI_EXCEL if strict_excel else 2.0 * PI_STANDARD
    pi_trig = 3.141 if strict_excel else PI_STANDARD
    pi_snell = 3.1416 if strict_excel else PI_STANDARD
    sqrt_g = 3.13 if strict_excel else float(np.sqrt(gravity))

    l0 = _hunt_wavelength(depth_offshore, period, two_pi)
    c0 = np.where(period > 0.0, l0 / np.maximum(period, 1e-12), 0.0)

    onshore = theta_rel_deg <= 90.0 if strict_excel else np.abs(theta_rel_deg) <= 90.0
    cos_o = np.maximum(np.cos(theta_rel_deg / 180.0 * pi_trig), 0.0)
    h_br = np.where(
        onshore & (hs_o > 0.0),
        (hs_o**2 * c0 * cos_o / 1.8 / sqrt_g / gamma_br**2) ** 0.4,
        0.0,
    )
    hs_br = gamma_br * h_br
    l_br = np.where(h_br > 0.0, _hunt_wavelength(h_br, period, two_pi), 0.0)

    snell_arg = np.where(
        (l0 > 0.0) & (h_br > 0.0),
        np.clip(l_br / np.maximum(l0, 1e-30) * np.sin(theta_rel_deg / 180.0 * pi_snell), -1.0, 1.0),
        0.0,
    )
    theta_br_deg = np.arcsin(snell_arg) * 180.0 / pi_snell
    sin_2theta = np.sin(2.0 * theta_br_deg / 180.0 * pi_trig)

    vl = 0.3 * sqrt_g * np.sqrt(np.maximum(hs_br, 0.0)) * sin_2theta
    v_eff = vl + v_tide

    k_swell = swell_period / 6.0
    k_grain = np.where(d50 < 0.002, 0.0002 / d50, 0.001 / d50**0.75)
    k_slope = np.clip(np.sqrt(tan_beta / 0.01), 0.75, 1.25)

    mud_red = 1.0 - p_mud
    sign_kamp = np.where(theta_rel_deg > 0.0, 1.0, -1.0)
    qt_kamphuis = (
        sign_kamp
        * 3.8
        * k_cal
        * mud_red
        * period**1.5
        * d50 ** (-0.25)
        * tan_beta**0.75
        * hs_br**2
        * np.abs(sin_2theta) ** 0.6
    )
    qt_cerc = (
        k_cal
        * mud_red
        * 0.023
        * sqrt_g
        * gamma_br ** (-0.5)
        * hs_br**2.5
        * sin_2theta
        * rho_bulk
    )
    qt_van_rijn = (
        k_cal
        * mud_red
        * 42.0
        * k_swell
        * k_grain
        * k_slope
        * hs_br**2.5
        * v_eff
    )
    return (
        l0,
        c0,
        h_br,
        hs_br,
        l_br,
        theta_br_deg,
        sin_2theta,
        vl,
        v_eff,
        k_swell,
        k_grain,
        k_slope,
        qt_kamphuis,
        qt_cerc,
        qt_van_rijn,
    )


def dune_erosion_kriebel_dean(
    h_br: ArrayLike,
    ws: ArrayLike,
    hwl: ArrayLike,
    ssl: ArrayLike,
    dune_height: ArrayLike,
) -> tuple[
    NDArray[np.float64] | float,
    NDArray[np.float64] | float,
    NDArray[np.float64] | float,
    NDArray[np.float64] | float,
]:
    """Calcula el retroceso y volumen de erosion de dunas segun Kriebel y Dean (1993)."""
    hb_a, ws_a, hwl_a, ssl_a, b_a = np.broadcast_arrays(
        np.asarray(h_br, dtype=np.float64),
        np.asarray(ws, dtype=np.float64),
        np.asarray(hwl, dtype=np.float64),
        np.asarray(ssl, dtype=np.float64),
        np.asarray(dune_height, dtype=np.float64),
    )
    is_scalar = hb_a.ndim == 0
    a_param = 0.067 * (100.0 * ws_a) ** 0.44
    w_surf = (hb_a / a_param) ** 1.5
    s_tot = ssl_a + hwl_a
    recession = w_surf * s_tot / (b_a + hb_a - 0.5 * s_tot)
    volume = b_a * recession
    return (
        _as_out(a_param, is_scalar),
        _as_out(w_surf, is_scalar),
        _as_out(recession, is_scalar),
        _as_out(volume, is_scalar),
    )


def dune_erosion_hallermeier_rhodes(
    hwl: ArrayLike,
    ssl: ArrayLike,
    dune_height: ArrayLike,
) -> tuple[NDArray[np.float64] | float, NDArray[np.float64] | float]:
    """Calcula retroceso y volumen de erosion de dunas segun Hallermeier y Rhodes (1988)."""
    hwl_a, ssl_a, b_a = np.broadcast_arrays(
        np.asarray(hwl, dtype=np.float64),
        np.asarray(ssl, dtype=np.float64),
        np.asarray(dune_height, dtype=np.float64),
    )
    is_scalar = hwl_a.ndim == 0
    s_tot = ssl_a + hwl_a
    volume = np.where(s_tot > 3.0, 0.13 * s_tot**4.8, 2.03 * s_tot**2.3)
    recession = volume / b_a
    return _as_out(recession, is_scalar), _as_out(volume, is_scalar)


def dune_erosion_van_rijn(
    d50: ArrayLike,
    tan_beta: ArrayLike,
    hs_o: ArrayLike,
    period: ArrayLike,
    theta_deg: ArrayLike,
    ssl: ArrayLike,
    dune_height: ArrayLike,
    duration_hours: ArrayLike = 5.0,
    *,
    strict_excel: bool = False,
) -> tuple[
    NDArray[np.float64] | float,
    NDArray[np.float64] | float,
    NDArray[np.float64] | float,
    NDArray[np.float64] | float,
    NDArray[np.float64] | float,
    NDArray[np.float64] | float,
    NDArray[np.float64] | float,
]:
    """Calcula erosion de dunas segun van Rijn (2009) tras 5 horas y tras ``t`` horas.

    Parameters
    ----------
    strict_excel : bool, default=False
        Si es ``True``, reproduce la constante de pendiente ``0.0222`` en la rama
        ``ssl < 5`` y usa el angulo con signo ``1 + theta / 100``. Si es ``False``,
        unifica la constante ``0.022`` y usa ``abs(theta)`` para garantizar simetria.
    """
    (
        d50_a,
        tb_a,
        hso_a,
        tp_a,
        th_a,
        ssl_a,
        b_a,
        t_a,
    ) = np.broadcast_arrays(
        np.asarray(d50, dtype=np.float64),
        np.asarray(tan_beta, dtype=np.float64),
        np.asarray(hs_o, dtype=np.float64),
        np.asarray(period, dtype=np.float64),
        np.asarray(theta_deg, dtype=np.float64),
        np.asarray(ssl, dtype=np.float64),
        np.asarray(dune_height, dtype=np.float64),
        np.asarray(duration_hours, dtype=np.float64),
    )
    is_scalar = d50_a.ndim == 0

    d_eff = np.where((b_a - ssl_a) > 1.0, b_a - ssl_a, 1.0)
    slope_ref = np.where(ssl_a < 5.0, 0.0222, 0.022) if strict_excel else 0.022
    ssl_exp = np.where(ssl_a < 5.0, 1.3, 0.5)
    angle_arg = th_a if strict_excel else np.abs(th_a)

    vol_5h = (
        170.0
        * (0.000225 / d50_a) ** 1.3
        * (ssl_a / 5.0) ** ssl_exp
        * (hso_a / 7.6) ** 0.5
        * (tp_a / 12.0) ** 0.5
        * (tb_a / slope_ref) ** 0.3
        * (1.0 + angle_arg / 100.0) ** 0.5
    )
    r_mean_5h = vol_5h / d_eff
    r_max_5h = 1.5 * r_mean_5h

    vol_t = np.where(t_a < 5.0, vol_5h * (t_a / 5.0) ** 0.5, vol_5h * (t_a / 5.0) ** 0.2)
    r_mean_t = vol_t / d_eff
    r_max_t = 1.5 * r_mean_t

    return (
        _as_out(d_eff, is_scalar),
        _as_out(vol_5h, is_scalar),
        _as_out(r_mean_5h, is_scalar),
        _as_out(r_max_5h, is_scalar),
        _as_out(vol_t, is_scalar),
        _as_out(r_mean_t, is_scalar),
        _as_out(r_max_t, is_scalar),
    )


def longshore_and_dune_erosion(
    depth_offshore: ArrayLike = 30.0,
    d50: ArrayLike = 0.0002,
    tan_beta: ArrayLike = 0.01,
    gamma_br: ArrayLike = 0.6,
    wave_height_offshore: ArrayLike = 4.0,
    wave_period: ArrayLike = 12.0,
    swell_period: ArrayLike = 6.0,
    wave_angle_offshore: ArrayLike = 20.0,
    v_tide: ArrayLike = 0.0,
    hwl: ArrayLike = 1.0,
    ssl: ArrayLike = 3.0,
    dune_height: ArrayLike = 5.0,
    duration_hours: ArrayLike = 12.0,
    *,
    rho_w: float = RHO_SEA_WATER,
    rho_s: float = RHO_SEDIMENT,
    rho_bulk: float = 1600.0,
    p_mud: float = 0.05,
    nu: float = KINEMATIC_VISCOSITY_20C,
    k_cal: float = 1.0,
    gravity: float = GRAVITY,
    strict_excel: bool = False,
) -> LongshoreDuneResult:
    """Calcula transporte longitudinal y erosion de dunas para casos individuales.

    Implementa la hoja ``Longsh tr. and dune erosion`` de ``VOLVOL.xls`` y de los libros
    ``Littoral*.xls``. Admite entrada escalar o vectorial mediante NumPy broadcasting.
    """
    (
        ho_a,
        d50_a,
        tb_a,
        gam_a,
        hso_a,
        tp_a,
        tsw_a,
        th_a,
        vt_a,
        hwl_a,
        ssl_a,
        b_a,
        dur_a,
    ) = np.broadcast_arrays(
        np.asarray(depth_offshore, dtype=np.float64),
        np.asarray(d50, dtype=np.float64),
        np.asarray(tan_beta, dtype=np.float64),
        np.asarray(gamma_br, dtype=np.float64),
        np.asarray(wave_height_offshore, dtype=np.float64),
        np.asarray(wave_period, dtype=np.float64),
        np.asarray(swell_period, dtype=np.float64),
        np.asarray(wave_angle_offshore, dtype=np.float64),
        np.asarray(v_tide, dtype=np.float64),
        np.asarray(hwl, dtype=np.float64),
        np.asarray(ssl, dtype=np.float64),
        np.asarray(dune_height, dtype=np.float64),
        np.asarray(duration_hours, dtype=np.float64),
    )
    is_scalar = ho_a.ndim == 0

    s_rel = rho_s / rho_w - 1.0
    ws = _settling_velocity_van_rijn(d50_a, s_rel, gravity, nu)

    (
        l0,
        c0,
        h_br,
        hs_br,
        l_br,
        theta_br_deg,
        sin_2theta,
        vl,
        v_eff,
        k_swell,
        k_grain,
        k_slope,
        qt_kamphuis,
        qt_cerc,
        qt_van_rijn,
    ) = _breaker_and_longshore_core(
        ho_a,
        d50_a,
        tb_a,
        gam_a,
        hso_a,
        tp_a,
        tsw_a,
        th_a,
        vt_a,
        rho_bulk=rho_bulk,
        p_mud=p_mud,
        k_cal=k_cal,
        gravity=gravity,
        strict_excel=strict_excel,
    )

    kd_a, kd_w, kd_r, kd_v = dune_erosion_kriebel_dean(h_br, ws, hwl_a, ssl_a, b_a)
    hr_r, hr_v = dune_erosion_hallermeier_rhodes(hwl_a, ssl_a, b_a)
    d_eff, vr_v5, vr_rm5, vr_rx5, vr_vt, vr_rmt, vr_rxt = dune_erosion_van_rijn(
        d50_a,
        tb_a,
        hso_a,
        tp_a,
        th_a,
        ssl_a,
        b_a,
        dur_a,
        strict_excel=strict_excel,
    )

    return LongshoreDuneResult(
        d_eff=_as_out(np.asarray(d_eff, dtype=np.float64), is_scalar),
        ws=_as_out(ws, is_scalar),
        L0=_as_out(l0, is_scalar),
        c0=_as_out(c0, is_scalar),
        h_br=_as_out(h_br, is_scalar),
        Hs_br=_as_out(hs_br, is_scalar),
        L_br=_as_out(l_br, is_scalar),
        theta_br_deg=_as_out(theta_br_deg, is_scalar),
        sin_2theta_br=_as_out(sin_2theta, is_scalar),
        v_longshore=_as_out(vl, is_scalar),
        v_eff=_as_out(v_eff, is_scalar),
        k_swell=_as_out(k_swell, is_scalar),
        k_grain=_as_out(k_grain, is_scalar),
        k_slope=_as_out(k_slope, is_scalar),
        qt_kamphuis_kg_s=_as_out(qt_kamphuis, is_scalar),
        qt_cerc_kg_s=_as_out(qt_cerc, is_scalar),
        qt_van_rijn_kg_s=_as_out(qt_van_rijn, is_scalar),
        kd_A_param=_as_out(np.asarray(kd_a, dtype=np.float64), is_scalar),
        kd_surf_width=_as_out(np.asarray(kd_w, dtype=np.float64), is_scalar),
        kd_recession=_as_out(np.asarray(kd_r, dtype=np.float64), is_scalar),
        kd_volume=_as_out(np.asarray(kd_v, dtype=np.float64), is_scalar),
        hr_recession=_as_out(np.asarray(hr_r, dtype=np.float64), is_scalar),
        hr_volume=_as_out(np.asarray(hr_v, dtype=np.float64), is_scalar),
        vr_volume_5h=_as_out(np.asarray(vr_v5, dtype=np.float64), is_scalar),
        vr_mean_recession_5h=_as_out(np.asarray(vr_rm5, dtype=np.float64), is_scalar),
        vr_max_recession_5h=_as_out(np.asarray(vr_rx5, dtype=np.float64), is_scalar),
        vr_volume_t=_as_out(np.asarray(vr_vt, dtype=np.float64), is_scalar),
        vr_mean_recession_t=_as_out(np.asarray(vr_rmt, dtype=np.float64), is_scalar),
        vr_max_recession_t=_as_out(np.asarray(vr_rxt, dtype=np.float64), is_scalar),
    )


def annual_longshore_transport(
    days: ArrayLike,
    wave_height_offshore: ArrayLike,
    wave_period: ArrayLike,
    wave_angle_north: ArrayLike,
    *,
    shore_normal_north: ArrayLike = 270.0,
    depth_offshore: ArrayLike = 20.0,
    d50: ArrayLike = 0.0002,
    tan_beta: ArrayLike = 0.01,
    gamma_br: ArrayLike = 0.6,
    swell_period: ArrayLike = 6.0,
    v_tide_pos: float = 0.3,
    v_tide_neg: float = -0.2,
    rho_bulk: float = 1600.0,
    p_mud: float = 0.0,
    k_cal: float = 1.0,
    gravity: float = GRAVITY,
    strict_excel: bool = False,
) -> AnnualLongshoreResult:
    """Calcula el transporte longitudinal anual neto y bruto sobre un clima de oleaje.

    Implementa la hoja ``annual longshore transport`` (``VOLVOL.xls`` y
    ``Littoral*.xls``), devolviendo volumenes en m^3 incluyendo poros.
    """
    (
        days_a,
        hso_a,
        tp_a,
        wa_n,
        sn_n,
        ho_a,
        d50_a,
        tb_a,
        gam_a,
        tsw_a,
    ) = np.broadcast_arrays(
        np.asarray(days, dtype=np.float64),
        np.asarray(wave_height_offshore, dtype=np.float64),
        np.asarray(wave_period, dtype=np.float64),
        np.asarray(wave_angle_north, dtype=np.float64),
        np.asarray(shore_normal_north, dtype=np.float64),
        np.asarray(depth_offshore, dtype=np.float64),
        np.asarray(d50, dtype=np.float64),
        np.asarray(tan_beta, dtype=np.float64),
        np.asarray(gamma_br, dtype=np.float64),
        np.asarray(swell_period, dtype=np.float64),
    )
    if days_a.ndim == 0:
        days_a = days_a.reshape(1)
        hso_a = hso_a.reshape(1)
        tp_a = tp_a.reshape(1)
        wa_n = wa_n.reshape(1)
        sn_n = sn_n.reshape(1)
        ho_a = ho_a.reshape(1)
        d50_a = d50_a.reshape(1)
        tb_a = tb_a.reshape(1)
        gam_a = gam_a.reshape(1)
        tsw_a = tsw_a.reshape(1)

    angle_diff, theta_rel = wrap_wave_angle_to_normal(wa_n, sn_n)
    angle_diff_a = np.asarray(angle_diff, dtype=np.float64)
    theta_rel_a = np.asarray(theta_rel, dtype=np.float64)
    v_tide_mean = np.full_like(days_a, 0.5 * (v_tide_pos + v_tide_neg))

    (
        l0,
        c0,
        h_br,
        hs_br,
        l_br,
        theta_br_deg,
        _sin_2theta,
        vl,
        v_eff,
        _k_swell,
        _k_grain,
        _k_slope,
        qt_kamphuis,
        qt_cerc,
        qt_van_rijn,
    ) = _breaker_and_longshore_core(
        ho_a,
        d50_a,
        tb_a,
        gam_a,
        hso_a,
        tp_a,
        tsw_a,
        theta_rel_a,
        v_tide_mean,
        rho_bulk=rho_bulk,
        p_mud=p_mud,
        k_cal=k_cal,
        gravity=gravity,
        strict_excel=strict_excel,
    )

    sec_per_case = days_a * 86400.0 / rho_bulk
    kamp_m3 = qt_kamphuis * sec_per_case
    cerc_m3 = qt_cerc * sec_per_case
    vr_m3 = qt_van_rijn * sec_per_case

    kamp_abs = np.abs(kamp_m3)
    cerc_abs = np.abs(cerc_m3)
    vr_abs = np.abs(vr_m3)

    kamp_net = float(np.sum(kamp_m3))
    cerc_net = float(np.sum(cerc_m3))
    vr_net = float(np.sum(vr_m3))

    kamp_gross = float(np.sum(kamp_abs))
    cerc_gross = float(np.sum(cerc_abs))
    vr_gross = float(np.sum(vr_abs))

    return AnnualLongshoreResult(
        angle_diff_deg=angle_diff_a,
        angle_to_normal_deg=theta_rel_a,
        onshore_mask=theta_rel_a <= 90.0,
        L0=l0,
        c0=c0,
        h_br=h_br,
        Hs_br=hs_br,
        L_br=l_br,
        theta_br_deg=theta_br_deg,
        v_longshore=vl,
        v_eff=v_eff,
        kamphuis_signed_m3=kamp_m3,
        cerc_signed_m3=cerc_m3,
        van_rijn_signed_m3=vr_m3,
        kamphuis_abs_m3=kamp_abs,
        cerc_abs_m3=cerc_abs,
        van_rijn_abs_m3=vr_abs,
        total_days=float(np.sum(days_a)),
        kamphuis_net_m3_yr=kamp_net,
        cerc_net_m3_yr=cerc_net,
        van_rijn_net_m3_yr=vr_net,
        kamphuis_gross_m3_yr=kamp_gross,
        cerc_gross_m3_yr=cerc_gross,
        van_rijn_gross_m3_yr=vr_gross,
        kamphuis_negative_m3_yr=0.5 * (kamp_net - kamp_gross),
        cerc_negative_m3_yr=0.5 * (cerc_net - cerc_gross),
        van_rijn_negative_m3_yr=0.5 * (vr_net - vr_gross),
        kamphuis_positive_m3_yr=0.5 * (kamp_net + kamp_gross),
        cerc_positive_m3_yr=0.5 * (cerc_net + cerc_gross),
        van_rijn_positive_m3_yr=0.5 * (vr_net + vr_gross),
    )
