"""Transporte mareal acoplado de arena, lodo y corrientes de densidad.

Implementa las herramientas Excel:
- ``TIDAL SAND TRANSPORT incl mud.xls`` (transporte mareal de arena predominante con fraccion fina)
- ``TIDAL MUD TRANSPORT incl sand_withdensitycurrent.xls`` (transporte mareal de lodo con
  floculacion, amortiguacion por Richardson, transporte impulsado por densidad y fraccion de arena)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sedmor.constants import (
    GRAVITY,
    KINEMATIC_VISCOSITY_20C,
    PI_STANDARD,
    RHO_SEA_WATER,
    RHO_SEDIMENT,
    VON_KARMAN,
)

__all__ = [
    "TidalMudSandResult",
    "TidalSandMudResult",
    "tidal_mud_sand_transport",
    "tidal_sand_mud_transport",
]


@dataclass(frozen=True)
class TidalSandMudResult:
    """Resultados a lo largo de un ciclo mareal para lecho mixto arena-lodo (predominio arena)."""

    time: NDArray[np.float64]
    water_level: NDArray[np.float64]
    depth: NDArray[np.float64]
    velocity: NDArray[np.float64]
    chezy: NDArray[np.float64]
    u_star_cw: NDArray[np.float64]
    ca_sand_eq_mass: NDArray[np.float64]
    ca_sand_adj_mass: NDArray[np.float64]
    ca_mud_mass: NDArray[np.float64]
    bedload_sand: NDArray[np.float64]
    suspended_sand_formula_cap: NDArray[np.float64]
    suspended_sand_formula_adj: NDArray[np.float64]
    total_sand_formula: NDArray[np.float64]
    z_profiles: NDArray[np.float64]
    u_profiles: NDArray[np.float64]
    c_sand_profiles_mass: NDArray[np.float64]
    c_mud_profiles_mass: NDArray[np.float64]
    suspended_sand_profile_rate: NDArray[np.float64]
    suspended_mud_profile_rate: NDArray[np.float64]
    sand_formula_pos: float
    sand_formula_neg: float
    sand_formula_net: float
    sand_profile_pos: float
    sand_profile_neg: float
    sand_profile_net: float
    mud_profile_pos: float
    mud_profile_neg: float
    mud_profile_net: float


@dataclass(frozen=True)
class TidalMudSandResult:
    """Resultados mareales de transporte cohesivo de lodo, arena y densidad."""

    time: NDArray[np.float64]
    water_level: NDArray[np.float64]
    depth: NDArray[np.float64]
    velocity: NDArray[np.float64]
    chezy: NDArray[np.float64]
    u_star_cw: NDArray[np.float64]
    mixing_max: NDArray[np.float64]
    ca_mud_eq_mass: NDArray[np.float64]
    ca_mud_adj_mass: NDArray[np.float64]
    ca_sand_eq_mass: NDArray[np.float64]
    ca_sand_profile_mass: NDArray[np.float64]
    density_driven_transport: NDArray[np.float64]
    z_profiles: NDArray[np.float64]
    u_profiles: NDArray[np.float64]
    mixing_profiles: NDArray[np.float64]
    ws_mud_profiles: NDArray[np.float64]
    richardson_profiles: NDArray[np.float64]
    damping_profiles: NDArray[np.float64]
    c_mud_profiles_mass: NDArray[np.float64]
    c_sand_profiles_mass: NDArray[np.float64]
    suspended_mud_rate: NDArray[np.float64]
    suspended_sand_rate: NDArray[np.float64]
    mud_pos: float
    mud_neg: float
    mud_net: float
    sand_pos: float
    sand_neg: float
    sand_net: float
    density_net: float


def _log_velocity_profile(
    z: NDArray[np.float64],
    depth: float,
    velocity: float,
    ks_vel: float,
) -> NDArray[np.float64]:
    denom = -1.0 + np.log(depth / (0.033 * ks_vel))
    return (velocity / denom) * np.log(z / (0.033 * ks_vel))


def _pos_neg_split(
    series: NDArray[np.float64],
    dt: float,
    split_index: int | None,
) -> tuple[float, float, float]:
    if split_index is not None:
        pos = float(np.sum(series[:split_index]) * dt)
        neg = float(np.sum(series[split_index:]) * dt)
    else:
        pos = float(np.sum(series[series > 0.0]) * dt)
        neg = float(np.sum(series[series < 0.0]) * dt)
    return pos, neg, pos + neg


def _trapz_uniform_bottom(
    values: NDArray[np.float64],
    z: NDArray[np.float64],
    bottom_coef: float = 0.666,
) -> float:
    dz = z[1:] - z[:-1]
    return float(bottom_coef * values[0] * z[0] + 0.5 * np.sum((values[:-1] + values[1:]) * dz))


def tidal_sand_mud_transport(
    depth: float = 20.0,
    wave_height: float = 2.0,
    wave_period: float = 7.0,
    u_net: float = 0.1,
    u_peak: float = 1.0,
    tidal_amplitude: float = 1.0,
    tidal_period: float = 43200.0,
    *,
    p_sand: float = 0.7,
    d50: float = 0.00015,
    d90: float = 0.0005,
    ws_sand: float = 0.01,
    rho_bulk_sand: float = 1600.0,
    p_mud: float = 0.3,
    ws_mud: float = 0.0005,
    rho_bulk_mud: float = 1400.0,
    rho_s: float = RHO_SEDIMENT,
    rho_w: float = RHO_SEA_WATER,
    nu: float = KINEMATIC_VISCOSITY_20C,
    ks_c: float = 0.01,
    ks_w: float = 0.01,
    tau_cr_sand: float = 0.18,
    ref_level: float = 0.03,
    dt: float = 300.0,
    k_cr: float = 1.0,
    k_cal: float = 1.0,
    k_ca: float = 1.0,
    k_mud_shields: float = 1.0,
    k_adj: float = 1.0,
    gravity: float = GRAVITY,
    strict_excel: bool = False,
) -> TidalSandMudResult:
    """Simula un ciclo mareal completo de transporte de arena con influencia de lodo.

    Implementa ``TIDAL SAND TRANSPORT incl mud.xls`` integrando tanto las formulas
    profundidad-promediadas con relajacion temporal de concentracion de referencia como los
    perfiles verticales discretizados en 20 niveles logaritmicos.
    """
    n_steps = int(round(tidal_period / dt)) + 1
    n_z = 20
    time = dt * np.arange(n_steps, dtype=np.float64)

    exp_dstar = 0.3333333 if strict_excel else (1.0 / 3.0)
    s_rel = rho_s / rho_w - 1.0
    d_star = d50 * (s_rel * gravity / nu**2) ** exp_dstar

    two_pi_tidal = 2.0 * 3.141 if strict_excel else 2.0 * PI_STANDARD
    two_pi_wave = 6.28 if strict_excel else 2.0 * PI_STANDARD
    pi_orb = 3.142 if strict_excel else PI_STANDARD

    eta = tidal_amplitude * np.sin(two_pi_tidal * time / 43200.0)
    if strict_excel:
        eta[0] = 1.5 * np.sin(two_pi_tidal * time[0] / 43200.0)

    h = depth + eta
    u = (0.0001 if strict_excel else 0.0) + u_net + u_peak * np.sin(two_pi_tidal * time / tidal_period)

    chezy = 18.0 * np.log10(12.0 * h / ks_c)
    f_c = 8.0 * gravity / chezy**2
    tau_c = rho_w * gravity * u**2 / chezy**2
    u_star_c = np.sqrt(tau_c / rho_w)

    if wave_period > 0.0:
        xw = 4.02 * h / wave_period**2
        kh = np.sqrt(xw) * (1.0 + 0.166 * xw + 0.031 * xw**2)
        u_orb = pi_orb * wave_height / (wave_period * np.sinh(kh))
    else:
        kh = np.zeros_like(h)
        u_orb = np.zeros_like(h)

    if wave_height > 0.0 and wave_period > 0.0:
        f_w = np.minimum(0.3, np.exp(-6.0 + 5.2 * (u_orb * wave_period / two_pi_wave / ks_w) ** (-0.19)))
    else:
        f_w = np.zeros_like(h)

    tau_w = 0.25 * rho_w * f_w * u_orb**2
    u_star_w = np.sqrt(tau_w / rho_w)
    u_star_cw = np.sqrt(u_star_c**2 + u_star_w**2)
    u_eff = np.abs(u) + 0.4 * u_orb

    fc_grain = 0.24 / (np.log10(12.0 * h / d90)) ** 2
    mu_c = fc_grain / f_c
    mu_w = np.maximum(0.0625, 0.125 * (1.5 - wave_height / h))
    tau_eff_cw = mu_c * tau_c + mu_w * tau_w

    tau_cr_eff = (1.0 + p_mud) ** (k_mud_shields * 3.0) * tau_cr_sand
    t_par = np.maximum(0.001, (tau_eff_cw - tau_cr_eff) / tau_cr_eff)

    ca_sand_eq_vol = np.minimum(k_ca * 0.015 * p_sand * (d50 / ref_level) * t_par**1.5 / d_star**0.3, 0.1)
    ca_sand_eq_mass = rho_s * ca_sand_eq_vol

    adj_sand = np.minimum(
        1.0,
        k_adj * 0.05 * (1.0 / h) * (ws_sand / u_star_cw) * (1.0 + 2.0 * ws_sand / u_star_cw) * (1.0 + wave_height / h) ** 2,
    )

    ca_sand_adj_mass = np.zeros_like(ca_sand_eq_mass)
    ca_sand_adj_mass[0] = ca_sand_eq_mass[0]
    for it in range(1, n_steps):
        w = adj_sand[it] * dt
        ca_sand_adj_mass[it] = (ca_sand_adj_mass[it - 1] + w * ca_sand_eq_mass[it]) / (1.0 + w)
    ca_sand_adj_vol = ca_sand_adj_mass / rho_s

    if d50 < 0.0005:
        u_cr_c = 0.19 * (1.0 + p_mud) ** (k_mud_shields * 1.5) * d50**0.1 * np.log10(12.0 * h / (3.0 * d90))
        u_cr_w = 0.24 * (1.0 + p_mud) ** (k_mud_shields * 1.5) * (s_rel * gravity) ** 0.66 * d50**0.33 * wave_period**0.33
    else:
        u_cr_c = 8.5 * (1.0 + p_mud) ** 0.5 * d50**0.6 * np.log10(12.0 * h / (3.0 * d90))
        g_prime = (rho_s - (rho_w / rho_w) * gravity) if strict_excel else (s_rel * gravity)
        u_cr_w = 0.95 * (1.0 + p_mud) ** 0.5 * g_prime**0.57 * d50**0.43 * wave_period**0.14

    weight_c = np.abs(u) / (np.abs(u) + u_orb)
    u_cr_cw = k_cr * (weight_c * u_cr_c + (1.0 - weight_c) * u_cr_w)
    excess = np.maximum(u_eff - u_cr_cw, 0.0)

    qb = np.where(
        u_eff > u_cr_cw,
        k_cal * rho_s * 0.015 * (1.0 - p_mud) * h * u * (d50 / h) ** 1.2 * excess**1.5 / (s_rel * gravity * d50) ** 0.75,
        0.0,
    )
    qs_cap = np.where(
        u_eff > u_cr_cw,
        k_cal
        * rho_s
        * 0.008
        * (1.0 - p_mud)
        * h
        * u
        * (d50 / h)
        * excess**2.4
        / ((s_rel * gravity * d50) ** 1.2)
        * d_star ** (-0.6),
        0.0,
    )

    qs_adj = np.zeros_like(qs_cap)
    qs_adj[0] = qs_cap[0]
    for it in range(1, n_steps):
        w = adj_sand[it] * dt
        qs_adj[it] = (qs_adj[it - 1] + w * qs_cap[it]) / (1.0 + w)
    qt_formula = qs_adj + qb

    # Perfiles verticales discretizados en 20 niveles logaritmicos
    k_idx = np.arange(n_z, dtype=np.float64)[:, None]
    z_prof = ref_level * (h[None, :] / ref_level) ** (k_idx / 19.0)
    u_prof = np.zeros_like(z_prof)
    c_sand_vol = np.zeros_like(z_prof)
    c_mud_unity = np.zeros_like(z_prof)

    for it in range(n_steps):
        z_col = z_prof[:, it]
        u_prof[:, it] = _log_velocity_profile(z_col, h[it], u[it], ks_c)

        beta_mud = ws_mud / (VON_KARMAN * u_star_cw[it])
        base_r = np.maximum(((h[it] - z_col) / z_col) * (ref_level / (h[it] - ref_level)), 0.0)
        c_mud_unity[:, it] = base_r**beta_mud
        c_mud_unity[-1, it] = 1e-8 if (strict_excel and it > 0) else 0.0
        if strict_excel and it == 0:
            c_sand_vol[:-1, it] = 1e-7
            c_sand_vol[-1, it] = 1e-8
        else:
            beta_s = ws_sand / (min(1.5, 1.0 + 2.0 * (ws_sand / u_star_cw[it]) ** 2) * VON_KARMAN * u_star_cw[it]) + 2.5 * (
                ws_sand / u_star_cw[it]
            ) ** 0.8 * (ca_sand_adj_vol[it] / 0.65) ** 0.4
            c_sand_vol[:, it] = ca_sand_adj_vol[it] * (base_r**beta_s)
            c_sand_vol[-1, it] = 1e-8

    c_sand_mass = c_sand_vol * rho_s
    q_sand_prof_dz = u_prof * c_sand_mass
    if strict_excel:
        # Fila 253 en Excel multiplica u * c_vol en lugar de u * c_mass (omite rho_s en la superficie).
        q_sand_prof_dz[-1, :] = u_prof[-1, :] * c_sand_vol[-1, :]

    total_load_sand_kg_m2 = np.array(
        [rho_s * 0.5 * np.sum((c_sand_vol[:-1, it] + c_sand_vol[1:, it]) * (z_prof[1:, it] - z_prof[:-1, it])) for it in range(n_steps)],
        dtype=np.float64,
    )
    total_load_mud_unity = np.array(
        [0.5 * np.sum((c_mud_unity[:-1, it] + c_mud_unity[1:, it]) * (z_prof[1:, it] - z_prof[:-1, it])) for it in range(n_steps)],
        dtype=np.float64,
    )
    ca_mud_mass = np.maximum((p_mud / p_sand) * (rho_bulk_mud / rho_bulk_sand) * (total_load_sand_kg_m2 / total_load_mud_unity), 1e-9)
    c_mud_mass = c_mud_unity * ca_mud_mass[None, :]
    if strict_excel:
        c_mud_mass[-1, :] = 1e-8
    q_mud_prof_dz = u_prof * c_mud_mass

    qs_sand_prof = np.zeros(n_steps, dtype=np.float64)
    qs_mud_prof = np.zeros(n_steps, dtype=np.float64)
    for it in range(n_steps):
        z_col = z_prof[:, it]
        dz_sand = z_col[1:] - z_col[:-1]
        if strict_excel:
            # En la fila 91 de Excel el intervalo k=11 (B127-B126) apunta por error a B127-B125.
            dz_sand = dz_sand.copy()
            dz_sand[11] = z_col[12] - z_col[10]
        qs_sand_prof[it] = 0.666 * q_sand_prof_dz[0, it] * z_col[0] + 0.5 * np.sum(
            (q_sand_prof_dz[:-1, it] + q_sand_prof_dz[1:, it]) * dz_sand
        )
        qs_mud_prof[it] = _trapz_uniform_bottom(q_mud_prof_dz[:, it], z_col, bottom_coef=0.666)

    split_idx = 75 if strict_excel else None
    sf_pos, sf_neg, sf_net = _pos_neg_split(qt_formula, dt, split_idx)
    sp_pos, sp_neg, sp_net = _pos_neg_split(qs_sand_prof, dt, split_idx)
    mp_pos, mp_neg, mp_net = _pos_neg_split(qs_mud_prof, dt, split_idx)

    return TidalSandMudResult(
        time=time,
        water_level=eta,
        depth=h,
        velocity=u,
        chezy=chezy,
        u_star_cw=u_star_cw,
        ca_sand_eq_mass=ca_sand_eq_mass,
        ca_sand_adj_mass=ca_sand_adj_mass,
        ca_mud_mass=ca_mud_mass,
        bedload_sand=qb,
        suspended_sand_formula_cap=qs_cap,
        suspended_sand_formula_adj=qs_adj,
        total_sand_formula=qt_formula,
        z_profiles=z_prof,
        u_profiles=u_prof,
        c_sand_profiles_mass=c_sand_mass,
        c_mud_profiles_mass=c_mud_mass,
        suspended_sand_profile_rate=qs_sand_prof,
        suspended_mud_profile_rate=qs_mud_prof,
        sand_formula_pos=sf_pos,
        sand_formula_neg=sf_neg,
        sand_formula_net=sf_net,
        sand_profile_pos=sp_pos,
        sand_profile_neg=sp_neg,
        sand_profile_net=sp_net,
        mud_profile_pos=mp_pos,
        mud_profile_neg=mp_neg,
        mud_profile_net=mp_net,
    )


def _excel_coarse_mud_grid_indices() -> NDArray[np.intp]:
    """Indices sobre los 50 niveles verticales usados en las integrales de TIDAL MUD."""
    return np.array(list(range(0, 20)) + [24, 29, 34, 39, 44, 49], dtype=np.intp)


def _integrate_coarse_profile(
    values: NDArray[np.float64],
    z_col: NDArray[np.float64],
    *,
    bottom_coef: float,
    bug_idx11: bool,
) -> float:
    idx = _excel_coarse_mud_grid_indices()
    v_sub = values[idx]
    z_sub = z_col[idx]
    dz = z_sub[1:] - z_sub[:-1]
    if bug_idx11:
        # B118-B116 en lugar de B118-B117 (fila 82: intervalo entre indice 11 y 12)
        dz = dz.copy()
        dz[11] = z_sub[12] - z_sub[10]
    total = 0.5 * np.sum((v_sub[:-1] + v_sub[1:]) * dz)
    if bottom_coef > 0.0:
        total += bottom_coef * v_sub[0] * z_sub[0]
    return float(total)


def tidal_mud_sand_transport(
    depth: float = 10.0,
    wave_height: float = 0.0,
    wave_period: float = 7.0,
    u_net: float = 0.0,
    u_peak: float = 0.9,
    tidal_amplitude: float = 1.0,
    tidal_phase_deg: float = -90.0,
    tidal_period: float = 43200.0,
    *,
    p_mud: float = 0.7,
    p_sand: float = 0.3,
    ws_mud_mode: int = 1,
    ws_mud_max: float = 0.001,
    ws_mud_min: float = 0.0001,
    d50_sand: float = 0.0001,
    ws_sand: float = 0.01,
    tau_cr_sand: float = 0.16,
    rho_bulk_mud: float = 800.0,
    rho_bulk_sand: float = 800.0,
    rho_s: float = RHO_SEDIMENT,
    rho_w: float = RHO_SEA_WATER,
    nu: float = KINEMATIC_VISCOSITY_20C,
    ks_c: float = 0.01,
    ks_w: float = 0.01,
    ks_vel: float = 0.1,
    tau_cr_mud: float = 0.18,
    ref_level: float = 0.01,
    dt: float = 300.0,
    bg_conc_mud_mass: float = 0.1,
    scale_conc_mud: float = 6e-5,
    damping_coef: float = 0.0,
    k_adj_mud: float = 1.0,
    disp_h: float = 100.0,
    salinity: float = 25.0,
    salinity_gradient: float = -15.0 / 20000.0,
    alpha_density: float = 1.0,
    gravity: float = GRAVITY,
    strict_excel: bool = False,
) -> TidalMudSandResult:
    """Simula un ciclo mareal completo de transporte de lodo, corriente de densidad y arena.

    Implementa ``TIDAL MUD TRANSPORT incl sand_withdensitycurrent.xls``.
    """
    n_steps = int(round(tidal_period / dt)) + 1
    n_z = 50
    time = dt * np.arange(n_steps, dtype=np.float64)

    two_pi_tidal = 2.0 * 3.141 if strict_excel else 2.0 * PI_STANDARD
    two_pi_wave = 6.28 if strict_excel else 2.0 * PI_STANDARD
    pi_orb = 3.142 if strict_excel else PI_STANDARD
    phase_rad = tidal_phase_deg / 180.0 * PI_STANDARD

    eta = tidal_amplitude * np.sin(two_pi_tidal * time / 43200.0 + phase_rad)
    h = depth + eta
    u = (0.0001 if strict_excel else 0.0) + u_net + u_peak * np.sin(two_pi_tidal * time / tidal_period)

    chezy = 18.0 * np.log10(12.0 * h / ks_c)
    tau_c = rho_w * gravity * u**2 / chezy**2
    u_star_c = np.sqrt(tau_c / rho_w)

    if wave_period > 0.0:
        xw = 4.02 * h / wave_period**2
        kh = np.sqrt(xw) * (1.0 + 0.166 * xw + 0.031 * xw**2)
        u_orb = np.where(wave_height > 0.0, pi_orb * wave_height / (wave_period * np.sinh(kh)), 0.0)
    else:
        u_orb = np.zeros_like(h)

    if wave_height > 0.0 and wave_period > 0.0:
        f_w = np.minimum(0.3, np.exp(-6.0 + 5.2 * (u_orb * wave_period / two_pi_wave / ks_w) ** (-0.19)))
    else:
        f_w = np.zeros_like(h)

    tau_w = 0.25 * rho_w * f_w * u_orb**2
    u_star_w = np.sqrt(tau_w / rho_w)
    u_star_cw = np.sqrt(u_star_c**2 + u_star_w**2)
    mixing_max = 0.1 * u_star_cw * h

    t_mud = np.maximum((tau_c + tau_w - tau_cr_mud) / tau_cr_mud, 0.001)
    ca_mud_eq_vol = p_mud * np.maximum(np.minimum(bg_conc_mud_mass / rho_s + scale_conc_mud * t_mud**1.5, 0.1), 1.89e-6)
    ca_mud_eq_mass = rho_s * ca_mud_eq_vol

    u_scw_safe = np.maximum(u_star_cw, 1e-12)
    adj_mud = np.maximum(
        np.minimum(1.0, k_adj_mud * 0.1 * (1.0 / h) * (ws_mud_max / u_scw_safe) * (1.0 + 2.0 * ws_mud_max / u_scw_safe) * (1.0 + wave_height / h) ** 2),
        0.0001,
    )
    ca_mud_adj_vol = np.zeros_like(ca_mud_eq_vol)
    ca_mud_adj_vol[0] = ca_mud_eq_vol[0]
    for it in range(1, n_steps):
        w = adj_mud[it] * dt
        ca_mud_adj_vol[it] = (ca_mud_adj_vol[it - 1] + w * ca_mud_eq_vol[it]) / (1.0 + w)
    ca_mud_adj_mass = ca_mud_adj_vol * rho_s

    tau_eff_sand = tau_c + tau_w
    t_sand = np.maximum(0.001, (tau_eff_sand - tau_cr_sand) / tau_cr_sand)
    exp_dstar = 0.3333 if strict_excel else (1.0 / 3.0)
    d_star_sand = d50_sand * ((rho_s / rho_w - 1.0) * gravity / nu**2) ** exp_dstar
    ca_sand_eq_mass = rho_s * np.minimum(0.015 * (d50_sand / ref_level) * t_sand**1.5 / (d_star_sand**0.3), 0.1)

    k_idx = np.arange(n_z, dtype=np.float64)[:, None]
    z_prof = ref_level * (h[None, :] / ref_level) ** (k_idx / 49.0)

    u_prof = np.zeros_like(z_prof)
    mix_prof = np.zeros_like(z_prof)
    ws_prof = np.zeros_like(z_prof)
    ri_prof = np.full_like(z_prof, 0.01)
    damp_prof = np.ones_like(z_prof)
    c_mud_vol = np.zeros_like(z_prof)
    c_sand_unity = np.zeros_like(z_prof)

    for it in range(n_steps):
        z_col = z_prof[:, it]
        h_it = float(h[it])
        u_prof[:, it] = _log_velocity_profile(z_col, h_it, float(u[it]), ks_vel)

        para = mixing_max[it] * (1.0 - (1.0 - 2.0 * z_col / h_it) ** 2)
        if strict_excel and it == 0:
            mix_prof[:, it] = np.where(z_col < 0.5 * h_it, para, mixing_max[it])
        else:
            mix_prof[:, it] = np.maximum(para, 1e-6)

        c_mud_vol[0, it] = ca_mud_adj_vol[it]
        if it == 0 and strict_excel:
            ri_prof[1, it] = 0.01
        elif it > 0:
            ri_prof[1, it] = max((ri_prof[0, it - 1] + ri_prof[1, it - 1] + ri_prof[2, it - 1]) / 3.0, 0.01)

        damp_prof[1, it] = 1.0 / (1.0 + damping_coef * np.sqrt(ri_prof[1, it]))

        for k in range(1, n_z):
            if (it == 0 and strict_excel) or ws_mud_mode == 1:
                ws_k = ws_mud_max
            else:
                c_ref = c_mud_vol[k - 1, it]
                if c_ref < 0.0025:
                    ws_raw = np.exp(
                        0.182 * np.log(ws_mud_max / ws_mud_min) * np.log(c_ref)
                        + (2.09 * np.log(ws_mud_max) - 1.09 * np.log(ws_mud_min))
                    )
                else:
                    ws_raw = ws_mud_max * (1.0 - c_ref) ** 4
                ws_k = max(ws_raw, 5e-5)

            ws_prof[k, it] = ws_k
            damp_prev = damp_prof[k - 1, it]
            c_next = c_mud_vol[k - 1, it] * (1.0 - (ws_k / (damp_prev * mix_prof[k, it])) * (z_col[k] - z_col[k - 1]))
            c_mud_vol[k, it] = max(c_next, 1e-5 * c_mud_vol[0, it])

            if k >= 2:
                du_dz = (u_prof[k - 1, it] - u_prof[k - 2, it]) / (z_col[k - 1] - z_col[k - 2])
                dc_dz = (c_mud_vol[k - 1, it] - c_mud_vol[k - 2, it]) / (z_col[k - 1] - z_col[k - 2])
                rho_mix = rho_w + (rho_s - rho_w) * c_mud_vol[k, it]
                if abs(du_dz) > 0.0:
                    ri_val = -((rho_s - rho_w) * gravity / rho_mix) * dc_dz / (du_dz**2)
                else:
                    ri_val = 10.0
                ri_prof[k, it] = float(np.clip(ri_val, 0.01, 10.0))
                damp_prof[k, it] = 1.0 / (1.0 + damping_coef * np.sqrt(ri_prof[k, it]))

        ws_prof[0, it] = ws_prof[1, it] if ws_mud_mode != 1 else ws_mud_max

        # Perfil unitario de arena (sin desbordamiento cuando u_star_cw es muy pequeno)
        if u_star_cw[it] > 1e-12:
            beta_s = ws_sand / (VON_KARMAN * u_star_cw[it])
            base_s = np.maximum(((h_it - z_col) / z_col) * (ref_level / (h_it - ref_level)), 0.0)
            with np.errstate(under="ignore", over="ignore"):
                c_sand_unity[:, it] = np.maximum(base_s**beta_s, 1e-8)
        else:
            c_sand_unity[:, it] = 1e-8
            c_sand_unity[0, it] = 1.0

    c_mud_mass = c_mud_vol * rho_s
    q_mud_z = u_prof * c_mud_mass

    c_mud_depth_avg = np.mean(c_mud_mass, axis=0)
    q_density = alpha_density * h * c_mud_depth_avg * (-disp_h / salinity * salinity_gradient)

    qs_mud = np.zeros(n_steps, dtype=np.float64)
    total_load_mud = np.zeros(n_steps, dtype=np.float64)
    total_load_sand_u = np.zeros(n_steps, dtype=np.float64)

    for it in range(n_steps):
        z_col = z_prof[:, it]
        if strict_excel:
            qs_mud[it] = _integrate_coarse_profile(q_mud_z[:, it], z_col, bottom_coef=0.666, bug_idx11=True)
            total_load_mud[it] = rho_s * _integrate_coarse_profile(c_mud_vol[:, it], z_col, bottom_coef=0.0, bug_idx11=False)
            # En la fila 92 el desfase de intervalo ocurre en B110-B108 (indice 2 en dz)
            idx = _excel_coarse_mud_grid_indices()
            z_sub = z_col[idx]
            s_sub = c_sand_unity[idx, it]
            dz_s = z_sub[1:] - z_sub[:-1]
            dz_s[3] = z_sub[4] - z_sub[2]
            total_load_sand_u[it] = float(0.5 * np.sum((s_sub[:-1] + s_sub[1:]) * dz_s))
        else:
            qs_mud[it] = _trapz_uniform_bottom(q_mud_z[:, it], z_col, bottom_coef=0.666)
            total_load_mud[it] = float(rho_s * 0.5 * np.sum((c_mud_vol[:-1, it] + c_mud_vol[1:, it]) * (z_col[1:] - z_col[:-1])))
            total_load_sand_u[it] = float(0.5 * np.sum((c_sand_unity[:-1, it] + c_sand_unity[1:, it]) * (z_col[1:] - z_col[:-1])))

    ca_sand_prof = np.minimum((p_sand / p_mud) * (rho_bulk_sand / rho_bulk_mud) * (total_load_mud / total_load_sand_u), ca_sand_eq_mass)
    c_sand_mass = c_sand_unity * ca_sand_prof[None, :]
    q_sand_z = u_prof * c_sand_mass

    qs_sand = np.zeros(n_steps, dtype=np.float64)
    for it in range(n_steps):
        z_col = z_prof[:, it]
        if strict_excel:
            qs_sand[it] = _integrate_coarse_profile(q_sand_z[:, it], z_col, bottom_coef=0.666, bug_idx11=False)
        else:
            qs_sand[it] = _trapz_uniform_bottom(q_sand_z[:, it], z_col, bottom_coef=0.666)

    split_idx = 73 if strict_excel else None
    mud_pos, mud_neg, mud_net = _pos_neg_split(qs_mud, dt, split_idx)
    sand_pos, sand_neg, sand_net = _pos_neg_split(qs_sand, dt, split_idx)
    density_net = float(np.sum(q_density) * dt)

    return TidalMudSandResult(
        time=time,
        water_level=eta,
        depth=h,
        velocity=u,
        chezy=chezy,
        u_star_cw=u_star_cw,
        mixing_max=mixing_max,
        ca_mud_eq_mass=ca_mud_eq_mass,
        ca_mud_adj_mass=ca_mud_adj_mass,
        ca_sand_eq_mass=ca_sand_eq_mass,
        ca_sand_profile_mass=ca_sand_prof,
        density_driven_transport=q_density,
        z_profiles=z_prof,
        u_profiles=u_prof,
        mixing_profiles=mix_prof,
        ws_mud_profiles=ws_prof,
        richardson_profiles=ri_prof,
        damping_profiles=damp_prof,
        c_mud_profiles_mass=c_mud_mass,
        c_sand_profiles_mass=c_sand_mass,
        suspended_mud_rate=qs_mud,
        suspended_sand_rate=qs_sand,
        mud_pos=mud_pos,
        mud_neg=mud_neg,
        mud_net=mud_net,
        sand_pos=sand_pos,
        sand_neg=sand_neg,
        sand_net=sand_net,
        density_net=density_net,
    )
