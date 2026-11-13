"""Propagacion mareal en estuarios, soluciones analiticas e intrusion salina.

Este modulo implementa todas las hojas de calculo del libro
``Computation Tidal motion.xlsx`` (L. C. van Rijn, 2013):

* Hoja 1 (``tidal_analytical_no_reflection``): Canal exponencial convergente sin reflexion
  y metodo de energia longitudinal.
* Hoja 2 (``tidal_compound_channel``): Canal compuesto (cauce principal + planicies intermareales).
* Hoja 4 (``tidal_reflection_prismatic``): Canal prismatico con reflexion en el extremo cerrado.
* Hoja 5 (``tidal_reflection_converging``): Canal convergente exponencial con reflexion.
* Hoja 6 (``tidal_fourpole``): Solucion matricial compleja (cuadripolos) y descomposicion en
  onda entrante ``K2`` y reflejada ``K1``.
* Hoja 7 (``tidal_wave_nonlinear_lamb``): Deformacion no lineal en aguas someras (Lamb, 1963).
* Hoja 8 (``tidal_progressive_wave``): Onda mareal progresiva pura sin friccion.
* Hoja 9 (``soulsby_coastal_flow``): Solucion analitica de Soulsby para flujo costero
  oscilatorio inducido por pendiente superficial.
* Hoja 10 (``salinity_van_rijn_steady``): Intrusion salina estacionaria de van Rijn
  (canal prismatico y convergente).
* Hoja 11 (``tidal_salinity_kuijper``): Modelo predictivo acoplado marea-salinidad de
  Kuijper (2010) y velocidades residuales netas.
* Hoja 13 (``salinity_time_dependent``): Intrusion salina dependiente del tiempo
  (Ippen-Harleman y Kuijper-Van Rijn).

Convencion ``strict_excel``
--------------------------
* ``strict_excel=False`` (por defecto): utiliza ``math.pi`` y ``2 * math.pi`` en lugar de
  ``3.14`` y ``6.28`` (o ``6.281`` en la hoja 7), y corrige la referencia fija ``500*h0`` /
  ``150*h0`` en la hoja 13 por el parametro ``xb_over_h``.
* ``strict_excel=True``: reproduce exactamente las formulas de la planilla Excel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sedmor.constants import GRAVITY, PI_EXCEL, PI_STANDARD, TWO_PI_EXCEL

__all__ = [
    "TidalHydraulicsParameters",
    "TidalAnalyticalResult",
    "TidalCompoundResult",
    "TidalReflectionResult",
    "FourpoleResult",
    "LambWaveResult",
    "ProgressiveWaveResult",
    "SoulsbyFlowResult",
    "SteadySalinityResult",
    "KuijperSalinityResult",
    "TimeDependentSalinityResult",
    "tidal_analytical_no_reflection",
    "tidal_compound_channel",
    "tidal_reflection_prismatic",
    "tidal_reflection_converging",
    "tidal_fourpole",
    "tidal_wave_nonlinear_lamb",
    "tidal_progressive_wave",
    "soulsby_coastal_flow",
    "salinity_van_rijn_steady",
    "tidal_salinity_kuijper",
    "salinity_time_dependent",
]


def _pi_pair(strict_excel: bool) -> tuple[float, float]:
    if strict_excel:
        return PI_EXCEL, TWO_PI_EXCEL
    return PI_STANDARD, 2.0 * PI_STANDARD


@dataclass(frozen=True)
class TidalHydraulicsParameters:
    """Parametros hidraulicos y de dispersion analitica de onda mareal."""

    beta: float
    omega: float
    hydraulic_radius: float
    chezy: float
    darcy_friction: float
    linear_friction_m: float
    c0: float
    alpha: float
    k: float
    mu: float
    phase_velocity: float
    tan_phi: float
    phi_rad: float
    phi_deg: float
    phi_hours: float
    slope_beta_deg: float
    beta_minus_phi_deg: float
    wavelength: float


@dataclass(frozen=True)
class TidalAnalyticalResult:
    """Solucion analitica sin reflexion y metodo energetico (Hoja 1)."""

    params: TidalHydraulicsParameters
    u_max_used: float
    x: NDArray[np.float64]
    width: NDArray[np.float64]
    eta_amplitude: NDArray[np.float64]
    u_river: NDArray[np.float64]
    u_flood_peak: NDArray[np.float64]
    u_ebb_peak: NDArray[np.float64]
    q_flood_peak: NDArray[np.float64]
    q_ebb_peak: NDArray[np.float64]
    u_stokes: NDArray[np.float64]
    setup_z: NDArray[np.float64]
    # Metodo de energia (columnas AM..BC, con x positivo tierra adentro)
    x_energy: NDArray[np.float64]
    h_energy: NDArray[np.float64]
    h_energy_alt: NDArray[np.float64]
    u_energy: NDArray[np.float64]
    dh_energy: NDArray[np.float64]
    dh_energy_alt: NDArray[np.float64]


@dataclass(frozen=True)
class TidalCompoundResult:
    """Solucion analitica para canal compuesto (Hoja 2)."""

    params: TidalHydraulicsParameters
    a_total: float
    h_effective: float
    c0_radius: float
    u_max_used: float
    x: NDArray[np.float64]
    width: NDArray[np.float64]
    eta_amplitude: NDArray[np.float64]
    u_river: NDArray[np.float64]
    u_flood_peak: NDArray[np.float64]
    u_ebb_peak: NDArray[np.float64]
    u_stokes: NDArray[np.float64]
    setup_z: NDArray[np.float64]


@dataclass(frozen=True)
class TidalReflectionResult:
    """Solucion analitica con reflexion total en el extremo cerrado (Hojas 4 y 5)."""

    params: TidalHydraulicsParameters
    f_l: float
    u_max_used: float
    x_eval: float
    time_hw_hours: float | None
    t_over_t: NDArray[np.float64]
    eta_incoming: NDArray[np.float64]
    eta_reflected: NDArray[np.float64]
    eta_total: NDArray[np.float64]
    u_incoming: NDArray[np.float64]
    u_reflected: NDArray[np.float64]
    u_total: NDArray[np.float64]
    q_total: NDArray[np.float64]
    eta_max_total: float
    u_max_total: float


@dataclass(frozen=True)
class FourpoleResult:
    """Solucion compleja mediante ecuaciones de cuadripolos (Hoja 6)."""

    omega: float
    area: float
    chezy: float
    hydraulic_radius: float
    c0: float
    mu: float
    k: float
    c_real: float
    sc1: complex
    sc2: complex
    sc3: complex
    sc4: complex
    eta_0: complex
    q_0: complex
    eta_l: complex
    q_l: complex
    k1_reflected: complex
    k2_incoming: complex
    eta_l_amp: float
    eta_l_phase_deg: float
    q_l_amp: float
    q_l_phase_deg: float
    reflection_ratio: float


@dataclass(frozen=True)
class LambWaveResult:
    """Deformacion no lineal de onda mareal en aguas someras segun Lamb (1963) (Hoja 7)."""

    omega: float
    k: float
    c0: float
    wavelength: float
    x: NDArray[np.float64]
    eta1_x: NDArray[np.float64]
    eta2_x: NDArray[np.float64]
    eta_x: NDArray[np.float64]
    time: NDArray[np.float64]
    eta1_t: NDArray[np.float64]
    eta2_t: NDArray[np.float64]
    eta_t: NDArray[np.float64]


@dataclass(frozen=True)
class ProgressiveWaveResult:
    """Onda mareal progresiva sin friccion (Hoja 8)."""

    omega: float
    c0: float
    k: float
    wavelength: float
    u_max: float
    time: NDArray[np.float64]
    water_level: NDArray[np.float64]
    surface_slope: NDArray[np.float64]
    velocity: NDArray[np.float64]


@dataclass(frozen=True)
class SoulsbyFlowResult:
    """Solucion analitica de Soulsby para flujo costero oscilatorio (Hoja 9)."""

    omega: float
    chezy: float
    h_star: float
    linear_r: float
    u_max: float
    u_max_no_friction: float
    alpha_deg: float
    phi_deg: float
    time: NDArray[np.float64]
    slope: NDArray[np.float64]
    velocity: NDArray[np.float64]
    velocity_no_friction: NDArray[np.float64]


@dataclass(frozen=True)
class SteadySalinityResult:
    """Intrusion salina estacionaria de van Rijn (Hoja 10)."""

    excursion_length: float
    u_river: float
    ls_prismatic: float
    ls_converging: float
    eps: float
    x: NDArray[np.float64]
    salinity_prismatic: NDArray[np.float64]
    dispersion_converging: NDArray[np.float64]
    salinity_converging: NDArray[np.float64]


@dataclass(frozen=True)
class KuijperSalinityResult:
    """Modelo predictivo acoplado marea-salinidad de Kuijper 2010 (Hoja 11)."""

    tidal_params: TidalHydraulicsParameters
    u_max_used: float
    excursion_length: float
    u_river_0: float
    v_delta: float
    volume_ratio_number: float
    gamma: float
    psi: float
    ksi: float
    richardson_n_r0: float
    la_over_le0: float
    d0_mouth: float
    ls_max_kuijper: float
    ls_min_prismatic: float
    ls_min_rigter_van_os: float
    ls_mean_prandle_stratified: float
    ls_mean_prandle_well_mixed: float
    ls_schijf_schonfeld: float
    x: NDArray[np.float64]
    width: NDArray[np.float64]
    eta_amplitude: NDArray[np.float64]
    u_flood_peak: NDArray[np.float64]
    u_ebb_peak: NDArray[np.float64]
    salinity_rel: NDArray[np.float64]
    salinity: NDArray[np.float64]
    density: NDArray[np.float64]
    density_gradient: NDArray[np.float64]
    u_river: NDArray[np.float64]
    u_salinity: NDArray[np.float64]
    u_stokes: NDArray[np.float64]
    du_asymmetry_1: NDArray[np.float64]
    du_asymmetry_2: NDArray[np.float64]
    salt_wedge_thickness: NDArray[np.float64]
    u_salt_wedge: NDArray[np.float64]


@dataclass(frozen=True)
class TimeDependentSalinityResult:
    """Intrusion salina dependiente del tiempo (Hoja 13)."""

    u_river: float
    excursion_length: float
    alpha1_ippen: float
    ls_min_ippen: float
    ls_max_ippen: float
    ls_min_kuijper: float
    ls_max_kuijper: float
    x: NDArray[np.float64]
    salinity_ippen_harleman: NDArray[np.float64]
    salinity_kuijper_van_rijn: NDArray[np.float64]


def _linear_dispersion_params(
    *,
    h0: float,
    b0: float,
    period_s: float,
    l_b: float,
    k_s: float,
    u_max: float,
    gravity: float,
    c0_override: float | None = None,
    r_override: float | None = None,
    strict_excel: bool = False,
) -> TidalHydraulicsParameters:
    pi_val, two_pi = _pi_pair(strict_excel)
    beta = 1.0 / l_b if l_b > 0.0 and math.isfinite(l_b) else 0.0
    omega = two_pi / period_s
    hyd_r = r_override if r_override is not None else (h0 * b0) / (b0 + 2.0 * h0)
    chezy = 18.0 * math.log10(12.0 * hyd_r / k_s)
    f_darcy = 8.0 * gravity / (chezy**2)
    m_lin = 8.0 * gravity * u_max / (3.0 * pi_val * (chezy**2) * hyd_r)
    c0 = c0_override if c0_override is not None else math.sqrt(gravity * h0)

    crit = 2.0 * omega / c0
    if beta < crit:
        alpha = 1.0 - 0.25 * (beta**2) * ((c0 / omega) ** 2)
        k = (
            math.sqrt(0.5)
            * (omega / c0)
            * math.sqrt(alpha)
            * math.sqrt(1.0 + math.sqrt(1.0 + (m_lin**2) / ((omega**2) * (alpha**2))))
        )
    else:
        alpha = -1.0 + 0.25 * (beta**2) * ((c0 / omega) ** 2)
        k = (
            math.sqrt(0.5)
            * (omega / c0)
            * math.sqrt(alpha)
            * math.sqrt(-1.0 + math.sqrt(1.0 + (m_lin**2) / ((omega**2) * (alpha**2))))
        )

    mu = (m_lin * omega) / (2.0 * (c0**2) * k)
    phase_vel = omega / k
    tan_phi = (0.5 * beta + mu) / k
    phi_rad = math.atan(tan_phi)
    phi_deg = phi_rad * 360.0 / two_pi
    phi_hours = (phi_deg / 360.0) * (period_s / 3600.0)
    if (mu - 0.5 * beta) < 0.0:
        slope_beta_deg = 180.0 - (360.0 / two_pi) * math.atan(k / (0.5 * beta - mu))
    elif (mu - 0.5 * beta) > 0.0:
        slope_beta_deg = (360.0 / two_pi) * math.atan(k / (mu - 0.5 * beta))
    else:
        slope_beta_deg = 90.0

    return TidalHydraulicsParameters(
        beta=float(beta),
        omega=float(omega),
        hydraulic_radius=float(hyd_r),
        chezy=float(chezy),
        darcy_friction=float(f_darcy),
        linear_friction_m=float(m_lin),
        c0=float(c0),
        alpha=float(alpha),
        k=float(k),
        mu=float(mu),
        phase_velocity=float(phase_vel),
        tan_phi=float(tan_phi),
        phi_rad=float(phi_rad),
        phi_deg=float(phi_deg),
        phi_hours=float(phi_hours),
        slope_beta_deg=float(slope_beta_deg),
        beta_minus_phi_deg=float(slope_beta_deg - phi_deg),
        wavelength=float(two_pi / k),
    )


def tidal_analytical_no_reflection(
    eta0: float = 2.55,
    h0: float = 15.0,
    b0: float = 550.0,
    period_s: float = 43200.0,
    l_b: float = 30000.0,
    length: float = 100000.0,
    k_s: float = 0.05,
    q_river: float = 0.0,
    u_max: float | None = 0.84,
    dx: float = 1000.0,
    gravity: float = GRAVITY,
    *,
    strict_excel: bool = False,
) -> TidalAnalyticalResult:
    """Calcula la propagacion de marea en canal convergente sin reflexion (Hoja 1).

    Incluye tanto el modelo analitico (coordenada ``x <= 0`` hacia tierra adentro)
    como los dos esquemas del metodo energetico (``x >= 0``). Si ``u_max=None``,
    resuelve iterativamente el coeficiente de friccion linealizado hasta que ``u_max``
    converge con ``|u_ebb(x=0)|``.
    """
    if h0 <= 0.0 or b0 <= 0.0 or length <= 0.0 or dx <= 0.0:
        raise ValueError("Parametros geometricos y paso espacial deben ser positivos.")

    pi_val, two_pi = _pi_pair(strict_excel)

    u_curr = 0.8 if u_max is None else float(u_max)
    if u_max is None:
        for _ in range(60):
            p_iter = _linear_dispersion_params(
                h0=h0,
                b0=b0,
                period_s=period_s,
                l_b=l_b,
                k_s=k_s,
                u_max=u_curr,
                gravity=gravity,
                strict_excel=strict_excel,
            )
            u_r0 = q_river / (b0 * h0)
            u_new = abs(u_r0 + (eta0 / h0) * p_iter.phase_velocity * math.cos(p_iter.phi_rad))
            if abs(u_new - u_curr) < 1e-10:
                u_curr = u_new
                break
            u_curr = 0.5 * (u_curr + u_new)

    params = _linear_dispersion_params(
        h0=h0,
        b0=b0,
        period_s=period_s,
        l_b=l_b,
        k_s=k_s,
        u_max=u_curr,
        gravity=gravity,
        strict_excel=strict_excel,
    )

    n_nodes = int(round(length / dx)) + 1
    x = -np.linspace(0.0, (n_nodes - 1) * dx, n_nodes, dtype=np.float64)
    width = b0 * np.exp(params.beta * x)
    amp_factor = np.exp((-0.5 * params.beta + params.mu) * x)
    eta_amp = eta0 * amp_factor
    u_riv = q_river / (width * h0)
    u_wave = (eta0 / h0) * params.phase_velocity * math.cos(params.phi_rad) * amp_factor
    u_flood = u_riv - u_wave
    u_ebb = u_riv + u_wave
    q_flood = h0 * width * u_flood
    q_ebb = h0 * width * u_ebb

    u_stokes = -0.5 * eta_amp / h0 * math.cos(params.phi_rad)
    setup_z = np.zeros_like(x)
    for i in range(1, n_nodes):
        setup_z[i] = setup_z[i - 1] + (3.0 * params.linear_friction_m / gravity * u_stokes[i]) * dx

    # Metodo de energia (columnas AM..BC, x positivo)
    x_en = np.abs(x)
    cos_phi_en = math.cos(params.phi_deg * two_pi / 360.0)
    h_en = np.zeros_like(x_en)
    h_en_alt = np.zeros_like(x_en)
    u_en = np.zeros_like(x_en)
    dh_en = np.zeros_like(x_en)
    dh_en_alt = np.zeros_like(x_en)

    h_en[0] = 2.0 * eta0
    h_en_alt[0] = 2.0 * eta0
    for i in range(n_nodes):
        u_en[i] = 0.5 * h_en[i] * params.phase_velocity * cos_phi_en / h0
        term1 = h_en[i] / l_b * 0.5 if (l_b > 0.0 and math.isfinite(l_b)) else 0.0
        term2 = (
            params.darcy_friction
            * (h_en[i] ** 2)
            * (params.phase_velocity**2)
            * cos_phi_en
            / (12.0 * pi_val * gravity * (h0**3))
        )
        dh_en[i] = (term1 - term2) * dx

        az = h_en[i] / l_b if (l_b > 0.0 and math.isfinite(l_b)) else 0.0
        ba = (
            gravity
            / (params.chezy**2)
            * h_en[i]
            * u_en[i]
            * cos_phi_en
            / (params.phase_velocity * h0)
        )
        bb = 1.0 + gravity * 0.5 * h_en_alt[i] / (params.phase_velocity * u_en[i] * cos_phi_en)
        dh_en_alt[i] = (az - ba) / bb * dx
        if i + 1 < n_nodes:
            h_en[i + 1] = h_en[i] + dh_en[i]
            h_en_alt[i + 1] = h_en_alt[i] + dh_en_alt[i]

    return TidalAnalyticalResult(
        params=params,
        u_max_used=float(u_curr),
        x=x,
        width=width,
        eta_amplitude=eta_amp,
        u_river=u_riv,
        u_flood_peak=u_flood,
        u_ebb_peak=u_ebb,
        q_flood_peak=q_flood,
        q_ebb_peak=q_ebb,
        u_stokes=u_stokes,
        setup_z=setup_z,
        x_energy=x_en,
        h_energy=h_en,
        h_energy_alt=h_en_alt,
        u_energy=u_en,
        dh_energy=dh_en,
        dh_energy_alt=dh_en_alt,
    )


def tidal_compound_channel(
    eta0: float = 2.6,
    b0: float = 4000.0,
    h_c: float = 12.0,
    b_c: float = 2000.0,
    h_flat: float = 2.0,
    period_s: float = 43200.0,
    l_b: float = 20000.0,
    length: float = 60000.0,
    k_s: float = 0.05,
    alpha_h: float = 1.0,
    q_river: float = 1000.0,
    u_max: float | None = 1.17,
    dx: float = 500.0,
    gravity: float = GRAVITY,
    *,
    strict_excel: bool = False,
) -> TidalCompoundResult:
    """Solucion analitica de marea en canal compuesto (cauce + planicies) sin reflexion (Hoja 2)."""
    if b0 <= 0.0 or h_c <= 0.0 or b_c <= 0.0 or length <= 0.0 or dx <= 0.0:
        raise ValueError("Parametros geometricos y paso espacial deben ser positivos.")

    a_total = b_c * h_c + (b0 - b_c) * h_flat
    hyd_r = a_total / (b0 + 2.0 * h_c + 2.0 * h_flat)
    h_eff = alpha_h * (b_c / b0) * h_c
    c0_h = math.sqrt(gravity * h_eff)
    c0_r = math.sqrt(gravity * hyd_r)

    u_curr = 1.0 if u_max is None else float(u_max)
    if u_max is None:
        for _ in range(60):
            p_iter = _linear_dispersion_params(
                h0=h_c,
                b0=b0,
                period_s=period_s,
                l_b=l_b,
                k_s=k_s,
                u_max=u_curr,
                gravity=gravity,
                c0_override=c0_h,
                r_override=hyd_r,
                strict_excel=strict_excel,
            )
            u_r0 = q_river / (b0 * h_c)
            u_new = abs(u_r0 + (eta0 / h_eff) * p_iter.phase_velocity * math.cos(p_iter.phi_rad))
            if abs(u_new - u_curr) < 1e-10:
                u_curr = u_new
                break
            u_curr = 0.5 * (u_curr + u_new)

    params = _linear_dispersion_params(
        h0=h_c,
        b0=b0,
        period_s=period_s,
        l_b=l_b,
        k_s=k_s,
        u_max=u_curr,
        gravity=gravity,
        c0_override=c0_h,
        r_override=hyd_r,
        strict_excel=strict_excel,
    )

    n_nodes = int(round(length / dx)) + 1
    x = -np.linspace(0.0, (n_nodes - 1) * dx, n_nodes, dtype=np.float64)
    width = b0 * np.exp(params.beta * x)
    amp_factor = np.exp((-0.5 * params.beta + params.mu) * x)
    eta_amp = eta0 * amp_factor
    u_riv = q_river / (width * h_c)
    u_wave = (eta0 / h_eff) * params.phase_velocity * math.cos(params.phi_rad) * amp_factor
    u_flood = u_riv - u_wave
    u_ebb = u_riv + u_wave

    u_stokes = 0.5 * eta_amp / hyd_r * math.cos(params.phi_rad)
    setup_z = np.zeros_like(x)
    for i in range(1, n_nodes):
        setup_z[i] = setup_z[i - 1] + (3.0 * params.linear_friction_m / gravity * u_stokes[i]) * dx

    return TidalCompoundResult(
        params=params,
        a_total=float(a_total),
        h_effective=float(h_eff),
        c0_radius=float(c0_r),
        u_max_used=float(u_curr),
        x=x,
        width=width,
        eta_amplitude=eta_amp,
        u_river=u_riv,
        u_flood_peak=u_flood,
        u_ebb_peak=u_ebb,
        u_stokes=u_stokes,
        setup_z=setup_z,
    )


def tidal_reflection_prismatic(
    eta0: float = 2.1,
    h0: float = 5.0,
    b0: float = 25000.0,
    period_s: float = 43200.0,
    length: float = 180000.0,
    k_s: float = 0.05,
    x_eval: float = 0.0,
    u_max: float | None = 1.3,
    n_time_steps: int = 101,
    gravity: float = GRAVITY,
    *,
    strict_excel: bool = False,
) -> TidalReflectionResult:
    """Solucion analitica con reflexion total en canal prismatico cerrado (Hoja 4)."""
    if h0 <= 0.0 or b0 <= 0.0 or length <= 0.0:
        raise ValueError("h0, b0 y length deben ser positivos.")

    _, two_pi = _pi_pair(strict_excel)
    t_over_t = np.linspace(0.0, 1.0, n_time_steps, dtype=np.float64)

    def _eval_at(u_guess: float, x_pos: float) -> tuple[TidalHydraulicsParameters, float, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        p = _linear_dispersion_params(
            h0=h0,
            b0=b0,
            period_s=period_s,
            l_b=math.inf,
            k_s=k_s,
            u_max=u_guess,
            gravity=gravity,
            strict_excel=strict_excel,
        )
        f_l = math.sqrt((math.cos(p.k * length) ** 2) + (math.sinh(p.mu * length) ** 2))
        dx_end = x_pos - length
        amp_u = 0.5 * eta0 / (f_l * h0) * p.omega / math.hypot(p.k, p.mu)
        eta_inc = (
            0.5 * eta0 / f_l * math.exp(-p.mu * dx_end) * np.cos(p.k * dx_end - two_pi * t_over_t)
        )
        eta_ref = (
            0.5 * eta0 / f_l * math.exp(p.mu * dx_end) * np.cos(p.k * dx_end + two_pi * t_over_t)
        )
        u_inc = amp_u * math.exp(-p.mu * dx_end) * np.cos(-p.k * dx_end + two_pi * t_over_t + p.phi_rad)
        u_ref = -amp_u * math.exp(p.mu * dx_end) * np.cos(p.k * dx_end + two_pi * t_over_t + p.phi_rad)
        return p, f_l, eta_inc, eta_ref, u_inc, u_ref

    u_curr = 1.0 if u_max is None else float(u_max)
    if u_max is None:
        for _ in range(60):
            _, _, _, _, u_i, u_r = _eval_at(u_curr, 0.0)
            u_new = float(np.max(u_i + u_r))
            if abs(u_new - u_curr) < 1e-10:
                u_curr = u_new
                break
            u_curr = 0.5 * (u_curr + u_new)

    params, f_l, eta_inc, eta_ref, u_inc, u_ref = _eval_at(u_curr, x_eval)
    eta_tot = eta_inc + eta_ref
    u_tot = u_inc + u_ref
    q_tot = u_tot * b0 * h0
    time_hw = (
        (1.0 / params.omega)
        * math.atan(-math.tanh(params.mu * (x_eval - length)) * math.tan(params.k * (x_eval - length)))
        / 3600.0
    )

    return TidalReflectionResult(
        params=params,
        f_l=float(f_l),
        u_max_used=float(u_curr),
        x_eval=float(x_eval),
        time_hw_hours=float(time_hw),
        t_over_t=t_over_t,
        eta_incoming=eta_inc,
        eta_reflected=eta_ref,
        eta_total=eta_tot,
        u_incoming=u_inc,
        u_reflected=u_ref,
        u_total=u_tot,
        q_total=q_tot,
        eta_max_total=float(np.max(eta_tot)),
        u_max_total=float(np.max(u_tot)),
    )


def tidal_reflection_converging(
    eta0: float = 0.65,
    h0: float = 6.1,
    b0: float = 32000.0,
    period_s: float = 45000.0,
    l_b: float = 45000.0,
    length: float = 210000.0,
    k_s: float = 0.003,
    x_eval: float = 0.0,
    u_max: float | None = 0.72,
    n_time_steps: int = 101,
    gravity: float = GRAVITY,
    *,
    strict_excel: bool = False,
) -> TidalReflectionResult:
    """Solucion analitica con reflexion total en canal convergente (Hoja 5)."""
    if h0 <= 0.0 or b0 <= 0.0 or length <= 0.0 or l_b <= 0.0:
        raise ValueError("h0, b0, length y l_b deben ser positivos.")

    _, two_pi = _pi_pair(strict_excel)
    t_over_t = np.linspace(0.0, 1.0, n_time_steps, dtype=np.float64)

    def _eval_at(u_guess: float, x_pos: float) -> tuple[TidalHydraulicsParameters, float, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        p = _linear_dispersion_params(
            h0=h0,
            b0=b0,
            period_s=period_s,
            l_b=l_b,
            k_s=k_s,
            u_max=u_guess,
            gravity=gravity,
            strict_excel=strict_excel,
        )
        f_l = math.sqrt((math.cos(p.k * length) ** 2) + (math.sinh(p.mu * length) ** 2))
        e_inc = math.exp((0.5 * p.beta - p.mu) * x_pos + p.mu * length)
        e_ref = math.exp((0.5 * p.beta + p.mu) * x_pos - p.mu * length)
        phase_arg = p.k * (length - x_pos)
        eta_inc = 0.5 * eta0 / f_l * e_inc * np.cos(phase_arg + two_pi * t_over_t)
        eta_ref = 0.5 * eta0 / f_l * e_ref * np.cos(-phase_arg + two_pi * t_over_t)
        amp_u = 0.5 * (eta0 / h0) * (p.omega / math.hypot(p.mu, p.k)) * (1.0 / f_l)
        u_inc = amp_u * e_inc * np.cos(phase_arg + two_pi * t_over_t + p.phi_rad)
        u_ref = -amp_u * e_ref * np.cos(-phase_arg + two_pi * t_over_t + p.phi_rad)
        return p, f_l, eta_inc, eta_ref, u_inc, u_ref

    u_curr = 0.7 if u_max is None else float(u_max)
    if u_max is None:
        for _ in range(60):
            _, _, _, _, u_i, u_r = _eval_at(u_curr, 0.0)
            u_new = float(np.max(u_i + u_r))
            if abs(u_new - u_curr) < 1e-10:
                u_curr = u_new
                break
            u_curr = 0.5 * (u_curr + u_new)

    params, f_l, eta_inc, eta_ref, u_inc, u_ref = _eval_at(u_curr, x_eval)
    eta_tot = eta_inc + eta_ref
    u_tot = u_inc + u_ref
    q_tot = u_tot * h0 * b0

    return TidalReflectionResult(
        params=params,
        f_l=float(f_l),
        u_max_used=float(u_curr),
        x_eval=float(x_eval),
        time_hw_hours=None,
        t_over_t=t_over_t,
        eta_incoming=eta_inc,
        eta_reflected=eta_ref,
        eta_total=eta_tot,
        u_incoming=u_inc,
        u_reflected=u_ref,
        u_total=u_tot,
        q_total=q_tot,
        eta_max_total=float(np.max(eta_tot)),
        u_max_total=float(np.max(u_tot)),
    )


def tidal_fourpole(
    h0: float = 10.0,
    b: float = 5000.0,
    k_s: float = 0.05,
    period_s: float = 44000.0,
    length: float = 5000.0,
    eta0_amp: float = 2.2,
    eta0_phase_deg: float = 0.0,
    q0_amp: float = 50000.0,
    q0_phase_deg: float = 70.0,
    q_peak_est: float | None = 50000.0,
    gravity: float = GRAVITY,
    *,
    strict_excel: bool = False,
) -> FourpoleResult:
    """Resuelve las ecuaciones complejas de cuadripolos para un tramo prismatico (Hoja 6)."""
    if h0 <= 0.0 or b <= 0.0 or length <= 0.0:
        raise ValueError("h0, b y length deben ser positivos.")

    pi_val, two_pi = _pi_pair(strict_excel)
    omega = two_pi / period_s
    area = h0 * b
    chezy = 18.0 * math.log10(12.0 * h0 / k_s)
    hyd_r = area / (b + 2.0 * h0)
    c0 = math.sqrt(gravity * area / b)

    q_est = float(q0_amp) if q_peak_est is None else float(q_peak_est)
    n_fric = 8.0 * gravity * q_est / (3.0 * pi_val * (chezy**2) * (area**2) * hyd_r)
    ratio = (n_fric * area / omega) ** 2
    root_term = math.sqrt(1.0 + ratio)
    mu = (1.0 / math.sqrt(2.0)) * (omega / c0) * math.sqrt(-1.0 + root_term)
    k = (1.0 / math.sqrt(2.0)) * (omega / c0) * math.sqrt(1.0 + root_term)
    c_real = omega / k

    mu_l = mu * length
    k_l = k * length
    b_w = b * omega

    sc1 = complex(math.cosh(mu_l) * math.cos(k_l), math.sinh(mu_l) * math.sin(k_l))
    sc2 = complex(
        (1.0 / b_w) * (-mu * math.cosh(mu_l) * math.sin(k_l) - k * math.sinh(mu_l) * math.cos(k_l)),
        (1.0 / b_w) * (mu * math.sinh(mu_l) * math.cos(k_l) - k * math.cosh(mu_l) * math.sin(k_l)),
    )
    sc3 = complex(
        b_w / (mu**2 + k**2) * (-k * math.sinh(mu_l) * math.cos(k_l) + mu * math.cosh(mu_l) * math.sin(k_l)),
        b_w / (mu**2 + k**2) * (-mu * math.sinh(mu_l) * math.cos(k_l) - k * math.cosh(mu_l) * math.sin(k_l)),
    )
    sc4 = sc1

    eta0_c = complex(
        eta0_amp * math.cos(eta0_phase_deg / 360.0 * two_pi),
        eta0_amp * math.sin(eta0_phase_deg / 360.0 * two_pi),
    )
    q0_c = complex(
        q0_amp * math.cos(q0_phase_deg / 360.0 * two_pi),
        q0_amp * math.sin(q0_phase_deg / 360.0 * two_pi),
    )

    eta_l = sc1 * eta0_c + sc2 * q0_c
    q_l = sc3 * eta0_c + sc4 * q0_c

    k1 = 0.5 * eta0_c - 0.5 * complex(k, -mu) / b_w * q0_c
    k2 = 0.5 * eta0_c + 0.5 * complex(k, -mu) / b_w * q0_c

    eta_l_phase_deg = math.atan(eta_l.imag / eta_l.real) * 360.0 / two_pi if eta_l.real != 0.0 else math.copysign(90.0, eta_l.imag)
    q_l_phase_deg = math.atan(q_l.imag / q_l.real) * 360.0 / two_pi if q_l.real != 0.0 else math.copysign(90.0, q_l.imag)

    return FourpoleResult(
        omega=float(omega),
        area=float(area),
        chezy=float(chezy),
        hydraulic_radius=float(hyd_r),
        c0=float(c0),
        mu=float(mu),
        k=float(k),
        c_real=float(c_real),
        sc1=sc1,
        sc2=sc2,
        sc3=sc3,
        sc4=sc4,
        eta_0=eta0_c,
        q_0=q0_c,
        eta_l=eta_l,
        q_l=q_l,
        k1_reflected=k1,
        k2_incoming=k2,
        eta_l_amp=float(abs(eta_l)),
        eta_l_phase_deg=float(eta_l_phase_deg),
        q_l_amp=float(abs(q_l)),
        q_l_phase_deg=float(q_l_phase_deg),
        reflection_ratio=float(abs(k1) / abs(k2)),
    )


def tidal_wave_nonlinear_lamb(
    eta0: float = 2.0,
    h0: float = 10.0,
    period_s: float = 44000.0,
    t_fixed: float = 11000.0,
    x_fixed: float = 70000.0,
    x: ArrayLike | None = None,
    time: ArrayLike | None = None,
    gravity: float = GRAVITY,
    *,
    strict_excel: bool = False,
) -> LambWaveResult:
    """Deformacion no lineal de una onda mareal en aguas someras segun Lamb (1963) (Hoja 7)."""
    if h0 <= 0.0 or period_s <= 0.0:
        raise ValueError("h0 y period_s deben ser positivos.")

    _, two_pi = _pi_pair(strict_excel)
    omega = two_pi / period_s
    c0 = math.sqrt(gravity * h0)
    k = omega / c0
    wavelength = (6.281 / k) if strict_excel else (two_pi / k)

    x_arr = np.linspace(0.0, wavelength, 101, dtype=np.float64) if x is None else np.asarray(x, dtype=np.float64)
    t_arr = np.linspace(0.0, period_s, 101, dtype=np.float64) if time is None else np.asarray(time, dtype=np.float64)

    factor = 0.75 * gravity * omega * (eta0**2) / (c0**3)
    eta1_x = eta0 * np.cos(omega * t_fixed - k * x_arr)
    eta2_x = -factor * x_arr * np.sin(2.0 * omega * t_fixed - 2.0 * k * x_arr)

    eta1_t = eta0 * np.cos(omega * t_arr - k * x_fixed)
    eta2_t = -factor * x_fixed * np.sin(2.0 * omega * t_arr - 2.0 * k * x_fixed)

    return LambWaveResult(
        omega=float(omega),
        k=float(k),
        c0=float(c0),
        wavelength=float(wavelength),
        x=x_arr,
        eta1_x=eta1_x,
        eta2_x=eta2_x,
        eta_x=eta1_x + eta2_x,
        time=t_arr,
        eta1_t=eta1_t,
        eta2_t=eta2_t,
        eta_t=eta1_t + eta2_t,
    )


def tidal_progressive_wave(
    eta0: float = 2.0,
    h0: float = 50.0,
    period_s: float = 44000.0,
    x: float = 100000.0,
    time: ArrayLike | None = None,
    dt_s: float = 500.0,
    gravity: float = GRAVITY,
    *,
    strict_excel: bool = False,
) -> ProgressiveWaveResult:
    """Solucion de onda mareal progresiva pura sin friccion (Hoja 8)."""
    if h0 <= 0.0 or period_s <= 0.0:
        raise ValueError("h0 y period_s deben ser positivos.")

    _, two_pi = _pi_pair(strict_excel)
    omega = two_pi / period_s
    c0 = math.sqrt(gravity * h0)
    k = omega / c0
    wavelength = two_pi / k
    u_max = (eta0 / h0) * c0

    if time is None:
        n_steps = int(round(period_s / dt_s)) + 1
        t_arr = np.linspace(0.0, (n_steps - 1) * dt_s, n_steps, dtype=np.float64)
    else:
        t_arr = np.asarray(time, dtype=np.float64)

    phase = omega * t_arr - k * x
    return ProgressiveWaveResult(
        omega=float(omega),
        c0=float(c0),
        k=float(k),
        wavelength=float(wavelength),
        u_max=float(u_max),
        time=t_arr,
        water_level=eta0 * np.cos(phase),
        surface_slope=eta0 * k * np.sin(phase),
        velocity=u_max * np.cos(phase),
    )


def soulsby_coastal_flow(
    h0: float = 10.0,
    period_s: float = 44000.0,
    s0: float = 1.0e-5,
    k_s: float = 0.05,
    time: ArrayLike | None = None,
    dt_s: float = 500.0,
    gravity: float = GRAVITY,
    *,
    strict_excel: bool = False,
) -> SoulsbyFlowResult:
    """Solucion analitica de Soulsby para flujo costero por pendiente oscilatoria (Hoja 9)."""
    if h0 <= 0.0 or period_s <= 0.0 or k_s <= 0.0:
        raise ValueError("h0, period_s y k_s deben ser positivos.")

    pi_val, two_pi = _pi_pair(strict_excel)
    omega = two_pi / period_s
    chezy = 18.0 * math.log10(12.0 * h0 / k_s)
    coef_hstar = 5.333 if strict_excel else (16.0 / 3.0)
    h_star = (coef_hstar / pi_val) * (gravity**2) * s0 / ((chezy**2) * (omega**2))
    r_lin = 8.0 * gravity / (3.0 * pi_val * (chezy**2) * h0)
    u_max = (omega / (math.sqrt(2.0) * r_lin)) * math.sqrt(-1.0 + math.sqrt(1.0 + (h_star / h0) ** 2))
    u_max_no_fric = gravity * s0 / omega

    a_coef = gravity * (u_max**2) / ((chezy**2) * h0)
    b_coef = u_max * omega
    alpha_rad = math.atan2(b_coef, a_coef)
    alpha_deg = alpha_rad * 360.0 / two_pi
    phi_deg = 180.0 - alpha_deg

    if time is None:
        n_steps = int(round(period_s / dt_s)) + 1
        t_arr = np.linspace(0.0, (n_steps - 1) * dt_s, n_steps, dtype=np.float64)
    else:
        t_arr = np.asarray(time, dtype=np.float64)

    slope = s0 * np.cos(omega * t_arr)
    velocity = u_max * np.cos(omega * t_arr + (phi_deg / 360.0) * two_pi)
    velocity_no_fric = u_max_no_fric * np.cos(omega * t_arr + pi_val * 0.5)

    return SoulsbyFlowResult(
        omega=float(omega),
        chezy=float(chezy),
        h_star=float(h_star),
        linear_r=float(r_lin),
        u_max=float(u_max),
        u_max_no_friction=float(u_max_no_fric),
        alpha_deg=float(alpha_deg),
        phi_deg=float(phi_deg),
        time=t_arr,
        slope=slope,
        velocity=velocity,
        velocity_no_friction=velocity_no_fric,
    )


def salinity_van_rijn_steady(
    q_river: float = 1000.0,
    h0: float = 10.0,
    b0: float = 500.0,
    s0: float = 25.0,
    period_s: float = 45000.0,
    l_b: float = 1.0e9,
    u_max: float = 1.0,
    k_exponent: float = 1.5,
    d0: float = 1000.0,
    dx: float = 200.0,
    x: ArrayLike | None = None,
    *,
    strict_excel: bool = False,
) -> SteadySalinityResult:
    """Distribucion longitudinal estacionaria de salinidad segun van Rijn (Hoja 10)."""
    if h0 <= 0.0 or b0 <= 0.0 or d0 <= 0.0 or q_river <= 0.0:
        raise ValueError("h0, b0, d0 y q_river deben ser positivos.")

    pi_val, _ = _pi_pair(strict_excel)
    excursion_length = (1.0 / pi_val) * u_max * period_s
    u_river = q_river / (h0 * b0)
    ls_prismatic = 1.0 / (k_exponent * u_river / d0)
    eps = k_exponent * l_b * u_river / d0
    ls_converging = l_b * math.log((1.0 + eps) / eps)

    if x is None:
        max_x = max(ls_prismatic * 1.1, 4000.0)
        n_steps = int(round(max_x / dx)) + 1
        x_arr = np.linspace(0.0, (n_steps - 1) * dx, n_steps, dtype=np.float64)
    else:
        x_arr = np.asarray(x, dtype=np.float64)

    ratio_p = x_arr / ls_prismatic
    sal_prismatic = np.where(ratio_p < 1.0, s0 * np.maximum(1.0 - ratio_p, 0.0) ** (1.0 / k_exponent), 0.0)

    exp_term = eps * (np.exp(x_arr / l_b) - 1.0 if strict_excel else np.expm1(x_arr / l_b))
    d_conv = np.where(exp_term < 1.0, d0 * (1.0 - exp_term), 0.0)
    sal_converging = s0 * np.power(np.maximum(d_conv / d0, 0.0), 1.0 / k_exponent)

    return SteadySalinityResult(
        excursion_length=float(excursion_length),
        u_river=float(u_river),
        ls_prismatic=float(ls_prismatic),
        ls_converging=float(ls_converging),
        eps=float(eps),
        x=x_arr,
        salinity_prismatic=sal_prismatic,
        dispersion_converging=d_conv,
        salinity_converging=sal_converging,
    )


def tidal_salinity_kuijper(
    eta0: float = 2.1,
    h0: float = 10.0,
    b0: float = 25000.0,
    period_s: float = 45000.0,
    l_b: float = 25000.0,
    l_a: float = 25000.0,
    length: float = 100000.0,
    k_s: float = 0.05,
    q_river: float = 60.0,
    s0: float = 30.0,
    alpha_d: float = 0.2,
    mixing_factor: float = 1.0,
    asymmetry_ratio_1: float = 0.8,
    asymmetry_ratio_2: float = 1.25,
    u_max: float | None = 0.79,
    dx: float = 1000.0,
    gravity: float = GRAVITY,
    *,
    strict_excel: bool = False,
) -> KuijperSalinityResult:
    """Modelo predictivo acoplado marea-salinidad de Kees Kuijper (2010) (Hoja 11)."""
    if h0 <= 0.0 or b0 <= 0.0 or l_b <= 0.0 or l_a <= 0.0 or length <= 0.0:
        raise ValueError("Parametros geometricos y de escala deben ser positivos.")

    pi_val, _ = _pi_pair(strict_excel)

    def _u_ebb_at_mouth(u_guess: float) -> float:
        p = _linear_dispersion_params(
            h0=h0,
            b0=b0,
            period_s=period_s,
            l_b=l_b,
            k_s=k_s,
            u_max=u_guess,
            gravity=gravity,
            strict_excel=strict_excel,
        )
        u_r0 = q_river / (b0 * h0)
        return abs(u_r0 + (eta0 / h0) * p.phase_velocity * math.cos(p.phi_rad))

    u_curr = 0.75 if u_max is None else float(u_max)
    if u_max is None:
        for _ in range(60):
            u_new = _u_ebb_at_mouth(u_curr)
            if abs(u_new - u_curr) < 1e-10:
                u_curr = u_new
                break
            u_curr = 0.5 * (u_curr + u_new)

    params = _linear_dispersion_params(
        h0=h0,
        b0=b0,
        period_s=period_s,
        l_b=l_b,
        k_s=k_s,
        u_max=u_curr,
        gravity=gravity,
        strict_excel=strict_excel,
    )

    n_nodes = int(round(length / dx)) + 1
    x = -np.linspace(0.0, (n_nodes - 1) * dx, n_nodes, dtype=np.float64)
    width = b0 * np.exp(params.beta * x)
    amp_factor = np.exp((-0.5 * params.beta + params.mu) * x)
    eta_amp = eta0 * amp_factor
    u_riv = q_river / (h0 * width)
    u_wave = (eta0 / h0) * params.phase_velocity * math.cos(params.phi_rad) * amp_factor
    u_flood = u_riv - u_wave
    u_ebb = u_riv + u_wave

    u_flood_0 = float(u_flood[0])
    excursion_length = abs((1.0 / pi_val) * u_flood_0 * period_s)
    u_r0 = float(u_riv[0])
    v_delta = math.sqrt(0.77 * (s0 / 1000.0) * h0 * gravity)
    vol_ratio = pi_val * q_river / (abs(u_flood_0) * h0 * b0)

    gamma = (1.0 / l_b) - (1.0 / l_a)
    psi = 0.5 * (0.5 / l_b - params.mu) + 1.5 * gamma + 0.5 / l_a
    ksi = 1.0 / (1.0 - psi * l_a)
    n_r0 = pi_val * 0.78 * s0 / (1000.0 + 0.78 * s0) * (gravity * h0) / (u_curr**2) * (u_r0 / u_curr)
    la_over_le0 = l_a / excursion_length
    sqrt_g = params.chezy / 3.13
    if la_over_le0 < 10.0:
        d0 = alpha_d * (60.0 / la_over_le0) * sqrt_g * math.sqrt(n_r0) * (u_curr * h0)
        ls_max = ksi * l_a * math.log(2.0 * d0 / (u_r0 * ksi * l_a) + 1.0)
    else:
        d0 = alpha_d * 6.0 * sqrt_g * math.sqrt(n_r0) * (u_curr * h0)
        ls_max = (2.0 / params.mu) * math.log(params.mu * d0 / u_r0 + 1.0)

    ls_min_prismatic = 0.063 * h0 * ((params.chezy**2) / gravity) * ((v_delta / u_curr) ** 2) * ((u_curr / u_r0) ** 1.3)
    ls_min_rigter = 0.18 * h0 * ((params.chezy**2) / gravity) * (v_delta**2) / (u_r0 * u_curr)
    ls_prandle_strat = 0.26 * h0 * ((params.chezy**2) / gravity) * (v_delta**2) / (u_r0 * u_curr)
    ls_prandle_mixed = 0.041 * h0 * ((params.chezy**2) / gravity) * (v_delta**2) / (u_r0 * u_curr)
    ls_schijf = 0.05 * h0 * ((params.chezy**2) / gravity) * ((v_delta / u_r0) ** 2)

    if la_over_le0 < 10.0:
        sal_rel = (1.0 - (0.5 * u_r0 * ksi * l_a / d0) * (np.exp(-x / (ksi * l_a)) - 1.0)) ** 2
        inner_zero = (0.5 * u_r0 * ksi * l_a / d0) * (np.exp(-x / (ksi * l_a)) - 1.0) >= 1.0
    else:
        sal_rel = (1.0 - (u_r0 / (params.mu * d0)) * (np.exp(0.5 * params.mu * (-x)) - 1.0)) ** 2
        inner_zero = (u_r0 / (params.mu * d0)) * (np.exp(0.5 * params.mu * (-x)) - 1.0) >= 1.0

    sal = np.zeros_like(x)
    sal[0] = sal_rel[0] * s0 if not inner_zero[0] else 0.0
    for i in range(1, n_nodes):
        if sal[i - 1] > 0.01 and not inner_zero[i]:
            sal[i] = float(sal_rel[i] * s0)
        else:
            sal[i] = 0.0

    density = 1000.0 + 0.77 * sal
    dens_grad = np.zeros_like(x)
    if n_nodes > 1:
        dens_grad[1:] = (density[:-1] - density[1:]) / (x[:-1] - x[1:])
        dens_grad[0] = dens_grad[1]

    u_sal = (
        -0.035
        * 3.13
        * params.chezy
        / (mixing_factor * 0.007 * (np.abs(u_flood) + u_riv) * h0)
        * (h0**2)
        * (h0 / 1000.0)
        * dens_grad
    )
    u_stokes = -0.5 * eta_amp / h0 * math.cos(params.phi_rad)
    du_1 = -2.0 * (asymmetry_ratio_1 - 1.0) / (asymmetry_ratio_1 + 1.0) * np.abs(u_flood)
    du_2 = -2.0 * (asymmetry_ratio_2 - 1.0) / (asymmetry_ratio_2 + 1.0) * np.abs(u_flood)

    wedge_thick = np.where(np.abs(x) < ls_max, 0.67 * h0 * (ls_max - np.abs(x)) / ls_max, 0.0)
    wedge_grad = np.zeros_like(x)
    if n_nodes > 1:
        wedge_grad[1:] = (wedge_thick[:-1] - wedge_thick[1:]) / (x[:-1] - x[1:])
        wedge_grad[0] = wedge_grad[1]

    u_wedge = (
        -0.125
        * (0.7 * s0 / 1000.0)
        * 3.13
        * params.chezy
        / (0.007 * (np.abs(u_flood) + u_riv) * h0)
        * (wedge_thick**2)
        * wedge_grad
    )

    return KuijperSalinityResult(
        tidal_params=params,
        u_max_used=float(u_curr),
        excursion_length=float(excursion_length),
        u_river_0=float(u_r0),
        v_delta=float(v_delta),
        volume_ratio_number=float(vol_ratio),
        gamma=float(gamma),
        psi=float(psi),
        ksi=float(ksi),
        richardson_n_r0=float(n_r0),
        la_over_le0=float(la_over_le0),
        d0_mouth=float(d0),
        ls_max_kuijper=float(ls_max),
        ls_min_prismatic=float(ls_min_prismatic),
        ls_min_rigter_van_os=float(ls_min_rigter),
        ls_mean_prandle_stratified=float(ls_prandle_strat),
        ls_mean_prandle_well_mixed=float(ls_prandle_mixed),
        ls_schijf_schonfeld=float(ls_schijf),
        x=x,
        width=width,
        eta_amplitude=eta_amp,
        u_flood_peak=u_flood,
        u_ebb_peak=u_ebb,
        salinity_rel=sal_rel,
        salinity=sal,
        density=density,
        density_gradient=dens_grad,
        u_river=u_riv,
        u_salinity=u_sal,
        u_stokes=u_stokes,
        du_asymmetry_1=du_1,
        du_asymmetry_2=du_2,
        salt_wedge_thickness=wedge_thick,
        u_salt_wedge=u_wedge,
    )


def salinity_time_dependent(
    h0: float = 15.0,
    b: float = 430.0,
    u_max: float = 0.6,
    period_s: float = 44000.0,
    d0: float = 1500.0,
    k_exponent: float = 0.5,
    q_river: float = 1160.0,
    xb_over_h: float = 150.0,
    s0: float = 25.0,
    time_s: float = 0.0,
    x: ArrayLike | None = None,
    dx: float = 500.0,
    length: float = 25000.0,
    *,
    strict_excel: bool = False,
) -> TimeDependentSalinityResult:
    """Intrusion salina dependiente del tiempo (Ippen-Harleman y Kuijper-Van Rijn) (Hoja 13).

    Nota sobre ``strict_excel``
    ---------------------------
    En la hoja original ``13 Salinity time-dependent``, la celda ``H24`` fija
    ``500 * h0`` en el denominador de Kuijper-Van Rijn y ``E14`` fija ``2 * 150 * h0``.
    Con ``strict_excel=False``, ambas expresiones utilizan consistentemente ``xb_over_h``.
    """
    if h0 <= 0.0 or b <= 0.0 or d0 <= 0.0 or q_river <= 0.0:
        raise ValueError("h0, b, d0 y q_river deben ser positivos.")

    pi_val, two_pi = _pi_pair(strict_excel)
    u_river = q_river / (h0 * b)
    excursion_length = u_max * period_s / pi_val

    xb_denom_ih = 150.0 if strict_excel else xb_over_h
    alpha1 = u_river / (d0 * 2.0 * xb_denom_ih * h0)
    ls_min_ippen = 40.0 * h0 * (math.sqrt(d0 / (u_river * h0)) - 4.0)
    ls_max_ippen = excursion_length + ls_min_ippen

    ls_min_kuijper = d0 / (k_exponent * u_river)
    ls_max_kuijper = ls_min_kuijper + excursion_length

    if x is None:
        n_steps = int(round(length / dx)) + 1
        x_arr = np.linspace(0.0, (n_steps - 1) * dx, n_steps, dtype=np.float64)
    else:
        x_arr = np.asarray(x, dtype=np.float64)

    tidal_shift = (u_max * period_s / two_pi) * (1.0 - math.cos(two_pi / period_s * time_s))
    b_term = x_arr + xb_over_h * h0 - tidal_shift
    sal_ih = np.where(b_term < 0.0, s0, s0 * np.exp(-alpha1 * (b_term**2)))

    xb_kvr_num = xb_over_h * h0
    xb_kvr_den = (500.0 * h0) if strict_excel else (xb_over_h * h0)
    cond_neg = (b_term / (ls_min_kuijper + xb_kvr_num)) < 0.0
    cond_inside = (x_arr - tidal_shift - ls_min_kuijper) < 0.0
    sal_kvr = np.where(
        cond_neg,
        s0,
        np.where(
            cond_inside,
            s0 * ((1.0 - b_term / (ls_min_kuijper + xb_kvr_den)) ** 2),
            0.0,
        ),
    )

    return TimeDependentSalinityResult(
        u_river=float(u_river),
        excursion_length=float(excursion_length),
        alpha1_ippen=float(alpha1),
        ls_min_ippen=float(ls_min_ippen),
        ls_max_ippen=float(ls_max_ippen),
        ls_min_kuijper=float(ls_min_kuijper),
        ls_max_kuijper=float(ls_max_kuijper),
        x=x_arr,
        salinity_ippen_harleman=sal_ih,
        salinity_kuijper_van_rijn=sal_kvr,
    )
