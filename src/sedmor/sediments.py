"""Propiedades físicas de sedimentos, velocidad de caída y análisis granulométrico.

Implementa de forma vectorizada y autónoma las formulaciones científicas estándar para
caracterizar agua–sedimento, calcular velocidades terminales de sedimentación y analizar
curvas granulométricas de tamizado:

- **Propiedades del fluido y partícula**: viscosidad cinemática :math:`\\nu(T)` según
  van Rijn (1993), densidad del agua y diámetro adimensional :math:`D_*`.
- **Velocidad de caída (*settling / fall velocity*)**:
  - **Stokes (1851)** (régimen laminar),
  - **Rubey (1933)**,
  - **van Rijn (1993)** (reutilizando :mod:`sedmor.littoral`),
  - **Soulsby (1997)**,
  - **Ferguson y Church (2004)** (esferas lisas o granos naturales/angulares),
  - Corrección por **caída obstaculizada** en altas concentraciones
    (Richardson y Zaki, 1954) y modelo de **floculación** para fangos cohesivos.
- **Herramientas de curvas granulométricas**:
  - Conversión :math:`\\text{mm} \\leftrightarrow \\phi` (Krumbein, 1934) y clasificación
    textural de **Wentworth (1922)**.
  - Conversión desde % retenido o masa retenida a % pasante acumulado, interpolación
    logarítmica de percentiles arbitrarios (:math:`D_{10}, D_{16}, D_{50}, D_{84}, D_{90}`),
    coeficientes de Hazen (:math:`C_u, C_c`) y estadísticos gráficos de **Folk y Ward (1957)**
    (:math:`M_z, \\sigma_I, Sk_I, K_G`).
  - Generación de curvas log-normales, mezcla ponderada de muestras y verificación de
    criterios de filtro granular de **Terzaghi**.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr

from .constants import GRAVITY, KINEMATIC_VISCOSITY_20C, RHO_FRESH_WATER, RHO_SEDIMENT
from .littoral import _settling_velocity_van_rijn
from .runup import _as_out, _validate_positive

FloatOrArray = NDArray[np.float64] | float


@dataclass(frozen=True)
class FallVelocityResult:
    """Resultado de velocidad de caída y parámetros hidrodinámicos de la partícula."""

    d50_m: FloatOrArray
    d_star: FloatOrArray
    ws_clear_m_s: FloatOrArray
    ws_hindered_m_s: FloatOrArray
    reynolds_particle: FloatOrArray
    drag_coefficient: FloatOrArray
    regime: NDArray[np.str_] | str


@dataclass(frozen=True)
class GrainSizeCurveResult:
    """Estadísticos completos e índices geotécnicos/sedimentológicos de una curva granulométrica."""

    diameters_mm: NDArray[np.float64]
    percent_passing: NDArray[np.float64]
    d5_mm: float
    d10_mm: float
    d16_mm: float
    d25_mm: float
    d30_mm: float
    d35_mm: float
    d50_mm: float
    d60_mm: float
    d65_mm: float
    d75_mm: float
    d84_mm: float
    d85_mm: float
    d90_mm: float
    d95_mm: float
    cu_uniformity: float
    cc_curvature: float
    sigma_g_geometric: float
    folk_ward_mean_phi: float
    folk_ward_mean_mm: float
    folk_ward_sorting_phi: float
    folk_ward_skewness: float
    folk_ward_kurtosis: float
    sorting_class: str
    skewness_class: str
    kurtosis_class: str
    gravel_pct: float
    sand_pct: float
    fines_pct: float
    uscs_gradation: str


def water_kinematic_viscosity(temperature_c: ArrayLike = 20.0) -> FloatOrArray:
    """Calcula la viscosidad cinemática del agua ``nu`` [m²/s] según temperatura (van Rijn, 1993).

    Implementa :math:`\\nu(T) = 1.792\\times 10^{-6} / (1 + 0.0337\\,T + 0.000221\\,T^2)`
    para ``T`` en grados Celsius (válida en ``0–40 °C``).
    """
    t_a = np.asarray(temperature_c, dtype=np.float64)
    if np.any(np.isnan(t_a)) or np.any(t_a < -2.0) or np.any(t_a > 60.0):
        raise ValueError("La temperatura 'temperature_c' debe estar en [-2, 60] °C.")
    is_scalar = t_a.ndim == 0
    nu = 1.792e-6 / (1.0 + 0.0337 * t_a + 0.000221 * (t_a**2))
    return _as_out(nu, is_scalar)


def water_density(temperature_c: ArrayLike = 20.0, salinity_psu: ArrayLike = 0.0) -> FloatOrArray:
    """Estima la densidad del agua dulce o marina [kg/m³] según temperatura y salinidad (UNESCO)."""
    t_a, s_a = np.broadcast_arrays(
        np.asarray(temperature_c, dtype=np.float64),
        np.asarray(salinity_psu, dtype=np.float64),
    )
    if np.any(np.isnan(t_a)) or np.any(t_a < -2.0) or np.any(t_a > 45.0):
        raise ValueError("'temperature_c' debe estar en [-2, 45] °C.")
    if np.any(np.isnan(s_a)) or np.any(s_a < 0.0) or np.any(s_a > 45.0):
        raise ValueError("'salinity_psu' debe estar en [0, 45] PSU.")
    is_scalar = t_a.ndim == 0

    rho_0 = (
        999.842594
        + 6.793952e-2 * t_a
        - 9.095290e-3 * (t_a**2)
        + 1.001685e-4 * (t_a**3)
        - 1.120083e-6 * (t_a**4)
        + 6.536332e-9 * (t_a**5)
    )
    a_coef = (
        8.24493e-1
        - 4.0899e-3 * t_a
        + 7.6438e-5 * (t_a**2)
        - 8.2467e-7 * (t_a**3)
        + 5.3875e-9 * (t_a**4)
    )
    b_coef = -5.72466e-3 + 1.0227e-4 * t_a - 1.6546e-6 * (t_a**2)
    c_coef = 4.8314e-4
    rho = rho_0 + a_coef * s_a + b_coef * np.power(s_a, 1.5) + c_coef * (s_a**2)
    return _as_out(rho, is_scalar)


def dimensionless_grain_size(
    d: ArrayLike,
    *,
    rho_s: ArrayLike = RHO_SEDIMENT,
    rho_w: ArrayLike = RHO_FRESH_WATER,
    nu: ArrayLike = KINEMATIC_VISCOSITY_20C,
    gravity: float = GRAVITY,
) -> FloatOrArray:
    """Calcula el diámetro adimensional de Bonneville / van Rijn :math:`D_* = d\\,(\\Delta g / \\nu^2)^{1/3}`."""
    if gravity <= 0.0:
        raise ValueError("'gravity' debe ser > 0.")
    d_a, rs_a, rw_a, nu_a = np.broadcast_arrays(
        np.asarray(d, dtype=np.float64),
        np.asarray(rho_s, dtype=np.float64),
        np.asarray(rho_w, dtype=np.float64),
        np.asarray(nu, dtype=np.float64),
    )
    is_scalar = d_a.ndim == 0
    _validate_positive("d", d_a, allow_zero=False)
    _validate_positive("rho_s", rs_a, allow_zero=False)
    _validate_positive("rho_w", rw_a, allow_zero=False)
    _validate_positive("nu", nu_a, allow_zero=False)
    delta = (rs_a / rw_a) - 1.0
    if np.any(delta <= 0.0):
        raise ValueError("'rho_s' debe ser mayor que 'rho_w'.")

    d_star = d_a * np.cbrt(delta * gravity / (nu_a**2))
    return _as_out(d_star, is_scalar)


def _prepare_fall_inputs(
    d: ArrayLike,
    rho_s: ArrayLike,
    rho_w: ArrayLike,
    nu: ArrayLike,
    c_vol: ArrayLike,
    gravity: float,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool
]:
    """Valida y difunde entradas comunes de modelos de velocidad de caída."""
    if gravity <= 0.0:
        raise ValueError("'gravity' debe ser > 0.")
    d_a, rs_a, rw_a, nu_a, cv_a = np.broadcast_arrays(
        np.asarray(d, dtype=np.float64),
        np.asarray(rho_s, dtype=np.float64),
        np.asarray(rho_w, dtype=np.float64),
        np.asarray(nu, dtype=np.float64),
        np.asarray(c_vol, dtype=np.float64),
    )
    is_scalar = d_a.ndim == 0
    _validate_positive("d", d_a, allow_zero=False)
    _validate_positive("rho_s", rs_a, allow_zero=False)
    _validate_positive("rho_w", rw_a, allow_zero=False)
    _validate_positive("nu", nu_a, allow_zero=False)
    if np.any(np.isnan(cv_a)) or np.any(cv_a < 0.0) or np.any(cv_a >= 0.7):
        raise ValueError("La concentración volumétrica 'c_vol' debe pertenecer a [0, 0.7).")
    delta = (rs_a / rw_a) - 1.0
    if np.any(delta <= 0.0):
        raise ValueError("'rho_s' debe ser mayor que 'rho_w'.")
    d_star = d_a * np.cbrt(delta * gravity / (nu_a**2))
    return d_a, delta, nu_a, cv_a, d_star, rs_a, rw_a, is_scalar


def _package_fall_result(
    d_a: np.ndarray,
    delta: np.ndarray,
    nu_a: np.ndarray,
    cv_a: np.ndarray,
    d_star: np.ndarray,
    ws: np.ndarray,
    hindered_exponent: float | None,
    gravity: float,
    is_scalar: bool,
) -> FallVelocityResult:
    """Empaqueta ``FallVelocityResult`` calculando Re_p, Cd y caída obstaculizada."""
    re_p = ws * d_a / nu_a
    cd = (4.0 / 3.0) * delta * gravity * d_a / np.maximum(ws**2, 1e-30)

    if hindered_exponent is None:
        # Exponente de Richardson & Zaki (1954) aproximado por régimen de Reynolds de partícula
        exp_rz = np.where(
            re_p < 0.2,
            4.65,
            np.where(re_p <= 500.0, 4.4 * np.power(np.maximum(re_p, 1e-12), -0.03), 2.39),
        )
    else:
        if hindered_exponent <= 0.0:
            raise ValueError("'hindered_exponent' debe ser > 0.")
        exp_rz = np.full_like(re_p, float(hindered_exponent))

    ws_hind = ws * np.power(1.0 - cv_a, exp_rz)
    reg = np.where(re_p < 0.5, "stokes_laminar", np.where(re_p <= 1000.0, "transitional", "turbulent"))
    reg_out: NDArray[np.str_] | str = (
        str(np.asarray(reg).item()) if is_scalar else np.asarray(reg, dtype=str)
    )

    return FallVelocityResult(
        d50_m=_as_out(d_a, is_scalar),
        d_star=_as_out(d_star, is_scalar),
        ws_clear_m_s=_as_out(ws, is_scalar),
        ws_hindered_m_s=_as_out(ws_hind, is_scalar),
        reynolds_particle=_as_out(re_p, is_scalar),
        drag_coefficient=_as_out(cd, is_scalar),
        regime=reg_out,
    )


def fall_velocity_stokes(
    d: ArrayLike,
    *,
    rho_s: ArrayLike = RHO_SEDIMENT,
    rho_w: ArrayLike = RHO_FRESH_WATER,
    nu: ArrayLike = KINEMATIC_VISCOSITY_20C,
    c_vol: ArrayLike = 0.0,
    hindered_exponent: float | None = None,
    gravity: float = GRAVITY,
) -> FallVelocityResult:
    """Calcula velocidad de caída laminar según la ley de Stokes (1851): :math:`w_s = \\Delta g d^2/(18\\nu)`."""
    d_a, delta, nu_a, cv_a, d_star, _, _, is_scalar = _prepare_fall_inputs(
        d, rho_s, rho_w, nu, c_vol, gravity
    )
    ws = delta * gravity * (d_a**2) / (18.0 * nu_a)
    return _package_fall_result(d_a, delta, nu_a, cv_a, d_star, ws, hindered_exponent, gravity, is_scalar)


def fall_velocity_rubey(
    d: ArrayLike,
    *,
    rho_s: ArrayLike = RHO_SEDIMENT,
    rho_w: ArrayLike = RHO_FRESH_WATER,
    nu: ArrayLike = KINEMATIC_VISCOSITY_20C,
    c_vol: ArrayLike = 0.0,
    hindered_exponent: float | None = None,
    gravity: float = GRAVITY,
) -> FallVelocityResult:
    """Calcula velocidad de caída clásica de Rubey (1933) para arenas y gravas."""
    d_a, delta, nu_a, cv_a, d_star, _, _, is_scalar = _prepare_fall_inputs(
        d, rho_s, rho_w, nu, c_vol, gravity
    )
    ws = np.sqrt((2.0 / 3.0) * delta * gravity * d_a + 36.0 * (nu_a**2) / (d_a**2)) - 6.0 * nu_a / d_a
    return _package_fall_result(d_a, delta, nu_a, cv_a, d_star, ws, hindered_exponent, gravity, is_scalar)


def fall_velocity_van_rijn(
    d: ArrayLike,
    *,
    rho_s: ArrayLike = RHO_SEDIMENT,
    rho_w: ArrayLike = RHO_FRESH_WATER,
    nu: ArrayLike = KINEMATIC_VISCOSITY_20C,
    c_vol: ArrayLike = 0.0,
    hindered_exponent: float | None = None,
    gravity: float = GRAVITY,
) -> FallVelocityResult:
    """Calcula velocidad de caída por tramos según van Rijn (1993) reutilizando :mod:`sedmor.littoral`."""
    d_a, delta, nu_a, cv_a, d_star, _, _, is_scalar = _prepare_fall_inputs(
        d, rho_s, rho_w, nu, c_vol, gravity
    )
    ws = np.zeros_like(d_a, dtype=np.float64)
    for idx in np.ndindex(d_a.shape):
        ws[idx] = float(
            _settling_velocity_van_rijn(
                np.asarray(d_a[idx], dtype=np.float64),
                float(delta[idx]),
                float(gravity),
                float(nu_a[idx]),
            )
        )
    return _package_fall_result(d_a, delta, nu_a, cv_a, d_star, ws, hindered_exponent, gravity, is_scalar)


def fall_velocity_soulsby(
    d: ArrayLike,
    *,
    rho_s: ArrayLike = RHO_SEDIMENT,
    rho_w: ArrayLike = RHO_FRESH_WATER,
    nu: ArrayLike = KINEMATIC_VISCOSITY_20C,
    c_vol: ArrayLike = 0.0,
    hindered_exponent: float | None = None,
    gravity: float = GRAVITY,
) -> FallVelocityResult:
    """Calcula velocidad de caída explícita según Soulsby (1997)."""
    d_a, delta, nu_a, cv_a, d_star, _, _, is_scalar = _prepare_fall_inputs(
        d, rho_s, rho_w, nu, c_vol, gravity
    )
    ws = (nu_a / d_a) * (np.sqrt(10.36**2 + 1.049 * np.power(d_star, 3)) - 10.36)
    return _package_fall_result(d_a, delta, nu_a, cv_a, d_star, ws, hindered_exponent, gravity, is_scalar)


def fall_velocity_ferguson_church(
    d: ArrayLike,
    *,
    shape_type: Literal["natural", "smooth_sphere", "very_angular"] = "natural",
    c1: float | None = None,
    c2: float | None = None,
    rho_s: ArrayLike = RHO_SEDIMENT,
    rho_w: ArrayLike = RHO_FRESH_WATER,
    nu: ArrayLike = KINEMATIC_VISCOSITY_20C,
    c_vol: ArrayLike = 0.0,
    hindered_exponent: float | None = None,
    gravity: float = GRAVITY,
) -> FallVelocityResult:
    """Calcula velocidad de caída universal según Ferguson y Church (2004).

    Implementa :math:`w_s = \\Delta g d^2 / (C_1 \\nu + \\sqrt{0.75\\,C_2\\,\\Delta g d^3})`,
    con ``(C1, C2) = (18, 1.0)`` para granos naturales, ``(18, 0.4)`` para esferas lisas
    y ``(24, 1.2)`` para clastos muy angulares (o valores ``c1, c2`` personalizados).
    """
    presets = {
        "natural": (18.0, 1.0),
        "smooth_sphere": (18.0, 0.4),
        "very_angular": (24.0, 1.2),
    }
    if c1 is None or c2 is None:
        if shape_type not in presets:
            raise ValueError("'shape_type' debe ser 'natural', 'smooth_sphere' o 'very_angular'.")
        p1, p2 = presets[shape_type]
        c1_val = p1 if c1 is None else float(c1)
        c2_val = p2 if c2 is None else float(c2)
    else:
        c1_val, c2_val = float(c1), float(c2)
    if c1_val <= 0.0 or c2_val <= 0.0:
        raise ValueError("'c1' y 'c2' deben ser > 0.")

    d_a, delta, nu_a, cv_a, d_star, _, _, is_scalar = _prepare_fall_inputs(
        d, rho_s, rho_w, nu, c_vol, gravity
    )
    rg = delta * gravity
    ws = (rg * (d_a**2)) / (c1_val * nu_a + np.sqrt(0.75 * c2_val * rg * np.power(d_a, 3)))
    return _package_fall_result(d_a, delta, nu_a, cv_a, d_star, ws, hindered_exponent, gravity, is_scalar)


def flocculated_settling_velocity(
    concentration_kg_m3: ArrayLike,
    *,
    ws_base: float = 0.0005,
    c_floc_max: float = 10.0,
    c_gel: float = 100.0,
    n_floc: float = 1.0,
    m_hindered: float = 4.0,
) -> FloatOrArray:
    """Estima velocidad de caída de sedimentos cohesivos (fangos) por floculación y caída obstaculizada.

    Para ``0 < C <= c_floc_max`` sigue :math:`w_s = w_{s,\\text{base}} (C / C_1)^{n_{\\text{floc}}}`
    (o escalado con ``ws_base`` como valor de referencia a ``1 kg/m³``); cuando ``C > c_floc_max``
    incorpora el frenado por interferencia :math:`(1 - C / c_{\\text{gel}})^{m_{\\text{hindered}}}`.
    """
    if ws_base <= 0.0 or c_floc_max <= 0.0 or c_gel <= c_floc_max or n_floc < 0.0 or m_hindered <= 0.0:
        raise ValueError("Parámetros de floculación inválidos (verifique c_gel > c_floc_max > 0).")
    c_a = np.asarray(concentration_kg_m3, dtype=np.float64)
    is_scalar = c_a.ndim == 0
    _validate_positive("concentration_kg_m3", c_a, allow_zero=True)

    ws_peak = ws_base * (c_floc_max**n_floc)
    hind = np.power(np.maximum(1.0 - c_a / c_gel, 0.0), m_hindered)
    ws = np.where(
        c_a <= 0.0,
        0.0,
        np.where(c_a <= c_floc_max, ws_base * np.power(np.maximum(c_a, 1e-12), n_floc), ws_peak * hind),
    )
    return _as_out(ws, is_scalar)


def compare_fall_velocity_models(
    d: ArrayLike,
    *,
    rho_s: ArrayLike = RHO_SEDIMENT,
    rho_w: ArrayLike = RHO_FRESH_WATER,
    nu: ArrayLike = KINEMATIC_VISCOSITY_20C,
    gravity: float = GRAVITY,
) -> dict[str, FloatOrArray]:
    """Compara Stokes, Rubey, van Rijn, Soulsby y Ferguson-Church para un mismo diámetro ``d``."""
    return {
        "d_star": dimensionless_grain_size(d, rho_s=rho_s, rho_w=rho_w, nu=nu, gravity=gravity),
        "stokes_m_s": fall_velocity_stokes(d, rho_s=rho_s, rho_w=rho_w, nu=nu, gravity=gravity).ws_clear_m_s,
        "rubey_m_s": fall_velocity_rubey(d, rho_s=rho_s, rho_w=rho_w, nu=nu, gravity=gravity).ws_clear_m_s,
        "van_rijn_m_s": fall_velocity_van_rijn(d, rho_s=rho_s, rho_w=rho_w, nu=nu, gravity=gravity).ws_clear_m_s,
        "soulsby_m_s": fall_velocity_soulsby(d, rho_s=rho_s, rho_w=rho_w, nu=nu, gravity=gravity).ws_clear_m_s,
        "ferguson_church_m_s": fall_velocity_ferguson_church(
            d, rho_s=rho_s, rho_w=rho_w, nu=nu, gravity=gravity
        ).ws_clear_m_s,
    }


def mm_to_phi(d_mm: ArrayLike) -> FloatOrArray:
    """Convierte diámetro en milímetros a escala sedimentológica ``phi = -log2(d_mm)`` (Krumbein, 1934)."""
    d_a = np.asarray(d_mm, dtype=np.float64)
    is_scalar = d_a.ndim == 0
    _validate_positive("d_mm", d_a, allow_zero=False)
    return _as_out(-np.log2(d_a), is_scalar)


def phi_to_mm(phi: ArrayLike) -> FloatOrArray:
    """Convierte escala ``phi`` a diámetro en milímetros ``d_mm = 2**(-phi)``."""
    p_a = np.asarray(phi, dtype=np.float64)
    if np.any(np.isnan(p_a)):
        raise ValueError("'phi' no debe contener NaN.")
    is_scalar = p_a.ndim == 0
    return _as_out(np.power(2.0, -p_a), is_scalar)


def classify_grain_size_wentworth(d_mm: float) -> str:
    """Clasifica un diámetro de partícula ``d_mm`` según la escala de Wentworth (1922)."""
    if d_mm <= 0.0 or math.isnan(d_mm):
        raise ValueError("'d_mm' debe ser > 0.")
    bounds = [
        (256.0, "Bloque (Boulder, >= 256 mm)"),
        (64.0, "Canto (Cobble, 64–256 mm)"),
        (4.0, "Guijarro / Grava gruesa (Pebble, 4–64 mm)"),
        (2.0, "Gránulo / Grava fina (Granule, 2–4 mm)"),
        (1.0, "Arena muy gruesa (1–2 mm)"),
        (0.5, "Arena gruesa (0.5–1 mm)"),
        (0.25, "Arena media (0.25–0.5 mm)"),
        (0.125, "Arena fina (0.125–0.25 mm)"),
        (0.0625, "Arena muy fina (0.0625–0.125 mm)"),
        (0.031, "Limo grueso (31–62.5 µm)"),
        (0.0156, "Limo medio (15.6–31 µm)"),
        (0.0078, "Limo fino (7.8–15.6 µm)"),
        (0.0039, "Limo muy fino (3.9–7.8 µm)"),
    ]
    for lim, name in bounds:
        if d_mm >= lim:
            return name
    return "Arcilla (< 3.9 µm)"


def sieve_data_to_percent_passing(
    diameters: ArrayLike,
    values: ArrayLike,
    *,
    input_type: Literal["passing", "retained_pct", "retained_mass"] = "passing",
    pan_value: float = 0.0,
    diameter_unit: Literal["mm", "m"] = "mm",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Ordena tamices de menor a mayor diámetro y devuelve ``(diameters_mm, percent_passing)``.

    Cuando ``input_type`` es ``"retained_pct"`` o ``"retained_mass"``, cada valor ``values[i]``
    corresponde al material **retenido sobre el tamiz** ``diameters[i]`` (convención estándar
    de laboratorio), y ``pan_value`` representa el material recogido en la cazoleta/fondo
    por debajo del tamiz más fino.
    """
    d_a = np.asarray(diameters, dtype=np.float64).ravel()
    v_a = np.asarray(values, dtype=np.float64).ravel()
    if d_a.size < 2 or d_a.shape != v_a.shape:
        raise ValueError("'diameters' y 'values' deben ser arreglos 1D de igual longitud (>= 2).")
    _validate_positive("diameters", d_a, allow_zero=False)
    if np.any(np.isnan(v_a)) or np.any(v_a < 0.0):
        raise ValueError("'values' no debe contener NaN ni valores negativos.")
    if pan_value < 0.0 or math.isnan(pan_value):
        raise ValueError("'pan_value' debe ser >= 0.")

    if diameter_unit == "m":
        d_mm = 1000.0 * d_a
    elif diameter_unit == "mm":
        d_mm = d_a.copy()
    else:
        raise ValueError("'diameter_unit' debe ser 'mm' o 'm'.")

    order = np.argsort(d_mm)
    d_sorted = d_mm[order]
    v_sorted = v_a[order]
    if np.any(np.diff(d_sorted) <= 0.0):
        raise ValueError("Los diámetros de tamiz deben ser estrictamente distintos.")

    if input_type == "passing":
        if np.any(v_sorted > 100.0 + 1e-9):
            raise ValueError("En 'input_type=\"passing\"', los valores deben estar en [0, 100].")
        if np.any(np.diff(v_sorted) < -1e-9):
            raise ValueError("El porcentaje pasante debe ser monótono creciente con el diámetro.")
        passing = np.clip(v_sorted, 0.0, 100.0)
    elif input_type in ("retained_pct", "retained_mass"):
        total = float(np.sum(v_sorted)) + float(pan_value)
        if total <= 0.0:
            raise ValueError("La suma de material retenido más fondo debe ser > 0.")
        # Todo lo que pasa por el tamiz d_sorted[i] es pan_value + suma de retenidos en tamices más finos (< d_sorted[i])
        cum_finer = float(pan_value) + np.concatenate(([0.0], np.cumsum(v_sorted[:-1])))
        passing = np.clip(100.0 * cum_finer / total, 0.0, 100.0)
    else:
        raise ValueError("'input_type' debe ser 'passing', 'retained_pct' o 'retained_mass'.")

    return d_sorted, passing


def percentile_diameter(
    diameters_mm: ArrayLike,
    percent_passing: ArrayLike,
    target_percent: ArrayLike,
    *,
    log_interpolation: bool = True,
) -> FloatOrArray:
    """Interpola el diámetro :math:`D_p` [mm] para un porcentaje pasante objetivo ``target_percent``.

    Por defecto realiza interpolación lineal en escala logarítmica (:math:`\\log_{10} d`),
    conforme al análisis granulométrico estándar.
    """
    d_sorted, p_sorted = sieve_data_to_percent_passing(diameters_mm, percent_passing, input_type="passing")
    tp_a = np.asarray(target_percent, dtype=np.float64)
    is_scalar = tp_a.ndim == 0
    if np.any(np.isnan(tp_a)) or np.any(tp_a < 0.0) or np.any(tp_a > 100.0):
        raise ValueError("'target_percent' debe pertenecer a [0, 100].")

    # Mantener extremos únicos cuando hay tramos planos en 0% o 100% para evitar sesgos
    if log_interpolation:
        y_vals = np.log10(d_sorted)
        interp_y = np.interp(tp_a, p_sorted, y_vals)
        d_out = np.power(10.0, interp_y)
    else:
        d_out = np.interp(tp_a, p_sorted, d_sorted)
    return _as_out(d_out, is_scalar)


def _classify_folk_ward(sigma_i: float, sk_i: float, k_g: float) -> tuple[str, str, str]:
    """Asigna descripciones verbales estándar de Folk y Ward (1957)."""
    if sigma_i < 0.35:
        s_cls = "Muy bien seleccionado"
    elif sigma_i < 0.50:
        s_cls = "Bien seleccionado"
    elif sigma_i < 0.71:
        s_cls = "Moderadamente bien seleccionado"
    elif sigma_i < 1.00:
        s_cls = "Moderadamente seleccionado"
    elif sigma_i < 2.00:
        s_cls = "Pobremente seleccionado"
    elif sigma_i < 4.00:
        s_cls = "Muy pobremente seleccionado"
    else:
        s_cls = "Extremadamente mal seleccionado"

    if sk_i > 0.30:
        sk_cls = "Muy asimétrico hacia finos (Sk > +0.3)"
    elif sk_i > 0.10:
        sk_cls = "Asimétrico hacia finos (+0.1 < Sk <= +0.3)"
    elif sk_i >= -0.10:
        sk_cls = "Casi simétrico (-0.1 <= Sk <= +0.1)"
    elif sk_i >= -0.30:
        sk_cls = "Asimétrico hacia gruesos (-0.3 <= Sk < -0.1)"
    else:
        sk_cls = "Muy asimétrico hacia gruesos (Sk < -0.3)"

    if k_g < 0.67:
        k_cls = "Muy platicúrtico"
    elif k_g < 0.90:
        k_cls = "Platicúrtico"
    elif k_g <= 1.11:
        k_cls = "Mesocúrtico (normal)"
    elif k_g <= 1.50:
        k_cls = "Leptocúrtico"
    elif k_g <= 3.00:
        k_cls = "Muy leptocúrtico"
    else:
        k_cls = "Extremadamente leptocúrtico"

    return s_cls, sk_cls, k_cls


def grain_size_curve_statistics(
    diameters: ArrayLike,
    values: ArrayLike,
    *,
    input_type: Literal["passing", "retained_pct", "retained_mass"] = "passing",
    pan_value: float = 0.0,
    diameter_unit: Literal["mm", "m"] = "mm",
) -> GrainSizeCurveResult:
    """Calcula percentiles, índices de Hazen y estadísticos de Folk y Ward (1957).

    Nota sobre la convención de signos de Folk y Ward: los percentiles sedimentológicos
    :math:`\\phi_p` corresponden al porcentaje **más grueso** (:math:`100 - p` en % pasante),
    de modo que :math:`\\phi_5 = \\phi(D_{95})` y :math:`\\phi_{95} = \\phi(D_5)`. Así, una cola
    hacia los finos produce :math:`Sk_I > 0` y una distribución log-normal arroja :math:`K_G \\approx 1.0`.
    """
    d_mm, p_pass = sieve_data_to_percent_passing(
        diameters,
        values,
        input_type=input_type,
        pan_value=pan_value,
        diameter_unit=diameter_unit,
    )

    pcts = np.array([5.0, 10.0, 16.0, 25.0, 30.0, 35.0, 50.0, 60.0, 65.0, 75.0, 84.0, 85.0, 90.0, 95.0])
    ds = np.asarray(percentile_diameter(d_mm, p_pass, pcts, log_interpolation=True), dtype=np.float64)
    d5, d10, d16, d25, d30, d35, d50, d60, d65, d75, d84, d85, d90, d95 = [float(v) for v in ds]

    cu = d60 / d10
    cc = (d30**2) / (d10 * d60)
    sigma_g = math.sqrt(d84 / d16)

    # Percentiles en escala phi (% más grueso: phi_p = phi(D_{100-p}))
    phi_5 = float(mm_to_phi(d95))
    phi_16 = float(mm_to_phi(d84))
    phi_25 = float(mm_to_phi(d75))
    phi_50 = float(mm_to_phi(d50))
    phi_75 = float(mm_to_phi(d25))
    phi_84 = float(mm_to_phi(d16))
    phi_95 = float(mm_to_phi(d5))

    mz = (phi_16 + phi_50 + phi_84) / 3.0
    mean_mm = float(phi_to_mm(mz))
    sigma_i = (phi_84 - phi_16) / 4.0 + (phi_95 - phi_5) / 6.6

    denom_sk1 = 2.0 * max(phi_84 - phi_16, 1e-12)
    denom_sk2 = 2.0 * max(phi_95 - phi_5, 1e-12)
    sk_i = (phi_16 + phi_84 - 2.0 * phi_50) / denom_sk1 + (phi_5 + phi_95 - 2.0 * phi_50) / denom_sk2
    k_g = (phi_95 - phi_5) / max(2.44 * (phi_75 - phi_25), 1e-12)

    s_cls, sk_cls, k_cls = _classify_folk_ward(sigma_i, sk_i, k_g)

    # Fracciones texturales en bordes Wentworth (0.0625 mm finos/arena; 2.0 mm arena/grava)
    p_at_00625 = float(np.interp(math.log10(0.0625), np.log10(d_mm), p_pass))
    p_at_2 = float(np.interp(math.log10(2.0), np.log10(d_mm), p_pass))
    fines_pct = p_at_00625
    sand_pct = max(p_at_2 - p_at_00625, 0.0)
    gravel_pct = max(100.0 - p_at_2, 0.0)

    if (1.0 <= cc <= 3.0) and ((gravel_pct > sand_pct and cu >= 4.0) or (sand_pct >= gravel_pct and cu >= 6.0)):
        uscs_grad = "Bien graduado (W)"
    else:
        uscs_grad = "Mal graduado / Uniforme (P)"

    return GrainSizeCurveResult(
        diameters_mm=d_mm,
        percent_passing=p_pass,
        d5_mm=d5,
        d10_mm=d10,
        d16_mm=d16,
        d25_mm=d25,
        d30_mm=d30,
        d35_mm=d35,
        d50_mm=d50,
        d60_mm=d60,
        d65_mm=d65,
        d75_mm=d75,
        d84_mm=d84,
        d85_mm=d85,
        d90_mm=d90,
        d95_mm=d95,
        cu_uniformity=cu,
        cc_curvature=cc,
        sigma_g_geometric=sigma_g,
        folk_ward_mean_phi=mz,
        folk_ward_mean_mm=mean_mm,
        folk_ward_sorting_phi=sigma_i,
        folk_ward_skewness=sk_i,
        folk_ward_kurtosis=k_g,
        sorting_class=s_cls,
        skewness_class=sk_cls,
        kurtosis_class=k_cls,
        gravel_pct=gravel_pct,
        sand_pct=sand_pct,
        fines_pct=fines_pct,
        uscs_gradation=uscs_grad,
    )


def lognormal_grain_size_curve(
    d50_mm: float,
    sigma_g: float,
    *,
    n_points: int = 61,
    n_sigma: float = 3.0,
) -> GrainSizeCurveResult:
    """Genera una curva granulométrica log-normal a partir de ``d50_mm`` y ``sigma_g = sqrt(D84/D16)``."""
    if d50_mm <= 0.0 or sigma_g <= 1.0:
        raise ValueError("'d50_mm' debe ser > 0 y 'sigma_g' > 1.0.")
    if n_points < 5 or n_sigma <= 1.0:
        raise ValueError("'n_points' debe ser >= 5 y 'n_sigma' > 1.0.")

    z = np.linspace(-n_sigma, n_sigma, n_points)
    p_grid = 100.0 * ndtr(z)
    d_grid = d50_mm * np.power(sigma_g, z)
    return grain_size_curve_statistics(d_grid, p_grid, input_type="passing", diameter_unit="mm")


def blend_grain_size_curves(
    curves: list[tuple[ArrayLike, ArrayLike]] | tuple[tuple[ArrayLike, ArrayLike], ...],
    weights: ArrayLike,
    *,
    diameter_grid_mm: ArrayLike | None = None,
) -> GrainSizeCurveResult:
    """Mezcla múltiples curvas granulométricas ``[(d_mm, passing), ...]`` según pesos en masa."""
    w_a = np.asarray(weights, dtype=np.float64).ravel()
    if len(curves) < 1 or w_a.shape != (len(curves),):
        raise ValueError("'curves' y 'weights' deben tener la misma cantidad de elementos (>= 1).")
    if np.any(np.isnan(w_a)) or np.any(w_a < 0.0) or np.sum(w_a) <= 0.0:
        raise ValueError("'weights' debe contener valores >= 0 con suma > 0.")
    w_norm = w_a / np.sum(w_a)

    parsed = [
        sieve_data_to_percent_passing(d_c, p_c, input_type="passing", diameter_unit="mm")
        for d_c, p_c in curves
    ]

    if diameter_grid_mm is None:
        grid = np.unique(np.concatenate([d_c for d_c, _ in parsed]))
    else:
        grid = np.sort(np.unique(np.asarray(diameter_grid_mm, dtype=np.float64).ravel()))
        _validate_positive("diameter_grid_mm", grid, allow_zero=False)

    blended_p = np.zeros_like(grid, dtype=np.float64)
    log_grid = np.log10(grid)
    for w_k, (d_k, p_k) in zip(w_norm, parsed):
        p_interp = np.interp(log_grid, np.log10(d_k), p_k)
        blended_p += float(w_k) * p_interp

    blended_p = np.clip(np.maximum.accumulate(blended_p), 0.0, 100.0)
    return grain_size_curve_statistics(grid, blended_p, input_type="passing", diameter_unit="mm")


def check_filter_compatibility(
    base_diameters_mm: ArrayLike,
    base_passing: ArrayLike,
    filter_diameters_mm: ArrayLike,
    filter_passing: ArrayLike,
) -> dict[str, float | bool]:
    """Verifica criterios de filtro granular de Terzaghi / USACE entre material base y filtro."""
    base_st = grain_size_curve_statistics(base_diameters_mm, base_passing)
    filt_st = grain_size_curve_statistics(filter_diameters_mm, filter_passing)

    d15_f = float(percentile_diameter(filt_st.diameters_mm, filt_st.percent_passing, 15.0))
    d15_b = float(percentile_diameter(base_st.diameters_mm, base_st.percent_passing, 15.0))

    r_ret = d15_f / base_st.d85_mm
    r_perm = d15_f / d15_b
    r_d50 = filt_st.d50_mm / base_st.d50_mm

    return {
        "base_d15_mm": d15_b,
        "base_d50_mm": base_st.d50_mm,
        "base_d85_mm": base_st.d85_mm,
        "filter_d15_mm": d15_f,
        "filter_d50_mm": filt_st.d50_mm,
        "retention_ratio_D15f_D85b": r_ret,
        "permeability_ratio_D15f_D15b": r_perm,
        "median_ratio_D50f_D50b": r_d50,
        "meets_retention": r_ret <= 5.0,
        "meets_permeability": r_perm >= 5.0,
        "meets_median_ratio": r_d50 <= 25.0,
        "is_compatible": (r_ret <= 5.0) and (r_perm >= 5.0) and (r_d50 <= 25.0),
    }


__all__ = [
    "FallVelocityResult",
    "GrainSizeCurveResult",
    "blend_grain_size_curves",
    "check_filter_compatibility",
    "classify_grain_size_wentworth",
    "compare_fall_velocity_models",
    "dimensionless_grain_size",
    "fall_velocity_ferguson_church",
    "fall_velocity_rubey",
    "fall_velocity_soulsby",
    "fall_velocity_stokes",
    "fall_velocity_van_rijn",
    "flocculated_settling_velocity",
    "grain_size_curve_statistics",
    "lognormal_grain_size_curve",
    "mm_to_phi",
    "percentile_diameter",
    "phi_to_mm",
    "sieve_data_to_percent_passing",
    "water_density",
    "water_kinematic_viscosity",
]
