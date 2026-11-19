"""Propagación, refracción, difracción y dispersión de oleaje con corriente.

Implementa de forma vectorizada y autónoma las formulaciones científicas clásicas y
modernas de propagación lineal en aguas intermedias y someras, incluyendo el efecto
Doppler y el flujo de acción de onda en presencia de corrientes:

- **Relación de dispersión con corriente** (:func:`dispersion_with_current`):
  resuelve :math:`(\\omega_a - k\\,U_\\parallel)^2 = g\\,k\\tanh(k\\,h)` identificando
  automáticamente condiciones de **bloqueo por corriente opuesta** (*wave blocking*;
  Peregrine, 1976; Jonsson, 1990).
- **Refracción y someramiento generalizado** (:func:`shoaling_refraction_snell`):
  ley de Snell en contornos batimétricos rectos y paralelos (:math:`k\\sin\\theta = \\text{cte}`)
  acoplada a corrientes :math:`(U_x, U_y)` y conservación del flujo de acción de onda
  :math:`E\\,(c_{g,x} + U_x) / \\sigma` (Bretherton y Garrett, 1968).
- **Difracción del oleaje** detrás de dique semi-infinito impermeable
  (:func:`diffraction_semi_infinite_breakwater`, solución exacta de Sommerfeld /
  Penney y Price, 1952), en bocana entre dos diques (:func:`diffraction_breakwater_gap`)
  y para oleaje irregular multidireccional con espectro angular de Mitsuyasu
  (:func:`diffraction_random_directional`, Goda, Takayama y Suzuki, 1978; Goda, 2000).
- **Propagación 1D en perfil batimétrico** (:func:`propagate_wave_profile`): integra
  someramiento, refracción, corriente, fricción de fondo opcional y rotura por profundidad
  límite (:math:`H \\le \\gamma_b\\,h`).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.special import fresnel

from .constants import GRAVITY, PI_STANDARD
from .runup import _as_bool_out, _as_out, _validate_positive
from .waves import wave_friction_factor

FloatOrArray = NDArray[np.float64] | float


@dataclass(frozen=True)
class WaveCurrentDispersionResult:
    """Resultado de la relación de dispersión lineal con corriente de fondo."""

    omega_abs: FloatOrArray
    sigma_intrinsic: FloatOrArray
    k: FloatOrArray
    L: FloatOrArray
    kh: FloatOrArray
    c_abs: FloatOrArray
    c_intrinsic: FloatOrArray
    cg_abs: FloatOrArray
    cg_intrinsic: FloatOrArray
    u_parallel: FloatOrArray
    is_blocked: NDArray[np.bool_] | bool
    regime: NDArray[np.str_] | str


@dataclass(frozen=True)
class RefractionShoalingResult:
    """Resultado de transformación por someramiento, refracción y corriente (Snell)."""

    H: FloatOrArray
    H_unbroken: FloatOrArray
    L: FloatOrArray
    k: FloatOrArray
    kh: FloatOrArray
    theta_deg: FloatOrArray
    c_abs: FloatOrArray
    cg_intrinsic: FloatOrArray
    cg_abs_x: FloatOrArray
    sigma_intrinsic: FloatOrArray
    Ks: FloatOrArray
    Kr: FloatOrArray
    Kc_current: FloatOrArray
    breaking_index: FloatOrArray
    is_broken: NDArray[np.bool_] | bool
    is_blocked: NDArray[np.bool_] | bool


@dataclass(frozen=True)
class DiffractionResult:
    """Resultado de difracción de oleaje monocromático alrededor de un dique."""

    Kd: FloatOrArray
    phase_deg: FloatOrArray
    r_over_L: FloatOrArray
    theta_deg: FloatOrArray
    complex_amplitude: NDArray[np.complex128] | complex


@dataclass(frozen=True)
class GapDiffractionResult:
    """Resultado de difracción a través de una abertura entre dos diques."""

    Kd: FloatOrArray
    phase_deg: FloatOrArray
    gap_over_L: FloatOrArray
    complex_amplitude: NDArray[np.complex128] | complex


@dataclass(frozen=True)
class WavePropagationTransectResult:
    """Perfil completo de propagación de oleaje sobre batimetría 1D con corriente."""

    x: NDArray[np.float64]
    depth: NDArray[np.float64]
    U_x: NDArray[np.float64]
    U_y: NDArray[np.float64]
    H: NDArray[np.float64]
    H_unbroken: NDArray[np.float64]
    L: NDArray[np.float64]
    k: NDArray[np.float64]
    theta_deg: NDArray[np.float64]
    c_abs: NDArray[np.float64]
    cg_abs_x: NDArray[np.float64]
    Ks: NDArray[np.float64]
    Kr: NDArray[np.float64]
    Kc_current: NDArray[np.float64]
    Kd: NDArray[np.float64]
    is_broken: NDArray[np.bool_]
    is_blocked: NDArray[np.bool_]
    breaking_index: int | None
    breaking_x: float | None
    breaking_depth: float | None
    breaking_H: float | None
    breaking_angle_deg: float | None


def _intrinsic_sigma_and_cg(k: float, h: float, g: float) -> tuple[float, float, float]:
    """Calcula frecuencia intrínseca sigma, celeridad c y celeridad de grupo cg."""
    kh = k * h
    th = math.tanh(kh)
    sigma = math.sqrt(g * k * th)
    c_int = sigma / k
    if kh > 150.0:
        n_val = 0.5
    elif kh < 1e-6:
        n_val = 1.0 - (kh**2) / 3.0
    else:
        n_val = 0.5 * (1.0 + 2.0 * kh / math.sinh(2.0 * kh))
    return sigma, c_int, n_val * c_int


def _solve_1d_wave_current(omega_a: float, h: float, u_par: float, g: float) -> tuple[float, bool]:
    """Resuelve sqrt(g*k*tanh(k*h)) + k*u_par = omega_a en la rama incidente (cg + u_par >= 0)."""

    def f_disp(k: float) -> float:
        return math.sqrt(g * k * math.tanh(k * h)) + k * u_par - omega_a

    # Límite profundo sin corriente como referencia inicial
    k0_deep = (omega_a**2) / g

    if u_par >= 0.0:
        # F(k) crece estrictamente de -omega_a hasta +inf
        k_lo = 1e-12
        k_hi = max(k0_deep, 1.0 / max(h, 1e-6), math.sqrt(omega_a / max(g * h, 1e-9)))
        while f_disp(k_hi) < 0.0:
            k_hi *= 2.0
        return float(brentq(f_disp, k_lo, k_hi, xtol=1e-13, rtol=1e-13)), False

    # Corriente opuesta (u_par < 0): bloqueo inmediato si |u_par| >= sqrt(g*h)
    c_shallow = math.sqrt(g * h)
    if -u_par >= c_shallow:
        return float("nan"), True

    def f_prime(k: float) -> float:
        _, _, cg = _intrinsic_sigma_and_cg(k, h, g)
        return cg + u_par

    # Busca k_block donde f'(k) = cg(k) + u_par = 0
    k_block_deep = g / (4.0 * (u_par**2))
    k_right = max(k_block_deep * 2.0, 2.0 / h, 1e-8)
    while f_prime(k_right) > 0.0:
        k_right *= 2.0
        if k_right > 1e10:
            break

    k_block = float(brentq(f_prime, 1e-12, k_right, xtol=1e-13, rtol=1e-13))
    f_max = f_disp(k_block)
    if f_max < -1e-12:
        return float("nan"), True
    if abs(f_max) <= 1e-12:
        return k_block, False
    return float(brentq(f_disp, 1e-12, k_block, xtol=1e-13, rtol=1e-13)), False


def _solve_2d_snell_wave_current(
    omega_a: float,
    h: float,
    k_y: float,
    u_x: float,
    u_y: float,
    g: float,
) -> tuple[float, bool]:
    """Resuelve sqrt(g*k*tanh(k*h)) + sqrt(k^2 - k_y^2)*u_x + k_y*u_y = omega_a con k >= |k_y|."""
    abs_ky = abs(k_y)
    if abs_ky == 0.0 and abs(u_y) == 0.0:
        return _solve_1d_wave_current(omega_a, h, u_x, g)

    omega_eff = omega_a - k_y * u_y
    if omega_eff <= 0.0:
        return float("nan"), True

    def g_disp(k: float) -> float:
        kx = math.sqrt(max(k * k - abs_ky * abs_ky, 0.0))
        return math.sqrt(g * k * math.tanh(k * h)) + kx * u_x - omega_eff

    k_min = max(abs_ky, 1e-12)
    if g_disp(k_min) > 1e-12:
        # Reflexión total / cáustica por corriente favorable o variación batimétrica
        return float("nan"), True

    if u_x >= 0.0:
        k_hi = max(k_min * 1.1, (omega_eff**2) / g, omega_eff / math.sqrt(g * h))
        while g_disp(k_hi) < 0.0:
            k_hi *= 2.0
            if k_hi > 1e10:
                return float("nan"), True
        return float(brentq(g_disp, k_min, k_hi, xtol=1e-13, rtol=1e-13)), False

    # u_x < 0: evaluar g_disp(k) en malla logarítmica para localizar rama con dG/dk > 0 (cg_x + u_x > 0)
    k_deep_blk = g / (4.0 * (u_x**2))
    k_hi_search = max(k_min * 10.0, k_deep_blk * 4.0, 5.0 / max(h, 1e-4))
    grid = np.concatenate(
        [
            k_min + np.geomspace(1e-10 * max(k_min, 1.0), k_hi_search - k_min, 250),
        ]
    )
    vals = np.array([g_disp(float(kv)) for kv in grid], dtype=np.float64)
    idx_max = int(np.argmax(vals))
    if vals[idx_max] < -1e-12:
        return float("nan"), True

    k_peak = float(grid[idx_max])
    # Refinar pico donde cg_x + u_x = 0 si no está en el extremo
    if 0 < idx_max < len(grid) - 1:

        def g_prime(k: float) -> float:
            kx = math.sqrt(max(k * k - abs_ky * abs_ky, 1e-30))
            _, _, cg = _intrinsic_sigma_and_cg(k, h, g)
            return cg + u_x * (k / kx)

        left_b = float(grid[idx_max - 1])
        right_b = float(grid[idx_max + 1])
        if g_prime(left_b) > 0.0 and g_prime(right_b) < 0.0:
            k_peak = float(brentq(g_prime, left_b, right_b, xtol=1e-12))

    if g_disp(k_peak) < -1e-12:
        return float("nan"), True

    # Buscar el primer cruce de negativo a positivo en [k_min, k_peak] con cg_x + u_x >= 0
    sub = np.linspace(k_min, k_peak, 60)
    sub_vals = np.array([g_disp(float(kv)) for kv in sub], dtype=np.float64)
    pos_idx = np.where(sub_vals >= 0.0)[0]
    if pos_idx.size == 0:
        return float("nan"), True
    j = int(pos_idx[0])
    if j == 0:
        return float(sub[0]), False
    return float(brentq(g_disp, float(sub[j - 1]), float(sub[j]), xtol=1e-13, rtol=1e-13)), False


def dispersion_with_current(
    h: ArrayLike,
    T: ArrayLike,
    *,
    U: ArrayLike = 0.0,
    current_angle_deg: ArrayLike = 0.0,
    g: float = GRAVITY,
) -> WaveCurrentDispersionResult:
    """Resuelve la relación de dispersión lineal con corriente: ``(omega_a - k*U_par)^2 = g*k*tanh(k*h)``.

    El ángulo ``current_angle_deg`` es el ángulo entre el vector de corriente y la dirección
    de propagación de la ola (``0°`` a favor, ``180°`` en contra). Devuelve ``is_blocked=True``
    y ``NaN`` cuando una corriente opuesta impide el avance de energía (``cg + U_par < 0``).
    """
    if g <= 0.0:
        raise ValueError("La aceleración de gravedad 'g' debe ser > 0.")

    h_a, t_a, u_a, ang_a = np.broadcast_arrays(
        np.asarray(h, dtype=np.float64),
        np.asarray(T, dtype=np.float64),
        np.asarray(U, dtype=np.float64),
        np.asarray(current_angle_deg, dtype=np.float64),
    )
    is_scalar = h_a.ndim == 0
    _validate_positive("h", h_a, allow_zero=False)
    _validate_positive("T", t_a, allow_zero=False)
    if np.any(np.isnan(u_a)) or np.any(np.isnan(ang_a)):
        raise ValueError("'U' y 'current_angle_deg' no deben contener NaN.")

    omega_abs = 2.0 * PI_STANDARD / t_a
    u_par = u_a * np.cos(np.radians(ang_a))

    shape = h_a.shape
    k_arr = np.full(shape, np.nan, dtype=np.float64)
    l_arr = np.full(shape, np.nan, dtype=np.float64)
    kh_arr = np.full(shape, np.nan, dtype=np.float64)
    sig_arr = np.full(shape, np.nan, dtype=np.float64)
    c_abs = np.full(shape, np.nan, dtype=np.float64)
    c_int = np.full(shape, np.nan, dtype=np.float64)
    cg_abs = np.full(shape, np.nan, dtype=np.float64)
    cg_int = np.full(shape, np.nan, dtype=np.float64)
    blk_arr = np.zeros(shape, dtype=bool)
    reg_arr = np.full(shape, "blocked", dtype="<U16")

    for idx in np.ndindex(shape):
        om_i = float(omega_abs[idx])
        h_i = float(h_a[idx])
        up_i = float(u_par[idx])
        k_val, blocked = _solve_1d_wave_current(om_i, h_i, up_i, g)
        if blocked or math.isnan(k_val):
            blk_arr[idx] = True
            reg_arr[idx] = "blocked"
            continue
        sig_i, ci_i, cgi_i = _intrinsic_sigma_and_cg(k_val, h_i, g)
        kh_i = k_val * h_i
        k_arr[idx] = k_val
        l_arr[idx] = 2.0 * PI_STANDARD / k_val
        kh_arr[idx] = kh_i
        sig_arr[idx] = sig_i
        c_int[idx] = ci_i
        c_abs[idx] = om_i / k_val
        cg_int[idx] = cgi_i
        cg_abs[idx] = cgi_i + up_i
        if kh_i >= math.pi:
            reg_arr[idx] = "deep"
        elif kh_i <= math.pi / 10.0:
            reg_arr[idx] = "shallow"
        else:
            reg_arr[idx] = "intermediate"

    reg_out: NDArray[np.str_] | str = (
        str(np.asarray(reg_arr).item()) if is_scalar else np.asarray(reg_arr, dtype=str)
    )
    return WaveCurrentDispersionResult(
        omega_abs=_as_out(omega_abs, is_scalar),
        sigma_intrinsic=_as_out(sig_arr, is_scalar),
        k=_as_out(k_arr, is_scalar),
        L=_as_out(l_arr, is_scalar),
        kh=_as_out(kh_arr, is_scalar),
        c_abs=_as_out(c_abs, is_scalar),
        c_intrinsic=_as_out(c_int, is_scalar),
        cg_abs=_as_out(cg_abs, is_scalar),
        cg_intrinsic=_as_out(cg_int, is_scalar),
        u_parallel=_as_out(u_par, is_scalar),
        is_blocked=_as_bool_out(blk_arr, is_scalar),
        regime=reg_out,
    )


def shoaling_refraction_snell(
    H0: ArrayLike,
    T: ArrayLike,
    theta0_deg: ArrayLike,
    h: ArrayLike,
    *,
    U: ArrayLike = 0.0,
    current_dir_deg: ArrayLike = 0.0,
    gamma_b: float = 0.78,
    limit_breaking: bool = True,
    g: float = GRAVITY,
) -> RefractionShoalingResult:
    """Calcula someramiento, refracción (ley de Snell) e interacción con corriente en profundidad ``h``.

    ``theta0_deg`` se mide respecto a la normal a los contornos batimétricos (eje ``+x``
    hacia la costa, en ``(-90, 90)`` grados). ``current_dir_deg`` sigue la misma convención
    direccional (``0°`` hacia tierra en ``+x``, ``90°`` corriente longitudinal en ``+y``).
    Conserva ``k * sin(theta) = k0 * sin(theta0)`` y el flujo de acción de onda
    ``E * (cg * cos(theta) + U_x) / sigma``.
    """
    if g <= 0.0 or gamma_b <= 0.0:
        raise ValueError("'g' y 'gamma_b' deben ser > 0.")

    h0_a, t_a, th0_a, h_a, u_a, cdir_a = np.broadcast_arrays(
        np.asarray(H0, dtype=np.float64),
        np.asarray(T, dtype=np.float64),
        np.asarray(theta0_deg, dtype=np.float64),
        np.asarray(h, dtype=np.float64),
        np.asarray(U, dtype=np.float64),
        np.asarray(current_dir_deg, dtype=np.float64),
    )
    is_scalar = h0_a.ndim == 0

    _validate_positive("H0", h0_a, allow_zero=True)
    _validate_positive("T", t_a, allow_zero=False)
    _validate_positive("h", h_a, allow_zero=False)
    if np.any(np.abs(th0_a) >= 90.0):
        raise ValueError("El ángulo profundo 'theta0_deg' debe pertenecer a (-90, 90) grados.")
    if np.any(np.isnan(u_a)) or np.any(np.isnan(cdir_a)):
        raise ValueError("'U' y 'current_dir_deg' no deben contener NaN.")

    omega_a = 2.0 * PI_STANDARD / t_a
    k0 = (omega_a**2) / g
    cg0 = g / (2.0 * omega_a)
    th0_rad = np.radians(th0_a)
    ky = k0 * np.sin(th0_rad)
    ux = u_a * np.cos(np.radians(cdir_a))
    uy = u_a * np.sin(np.radians(cdir_a))

    shape = h0_a.shape
    H_out = np.full(shape, np.nan, dtype=np.float64)
    Hunb_out = np.full(shape, np.nan, dtype=np.float64)
    L_out = np.full(shape, np.nan, dtype=np.float64)
    k_out = np.full(shape, np.nan, dtype=np.float64)
    kh_out = np.full(shape, np.nan, dtype=np.float64)
    th_out = np.full(shape, np.nan, dtype=np.float64)
    ca_out = np.full(shape, np.nan, dtype=np.float64)
    cgi_out = np.full(shape, np.nan, dtype=np.float64)
    cgax_out = np.full(shape, np.nan, dtype=np.float64)
    sig_out = np.full(shape, np.nan, dtype=np.float64)
    Ks_out = np.full(shape, np.nan, dtype=np.float64)
    Kr_out = np.full(shape, np.nan, dtype=np.float64)
    Kc_out = np.full(shape, np.nan, dtype=np.float64)
    bi_out = np.full(shape, np.nan, dtype=np.float64)
    brk_arr = np.zeros(shape, dtype=bool)
    blk_arr = np.zeros(shape, dtype=bool)

    for idx in np.ndindex(shape):
        om_i = float(omega_a[idx])
        h_i = float(h_a[idx])
        ky_i = float(ky[idx])
        ux_i = float(ux[idx])
        uy_i = float(uy[idx])
        c0_i = float(cg0[idx])
        cos0_i = float(math.cos(float(th0_rad[idx])))

        k_val, blocked = _solve_2d_snell_wave_current(om_i, h_i, ky_i, ux_i, uy_i, g)
        if blocked or math.isnan(k_val):
            blk_arr[idx] = True
            continue

        sin_th = max(-1.0, min(1.0, ky_i / k_val))
        cos_th = math.sqrt(max(1.0 - sin_th * sin_th, 0.0))
        th_deg_val = math.degrees(math.asin(sin_th))
        sig_i, _, cgi_i = _intrinsic_sigma_and_cg(k_val, h_i, g)
        cg_abs_x = cgi_i * cos_th + ux_i
        if cg_abs_x <= 1e-12:
            blk_arr[idx] = True
            continue

        ks_i = math.sqrt(c0_i / max(cgi_i, 1e-30))
        kr_i = math.sqrt(cos0_i / max(cos_th, 1e-30))
        kc_i = math.sqrt((sig_i / om_i) * (cgi_i * cos_th / cg_abs_x))
        h_unb = float(h0_a[idx]) * ks_i * kr_i * kc_i
        h_max = gamma_b * h_i
        is_brk = h_unb >= h_max and float(h0_a[idx]) > 0.0
        h_final = min(h_unb, h_max) if limit_breaking else h_unb

        k_out[idx] = k_val
        L_out[idx] = 2.0 * PI_STANDARD / k_val
        kh_out[idx] = k_val * h_i
        th_out[idx] = th_deg_val
        ca_out[idx] = om_i / k_val
        cgi_out[idx] = cgi_i
        cgax_out[idx] = cg_abs_x
        sig_out[idx] = sig_i
        Ks_out[idx] = ks_i
        Kr_out[idx] = kr_i
        Kc_out[idx] = kc_i
        Hunb_out[idx] = h_unb
        H_out[idx] = h_final
        bi_out[idx] = h_final / h_i
        brk_arr[idx] = is_brk

    return RefractionShoalingResult(
        H=_as_out(H_out, is_scalar),
        H_unbroken=_as_out(Hunb_out, is_scalar),
        L=_as_out(L_out, is_scalar),
        k=_as_out(k_out, is_scalar),
        kh=_as_out(kh_out, is_scalar),
        theta_deg=_as_out(th_out, is_scalar),
        c_abs=_as_out(ca_out, is_scalar),
        cg_intrinsic=_as_out(cgi_out, is_scalar),
        cg_abs_x=_as_out(cgax_out, is_scalar),
        sigma_intrinsic=_as_out(sig_out, is_scalar),
        Ks=_as_out(Ks_out, is_scalar),
        Kr=_as_out(Kr_out, is_scalar),
        Kc_current=_as_out(Kc_out, is_scalar),
        breaking_index=_as_out(bi_out, is_scalar),
        is_broken=_as_bool_out(brk_arr, is_scalar),
        is_blocked=_as_bool_out(blk_arr, is_scalar),
    )


def _fresnel_diffraction_term(u: NDArray[np.float64]) -> NDArray[np.complex128]:
    """Evalúa ``f(u) = (1+i)/2 * int_{-inf}^{u} exp(-i*pi/2 * t^2) dt``."""
    s, c = fresnel(u)
    return 0.5 + 0.5 * (1.0 + 1j) * (c - 1j * s)


def _semi_infinite_potential(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    L: NDArray[np.float64],
    theta0_rad: NDArray[np.float64],
    *,
    tip_x: float | NDArray[np.float64] = 0.0,
    tip_y: float | NDArray[np.float64] = 0.0,
    mirror_x: bool = False,
) -> tuple[NDArray[np.complex128], NDArray[np.float64], NDArray[np.float64]]:
    """Potencial complejo exacto de Sommerfeld / Penney-Price (1952)."""
    dx = (tip_x - x) if mirror_x else (x - tip_x)
    dy = y - tip_y
    th0 = (PI_STANDARD - theta0_rad) if mirror_x else theta0_rad

    r = np.hypot(dx, dy)
    th = np.mod(np.arctan2(dy, dx), 2.0 * PI_STANDARD)
    k = 2.0 * PI_STANDARD / L

    root = np.sqrt(8.0 * r / L)
    u1 = root * np.sin(0.5 * (th - th0))
    u2 = -root * np.sin(0.5 * (th + th0))

    phi = (
        np.exp(-1j * k * r * np.cos(th - th0)) * _fresnel_diffraction_term(u1)
        + np.exp(-1j * k * r * np.cos(th + th0)) * _fresnel_diffraction_term(u2)
    )
    return phi, r, th


def diffraction_semi_infinite_breakwater(
    x: ArrayLike,
    y: ArrayLike,
    L: ArrayLike,
    *,
    wave_angle_deg: ArrayLike = 90.0,
    tip_x: float = 0.0,
    tip_y: float = 0.0,
) -> DiffractionResult:
    """Coeficiente de difracción detrás de un dique semi-infinito impermeable (Penney y Price, 1952).

    El dique ocupa la semirrecta ``x >= tip_x, y = tip_y``; el oleaje incide desde el
    semiplano expuesto ``y < tip_y`` formando el ángulo ``wave_angle_deg`` (en ``(0, 180)`` grados)
    con el eje positivo del dique (``90°`` equivale a incidencia perpendicular hacia ``+y``).
    """
    x_a, y_a, l_a, ang_a = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        np.asarray(L, dtype=np.float64),
        np.asarray(wave_angle_deg, dtype=np.float64),
    )
    is_scalar = x_a.ndim == 0
    _validate_positive("L", l_a, allow_zero=False)
    if np.any((ang_a <= 0.0) | (ang_a >= 180.0) | np.isnan(ang_a)):
        raise ValueError("'wave_angle_deg' debe pertenecer al intervalo abierto (0, 180) grados.")

    phi, r, th = _semi_infinite_potential(
        x_a, y_a, l_a, np.radians(ang_a), tip_x=tip_x, tip_y=tip_y, mirror_x=False
    )
    kd = np.abs(phi)
    phase = np.degrees(np.angle(phi))
    r_over_l = r / l_a
    th_deg = np.degrees(th)

    return DiffractionResult(
        Kd=_as_out(kd, is_scalar),
        phase_deg=_as_out(phase, is_scalar),
        r_over_L=_as_out(r_over_l, is_scalar),
        theta_deg=_as_out(th_deg, is_scalar),
        complex_amplitude=complex(np.asarray(phi).item()) if is_scalar else phi,
    )


def diffraction_breakwater_gap(
    x: ArrayLike,
    y: ArrayLike,
    L: ArrayLike,
    gap_width: ArrayLike,
    *,
    wave_angle_deg: ArrayLike = 90.0,
) -> GapDiffractionResult:
    """Coeficiente de difracción en la bocana entre dos diques colineales (Penney y Price, 1952).

    La abertura de ancho ``gap_width`` se ubica entre ``x = -gap_width/2`` y ``x = +gap_width/2``
    sobre ``y = 0``; la zona abrigada de interés es el semiplano ``y >= 0`` para olas que
    inciden desde ``y < 0`` con ángulo ``wave_angle_deg`` en ``(0, 180)`` grados.
    """
    x_a, y_a, l_a, b_a, ang_a = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        np.asarray(L, dtype=np.float64),
        np.asarray(gap_width, dtype=np.float64),
        np.asarray(wave_angle_deg, dtype=np.float64),
    )
    is_scalar = x_a.ndim == 0
    _validate_positive("L", l_a, allow_zero=False)
    _validate_positive("gap_width", b_a, allow_zero=False)
    if np.any(y_a < 0.0):
        raise ValueError("La coordenada 'y' en el recinto abrigado de la bocana debe ser >= 0.")
    if np.any((ang_a <= 0.0) | (ang_a >= 180.0) | np.isnan(ang_a)):
        raise ValueError("'wave_angle_deg' debe pertenecer al intervalo abierto (0, 180) grados.")

    th0 = np.radians(ang_a)
    k = 2.0 * PI_STANDARD / l_a
    cos_th0 = np.cos(th0)
    sin_th0 = np.sin(th0)

    # Superposición coherente de ambos semi-diques referidos al origen global (0, 0)
    phi_inc = np.exp(-1j * k * (x_a * cos_th0 + y_a * sin_th0))
    phi_r_loc, _, _ = _semi_infinite_potential(x_a, y_a, l_a, th0, tip_x=0.5 * b_a, mirror_x=False)
    phi_l_loc, _, _ = _semi_infinite_potential(x_a, y_a, l_a, th0, tip_x=-0.5 * b_a, mirror_x=True)
    phi_r = np.exp(-1j * k * (0.5 * b_a) * cos_th0) * phi_r_loc
    phi_l = np.exp(-1j * k * (-0.5 * b_a) * cos_th0) * phi_l_loc

    phi_gap = phi_r + phi_l - phi_inc
    kd = np.abs(phi_gap)
    phase = np.degrees(np.angle(phi_gap))

    return GapDiffractionResult(
        Kd=_as_out(kd, is_scalar),
        phase_deg=_as_out(phase, is_scalar),
        gap_over_L=_as_out(b_a / l_a, is_scalar),
        complex_amplitude=complex(np.asarray(phi_gap).item()) if is_scalar else phi_gap,
    )


def diffraction_random_directional(
    x: ArrayLike,
    y: ArrayLike,
    L: float,
    *,
    wave_angle_deg: float = 90.0,
    s_max: float = 25.0,
    gap_width: float | None = None,
    n_components: int = 45,
    tip_x: float = 0.0,
    tip_y: float = 0.0,
) -> FloatOrArray:
    """Coeficiente de difracción para oleaje irregular direccional (Goda, 1978, 2000).

    Pondera la energía angular mediante la distribución de Mitsuyasu
    :math:`D(\\theta) \\propto \\cos^{2s}((\\theta - \\theta_0)/2)` (típicamente
    ``s_max = 10`` mar de viento, ``25`` transición y ``75`` swell lejano) sobre
    un dique semi-infinito o sobre una bocana (``gap_width is not None``).
    """
    if L <= 0.0 or s_max <= 0.0:
        raise ValueError("'L' y 's_max' deben ser > 0.")
    if wave_angle_deg <= 0.0 or wave_angle_deg >= 180.0:
        raise ValueError("'wave_angle_deg' debe pertenecer a (0, 180) grados.")
    if n_components < 3:
        raise ValueError("'n_components' debe ser >= 3.")

    x_a, y_a = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
    )
    is_scalar = x_a.ndim == 0

    lo_ang = max(2.0, wave_angle_deg - 75.0)
    hi_ang = min(178.0, wave_angle_deg + 75.0)
    thetas = np.linspace(lo_ang, hi_ang, n_components)
    half_diff = 0.5 * np.radians(thetas - wave_angle_deg)
    raw_w = np.power(np.maximum(np.cos(half_diff), 0.0), 2.0 * s_max)
    weights = raw_w / np.sum(raw_w)

    kd_sq = np.zeros_like(x_a, dtype=np.float64)
    for w_j, th_j in zip(weights, thetas):
        if gap_width is not None:
            kd_j = diffraction_breakwater_gap(
                x_a, y_a, L, gap_width, wave_angle_deg=float(th_j)
            ).Kd
        else:
            kd_j = diffraction_semi_infinite_breakwater(
                x_a, y_a, L, wave_angle_deg=float(th_j), tip_x=tip_x, tip_y=tip_y
            ).Kd
        kd_sq += float(w_j) * np.power(np.asarray(kd_j, dtype=np.float64), 2)

    kd_rand = np.sqrt(kd_sq)
    return _as_out(kd_rand, is_scalar)


def propagate_wave_profile(
    x: ArrayLike,
    depth: ArrayLike,
    H0: float,
    T: float,
    *,
    theta0_deg: float = 0.0,
    U_x: ArrayLike = 0.0,
    U_y: ArrayLike = 0.0,
    Kd: ArrayLike = 1.0,
    gamma_b: float = 0.78,
    friction_f: float | None = 0.0,
    k_s: float | None = None,
    g: float = GRAVITY,
) -> WavePropagationTransectResult:
    """Propaga un tren de olas sobre un perfil batimétrico 1D ``(x, depth)`` hacia la costa.

    Integra paso a paso desde la estación ``0`` (mar afuera) hasta el extremo costero:
    - Relación de dispersión con corriente 2D ``(U_x, U_y)`` y ley de Snell.
    - Someramiento, refracción y conservación del flujo de acción de onda.
    - Factor de difracción espacial ``Kd(x)`` opcional.
    - Disipación por fricción del lecho cuando ``friction_f > 0`` o mediante rugosidad ``k_s``
      (calculando ``f_w`` con :func:`sedmor.waves.wave_friction_factor`).
    - Criterio de rotura por profundidad límite :math:`H \\le \\gamma_b\\,h`.
    """
    if H0 < 0.0 or T <= 0.0 or g <= 0.0 or gamma_b <= 0.0:
        raise ValueError("'T', 'g', 'gamma_b' deben ser > 0 y 'H0' >= 0.")
    if abs(theta0_deg) >= 90.0:
        raise ValueError("'theta0_deg' debe pertenecer a (-90, 90) grados.")

    x_a = np.asarray(x, dtype=np.float64).ravel()
    h_a = np.asarray(depth, dtype=np.float64).ravel()
    n_pts = x_a.size
    if n_pts < 2 or h_a.shape != (n_pts,):
        raise ValueError("'x' y 'depth' deben ser arreglos 1D de igual longitud (>= 2).")
    _validate_positive("depth", h_a, allow_zero=False)
    if np.any(np.diff(x_a) <= 0.0):
        raise ValueError("La coordenada 'x' debe ser estrictamente creciente hacia la costa.")

    ux_a = np.broadcast_to(np.asarray(U_x, dtype=np.float64), (n_pts,)).astype(np.float64)
    uy_a = np.broadcast_to(np.asarray(U_y, dtype=np.float64), (n_pts,)).astype(np.float64)
    kd_a = np.broadcast_to(np.asarray(Kd, dtype=np.float64), (n_pts,)).astype(np.float64)
    _validate_positive("Kd", kd_a, allow_zero=False)

    if k_s is not None and k_s <= 0.0:
        raise ValueError("'k_s' debe ser > 0 cuando se especifica.")
    if friction_f is not None and friction_f < 0.0:
        raise ValueError("'friction_f' debe ser >= 0.")

    omega_a = 2.0 * PI_STANDARD / T
    k0 = (omega_a**2) / g
    cg0 = g / (2.0 * omega_a)
    th0_rad = math.radians(theta0_deg)
    ky = k0 * math.sin(th0_rad)
    cos0 = math.cos(th0_rad)

    H_arr = np.full(n_pts, np.nan, dtype=np.float64)
    Hunb_arr = np.full(n_pts, np.nan, dtype=np.float64)
    L_arr = np.full(n_pts, np.nan, dtype=np.float64)
    k_arr = np.full(n_pts, np.nan, dtype=np.float64)
    th_arr = np.full(n_pts, np.nan, dtype=np.float64)
    ca_arr = np.full(n_pts, np.nan, dtype=np.float64)
    cgax_arr = np.full(n_pts, np.nan, dtype=np.float64)
    ks_arr = np.full(n_pts, np.nan, dtype=np.float64)
    kr_arr = np.full(n_pts, np.nan, dtype=np.float64)
    kc_arr = np.full(n_pts, np.nan, dtype=np.float64)
    brk_arr = np.zeros(n_pts, dtype=bool)
    blk_arr = np.zeros(n_pts, dtype=bool)

    # Flujo de acción normalizado H^2 * (cg_x + Ux) / sigma; se inicia con la referencia profunda
    flux_act = (H0**2) * cg0 * cos0 / omega_a
    # Ajustar altura en x[0] según difracción/bloqueo inicial
    use_friction = (k_s is not None) or (friction_f is not None and friction_f > 0.0)

    brk_idx: int | None = None
    brk_x: float | None = None
    brk_h: float | None = None
    brk_H: float | None = None
    brk_ang: float | None = None

    for i in range(n_pts):
        h_i = float(h_a[i])
        ux_i = float(ux_a[i])
        uy_i = float(uy_a[i])
        kd_i = float(kd_a[i])

        k_val, blocked = _solve_2d_snell_wave_current(omega_a, h_i, ky, ux_i, uy_i, g)
        if blocked or math.isnan(k_val):
            blk_arr[i:] = True
            break

        sin_th = max(-1.0, min(1.0, ky / k_val))
        cos_th = math.sqrt(max(1.0 - sin_th * sin_th, 0.0))
        th_deg_val = math.degrees(math.asin(sin_th))
        sig_i, _, cgi_i = _intrinsic_sigma_and_cg(k_val, h_i, g)
        cg_abs_x = cgi_i * cos_th + ux_i
        if cg_abs_x <= 1e-12:
            blk_arr[i:] = True
            break

        ks_i = math.sqrt(cg0 / max(cgi_i, 1e-30))
        kr_i = math.sqrt(cos0 / max(cos_th, 1e-30))
        kc_i = math.sqrt((sig_i / omega_a) * (cgi_i * cos_th / cg_abs_x))

        h_from_flux = math.sqrt(max(flux_act * sig_i / cg_abs_x, 0.0))
        h_unb = h_from_flux * kd_i
        h_max = gamma_b * h_i
        broken = h_unb >= h_max and H0 > 0.0
        h_curr = min(h_unb, h_max)

        k_arr[i] = k_val
        L_arr[i] = 2.0 * PI_STANDARD / k_val
        th_arr[i] = th_deg_val
        ca_arr[i] = omega_a / k_val
        cgax_arr[i] = cg_abs_x
        ks_arr[i] = ks_i
        kr_arr[i] = kr_i
        kc_arr[i] = kc_i
        Hunb_arr[i] = h_unb
        H_arr[i] = h_curr
        brk_arr[i] = broken

        if broken and brk_idx is None:
            brk_idx = i
            brk_x = float(x_a[i])
            brk_h = h_i
            brk_H = h_curr
            brk_ang = th_deg_val

        # Evolución al paso siguiente si hay fricción o si la rotura disipó energía
        if i + 1 < n_pts:
            # Si rompió en i, limita el flujo de acción saliente a la altura máxima h_curr
            if broken:
                flux_act = (h_curr / max(kd_i, 1e-12)) ** 2 * cg_abs_x / sig_i
            if use_friction and h_curr > 0.0:
                dx = float(x_a[i + 1] - x_a[i])
                kh_i = k_val * h_i
                sh = math.sinh(min(kh_i, 300.0))
                a_orb = h_curr / (2.0 * max(sh, 1e-12))
                u_orb = sig_i * a_orb
                if k_s is not None:
                    fw_i = float(wave_friction_factor(a_orb, k_s))
                else:
                    fw_i = float(friction_f or 0.0)
                # d/dx(H^2 * (cg_x + Ux)/sig) = - (16/(3*pi*g)) * fw * u_orb^3 / sig
                d_flux = (16.0 / (3.0 * PI_STANDARD * g)) * fw_i * (u_orb**3) / sig_i
                flux_act = max(flux_act - d_flux * dx, 0.0)

    return WavePropagationTransectResult(
        x=x_a,
        depth=h_a,
        U_x=ux_a,
        U_y=uy_a,
        H=H_arr,
        H_unbroken=Hunb_arr,
        L=L_arr,
        k=k_arr,
        theta_deg=th_arr,
        c_abs=ca_arr,
        cg_abs_x=cgax_arr,
        Ks=ks_arr,
        Kr=kr_arr,
        Kc_current=kc_arr,
        Kd=kd_a,
        is_broken=brk_arr,
        is_blocked=blk_arr,
        breaking_index=brk_idx,
        breaking_x=brk_x,
        breaking_depth=brk_h,
        breaking_H=brk_H,
        breaking_angle_deg=brk_ang,
    )


__all__ = [
    "DiffractionResult",
    "GapDiffractionResult",
    "RefractionShoalingResult",
    "WaveCurrentDispersionResult",
    "WavePropagationTransectResult",
    "diffraction_breakwater_gap",
    "diffraction_random_directional",
    "diffraction_semi_infinite_breakwater",
    "dispersion_with_current",
    "propagate_wave_profile",
    "shoaling_refraction_snell",
]
