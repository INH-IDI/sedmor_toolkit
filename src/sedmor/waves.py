"""Modulo de hidrodinamica del oleaje y capa limite oscilatoria.

Implementa las 8 herramientas de calculo en Excel de Leo C. van Rijn (2013)
relacionadas con propagacion, cinematica, capa limite, rotura, asimetria y
corrientes inducidas por oleaje:

1. ``Computation Basic wave parameters 1.xlsx``
2. ``Computation Basic wave parameters 2.xlsx``
3. ``Computation Orbital vel and Bed sh stress boundary layer.xlsx``
4. ``Computation Wave height Battjes Janssen model.xlsx``
5. ``Computation Wave height decrease due to friction.xlsx``
6. ``Computation Skewness and Asymmetry.xlsx``
7. ``Computation Undertow velocity.xlsx``
8. ``Computation Eulerian streaming.xlsx``
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from sedmor.constants import (
    GRAVITY,
    KINEMATIC_VISCOSITY_15C,
    PI_EXCEL,
    PI_STANDARD,
    RHO_FRESH_WATER,
    RHO_SEA_WATER,
    RHO_SEDIMENT,
    TWO_PI_EXCEL,
)

# Umbral inferior aproximado en Excel antes de emitir #NUM! por underflow en potencia
_EXCEL_UNDERFLOW_LIMIT = 2.2250738585072014e-308


def _is_scalar(x: Any) -> bool:
    return np.ndim(x) == 0


def _pi_consts(strict_excel: bool, pi_approx: float | None = None) -> tuple[float, float]:
    """Devuelve ``(pi_val, two_pi_val)`` segun ``strict_excel``."""
    if strict_excel:
        if pi_approx is not None:
            return pi_approx, 2.0 * pi_approx
        return PI_EXCEL, TWO_PI_EXCEL
    return PI_STANDARD, 2.0 * PI_STANDARD


def _hunt_kh_polynomial(y: np.ndarray, *, strict_excel: bool, typo_y5: bool = False) -> np.ndarray:
    """Evalua el polinomio de la aproximacion de Hunt (1979) para ``kh``."""
    if strict_excel and typo_y5:
        poly = (
            1.0
            + 0.666 * y
            + 0.355 * y**2
            + 0.16 * y**3
            + 0.06332 * y**4
            + 0.0218 * y**5
            + 0.00654 * y**5
        )
        return np.sqrt(y**2 + y / poly)

    if strict_excel:
        # En la mayoria de las hojas Excel se escribio 0.000654^Y^6 por error tipografico;
        # Excel evalua izquierda-a-derecha y produce #NUM! cuando hay underflow extremo.
        base_poly = (
            1.0
            + 0.666 * y
            + 0.3555 * y**2
            + 0.161 * y**3
            + 0.0632 * y**4
            + 0.0218 * y**5
        )
        with np.errstate(under="ignore", over="ignore", invalid="ignore"):
            p1 = np.power(0.000654, y)
            last = np.power(p1, 6.0)
            bad = (p1 == 0.0) | (last < _EXCEL_UNDERFLOW_LIMIT) | ~np.isfinite(last)
            kh = np.sqrt(y**2 + y * np.power(base_poly + last, -1.0))
            kh = np.where(bad, np.nan, kh)
        return kh

    poly = (
        1.0
        + 0.666 * y
        + 0.3555 * y**2
        + 0.161 * y**3
        + 0.0632 * y**4
        + 0.0218 * y**5
        + 0.00654 * y**6
    )
    return np.sqrt(y**2 + y / poly)


def wavelength_nielsen(
    h: float | np.ndarray,
    T: float | np.ndarray,
    *,
    g: float = GRAVITY,
    strict_excel: bool = False,
) -> dict[str, Any]:
    """Calcula la longitud de onda mediante la aproximacion explicita de Nielsen.

    Parameters
    ----------
    h : float or array_like
        Profundidad de agua [m].
    T : float or array_like
        Periodo del oleaje [s].
    g : float, optional
        Aceleracion gravitacional [m/s^2]. Por defecto ``9.81``.
    strict_excel : bool, optional
        Si es ``True``, usa ``pi = 3.14`` para reproducir las celdas de
        ``Computation Basic wave parameters 1.xlsx`` (hoja ``wave length``).

    Returns
    -------
    dict
        Diccionario con ``Y``, ``kh``, ``k`` [1/m] y ``L`` [m].
    """
    scalar = _is_scalar(h) and _is_scalar(T)
    h_arr = np.asarray(h, dtype=float)
    T_arr = np.asarray(T, dtype=float)
    _, two_pi = _pi_consts(strict_excel)

    y = (two_pi / T_arr) ** 2 * h_arr / g
    kh = np.sqrt(y) * (1.0 + 0.166 * y + 0.031 * y**2)
    k = kh / h_arr
    L = two_pi / k

    if scalar:
        return {"Y": float(y), "kh": float(kh), "k": float(k), "L": float(L)}
    return {"Y": y, "kh": kh, "k": k, "L": L}


def wavelength_hunt(
    h: float | np.ndarray,
    T: float | np.ndarray,
    *,
    g: float = GRAVITY,
    typo_y5: bool = False,
    strict_excel: bool = False,
    fixed_hunt_coeff: bool = False,
) -> dict[str, Any]:
    """Calcula la longitud de onda mediante la aproximacion de Hunt (1979).

    Parameters
    ----------
    h : float or array_like
        Profundidad de agua [m].
    T : float or array_like
        Periodo del oleaje [s].
    g : float, optional
        Aceleracion gravitacional [m/s^2].
    typo_y5 : bool, optional
        Si ``True`` y ``strict_excel=True``, reproduce el coeficiente ``0.00654*y^5``
        y ``4.02*h/T^2`` de la hoja ``Orbital velocities``.
    strict_excel : bool, optional
        Si es ``True``, reproduce exactamente las formulas Excel originales
        (incluyendo ``0.000654^Y^6`` que genera ``NaN``/``#NUM!`` en aguas
        profundas salvo que ``fixed_hunt_coeff=True``).
    fixed_hunt_coeff : bool, optional
        Algunas hojas (p. ej. asimetria) ya corregian el ultimo termino a
        ``0.00654*Y^6`` manteniendo ``3.14`` y ``6.28``.

    Returns
    -------
    dict
        Diccionario con ``Y``, ``kh``, ``k`` [1/m] y ``L`` [m].
    """
    scalar = _is_scalar(h) and _is_scalar(T)
    h_arr = np.asarray(h, dtype=float)
    T_arr = np.asarray(T, dtype=float)
    _, two_pi = _pi_consts(strict_excel)

    if strict_excel and typo_y5:
        y = 4.02 * h_arr / (T_arr**2)
        kh = _hunt_kh_polynomial(y, strict_excel=True, typo_y5=True)
    else:
        y = (two_pi / T_arr) ** 2 * h_arr / g
        use_excel_bug = strict_excel and not fixed_hunt_coeff
        kh = _hunt_kh_polynomial(y, strict_excel=use_excel_bug, typo_y5=False)

    k = kh / h_arr
    L = two_pi / k

    if scalar:
        return {"Y": float(y), "kh": float(kh), "k": float(k), "L": float(L)}
    return {"Y": y, "kh": kh, "k": k, "L": L}


def dispersion_relation(
    h: float | np.ndarray,
    T: float | np.ndarray,
    *,
    method: str = "exact",
    g: float = GRAVITY,
    strict_excel: bool = False,
) -> dict[str, Any]:
    """Resuelve la relacion de dispersion lineal ``omega^2 = g * k * tanh(k * h)``.

    Parameters
    ----------
    h : float or array_like
        Profundidad de agua [m].
    T : float or array_like
        Periodo del oleaje [s].
    method : {"exact", "hunt", "nielsen"}, optional
        Metodo de solucion. ``"exact"`` resuelve iterativamente mediante Newton-Raphson.
    g : float, optional
        Aceleracion gravitacional [m/s^2].
    strict_excel : bool, optional
        Propaga las convenciones Excel a las aproximaciones explicitas.

    Returns
    -------
    dict
        Diccionario con ``Y``, ``kh``, ``k`` [1/m] y ``L`` [m].
    """
    method_low = method.lower()
    if method_low == "hunt":
        return wavelength_hunt(h, T, g=g, strict_excel=strict_excel)
    if method_low == "nielsen":
        return wavelength_nielsen(h, T, g=g, strict_excel=strict_excel)
    if method_low != "exact":
        raise ValueError(f"Metodo desconocido '{method}'. Use 'exact', 'hunt' o 'nielsen'.")

    scalar = _is_scalar(h) and _is_scalar(T)
    h_arr = np.asarray(h, dtype=float)
    T_arr = np.asarray(T, dtype=float)
    _, two_pi = _pi_consts(strict_excel)
    y = (two_pi / T_arr) ** 2 * h_arr / g

    # Semilla robusta mediante Hunt corregida
    kh = _hunt_kh_polynomial(y, strict_excel=False)
    for _ in range(15):
        th = np.tanh(kh)
        ch2 = np.cosh(kh) ** 2
        f = kh * th - y
        df = th + kh / ch2
        kh = kh - f / df

    k = kh / h_arr
    L = two_pi / k

    if scalar:
        return {"Y": float(y), "kh": float(kh), "k": float(k), "L": float(L)}
    return {"Y": y, "kh": kh, "k": k, "L": L}


def orbital_velocity_profile(
    z: float | np.ndarray,
    h: float,
    T: float,
    H: float,
    L: float | None = None,
    *,
    g: float = GRAVITY,
    use_orbital_sheet_hunt: bool = False,
    strict_excel: bool = False,
) -> dict[str, Any]:
    """Calcula velocidades y excursiones orbitales segun teoria lineal de ondas.

    Reproduce las hojas ``orb. velocity`` (``Basic wave parameters 1.xlsx``) y
    ``Orbital velocities`` (``Orbital vel and Bed sh stress boundary layer.xlsx``).

    Parameters
    ----------
    z : float or array_like
        Cota vertical respecto al nivel medio del mar (``z = 0`` superficie,
        ``z = -h`` fondo) [m].
    h : float
        Profundidad del agua [m].
    T : float
        Periodo de ola [s].
    H : float
        Altura de ola [m].
    L : float or None, optional
        Longitud de onda [m]. Si es ``None``, se calcula mediante Hunt.
    g : float, optional
        Aceleracion gravitacional [m/s^2].
    use_orbital_sheet_hunt : bool, optional
        Si ``True`` y ``L is None``, replica ``4.02*h/T^2`` y ``y^5`` en ``strict_excel``.
    strict_excel : bool, optional
        Usa ``pi = 3.14`` y convenciones originales de Excel.
    """
    scalar = _is_scalar(z)
    z_arr = np.asarray(z, dtype=float)
    pi_val, two_pi = _pi_consts(strict_excel)

    if L is None:
        disp = wavelength_hunt(
            h,
            T,
            g=g,
            typo_y5=use_orbital_sheet_hunt,
            strict_excel=strict_excel,
        )
        L_val = float(disp["L"])
        kh_val = float(disp["kh"])
    else:
        L_val = float(L)
        kh_val = two_pi / L_val * h

    k_val = two_pi / L_val
    sinh_kh = math.sinh(kh_val)
    arg = k_val * (z_arr + h)

    u_max = (pi_val / T * H) * np.cosh(arg) / sinh_kh
    w_max = -(pi_val / T * H) * np.sinh(arg) / sinh_kh
    a_max = (0.5 * H) * np.cosh(arg) / sinh_kh

    if scalar:
        return {
            "Umax": float(u_max),
            "Wmax": float(w_max),
            "Amax": float(a_max),
            "k": float(k_val),
            "L": float(L_val),
            "kh": float(kh_val),
        }
    return {
        "Umax": u_max,
        "Wmax": w_max,
        "Amax": a_max,
        "k": float(k_val),
        "L": float(L_val),
        "kh": float(kh_val),
    }


def stokes_second_order(
    x_over_L: float | np.ndarray,
    h: float,
    T: float,
    H: float,
    L: float,
    t: float,
    *,
    strict_excel: bool = False,
) -> dict[str, Any]:
    """Elevacion superficial y velocidades extremas segun Stokes de segundo orden.

    Implementa la hoja ``Sec order stokes`` de ``Computation Basic wave parameters 1.xlsx``.
    """
    scalar = _is_scalar(x_over_L)
    xL = np.asarray(x_over_L, dtype=float)
    _, two_pi = _pi_consts(strict_excel)

    k = two_pi / L
    kh = k * h
    omega = two_pi / T
    sinh_kh = math.sinh(kh)
    epsilon = (k / 4.0) * (H / 2.0) * math.cosh(kh) * (2.0 + math.cosh(2.0 * kh)) / (sinh_kh**3)

    phase = omega * t - two_pi * xL
    eta_1 = (H / 2.0) * np.cos(phase)
    eta_2 = epsilon * (H / 2.0) * np.cos(2.0 * phase)
    eta = eta_1 + eta_2

    u1_bed = omega * H * 0.5 / sinh_kh
    u2_bed = 3.0 * omega * k * H**2 / (16.0 * (sinh_kh**4))
    u_crest = u1_bed + u2_bed
    u_trough = u1_bed - u2_bed

    if scalar:
        return {
            "k": k,
            "kh": kh,
            "omega": omega,
            "epsilon": epsilon,
            "u_crest": u_crest,
            "u_trough": u_trough,
            "eta_1": float(eta_1),
            "eta_2": float(eta_2),
            "eta": float(eta),
        }
    return {
        "k": k,
        "kh": kh,
        "omega": omega,
        "epsilon": epsilon,
        "u_crest": u_crest,
        "u_trough": u_trough,
        "eta_1": eta_1,
        "eta_2": eta_2,
        "eta": eta,
    }


def wave_mass_flux(
    z: float | np.ndarray,
    h: float,
    T: float,
    H: float,
    L: float,
    *,
    strict_excel: bool = False,
) -> dict[str, Any]:
    """Calcula la velocidad de deriva de Stokes y flujo masico de Longuet-Higgins.

    Implementa la hoja ``Mass flux`` de ``Computation Basic wave parameters 1.xlsx``.
    """
    scalar = _is_scalar(z)
    z_arr = np.asarray(z, dtype=float)
    _, two_pi = _pi_consts(strict_excel)

    k = two_pi / L
    kh = k * h
    omega = two_pi / T
    sinh_kh = math.sinh(kh)
    sinh_2kh = math.sinh(2.0 * kh)
    G = omega * k * H**2 / (8.0 * sinh_kh**2)

    u_unbounded = G * np.cosh(2.0 * k * (z_arr + h))
    u_bounded = G * (np.cosh(2.0 * k * (z_arr + h)) - sinh_2kh / (2.0 * kh))

    zr = z_arr / h
    term1 = 1.5 + np.cosh(2.0 * k * (z_arr + h))
    term2 = 0.5 * kh * np.sinh((2.0 * kh) * (3.0 * zr**2 + 4.0 * zr + 1.0))
    term3 = 1.5 * ((sinh_2kh / (2.0 * kh)) + 1.5) * (zr**2 - 1.0)
    u_lh = G * (term1 + term2 + term3)

    if scalar:
        return {
            "k": k,
            "kh": kh,
            "u_unbounded": float(u_unbounded),
            "u_bounded": float(u_bounded),
            "u_longuet_higgins": float(u_lh),
        }
    return {
        "k": k,
        "kh": kh,
        "u_unbounded": u_unbounded,
        "u_bounded": u_bounded,
        "u_longuet_higgins": u_lh,
    }


def wave_modified_velocity_profile(
    z: float | np.ndarray,
    v_mean: float,
    h: float,
    delta: float,
    u_delta: float,
    k_a: float,
    k_s: float,
) -> dict[str, Any]:
    """Perfil logaritmico de corriente en dos capas modificado por capa limite de oleaje.

    Implementa la hoja ``Velocity profile`` de ``Computation Basic wave parameters 1.xlsx``.
    """
    scalar = _is_scalar(z)
    z_arr = np.asarray(z, dtype=float)

    u_no_wave = v_mean * np.log(30.0 * z_arr / k_s) / (-1.0 + math.log(30.0 * h / k_s))
    u_inner = u_delta * np.log(30.0 * z_arr / k_s) / math.log(30.0 * delta / k_s)
    u_outer = v_mean * np.log(30.0 * z_arr / k_a) / (-1.0 + math.log(30.0 * h / k_a))
    u_with_wave = np.where(z_arr < delta, u_inner, u_outer)
    z_over_h = z_arr / h

    if scalar:
        return {
            "z_over_h": float(z_over_h),
            "u_no_wave": float(u_no_wave),
            "u_with_wave": float(u_with_wave),
        }
    return {
        "z_over_h": z_over_h,
        "u_no_wave": u_no_wave,
        "u_with_wave": u_with_wave,
    }


def jonswap_spectrum(
    f: float | np.ndarray,
    H_s: float,
    T_p: float,
) -> float | np.ndarray:
    """Espectro energetico parametrico de oleaje JONSWAP simplificado.

    Implementa la hoja ``jonswap spectrum`` de ``Computation Basic wave parameters 1.xlsx``:
    ``E(f) = 0.257 * Hs^2 * Tp * (Tp * f)^(-5) * exp(-1.03 * (Tp * f)^(-4))``.
    """
    scalar = _is_scalar(f)
    f_arr = np.asarray(f, dtype=float)
    tf = T_p * f_arr
    E = 0.257 * (H_s**2) * T_p * (tf ** -5.0) * np.exp(-1.03 * (tf ** -4.0))
    return float(E) if scalar else E


def wave_trains(
    x: float | np.ndarray,
    t: float,
    amplitude: float,
    L: float,
    T: float,
    *,
    strict_excel: bool = False,
) -> dict[str, Any]:
    """Calcula trenes de ondas armonicos primario y de segundo armonico.

    Implementa la hoja ``Wave trains`` de ``Computation Basic wave parameters 2.xlsx``.
    """
    scalar = _is_scalar(x)
    x_arr = np.asarray(x, dtype=float)
    _, two_pi = _pi_consts(strict_excel)

    phase = two_pi * t / T - two_pi * x_arr / L
    eta_1 = amplitude * np.cos(phase)
    eta_2 = amplitude * np.cos(2.0 * phase)

    if scalar:
        return {"eta_primary": float(eta_1), "eta_harmonic": float(eta_2)}
    return {"eta_primary": eta_1, "eta_harmonic": eta_2}


def wave_pressure_profile(
    z: float | np.ndarray,
    h: float,
    H: float,
    L: float,
    *,
    rho: float = RHO_FRESH_WATER,
    g: float | None = None,
    strict_excel: bool = False,
) -> dict[str, Any]:
    """Presion hidrostatica y dinamica bajo cresta y seno de una onda lineal.

    Implementa la hoja ``Pressure`` de ``Computation Basic wave parameters 2.xlsx``.
    En Excel la hoja utiliza ``g = 10.0`` m/s^2 cuando ``strict_excel=True``.
    """
    scalar = _is_scalar(z)
    z_arr = np.asarray(z, dtype=float)
    _, two_pi = _pi_consts(strict_excel)
    g_val = (10.0 if strict_excel else GRAVITY) if g is None else float(g)

    k = two_pi / L
    p_dyn_crest = rho * g_val * (0.5 * H) * np.cosh(k * (h + z_arr)) / math.cosh(k * h)
    p_dyn_trough = -p_dyn_crest
    p_stat = -rho * g_val * z_arr
    p_total_crest = p_dyn_crest + p_stat
    p_total_trough = p_dyn_trough + p_stat

    if scalar:
        return {
            "p_dyn_crest": float(p_dyn_crest),
            "p_dyn_trough": float(p_dyn_trough),
            "p_stat": float(p_stat),
            "p_total_crest": float(p_total_crest),
            "p_total_trough": float(p_total_trough),
        }
    return {
        "p_dyn_crest": p_dyn_crest,
        "p_dyn_trough": p_dyn_trough,
        "p_stat": p_stat,
        "p_total_crest": p_total_crest,
        "p_total_trough": p_total_trough,
    }


def classify_linear_wave_regime(
    H: float,
    h: float,
    T: float,
    *,
    g: float = GRAVITY,
) -> dict[str, Any]:
    """Evalua los criterios de aplicabilidad de la teoria lineal de Le Mehaute.

    Implementa las curvas limite de la hoja ``Linear waves`` de
    ``Computation Basic wave parameters 2.xlsx``.
    """
    X = h / (g * T**2)
    Y = H / (g * T**2)
    y_breaking_depth = 0.8 * X
    y_breaking_steepness = 0.0226
    y_ursell_10 = 0.05 * X
    y_steepness_1_40 = 0.004 * X

    is_broken = bool(Y > min(y_breaking_depth, y_breaking_steepness))
    is_linear_ursell = bool(Y <= y_ursell_10)
    is_low_steepness = bool(Y <= y_steepness_1_40)
    return {
        "X": float(X),
        "Y": float(Y),
        "Y_breaking_depth": float(y_breaking_depth),
        "Y_breaking_steepness": float(y_breaking_steepness),
        "Y_ursell_10": float(y_ursell_10),
        "Y_steepness_1_40": float(y_steepness_1_40),
        "is_broken": is_broken,
        "is_linear_ursell": is_linear_ursell,
        "is_low_steepness": is_low_steepness,
    }


def group_velocity_packet(
    x: float | np.ndarray,
    T: float,
    L: float,
    d_omega: float,
    d_k: float,
    amplitude: float,
    h: float,
    t: float,
    *,
    g: float = GRAVITY,
    strict_excel: bool = False,
) -> dict[str, Any]:
    """Velocidad de grupo y envolvente de un paquete de dos ondas cercanas.

    Implementa la hoja ``Group velocity`` de ``Computation Basic wave parameters 2.xlsx``.
    """
    scalar = _is_scalar(x)
    x_arr = np.asarray(x, dtype=float)
    _, two_pi = _pi_consts(strict_excel)

    omega = two_pi / T
    k = two_pi / L
    kh = k * h
    n = 0.5 * (1.0 + 2.0 * kh / math.sinh(2.0 * kh))
    c_gr = n * L / T
    L_computed = (g * T**2 / two_pi) * math.tanh(kh)
    eta = 2.0 * amplitude * np.cos(0.5 * d_omega * t - 0.5 * d_k * x_arr) * np.cos(
        omega * t - k * x_arr
    )

    result = {
        "omega": float(omega),
        "k": float(k),
        "n": float(n),
        "c_gr": float(c_gr),
        "L_computed": float(L_computed),
        "eta": float(eta) if scalar else eta,
    }
    return result


def sine_power_integrals(
    T: float,
    phase_deg: float,
    dt: float = 0.05,
    *,
    strict_excel: bool = False,
) -> dict[str, Any]:
    """Integracion numerica en medio periodo de productos y potencias senoidales.

    Implementa la hoja ``powers of sine functions`` (``Basic wave parameters 2.xlsx``)
    y la hoja ``phase effects`` (``Skewness and Asymmetry.xlsx``).
    Cuando ``strict_excel=True`` reproduce el pi redondeado ``3.141`` y el bug en
    la celda ``F91`` (``cos(phase_deg * 3.141)`` sin dividir entre ``180``).
    """
    pi_val = 3.141 if strict_excel else PI_STANDARD
    n_steps = int(round(0.5 * T / dt)) + 1
    t = np.arange(n_steps, dtype=float) * dt
    phase_rad = phase_deg / 180.0 * pi_val

    sin_phase = np.sin(2.0 * pi_val * t / T + phase_rad)
    sin_ref = np.sin(2.0 * pi_val * t / T)
    product = sin_phase * sin_ref

    int_product = float(np.sum(product) * dt)
    if strict_excel:
        analytic_product = float(0.25 * T * math.cos(phase_deg * pi_val))
    else:
        analytic_product = float(0.25 * T * math.cos(phase_rad))

    # Las columnas H e I de Excel comienzan en R9 tomando C8..C87
    int_sin4 = float(np.sum(sin_ref[:-1] ** 4) * dt)
    int_abs_sin3 = float(np.sum(np.abs(sin_ref[:-1] ** 3)) * dt)
    int_odd_moment = float(np.sum(sin_ref * np.abs(sin_phase**3)) * dt)

    return {
        "t": t,
        "sin_phase": sin_phase,
        "sin_ref": sin_ref,
        "product": product,
        "int_product": int_product,
        "analytic_product": analytic_product,
        "int_sin4": int_sin4,
        "int_abs_sin3": int_abs_sin3,
        "int_odd_moment": int_odd_moment,
    }


def wave_friction_factor(
    A: float | np.ndarray,
    k_s: float | np.ndarray,
) -> float | np.ndarray:
    """Factor de friccion del fondo por oleaje segun Swart / Van Rijn.

    ``f_w = exp(-6.0 + 5.2 * (A / k_s)**(-0.19))``.
    """
    scalar = _is_scalar(A) and _is_scalar(k_s)
    A_arr = np.asarray(A, dtype=float)
    ks_arr = np.asarray(k_s, dtype=float)
    fw = np.exp(-6.0 + 5.2 * np.power(A_arr / ks_arr, -0.19))
    return float(fw) if scalar else fw


def streaming_vertical_functions(zeta: float | np.ndarray) -> dict[str, Any]:
    """Funciones verticales adimensionales ``Fz`` y ``Gz`` de Longuet-Higgins.

    Implementa la hoja ``Sheet3`` de ``Computation Eulerian streaming.xlsx``.
    """
    scalar = _is_scalar(zeta)
    z = np.asarray(zeta, dtype=float)
    emz = np.exp(-z)
    fz = 1.0 - emz * (2.0 * np.cos(z) - emz + 2.0 * z * np.sin(z))
    gz = 3.0 + emz * (
        -4.0 * np.cos(z) + 2.0 * np.sin(z) + emz - 2.0 * z * np.sin(z) + 2.0 * z * np.cos(z)
    )
    if scalar:
        return {"Fz": float(fz), "Gz": float(gz)}
    return {"Fz": fz, "Gz": gz}


def wave_boundary_layer(
    z_over_delta: float | np.ndarray,
    t: float,
    U_max: float = 1.0,
    T: float = 8.0,
    eps: float = 1e-4,
    k: float = 0.25,
    *,
    rho: float = RHO_FRESH_WATER,
    strict_excel: bool = False,
) -> dict[str, Any]:
    """Estructura laminar/lineal de la capa limite oscilatoria y streaming inducido.

    Implementa la hoja ``Boundary layer`` de
    ``Computation Orbital vel and Bed sh stress boundary layer.xlsx``.
    """
    scalar = _is_scalar(z_over_delta)
    zeta = np.asarray(z_over_delta, dtype=float)
    pi_val, two_pi = _pi_consts(strict_excel)

    omega = two_pi / T
    A_max = T * U_max / two_pi
    tau_max = rho * math.sqrt(eps * omega) * U_max
    delta = math.sqrt(2.0 * eps / omega)
    c = omega / k
    uw_max = -0.25 * (U_max**2) * k * delta * rho
    uw_over_tau = uw_max / tau_max

    phase = omega * t
    u_t = U_max * (np.sin(phase) - np.exp(-zeta) * np.sin(phase - zeta))
    tau_t = tau_max * np.exp(-zeta) * np.sin(phase - zeta + 0.25 * pi_val)
    vert = streaming_vertical_functions(zeta)

    return {
        "omega": float(omega),
        "A_max": float(A_max),
        "tau_max": float(tau_max),
        "delta": float(delta),
        "c": float(c),
        "uw_max": float(uw_max),
        "uw_over_tau": float(uw_over_tau),
        "u_t": float(u_t) if scalar else u_t,
        "tau_t": float(tau_t) if scalar else tau_t,
        "Fz": vert["Fz"],
        "Gz": vert["Gz"],
    }


def wave_decay_friction(
    h: float = 10.0,
    H0: float = 4.0,
    T: float = 10.0,
    L: float | None = 93.0,
    k_s: float = 0.03,
    dx: float = 100.0,
    n_steps: int = 51,
    *,
    g: float = GRAVITY,
    strict_excel: bool = False,
) -> dict[str, Any]:
    """Atenuacion espacial de altura de ola en fondo plano por friccion del lecho.

    Implementa ``Computation Wave height decrease due to friction.xlsx``.
    """
    pi_val, two_pi = _pi_consts(strict_excel)
    if L is None:
        L_val = float(dispersion_relation(h, T, method="exact", g=g, strict_excel=strict_excel)["L"])
    else:
        L_val = float(L)

    k = two_pi / L_val
    kh = k * h
    omega = two_pi / T
    sinh_kh = math.sinh(kh)
    n = 0.5 * (1.0 + 2.0 * kh / math.sinh(2.0 * kh))
    c = L_val / T

    x = np.arange(n_steps, dtype=float) * dx
    H = np.zeros(n_steps, dtype=float)
    dH = np.zeros(n_steps, dtype=float)
    A = np.zeros(n_steps, dtype=float)
    f_w = np.zeros(n_steps, dtype=float)
    alpha = np.zeros(n_steps, dtype=float)

    H[0] = float(H0)
    const_alpha = (omega**3) / (3.0 * pi_val * g * c * n * (sinh_kh**3))

    for i in range(n_steps):
        A[i] = H[i] / (2.0 * sinh_kh)
        f_w[i] = float(wave_friction_factor(A[i], k_s))
        alpha[i] = f_w[i] * const_alpha
        dH[i] = -alpha[i] * (H[i] ** 2) * dx
        if i + 1 < n_steps:
            H[i + 1] = H[i] + dH[i]

    return {
        "x": x,
        "H": H,
        "dH": dH,
        "A": A,
        "f_w": f_w,
        "alpha": alpha,
        "k": float(k),
        "kh": float(kh),
        "omega": float(omega),
        "n": float(n),
        "L": float(L_val),
    }


def battjes_janssen_profile(
    h_profile: float | np.ndarray,
    H_rms0: float,
    theta0_deg: float,
    T: float,
    *,
    gamma: float = 0.7,
    k_s: float = 0.01,
    dx: float = 20.0,
    cal_vl: float = 0.5,
    rho: float = RHO_SEA_WATER,
    g: float = GRAVITY,
    strict_excel: bool = False,
) -> dict[str, np.ndarray]:
    """Modelo 1D de Battjes y Janssen (1978) para rotura, setup y corriente longitudinal.

    Implementa todas las columnas de ``Computation Wave height Battjes Janssen model.xlsx``.

    Notes
    -----
    Cuando ``strict_excel=True``:
    - emplea ``pi = 3.14`` y el coeficiente ``0.000654^Y^6`` de la hoja;
    - calcula la variacion del setup ``d_setup[i]`` dividiendo entre la profundidad sin
      marea/setup previa ``h[i-1]`` (tal como figura en las celdas ``H20:H122``).
    Cuando ``strict_excel=False`` (por defecto), utiliza ``pi`` estandar, Hunt corregido
    y la profundidad total efectiva previa ``h[i-1] + setup[i-1]`` para ``d_setup``.
    """
    h = np.asarray(h_profile, dtype=float)
    n_pts = h.size
    if n_pts == 0:
        raise ValueError("h_profile no puede estar vacio.")

    pi_val, two_pi = _pi_consts(strict_excel)
    x = np.arange(n_pts, dtype=float) * dx

    h_eff = np.zeros(n_pts, dtype=float)
    theta_deg = np.zeros(n_pts, dtype=float)
    H_rms = np.zeros(n_pts, dtype=float)
    d_setup = np.zeros(n_pts, dtype=float)
    setup = np.zeros(n_pts, dtype=float)
    U_orb = np.zeros(n_pts, dtype=float)
    v_longshore = np.zeros(n_pts, dtype=float)
    S_xx = np.zeros(n_pts, dtype=float)
    S_xy = np.zeros(n_pts, dtype=float)
    Y = np.zeros(n_pts, dtype=float)
    kh = np.zeros(n_pts, dtype=float)
    k = np.zeros(n_pts, dtype=float)
    L = np.zeros(n_pts, dtype=float)
    c = np.zeros(n_pts, dtype=float)
    n_grp = np.zeros(n_pts, dtype=float)
    H_max = np.zeros(n_pts, dtype=float)
    Q_b = np.zeros(n_pts, dtype=float)
    A_orb = np.zeros(n_pts, dtype=float)
    f_w = np.zeros(n_pts, dtype=float)
    f_c = np.zeros(n_pts, dtype=float)
    E_flux = np.zeros(n_pts, dtype=float)
    dE_flux_dx = np.zeros(n_pts, dtype=float)
    dE_flux = np.zeros(n_pts, dtype=float)

    theta0_rad = theta0_deg / 180.0 * pi_val

    for i in range(n_pts):
        h_prev_eff = h[i] if i == 0 else h[i] + setup[i - 1]
        h_eff[i] = h_prev_eff

        Y[i] = (two_pi / T) ** 2 * h_prev_eff / g
        kh[i] = float(_hunt_kh_polynomial(np.asarray(Y[i]), strict_excel=strict_excel))
        k[i] = kh[i] / h_prev_eff
        L[i] = two_pi / k[i]
        c[i] = L[i] / T
        n_grp[i] = 0.5 * (1.0 + 2.0 * kh[i] / math.sinh(2.0 * kh[i]))
        H_max[i] = (0.88 / k[i]) * math.tanh(gamma * kh[i] / 0.88)

        if i == 0:
            theta_deg[i] = float(theta0_deg)
            H_rms[i] = float(H_rms0)
            theta_rad = theta_deg[i] / 180.0 * pi_val
            E_flux[i] = math.cos(theta_rad) * n_grp[i] * c[i] * (H_rms[i] ** 2)
        else:
            E_flux[i] = max(E_flux[i - 1] + dE_flux[i - 1], 1e-12)
            sin_th = min(max((c[i] / c[0]) * math.sin(theta0_rad), -1.0), 1.0)
            theta_rad = math.asin(sin_th)
            theta_deg[i] = theta_rad * 180.0 / pi_val
            H_rms[i] = math.sqrt(E_flux[i] / (n_grp[i] * c[i] * math.cos(theta_rad)))

        if H_rms[i] < H_max[i]:
            r2 = (H_rms[i] / H_max[i]) ** 2
            Q_b[i] = math.exp((0.97 * r2 - 1.0) / (0.657 * r2))
        else:
            Q_b[i] = 1.0

        A_orb[i] = 0.5 * H_rms[i] / math.sinh(kh[i])
        f_w[i] = float(wave_friction_factor(A_orb[i], k_s))
        f_c[i] = 0.242 / (math.log10(12.0 * h_prev_eff / k_s) ** 2)

        if i > 0:
            theta_prev_rad = theta_deg[i - 1] / 180.0 * pi_val
            denom_h = h[i - 1] if strict_excel else h_eff[i - 1]
            s_cur = n_grp[i] * (H_rms[i] ** 2) * (1.0 + math.cos(theta_rad) ** 2)
            s_prev = n_grp[i - 1] * (H_rms[i - 1] ** 2) * (1.0 + math.cos(theta_prev_rad) ** 2)
            d_setup[i] = -(s_cur - s_prev) / (8.0 * denom_h)
            setup[i] = setup[i - 1] + d_setup[i]

        U_orb[i] = pi_val * H_rms[i] / (T * math.sinh(kh[i]))
        if i > 0:
            theta_prev_rad = theta_deg[i - 1] / 180.0 * pi_val
            sxy_cur = n_grp[i] * math.cos(theta_rad) * math.sin(theta_rad) * (H_rms[i] ** 2)
            sxy_prev = (
                n_grp[i - 1]
                * math.cos(theta_prev_rad)
                * math.sin(theta_prev_rad)
                * (H_rms[i - 1] ** 2)
            )
            v_longshore[i] = (
                -(sxy_cur - sxy_prev)
                * g
                / (8.0 * cal_vl * dx * math.sqrt(f_w[i - 1]) * math.sqrt(f_c[i - 1]) * U_orb[i - 1])
            )

        S_xx[i] = (
            (n_grp[i] - 0.5 + n_grp[i] * (math.cos(theta_rad) ** 2))
            * 0.125
            * rho
            * g
            * (H_rms[i] ** 2)
        )
        S_xy[i] = (
            n_grp[i] * math.sin(theta_rad) * math.cos(theta_rad) * 0.125 * rho * g * (H_rms[i] ** 2)
        )

        dE_flux_dx[i] = -2.0 * Q_b[i] * (H_max[i] ** 2 / T) - (
            (8.0 / g) * f_w[i] * (two_pi / T) ** 3 * (H_rms[i] ** 3) / (12.0 * pi_val * (math.sinh(kh[i]) ** 3))
        )
        dE_flux[i] = dE_flux_dx[i] * dx

    return {
        "x": x,
        "h": h,
        "h_eff": h_eff,
        "theta_deg": theta_deg,
        "H_rms": H_rms,
        "d_setup": d_setup,
        "setup": setup,
        "U_orb": U_orb,
        "v_longshore": v_longshore,
        "S_xx": S_xx,
        "S_xy": S_xy,
        "Y": Y,
        "kh": kh,
        "k": k,
        "L": L,
        "c": c,
        "n": n_grp,
        "H_max": H_max,
        "Q_b": Q_b,
        "A_orb": A_orb,
        "f_w": f_w,
        "f_c": f_c,
        "E_flux": E_flux,
        "dE_flux_dx": dE_flux_dx,
    }


def wave_skewness_asymmetry(
    H_rms: float,
    h: float,
    T: float,
    *,
    g: float = GRAVITY,
    strict_excel: bool = False,
) -> dict[str, Any]:
    """Asimetria y sesgo orbital segun Ruessink et al. (hoja ``Skewness and asymmetry Rues old``)."""
    pi_val, two_pi = _pi_consts(strict_excel)
    disp = wavelength_hunt(h, T, g=g, strict_excel=strict_excel, fixed_hunt_coeff=True)
    kh = float(disp["kh"])
    k = float(disp["k"])
    L = float(disp["L"])

    sigma_u = H_rms * pi_val / (1.41 * T * math.sinh(two_pi * h / L))
    Ur = 0.75 * 0.5 * 1.41 * H_rms * k / (kh**3)

    Hulp = 0.7939 / (1.0 + math.exp((-0.6065 - math.log10(Ur)) / 0.3539))
    beta_arg_sc = -90.0 + 90.0 * math.tanh(0.6373 / (Ur**0.5995))
    beta_display = -90.0 + 90.0 * math.tanh(0.64 / (Ur**0.6))
    beta_rad_sc = beta_arg_sc / 360.0 * two_pi
    beta_rad_disp = beta_display / 360.0 * two_pi

    SK = Hulp * math.cos(beta_rad_sc)
    AS = Hulp * math.sin(beta_rad_sc)

    P = (2.0 / 3.0) * (sigma_u**2)
    Q = (4.0 / 3.0) * (sigma_u**3) * SK / math.cos(beta_rad_disp)
    arg_acos = max(min(-0.5 * Q * (P ** -1.5), 1.0), -1.0)
    phi1 = (1.0 / 3.0) * math.acos(arg_acos)
    shift = two_pi / 3.0
    phis = (phi1, phi1 + shift, phi1 + 2.0 * shift)
    U2_roots = tuple(2.0 * math.sqrt(P) * math.cos(p) for p in phis)

    return {
        "Y": float(disp["Y"]),
        "kh": kh,
        "k": k,
        "L": L,
        "sigma_u": float(sigma_u),
        "Ur": float(Ur),
        "Hulp": float(Hulp),
        "SK": float(SK),
        "AS": float(AS),
        "beta_deg": float(beta_display),
        "P": float(P),
        "Q": float(Q),
        "phi": phis,
        "U2_roots": U2_roots,
        "U2_solution": float(min(U2_roots, key=lambda v: abs(v))),
    }


def abreu_skewness_asymmetry(
    H_rms: float,
    h: float,
    T: float,
    *,
    dt: float = 0.05,
    t: float | np.ndarray | None = None,
    g: float = GRAVITY,
    strict_excel: bool = False,
) -> dict[str, Any]:
    """Forma de onda no lineal de velocidad orbital segun Abreu et al. (2010).

    Implementa la hoja ``Skewness and asymmetry Abreu-R`` de
    ``Computation Skewness and Asymmetry.xlsx``.
    """
    pi_val, two_pi = _pi_consts(strict_excel)
    disp = wavelength_hunt(h, T, g=g, strict_excel=strict_excel, fixed_hunt_coeff=True)
    kh = float(disp["kh"])
    k = float(disp["k"])
    L = float(disp["L"])

    sigma_u = H_rms * pi_val / (1.41 * T * math.sinh(two_pi * h / L))
    Ur = 0.75 * 0.5 * 1.41 * H_rms * k / (kh**3)
    r = math.tanh(0.9305 * 0.8507 / (1.0 + math.exp(-1.586 - 3.367 * math.log10(Ur))))
    phi_rad = (-90.0 / 360.0) * two_pi * math.tanh(0.815 / (Ur**0.672))
    phi_deg = phi_rad / two_pi * 360.0
    F_par = math.sqrt(1.0 - r**2)
    H_par = r * math.sin(phi_rad) / (1.0 + F_par)
    U_w = pi_val * (1.41 * H_rms) / (T * math.sinh(kh))

    if t is None:
        t_arr = np.arange(0.0, T + 0.5 * dt, dt, dtype=float)
    else:
        t_arr = np.asarray(t, dtype=float)

    omega = two_pi / T
    u = U_w * F_par * (np.sin(omega * t_arr) + H_par) / (1.0 - r * np.cos(omega * t_arr + phi_rad))

    return {
        "Y": float(disp["Y"]),
        "kh": kh,
        "k": k,
        "L": L,
        "sigma_u": float(sigma_u),
        "Ur": float(Ur),
        "r": float(r),
        "phi_rad": float(phi_rad),
        "phi_deg": float(phi_deg),
        "F": float(F_par),
        "H_par": float(H_par),
        "U_w": float(U_w),
        "t": t_arr,
        "u": u,
    }


def isobe_horikawa_skewness(
    H_s: float = 1.41,
    T_p: float = 7.0,
    h: float = 1.5,
    L: float | None = 26.3,
    *,
    dt: float = 0.1,
    t: float | np.ndarray | None = None,
    g: float = GRAVITY,
    strict_excel: bool = False,
) -> dict[str, Any]:
    """Parametrizacion de Isobe y Horikawa (1982) para velocidad orbital asimetrica.

    Implementa la hoja ``Skewness Isobe-Horikawa`` de ``Computation Skewness and Asymmetry.xlsx``.
    """
    pi_val, two_pi = _pi_consts(strict_excel)
    if L is None:
        L_val = float(dispersion_relation(h, T_p, method="hunt", g=g, strict_excel=strict_excel)["L"])
    else:
        L_val = float(L)

    U_linear = pi_val * H_s / (T_p * math.sinh(two_pi / L_val * h))
    r = -0.4 * H_s / h + 1.0
    U_max = 2.0 * r * U_linear
    U1 = U_max / math.sqrt(g * h)
    T1 = T_p * math.sqrt(g / h)
    A1 = -0.0049 * (T1**2) - 0.069 * T1 + 0.2911
    r_a = max(0.5, -5.25 - 6.1 * math.tanh(A1 * U1 - 1.76))
    r_max = min(0.75, max(0.62, -2.5 * (h / L_val) + 0.85))
    U_on = U_max * (0.5 + (r_max - 0.5) * math.tanh((r_a - 0.5) / (r_max - 0.5)))
    U_off = U_max - U_on
    T_on = (U_off / (U_on + U_off)) * T_p
    T_off = T_p - T_on

    if t is None:
        t_arr = np.arange(0.0, T_p + 0.5 * dt, dt, dtype=float)
    else:
        t_arr = np.asarray(t, dtype=float)

    t_mod = np.mod(t_arr, T_p)
    u = np.where(
        t_mod < T_on,
        U_on * np.sin(pi_val / T_on * t_mod),
        -U_off * np.sin(pi_val / T_off * (t_mod - T_on)),
    )

    return {
        "L": float(L_val),
        "U_linear": float(U_linear),
        "r": float(r),
        "U_max": float(U_max),
        "U1": float(U1),
        "T1": float(T1),
        "A1": float(A1),
        "r_a": float(r_a),
        "r_max": float(r_max),
        "U_on": float(U_on),
        "U_off": float(U_off),
        "T_on": float(T_on),
        "T_off": float(T_off),
        "t": t_arr,
        "u": u,
    }


def nielsen_acceleration_transport(
    U1: float = 1.1,
    U2: float = 0.35,
    U_for: float = 1.45,
    U_back: float = 0.7,
    T: float = 9.0,
    phase_deg: float = 40.0,
    f_w: float = 0.01,
    d50: float = 2.1e-4,
    theta_cr: float = 0.052,
    tau_cr: float = 0.173,
    *,
    dt: float = 0.05,
    n_cycles: int = 2,
    rho: float = RHO_FRESH_WATER,
    rho_s: float = RHO_SEDIMENT,
    nu: float = KINEMATIC_VISCOSITY_15C,
    g: float = GRAVITY,
    strict_excel: bool = False,
) -> dict[str, Any]:
    """Transporte de fondo bajo ondas asimetricas con efecto de aceleracion de Nielsen.

    Implementa la hoja ``Nielsen effect`` de ``Computation Skewness and Asymmetry.xlsx``,
    comparando las formulaciones de carga de fondo de Nielsen (1992/2006) y Van Rijn
    bajo una onda Stokes de segundo orden (coseno) y una onda senoidal asimetrica.
    """
    pi_val = 3.14 if strict_excel else PI_STANDARD
    pi_cos = 3.141 if strict_excel else PI_STANDARD
    two_pi = 6.28 if strict_excel else 2.0 * PI_STANDARD

    s = rho_s / rho
    d_star = d50 * (((s - 1.0) * g) / (nu * nu)) ** (0.3333 if strict_excel else (1.0 / 3.0))
    T_for = (U_back / (U_for + U_back)) * T

    # Rejilla temporal identica a Excel (-0.05 hasta n_cycles*T + 0.05)
    t_start = -dt
    t_end = n_cycles * T + dt
    n_pts = int(round((t_end - t_start) / dt)) + 1
    t = t_start + np.arange(n_pts, dtype=float) * dt

    def _eval_sine_asym(t_vec: np.ndarray) -> np.ndarray:
        return np.where(
            t_vec < T_for,
            U_for * np.sin(pi_val * t_vec / T_for),
            -U_back * np.sin(pi_val * (t_vec - T_for) / (T - T_for)),
        )

    U_stokes = U1 * np.cos(2.0 * pi_cos * t / T) + U2 * np.cos(4.0 * pi_cos * t / T)
    U_sine = _eval_sine_asym(t)
    if strict_excel and n_pts > 1 and abs(t[1]) < 1e-12:
        U_sine[1] = 0.001

    dU_stokes = np.zeros_like(t)
    scale = T / two_pi
    dU_stokes[0] = scale * (U_stokes[1] - U_stokes[0]) / dt
    for i in range(1, n_pts - 1):
        dU_stokes[i] = scale * (U_stokes[i + 1] - U_stokes[i - 1]) / (2.0 * dt)
    if strict_excel and n_pts > 2 and abs(t[1]) < 1e-12:
        dU_stokes[1] = (U_stokes[2] - U_stokes[0]) / (2.0 * dt * two_pi / T)
    dU_stokes[-1] = 0.0 if strict_excel else scale * (U_stokes[-1] - U_stokes[-2]) / dt

    if strict_excel:
        # En la hoja Nielsen effect, F y G evaluan la rama IF(t < Tfor, ...) en t +/- dt
        cond = t < T_for
        U_sine_plus = np.where(
            cond,
            U_for * np.sin(pi_val * (t + dt) / T_for),
            -U_back * np.sin(pi_val * (t + dt - T_for) / (T - T_for)),
        )
        U_sine_minus = np.where(
            cond,
            U_for * np.sin(pi_val * (t - dt) / T_for),
            -U_back * np.sin(pi_val * (t - dt - T_for) / (T - T_for)),
        )
    else:
        U_sine_plus = _eval_sine_asym(t + dt)
        U_sine_minus = _eval_sine_asym(t - dt)
    dU_sine = scale * (U_sine_plus - U_sine_minus) / (2.0 * dt)

    ph_rad = phase_deg / 180.0 * pi_cos
    c_ph = math.cos(ph_rad)
    s_ph = math.sin(ph_rad)

    U_app_stokes = c_ph * U_stokes + s_ph * dU_stokes
    U_app_sine = c_ph * U_sine + s_ph * dU_sine

    theta_nielsen = (0.5 * f_w / ((s - 1.0) * g * d50)) * U_app_stokes * np.abs(U_app_stokes)
    tau_nielsen = 0.5 * rho * f_w * U_app_stokes * np.abs(U_app_stokes)
    tau_sine = 0.5 * rho * f_w * U_app_sine * np.abs(U_app_sine)

    qb_nielsen = np.where(
        np.abs(theta_nielsen) > theta_cr,
        12.0
        * rho_s
        * math.sqrt(s - 1.0)
        * math.sqrt(g)
        * (d50**1.5)
        * np.sign(theta_nielsen)
        * np.power(np.maximum(np.abs(theta_nielsen) - theta_cr, 0.0), 1.5),
        0.0,
    )

    vr_coeff = 0.5 * rho_s * d50 * (d_star**-0.3) * math.sqrt(1.0 / rho)
    qb_vr_stokes = np.where(
        np.abs(tau_nielsen) > tau_cr,
        np.sign(tau_nielsen)
        * vr_coeff
        * np.sqrt(np.abs(tau_nielsen))
        * (np.abs(tau_nielsen) - tau_cr)
        / tau_cr,
        0.0,
    )
    qb_vr_sine = np.where(
        np.abs(tau_sine) > tau_cr,
        np.sign(tau_sine)
        * vr_coeff
        * np.sqrt(np.abs(tau_sine))
        * (np.abs(tau_sine) - tau_cr)
        / tau_cr,
        0.0,
    )

    # En la hoja Excel Z382 divide la suma del primer ciclo (182 puntos) entre 183
    n_cycle_pts = int(round(T / dt)) + 2
    denom_sine = n_cycle_pts + 1 if strict_excel else min(n_cycle_pts, n_pts)
    net_nielsen = float(np.mean(qb_nielsen))
    net_vr_stokes = float(np.mean(qb_vr_stokes))
    net_vr_sine = float(np.sum(qb_vr_sine[:n_cycle_pts]) / denom_sine)

    return {
        "t": t,
        "d_star": float(d_star),
        "T_for": float(T_for),
        "U_stokes": U_stokes,
        "U_sine_asym": U_sine,
        "dU_stokes_dt_norm": dU_stokes,
        "dU_sine_dt_norm": dU_sine,
        "theta_nielsen": theta_nielsen,
        "tau_nielsen": tau_nielsen,
        "tau_sine_apparent": tau_sine,
        "qb_nielsen": qb_nielsen,
        "qb_vr_stokes": qb_vr_stokes,
        "qb_vr_sine": qb_vr_sine,
        "net_qb_nielsen": net_nielsen,
        "net_qb_vr_stokes": net_vr_stokes,
        "net_qb_vr_sine": net_vr_sine,
    }


def undertow_velocity(
    z_or_rel: float | np.ndarray,
    *,
    profile: int = 1,
    tau_rol: float = 13.0,
    eps_rol: float = 0.1,
    delta: float = 0.05,
    U_del: float = 0.05,
    h: float = 2.0,
    dh_dx: float = 0.001,
    eps: float = 0.05,
    dE_dx: float = 100.0,
    ratio_Er_E: float = 0.4,
    rho: float = RHO_FRESH_WATER,
    g: float = GRAVITY,
    strict_excel: bool = False,
) -> float | np.ndarray:
    """Perfil vertical de velocidad de resaca (undertow).

    Implementa ambas formulaciones de ``Computation Undertow velocity.xlsx``:

    - ``profile=1``: espera cota relativa ``z_rel = z / h`` en ``[delta/h, 1]``.
    - ``profile=2``: espera elevacion ``z`` sobre el fondo en ``[0, h]``.
    """
    scalar = _is_scalar(z_or_rel)
    arr = np.asarray(z_or_rel, dtype=float)

    if profile == 1:
        A = g / eps_rol * dh_dx
        B = tau_rol / rho / eps_rol
        delta_h = delta / h
        log_term = np.log(arr * h / delta)
        u = U_del + A * (h**2) * (arr - delta_h - log_term) + B * h * log_term
        return float(u) if scalar else u

    if profile == 2:
        term1 = -(1.0 / eps) * g * h * (arr - 0.5 * (arr**2) / h) * dh_dx
        if strict_excel:
            # Celda C9 original usaba 0.1*z para z=0 (eval igual a 0, difiere si ratio_Er_E != 0.45)
            mix = np.where(arr == 0.0, 0.1 * arr, (1.0 - 2.0 * ratio_Er_E) * arr)
            term2 = -(1.0 / (rho * eps)) * (
                ((eps * arr - 0.5 * arr**2) / (2.0 * h) + mix) * dE_dx
            )
            term2 = np.where(
                arr == 0.0,
                -(1.0 / (rho * eps)) * ((eps * arr - 0.5 * arr**2) / (2.0 * h) + (0.1 * arr) * dE_dx),
                term2,
            )
        else:
            term2 = -(1.0 / (rho * eps)) * (
                ((eps * arr - 0.5 * arr**2) / (2.0 * h) + (1.0 - 2.0 * ratio_Er_E) * arr) * dE_dx
            )
        u = term1 + term2
        return float(u) if scalar else u

    raise ValueError("profile debe ser 1 o 2.")


def eulerian_streaming_profile(
    zeta: float | np.ndarray,
    U_max: float = 1.0,
    T: float = 8.0,
    L: float = 25.0,
    dh_dx: float = 0.004,
    *,
    g: float = GRAVITY,
    strict_excel: bool = False,
) -> float | np.ndarray:
    """Perfil vertical de corriente Euleriana de deriva en capa limite con gradiente de presion.

    Implementa la hoja ``streaming profile`` de ``Computation Eulerian streaming.xlsx``.
    En ``strict_excel=True`` reproduce la precedencia sin parentesis ``U_max^2`` en el numerador;
    en ``strict_excel=False`` situa ``U_max^2`` en el denominador para mantener adimensionalidad.
    """
    scalar = _is_scalar(zeta)
    z = np.asarray(zeta, dtype=float)
    _, two_pi = _pi_consts(strict_excel)

    pref = 0.75 * (U_max**2) * T / L
    if strict_excel:
        pg = (12.0 * g * L / two_pi * (U_max**2) * dh_dx) - 1.0
    else:
        pg = (12.0 * g * L / (two_pi * (U_max**2)) * dh_dx) - 1.0
    u = pref * pg * (z**2 - 2.0 * z)
    return float(u) if scalar else u


def eulerian_streaming(
    H: float | np.ndarray,
    T: float | np.ndarray,
    h: float | np.ndarray,
    k_s: float | np.ndarray,
    *,
    g: float = GRAVITY,
    strict_excel: bool = False,
) -> dict[str, Any]:
    """Velocidad de deriva Euleriana en el borde de la capa limite (Van Rijn y Kranenburg).

    Implementa la hoja ``streaming Kranenburg`` de ``Computation Eulerian streaming.xlsx``.
    """
    scalar = _is_scalar(H) and _is_scalar(T) and _is_scalar(h) and _is_scalar(k_s)
    H_arr = np.asarray(H, dtype=float)
    T_arr = np.asarray(T, dtype=float)
    h_arr = np.asarray(h, dtype=float)
    ks_arr = np.asarray(k_s, dtype=float)
    pi_val, two_pi = _pi_consts(strict_excel)

    disp = wavelength_hunt(h_arr, T_arr, g=g, strict_excel=strict_excel)
    kh = np.asarray(disp["kh"], dtype=float)
    k = np.asarray(disp["k"], dtype=float)
    L = np.asarray(disp["L"], dtype=float)
    c = L / T_arr

    U_max = pi_val * H_arr / (T_arr * np.sinh(kh))
    A_max = U_max * T_arr / two_pi
    A_over_ks = A_max / ks_arr

    u2_c = (U_max**2) / c
    u_van_rijn = np.where(
        A_over_ks < 100.0,
        u2_c * (-1.0 + 0.87 * np.log10(A_over_ks)),
        0.75 * u2_c,
    )
    u_kranenburg_shape = u2_c * (
        0.345 + 0.7 * np.power(A_over_ks, -0.9) - 0.25 / (np.sinh(kh) ** 2)
    )
    u_kranenburg_no_shape = u2_c * (0.345 + 0.7 * np.power(A_over_ks, -0.9))

    if scalar:
        return {
            "Y": float(disp["Y"]),
            "kh": float(kh),
            "k": float(k),
            "L": float(L),
            "c": float(c),
            "U_max": float(U_max),
            "A_max": float(A_max),
            "A_over_ks": float(A_over_ks),
            "u_van_rijn": float(u_van_rijn),
            "u_kranenburg_shape": float(u_kranenburg_shape),
            "u_kranenburg_no_shape": float(u_kranenburg_no_shape),
        }
    return {
        "Y": disp["Y"],
        "kh": kh,
        "k": k,
        "L": L,
        "c": c,
        "U_max": U_max,
        "A_max": A_max,
        "A_over_ks": A_over_ks,
        "u_van_rijn": u_van_rijn,
        "u_kranenburg_shape": u_kranenburg_shape,
        "u_kranenburg_no_shape": u_kranenburg_no_shape,
    }


__all__ = [
    "abreu_skewness_asymmetry",
    "battjes_janssen_profile",
    "classify_linear_wave_regime",
    "dispersion_relation",
    "eulerian_streaming",
    "eulerian_streaming_profile",
    "group_velocity_packet",
    "isobe_horikawa_skewness",
    "jonswap_spectrum",
    "nielsen_acceleration_transport",
    "orbital_velocity_profile",
    "sine_power_integrals",
    "stokes_second_order",
    "streaming_vertical_functions",
    "undertow_velocity",
    "wave_boundary_layer",
    "wave_decay_friction",
    "wave_friction_factor",
    "wave_mass_flux",
    "wave_modified_velocity_profile",
    "wave_pressure_profile",
    "wave_skewness_asymmetry",
    "wave_trains",
    "wavelength_hunt",
    "wavelength_nielsen",
]
