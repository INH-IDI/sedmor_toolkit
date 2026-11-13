"""Dimensionamiento de escolleras, peso de roca de coraza y filtros costeros.

Implementa de forma vectorizada y autónoma las formulaciones científicas y
manuales técnicos estándar para estimar diámetro nominal (:math:`D_{n50}`), masa
mediana (:math:`M_{50}`), peso unitario (:math:`W_{50}`), espesor de capa y diseño
de subcapas/filtros granulares:

- **Hudson (1959, 1974) / USACE SPM & CEM**: fórmula clásica para unidades de roca
  o prefabricados mediante coeficiente de estabilidad :math:`K_D`.
- **van der Meer (1988)**: estabilidad de escollera en aguas profundas para rotura
  en voluta (*plunging*) y oscilación (*surging*), con permeabilidad nocional
  :math:`P`, nivel de daño :math:`S` y número de olas :math:`N`.
- **Van Gent et al. (2003, 2004) / The Rock Manual (CIRIA C683, 2007)**:
  formulación modificada para aguas someras con :math:`H_{2\\%}` y período
  espectral :math:`T_{m-1,0}`.
- **Isbash (1936)**: estabilidad de enrocado sometido a corrientes hidráulicas con
  corrección opcional por talud lateral (Lane / Carter).
- Propiedades geométricas de coraza, clasificación europea **EN 13383** (*LMA*/*HMA*)
  y verificación de filtros de **Terzaghi** / *The Rock Manual*.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import GRAVITY, PI_STANDARD, RHO_SEA_WATER
from .runup import _as_out, _validate_positive

FloatOrArray = NDArray[np.float64] | float


HUDSON_KD_VALUES: dict[str, float] = {
    "quarry_angular_rough_2_breaking_trunk": 2.0,
    "quarry_angular_rough_2_nonbreaking_trunk": 4.0,
    "quarry_angular_rough_2_breaking_head_1_1p5": 1.9,
    "quarry_angular_rough_2_nonbreaking_head_1_1p5": 3.2,
    "quarry_angular_rough_2_breaking_head_1_2": 1.6,
    "quarry_angular_rough_2_nonbreaking_head_1_2": 2.8,
    "quarry_rounded_2_breaking_trunk": 1.2,
    "quarry_rounded_2_nonbreaking_trunk": 2.4,
    "cubes_modified_2_breaking_trunk": 6.5,
    "cubes_modified_2_nonbreaking_trunk": 7.5,
    "tetrapods_2_breaking_trunk": 7.0,
    "tetrapods_2_nonbreaking_trunk": 8.0,
    "dolos_2_breaking_trunk": 15.8,
    "dolos_2_nonbreaking_trunk": 31.8,
}
"""Coeficientes de estabilidad ``K_D`` de Hudson estándar (USACE SPM / CEM)."""


EN13383_STANDARD_CLASSES: tuple[tuple[str, float, float, float], ...] = (
    ("LMA 5-40", 5.0, 40.0, 22.5),
    ("LMA 10-60", 10.0, 60.0, 35.0),
    ("LMA 40-200", 40.0, 200.0, 120.0),
    ("LMA 60-300", 60.0, 300.0, 180.0),
    ("HMA 300-1000", 300.0, 1000.0, 650.0),
    ("HMA 1000-3000", 1000.0, 3000.0, 2000.0),
    ("HMA 3000-6000", 3000.0, 6000.0, 4500.0),
    ("HMA 6000-10000", 6000.0, 10000.0, 8000.0),
    ("HMA 10000-15000", 10000.0, 15000.0, 12500.0),
)
"""Clases granulométricas estándar EN 13383-1 / Rock Manual: (clase, NLL_kg, NUL_kg, M50_ref_kg)."""


@dataclass(frozen=True)
class RockArmorProperties:
    """Propiedades físicas y geométricas de bloques de escollera."""

    delta: FloatOrArray
    dn50_m: FloatOrArray
    d50_sieve_m: FloatOrArray
    mass_kg: FloatOrArray
    mass_tonnes: FloatOrArray
    weight_kn: FloatOrArray
    layer_thickness_m: FloatOrArray
    stones_per_m2: FloatOrArray


@dataclass(frozen=True)
class ArmorStabilityResult:
    """Resultado de dimensionamiento de coraza por la fórmula de Hudson."""

    cot_alpha: FloatOrArray
    delta: FloatOrArray
    stability_number: FloatOrArray
    dn50_m: FloatOrArray
    d50_sieve_m: FloatOrArray
    mass_kg: FloatOrArray
    mass_tonnes: FloatOrArray
    weight_kn: FloatOrArray
    layer_thickness_m: FloatOrArray
    stones_per_m2: FloatOrArray


@dataclass(frozen=True)
class VanDerMeerArmorResult:
    """Resultado de estabilidad de coraza según van der Meer (1988) o Van Gent (2004)."""

    cot_alpha: FloatOrArray
    delta: FloatOrArray
    iribarren: FloatOrArray
    xi_critical: FloatOrArray
    regime: NDArray[np.str_] | str
    stability_number_Hs: FloatOrArray
    stability_number_H2: FloatOrArray
    dn50_m: FloatOrArray
    d50_sieve_m: FloatOrArray
    mass_kg: FloatOrArray
    mass_tonnes: FloatOrArray
    weight_kn: FloatOrArray
    layer_thickness_m: FloatOrArray
    stones_per_m2: FloatOrArray


@dataclass(frozen=True)
class IsbashArmorResult:
    """Resultado de estabilidad de enrocado bajo flujo de corriente (Isbash, 1936)."""

    isbash_c: FloatOrArray
    slope_factor_ks: FloatOrArray
    delta: FloatOrArray
    d50_stone_m: FloatOrArray
    dn50_m: FloatOrArray
    mass_kg: FloatOrArray
    mass_tonnes: FloatOrArray
    weight_kn: FloatOrArray


@dataclass(frozen=True)
class FilterLayerDesignResult:
    """Dimensionamiento preliminar de filtro/subcapa y verificación de Terzaghi."""

    dn50_armor_m: float
    mass_underlayer_kg: float
    dn50_underlayer_m: float
    thickness_underlayer_m: float
    mass_core_kg: float
    dn50_core_m: float
    filter_d15_approx_m: float | None
    retention_ratio_d15_d85: float | None
    permeability_ratio_d15_d15: float | None
    meets_terzaghi_retention: bool | None
    meets_terzaghi_permeability: bool | None


def _stone_metrics_from_dn50(
    dn50: np.ndarray,
    rho_r: np.ndarray,
    rho_w: np.ndarray,
    k_delta: np.ndarray,
    porosity: np.ndarray,
    n_layers: int,
    dn50_to_d50: float,
    gravity: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Calcula masa, peso, tamiz D50, espesor y densidad superficial desde Dn50."""
    delta = (rho_r / rho_w) - 1.0
    if np.any(delta <= 0.0):
        raise ValueError("La densidad de la roca 'rho_r' debe ser mayor que 'rho_w'.")
    if n_layers < 1:
        raise ValueError("El número de capas 'n_layers' debe ser >= 1.")
    if dn50_to_d50 <= 0.0 or dn50_to_d50 > 1.0:
        raise ValueError("La razón 'dn50_to_d50' debe pertenecer a (0, 1].")
    if np.any((porosity < 0.0) | (porosity >= 1.0)):
        raise ValueError("La porosidad 'porosity' debe estar en [0, 1).")

    d50 = dn50 / dn50_to_d50
    mass_kg = rho_r * np.power(dn50, 3)
    mass_t = mass_kg / 1000.0
    weight_kn = mass_kg * gravity / 1000.0
    thickness = float(n_layers) * k_delta * dn50
    stones_m2 = np.where(
        dn50 > 0.0,
        float(n_layers) * k_delta * (1.0 - porosity) / np.power(np.maximum(dn50, 1e-30), 2),
        0.0,
    )
    return delta, d50, mass_kg, mass_t, weight_kn, thickness, stones_m2


def rock_size_properties(
    *,
    mass_kg: ArrayLike | None = None,
    dn50: ArrayLike | None = None,
    rho_r: ArrayLike = 2650.0,
    rho_w: ArrayLike = RHO_SEA_WATER,
    k_delta: ArrayLike = 1.0,
    porosity: ArrayLike = 0.37,
    n_layers: int = 2,
    dn50_to_d50: float = 0.84,
    gravity: float = GRAVITY,
) -> RockArmorProperties:
    """Convierte entre masa ``M50`` [kg] y diámetro nominal ``Dn50`` [m] y calcula capa."""
    if (mass_kg is None) == (dn50 is None):
        raise ValueError("Debe especificarse exactamente uno de 'mass_kg' o 'dn50'.")
    if gravity <= 0.0:
        raise ValueError("La aceleración de gravedad 'gravity' debe ser > 0.")

    if dn50 is not None:
        size_in, rr_a, rw_a, kd_a, por_a = np.broadcast_arrays(
            np.asarray(dn50, dtype=np.float64),
            np.asarray(rho_r, dtype=np.float64),
            np.asarray(rho_w, dtype=np.float64),
            np.asarray(k_delta, dtype=np.float64),
            np.asarray(porosity, dtype=np.float64),
        )
        _validate_positive("dn50", size_in, allow_zero=False)
        _validate_positive("rho_r", rr_a, allow_zero=False)
        _validate_positive("rho_w", rw_a, allow_zero=False)
        _validate_positive("k_delta", kd_a, allow_zero=False)
        dn_a = size_in
    else:
        mass_in, rr_a, rw_a, kd_a, por_a = np.broadcast_arrays(
            np.asarray(mass_kg, dtype=np.float64),
            np.asarray(rho_r, dtype=np.float64),
            np.asarray(rho_w, dtype=np.float64),
            np.asarray(k_delta, dtype=np.float64),
            np.asarray(porosity, dtype=np.float64),
        )
        _validate_positive("mass_kg", mass_in, allow_zero=False)
        _validate_positive("rho_r", rr_a, allow_zero=False)
        _validate_positive("rho_w", rw_a, allow_zero=False)
        _validate_positive("k_delta", kd_a, allow_zero=False)
        dn_a = np.cbrt(mass_in / rr_a)

    is_scalar = dn_a.ndim == 0
    delta, d50, m_kg, m_t, w_kn, thick, n_stones = _stone_metrics_from_dn50(
        dn_a, rr_a, rw_a, kd_a, por_a, n_layers, dn50_to_d50, gravity
    )
    return RockArmorProperties(
        delta=_as_out(delta, is_scalar),
        dn50_m=_as_out(dn_a, is_scalar),
        d50_sieve_m=_as_out(d50, is_scalar),
        mass_kg=_as_out(m_kg, is_scalar),
        mass_tonnes=_as_out(m_t, is_scalar),
        weight_kn=_as_out(w_kn, is_scalar),
        layer_thickness_m=_as_out(thick, is_scalar),
        stones_per_m2=_as_out(n_stones, is_scalar),
    )


def hudson_armor_weight(
    H: ArrayLike,
    slope: ArrayLike,
    *,
    KD: ArrayLike = 2.0,
    rho_r: ArrayLike = 2650.0,
    rho_w: ArrayLike = RHO_SEA_WATER,
    k_delta: ArrayLike = 1.0,
    porosity: ArrayLike = 0.37,
    n_layers: int = 2,
    dn50_to_d50: float = 0.84,
    gravity: float = GRAVITY,
) -> ArmorStabilityResult:
    """Estima masa y diámetro de coraza mediante la fórmula clásica de Hudson (1959, 1974).

    Implementa :math:`M_{50} = \\rho_r\\,H^3 / (K_D\\,\\Delta^3\\,\\cot\\alpha)`,
    donde ``slope = tan(alpha)`` (p. ej. ``0.5`` para talud ``1V:2H``, ``cot(alpha)=2``).
    Consultar :data:`HUDSON_KD_VALUES` para coeficientes orientativos SPM / CEM.
    """
    if gravity <= 0.0:
        raise ValueError("La aceleración de gravedad 'gravity' debe ser > 0.")

    h_a, s_a, kd_val, rr_a, rw_a, k_d, por_a = np.broadcast_arrays(
        np.asarray(H, dtype=np.float64),
        np.asarray(slope, dtype=np.float64),
        np.asarray(KD, dtype=np.float64),
        np.asarray(rho_r, dtype=np.float64),
        np.asarray(rho_w, dtype=np.float64),
        np.asarray(k_delta, dtype=np.float64),
        np.asarray(porosity, dtype=np.float64),
    )
    is_scalar = h_a.ndim == 0

    _validate_positive("H", h_a, allow_zero=True)
    _validate_positive("slope", s_a, allow_zero=False)
    _validate_positive("KD", kd_val, allow_zero=False)
    _validate_positive("rho_r", rr_a, allow_zero=False)
    _validate_positive("rho_w", rw_a, allow_zero=False)
    _validate_positive("k_delta", k_d, allow_zero=False)

    cot_a = 1.0 / s_a
    delta = (rr_a / rw_a) - 1.0
    if np.any(delta <= 0.0):
        raise ValueError("La densidad de la roca 'rho_r' debe ser mayor que 'rho_w'.")

    ns = np.power(kd_val * cot_a, 1.0 / 3.0)
    dn50 = np.where(h_a > 0.0, h_a / (delta * ns), 0.0)
    _, d50, m_kg, m_t, w_kn, thick, n_stones = _stone_metrics_from_dn50(
        dn50, rr_a, rw_a, k_d, por_a, n_layers, dn50_to_d50, gravity
    )

    return ArmorStabilityResult(
        cot_alpha=_as_out(cot_a, is_scalar),
        delta=_as_out(delta, is_scalar),
        stability_number=_as_out(ns, is_scalar),
        dn50_m=_as_out(dn50, is_scalar),
        d50_sieve_m=_as_out(d50, is_scalar),
        mass_kg=_as_out(m_kg, is_scalar),
        mass_tonnes=_as_out(m_t, is_scalar),
        weight_kn=_as_out(w_kn, is_scalar),
        layer_thickness_m=_as_out(thick, is_scalar),
        stones_per_m2=_as_out(n_stones, is_scalar),
    )


def _evaluate_vdm_family(
    Hs: ArrayLike,
    T: ArrayLike,
    slope: ArrayLike,
    H_ref_for_dn: ArrayLike,
    c_p: float,
    c_s: float,
    P: ArrayLike,
    S: ArrayLike,
    N: ArrayLike,
    rho_r: ArrayLike,
    rho_w: ArrayLike,
    k_delta: ArrayLike,
    porosity: ArrayLike,
    n_layers: int,
    dn50_to_d50: float,
    gravity: float,
) -> VanDerMeerArmorResult:
    """Núcleo compartido para van der Meer (1988) y Van Gent (2004) modificado."""
    if gravity <= 0.0:
        raise ValueError("La aceleración de gravedad 'gravity' debe ser > 0.")

    hs_a, t_a, s_a, href_a, p_a, dam_a, n_a, rr_a, rw_a, kd_a, por_a = np.broadcast_arrays(
        np.asarray(Hs, dtype=np.float64),
        np.asarray(T, dtype=np.float64),
        np.asarray(slope, dtype=np.float64),
        np.asarray(H_ref_for_dn, dtype=np.float64),
        np.asarray(P, dtype=np.float64),
        np.asarray(S, dtype=np.float64),
        np.asarray(N, dtype=np.float64),
        np.asarray(rho_r, dtype=np.float64),
        np.asarray(rho_w, dtype=np.float64),
        np.asarray(k_delta, dtype=np.float64),
        np.asarray(porosity, dtype=np.float64),
    )
    is_scalar = hs_a.ndim == 0

    _validate_positive("Hs", hs_a, allow_zero=True)
    _validate_positive("T", t_a, allow_zero=False)
    _validate_positive("slope", s_a, allow_zero=False)
    _validate_positive("H2_percent", href_a, allow_zero=True)
    _validate_positive("P", p_a, allow_zero=False)
    if np.any(p_a > 0.65):
        raise ValueError("La permeabilidad nocional 'P' suele pertenecer a (0, 0.6].")
    _validate_positive("S", dam_a, allow_zero=False)
    _validate_positive("N", n_a, allow_zero=False)
    _validate_positive("rho_r", rr_a, allow_zero=False)
    _validate_positive("rho_w", rw_a, allow_zero=False)
    _validate_positive("k_delta", kd_a, allow_zero=False)

    cot_a = 1.0 / s_a
    L0 = gravity * (t_a**2) / (2.0 * PI_STANDARD)
    xi = np.where(hs_a > 0.0, s_a / np.sqrt(np.maximum(hs_a / L0, 1e-30)), 0.0)
    xi_cr = np.power((c_p / c_s) * np.power(p_a, 0.31) * np.sqrt(s_a), 1.0 / (p_a + 0.5))

    dam_factor = np.power(dam_a / np.sqrt(n_a), 0.2)
    ns_plg = np.where(
        xi > 0.0,
        c_p * np.power(p_a, 0.18) * dam_factor * np.power(np.maximum(xi, 1e-30), -0.5),
        0.0,
    )
    ns_srg = np.where(
        xi > 0.0,
        c_s * np.power(p_a, -0.13) * dam_factor * np.sqrt(cot_a) * np.power(np.maximum(xi, 1e-30), p_a),
        0.0,
    )

    use_plunging = (cot_a >= 4.0) | (xi < xi_cr)
    ns_ref = np.where(use_plunging, ns_plg, ns_srg)

    delta = (rr_a / rw_a) - 1.0
    if np.any(delta <= 0.0):
        raise ValueError("La densidad de la roca 'rho_r' debe ser mayor que 'rho_w'.")

    dn50 = np.where((href_a > 0.0) & (ns_ref > 0.0), href_a / (delta * np.maximum(ns_ref, 1e-30)), 0.0)
    ns_hs = np.where(dn50 > 0.0, hs_a / (delta * np.maximum(dn50, 1e-30)), 0.0)
    ns_h2 = np.where(dn50 > 0.0, href_a / (delta * np.maximum(dn50, 1e-30)), 0.0)

    _, d50, m_kg, m_t, w_kn, thick, n_stones = _stone_metrics_from_dn50(
        dn50, rr_a, rw_a, kd_a, por_a, n_layers, dn50_to_d50, gravity
    )

    reg_arr = np.where(use_plunging, "plunging", "surging")
    reg_out: NDArray[np.str_] | str = (
        str(np.asarray(reg_arr).item()) if is_scalar else np.asarray(reg_arr, dtype=str)
    )

    return VanDerMeerArmorResult(
        cot_alpha=_as_out(cot_a, is_scalar),
        delta=_as_out(delta, is_scalar),
        iribarren=_as_out(xi, is_scalar),
        xi_critical=_as_out(xi_cr, is_scalar),
        regime=reg_out,
        stability_number_Hs=_as_out(ns_hs, is_scalar),
        stability_number_H2=_as_out(ns_h2, is_scalar),
        dn50_m=_as_out(dn50, is_scalar),
        d50_sieve_m=_as_out(d50, is_scalar),
        mass_kg=_as_out(m_kg, is_scalar),
        mass_tonnes=_as_out(m_t, is_scalar),
        weight_kn=_as_out(w_kn, is_scalar),
        layer_thickness_m=_as_out(thick, is_scalar),
        stones_per_m2=_as_out(n_stones, is_scalar),
    )


def van_der_meer_rock_armor(
    Hs: ArrayLike,
    Tm: ArrayLike,
    slope: ArrayLike,
    *,
    P: ArrayLike = 0.4,
    S: ArrayLike = 2.0,
    N: ArrayLike = 3000.0,
    rho_r: ArrayLike = 2650.0,
    rho_w: ArrayLike = RHO_SEA_WATER,
    approach: Literal["mean", "design"] = "mean",
    k_delta: ArrayLike = 1.0,
    porosity: ArrayLike = 0.37,
    n_layers: int = 2,
    dn50_to_d50: float = 0.84,
    gravity: float = GRAVITY,
) -> VanDerMeerArmorResult:
    """Estima ``Dn50`` y ``M50`` de coraza de roca según van der Meer (1988).

    Aplica las ramas de rotura *plunging* (:math:`\\xi_m < \\xi_{cr}` o :math:`\\cot\\alpha \\ge 4`)
    y *surging* en aguas profundas con permeabilidad nocional ``P`` (0.1 núcleo impermeable,
    0.4 dos capas sobre filtro permeable, 0.6 estructura homogénea) y nivel de daño ``S``
    (típico 2.0 inicio de daño para taludes 1:2–1:3). ``approach="mean"`` usa ``c_p=6.2, c_s=1.0``
    y ``approach="design"`` usa ``c_p=5.2, c_s=0.87``.
    """
    if approach == "mean":
        c_p, c_s = 6.2, 1.0
    elif approach == "design":
        c_p, c_s = 5.2, 0.87
    else:
        raise ValueError("El parámetro 'approach' debe ser 'mean' o 'design'.")

    return _evaluate_vdm_family(
        Hs,
        Tm,
        slope,
        Hs,
        c_p,
        c_s,
        P,
        S,
        N,
        rho_r,
        rho_w,
        k_delta,
        porosity,
        n_layers,
        dn50_to_d50,
        gravity,
    )


def van_gent_shallow_rock_armor(
    Hs: ArrayLike,
    Tm_10: ArrayLike,
    slope: ArrayLike,
    *,
    H2_percent: ArrayLike | None = None,
    P: ArrayLike = 0.4,
    S: ArrayLike = 2.0,
    N: ArrayLike = 3000.0,
    rho_r: ArrayLike = 2650.0,
    rho_w: ArrayLike = RHO_SEA_WATER,
    k_delta: ArrayLike = 1.0,
    porosity: ArrayLike = 0.37,
    n_layers: int = 2,
    dn50_to_d50: float = 0.84,
    gravity: float = GRAVITY,
) -> VanDerMeerArmorResult:
    """Estabilidad de roca en aguas someras según Van Gent et al. (2004) / Rock Manual (2007).

    Utiliza el período espectral energético ``Tm_10``, constantes calibradas ``c_p=8.4`` y
    ``c_s=1.3``, y la altura excedida por el 2% del oleaje al pie ``H2_percent`` (si es ``None``,
    adopta el límite Rayleigh profundo :math:`H_{2\\%} = 1.40\\,H_s`).
    """
    h2_val: ArrayLike = (
        1.40 * np.asarray(Hs, dtype=np.float64) if H2_percent is None else H2_percent
    )
    return _evaluate_vdm_family(
        Hs,
        Tm_10,
        slope,
        h2_val,
        8.4,
        1.3,
        P,
        S,
        N,
        rho_r,
        rho_w,
        k_delta,
        porosity,
        n_layers,
        dn50_to_d50,
        gravity,
    )


def isbash_stone_stability(
    velocity: ArrayLike,
    *,
    regime: Literal["high_turbulence", "low_turbulence"] = "high_turbulence",
    isbash_c: ArrayLike | None = None,
    slope: ArrayLike = 0.0,
    angle_of_repose_deg: ArrayLike = 40.0,
    rho_r: ArrayLike = 2650.0,
    rho_w: ArrayLike = RHO_SEA_WATER,
    dn50_to_d50: float = 0.84,
    gravity: float = GRAVITY,
) -> IsbashArmorResult:
    """Estima tamaño y masa estable de enrocado frente a corrientes según Isbash (1936).

    Implementa :math:`D_{50} = V^2 / (2 g C^2 \\Delta K_s)` con ``C = 0.86`` (alta turbulencia /
    piedra expuesta) o ``C = 1.20`` (baja turbulencia / piedra embebida), e incluye el factor
    de reducción por talud lateral :math:`K_s = \\sqrt{1 - (\\sin\\alpha / \\sin\\phi)^2}`
    cuando ``slope > 0``. La masa se calcula como :math:`M_{50} = \\rho_r\\,(0.84\\,D_{50})^3`.
    """
    if gravity <= 0.0:
        raise ValueError("La aceleración de gravedad 'gravity' debe ser > 0.")
    if dn50_to_d50 <= 0.0 or dn50_to_d50 > 1.0:
        raise ValueError("La razón 'dn50_to_d50' debe pertenecer a (0, 1].")

    if isbash_c is None:
        if regime == "high_turbulence":
            c_expr: ArrayLike = 0.86
        elif regime == "low_turbulence":
            c_expr = 1.20
        else:
            raise ValueError("El parámetro 'regime' debe ser 'high_turbulence' o 'low_turbulence'.")
    else:
        c_expr = isbash_c

    v_a, c_a, s_a, phi_a, rr_a, rw_a = np.broadcast_arrays(
        np.asarray(velocity, dtype=np.float64),
        np.asarray(c_expr, dtype=np.float64),
        np.asarray(slope, dtype=np.float64),
        np.asarray(angle_of_repose_deg, dtype=np.float64),
        np.asarray(rho_r, dtype=np.float64),
        np.asarray(rho_w, dtype=np.float64),
    )
    is_scalar = v_a.ndim == 0

    _validate_positive("velocity", v_a, allow_zero=True)
    _validate_positive("isbash_c", c_a, allow_zero=False)
    _validate_positive("slope", s_a, allow_zero=True)
    _validate_positive("angle_of_repose_deg", phi_a, allow_zero=False)
    _validate_positive("rho_r", rr_a, allow_zero=False)
    _validate_positive("rho_w", rw_a, allow_zero=False)

    delta = (rr_a / rw_a) - 1.0
    if np.any(delta <= 0.0):
        raise ValueError("La densidad de la roca 'rho_r' debe ser mayor que 'rho_w'.")

    alpha_rad = np.arctan(s_a)
    phi_rad = np.radians(phi_a)
    if np.any(alpha_rad >= phi_rad):
        raise ValueError("El ángulo del talud debe ser estrictamente menor que el ángulo de reposo.")

    ks = np.sqrt(1.0 - np.power(np.sin(alpha_rad) / np.sin(phi_rad), 2))
    d50 = (v_a**2) / (2.0 * gravity * (c_a**2) * delta * ks)
    dn50 = dn50_to_d50 * d50
    mass_kg = rr_a * np.power(dn50, 3)
    mass_t = mass_kg / 1000.0
    weight_kn = mass_kg * gravity / 1000.0

    return IsbashArmorResult(
        isbash_c=_as_out(c_a, is_scalar),
        slope_factor_ks=_as_out(ks, is_scalar),
        delta=_as_out(delta, is_scalar),
        d50_stone_m=_as_out(d50, is_scalar),
        dn50_m=_as_out(dn50, is_scalar),
        mass_kg=_as_out(mass_kg, is_scalar),
        mass_tonnes=_as_out(mass_t, is_scalar),
        weight_kn=_as_out(weight_kn, is_scalar),
    )


def classify_rock_grading(mass_kg: float) -> dict[str, float | str]:
    """Asigna la clase granulométrica estándar EN 13383 / Rock Manual más cercana a ``mass_kg``."""
    if mass_kg <= 0.0:
        raise ValueError("La masa 'mass_kg' debe ser > 0.")
    for name, nll, nul, m50_ref in EN13383_STANDARD_CLASSES:
        if nll <= mass_kg <= nul:
            return {
                "grading_class": name,
                "nll_kg": nll,
                "nul_kg": nul,
                "nominal_m50_kg": m50_ref,
                "status": "within_standard_range",
            }
    if mass_kg < EN13383_STANDARD_CLASSES[0][1]:
        return {
            "grading_class": "Grava / Escollera fina (< 5 kg)",
            "nll_kg": 0.0,
            "nul_kg": 5.0,
            "nominal_m50_kg": mass_kg,
            "status": "below_lma_5_40",
        }
    return {
        "grading_class": "Escollera especial o prefabricado (> 15000 kg)",
        "nll_kg": 15000.0,
        "nul_kg": math.inf,
        "nominal_m50_kg": mass_kg,
        "status": "above_hma_10000_15000",
    }


def filter_and_underlayer_design(
    M50_armor_kg: float,
    *,
    rho_r: float = 2650.0,
    armor_to_underlayer_ratio: float = 12.0,
    armor_to_core_ratio: float = 500.0,
    k_delta: float = 1.0,
    n_underlayers: int = 2,
    d15_subsoil_m: float | None = None,
    d50_subsoil_m: float | None = None,
    d85_subsoil_m: float | None = None,
    underlayer_d15_ratio: float = 0.65,
) -> FilterLayerDesignResult:
    """Dimensiona subcapa de filtro y núcleo y verifica reglas geométricas de Terzaghi."""
    if M50_armor_kg <= 0.0 or rho_r <= 0.0:
        raise ValueError("'M50_armor_kg' y 'rho_r' deben ser > 0.")
    if armor_to_underlayer_ratio < 5.0 or armor_to_core_ratio < armor_to_underlayer_ratio:
        raise ValueError("Las razones de masa deben satisfacer 5 <= r_underlayer <= r_core.")

    dn50_armor = (M50_armor_kg / rho_r) ** (1.0 / 3.0)
    m_under = M50_armor_kg / armor_to_underlayer_ratio
    dn50_under = (m_under / rho_r) ** (1.0 / 3.0)
    thick_under = float(n_underlayers) * k_delta * dn50_under

    m_core = M50_armor_kg / armor_to_core_ratio
    dn50_core = (m_core / rho_r) ** (1.0 / 3.0)

    filter_d15: float | None = None
    ret_ratio: float | None = None
    perm_ratio: float | None = None
    ok_ret: bool | None = None
    ok_perm: bool | None = None

    if d85_subsoil_m is not None or d15_subsoil_m is not None or d50_subsoil_m is not None:
        filter_d15 = underlayer_d15_ratio * (dn50_core / 0.84)
        if d85_subsoil_m is not None:
            if d85_subsoil_m <= 0.0:
                raise ValueError("'d85_subsoil_m' debe ser > 0.")
            ret_ratio = filter_d15 / d85_subsoil_m
            ok_ret = ret_ratio <= 5.0
        if d15_subsoil_m is not None:
            if d15_subsoil_m <= 0.0:
                raise ValueError("'d15_subsoil_m' debe ser > 0.")
            perm_ratio = filter_d15 / d15_subsoil_m
            ok_perm = perm_ratio >= 5.0

    return FilterLayerDesignResult(
        dn50_armor_m=float(dn50_armor),
        mass_underlayer_kg=float(m_under),
        dn50_underlayer_m=float(dn50_under),
        thickness_underlayer_m=float(thick_under),
        mass_core_kg=float(m_core),
        dn50_core_m=float(dn50_core),
        filter_d15_approx_m=None if filter_d15 is None else float(filter_d15),
        retention_ratio_d15_d85=None if ret_ratio is None else float(ret_ratio),
        permeability_ratio_d15_d15=None if perm_ratio is None else float(perm_ratio),
        meets_terzaghi_retention=ok_ret,
        meets_terzaghi_permeability=ok_perm,
    )


def compare_armor_methods(
    Hs: ArrayLike,
    T: ArrayLike,
    slope: ArrayLike,
    *,
    KD: ArrayLike = 2.0,
    P: ArrayLike = 0.4,
    S: ArrayLike = 2.0,
    N: ArrayLike = 3000.0,
    H2_percent: ArrayLike | None = None,
    rho_r: ArrayLike = 2650.0,
    rho_w: ArrayLike = RHO_SEA_WATER,
    approach: Literal["mean", "design"] = "mean",
    gravity: float = GRAVITY,
) -> dict[str, FloatOrArray | NDArray[np.str_] | str]:
    """Compara rápidamente Hudson, van der Meer (1988) y Van Gent (2004)."""
    hud = hudson_armor_weight(
        Hs, slope, KD=KD, rho_r=rho_r, rho_w=rho_w, gravity=gravity
    )
    vdm = van_der_meer_rock_armor(
        Hs, T, slope, P=P, S=S, N=N, rho_r=rho_r, rho_w=rho_w, approach=approach, gravity=gravity
    )
    vg = van_gent_shallow_rock_armor(
        Hs,
        T,
        slope,
        H2_percent=H2_percent,
        P=P,
        S=S,
        N=N,
        rho_r=rho_r,
        rho_w=rho_w,
        gravity=gravity,
    )
    return {
        "hudson_dn50_m": hud.dn50_m,
        "hudson_mass_kg": hud.mass_kg,
        "vdm_dn50_m": vdm.dn50_m,
        "vdm_mass_kg": vdm.mass_kg,
        "vdm_regime": vdm.regime,
        "van_gent_dn50_m": vg.dn50_m,
        "van_gent_mass_kg": vg.mass_kg,
        "van_gent_regime": vg.regime,
    }


__all__ = [
    "ArmorStabilityResult",
    "EN13383_STANDARD_CLASSES",
    "FilterLayerDesignResult",
    "HUDSON_KD_VALUES",
    "IsbashArmorResult",
    "RockArmorProperties",
    "VanDerMeerArmorResult",
    "classify_rock_grading",
    "compare_armor_methods",
    "filter_and_underlayer_design",
    "hudson_armor_weight",
    "isbash_stone_stability",
    "rock_size_properties",
    "van_der_meer_rock_armor",
    "van_gent_shallow_rock_armor",
]
