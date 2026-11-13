"""Cálculo de rebase del oleaje (*wave overtopping*) y volúmenes individuales.

Implementa de forma vectorizada y autónoma las formulaciones del manual europeo
**EurOtop (2018)** (*Manual on Wave Overtopping of Sea Defences and Related Structures*,
van der Meer et al., 2018) publicadas en la literatura abierta:

- Caudal medio de rebase sobre diques y taludes costeros lisos o rugosos
  (Cap. 5: enfoque medio ``method="mean"`` y de diseño ``method="design"``).
- Caudal medio de rebase sobre diques en talud de escollera y piezas de coraza
  (Cap. 6, con factores de rugosidad :math:`\\gamma_f` de la Tabla 6.2).
- Caudal medio de rebase sobre muros verticales en régimen pulsátil (no impulsivo)
  e impulsivo según el criterio :math:`d^* = h^2 / (H_{m0}\\,L_{m-1,0}) < 0.23` (Cap. 7).
- Distribución estadística de volúmenes individuales de rebase (fracción :math:`P_{ow}`,
  parámetros de Weibull :math:`a, b`, volumen máximo :math:`V_{\\max}` y :math:`V_{1\\%}`).
- Cálculo inverso del francobordo requerido (:func:`solve_required_freeboard`) y
  clasificación frente a límites tolerables (Cap. 3, Tablas 3.1 y 3.2).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gamma as gamma_func

from .constants import GRAVITY, PI_STANDARD
from .runup import (
    _as_bool_out,
    _as_out,
    _deep_water_length_and_iribarren,
    _validate_positive,
)

FloatOrArray = NDArray[np.float64] | float


EUROTOP_ROUGHNESS_FACTORS: dict[str, float] = {
    "smooth_concrete_asphalt": 1.00,
    "grass": 1.00,
    "basalt_stone_pitching": 0.90,
    "placed_block_revetment": 0.85,
    "rock_1_layer_impermeable": 0.60,
    "rock_2_layer_impermeable": 0.55,
    "rock_1_layer_permeable": 0.45,
    "rock_2_layer_permeable": 0.40,
    "cubes_1_layer": 0.50,
    "cubes_2_layer": 0.47,
    "accropode_1": 0.46,
    "core_loc": 0.44,
    "xbloc": 0.44,
    "tetrapods": 0.38,
}
"""Factores orientativos de rugosidad y permeabilidad ``gamma_f`` (EurOtop, 2018, Tabla 6.2)."""


EUROTOP_TOLERABLE_LIMITS: dict[str, dict[str, float | str]] = {
    "pedestrians_aware": {
        "limit_q_l_s_m": 0.1,
        "limit_V_max_l_m": 50.0,
        "description": "Peatones atentos con visibilidad del mar en paseo amplio (Hm0 ~ 1–3 m).",
    },
    "pedestrians_unaware": {
        "limit_q_l_s_m": 0.03,
        "limit_V_max_l_m": 20.0,
        "description": "Público general no prevenido o cerca del coronamiento.",
    },
    "vehicles_slow": {
        "limit_q_l_s_m": 10.0,
        "limit_V_max_l_m": 2000.0,
        "description": "Vehículos a baja velocidad con pavimento drenado (Hm0 <= 2 m).",
    },
    "buildings_structural": {
        "limit_q_l_s_m": 1.0,
        "limit_V_max_l_m": 1000.0,
        "description": "Edificaciones costeras inmediatas tras paseo marítimo.",
    },
    "embankment_poor_grass": {
        "limit_q_l_s_m": 0.1,
        "limit_V_max_l_m": 500.0,
        "description": "Talud interior de dique con cobertura vegetal deficiente o discontinua.",
    },
    "embankment_well_maintained_grass": {
        "limit_q_l_s_m": 5.0,
        "limit_V_max_l_m": 2000.0,
        "description": "Talud interior de dique con césped denso bien mantenido (Hm0 < 3 m).",
    },
    "breakwater_rear_armoured": {
        "limit_q_l_s_m": 10.0,
        "limit_V_max_l_m": 5000.0,
        "description": "Dique de escollera con espaldón/talud trasero debidamente acorazado.",
    },
}
"""Límites tolerables de caudal medio ``q`` y volumen máximo ``V_max`` (EurOtop, 2018, Cap. 3)."""


@dataclass(frozen=True)
class SlopeOvertoppingResult:
    """Resultado de caudal medio de rebase para taludes costeros y diques en talud."""

    L0: FloatOrArray
    steepness: FloatOrArray
    iribarren: FloatOrArray
    gamma_beta: FloatOrArray
    q_breaking_dimless: FloatOrArray
    q_non_breaking_dimless: FloatOrArray
    q_dimless: FloatOrArray
    q_m3_s_m: FloatOrArray
    q_l_s_m: FloatOrArray
    is_breaking_limited: NDArray[np.bool_] | bool


@dataclass(frozen=True)
class VerticalWallOvertoppingResult:
    """Resultado de caudal medio de rebase para muros verticales y espaldones."""

    L0: FloatOrArray
    steepness: FloatOrArray
    impulsiveness_d_star: FloatOrArray
    is_impulsive: NDArray[np.bool_] | bool
    gamma_beta: FloatOrArray
    q_dimless: FloatOrArray
    q_m3_s_m: FloatOrArray
    q_l_s_m: FloatOrArray


@dataclass(frozen=True)
class OvertoppingVolumesResult:
    """Distribución de volúmenes individuales de rebase según Weibull (EurOtop, 2018)."""

    P_ow: FloatOrArray
    N_waves: FloatOrArray
    N_overtopping: FloatOrArray
    V_mean: FloatOrArray
    weibull_a: FloatOrArray
    weibull_b: FloatOrArray
    V_max: FloatOrArray
    V_1percent: FloatOrArray


@dataclass(frozen=True)
class OvertoppingHazardResult:
    """Evaluación frente a umbrales tolerables de rebase (EurOtop, 2018, Cap. 3)."""

    q_l_s_m: float
    V_max_l_m: float | None
    category: str
    is_acceptable: bool
    limit_q_l_s_m: float
    limit_V_max_l_m: float
    description: str


def _slope_coefficients(method: Literal["mean", "design"]) -> tuple[float, float, float, float]:
    """Devuelve constantes (a_brk, b_brk, a_non, b_non) según EurOtop (2018, Cap. 5)."""
    if method == "mean":
        return 0.023, 2.7, 0.09, 1.5
    if method == "design":
        return 0.026, 2.5, 0.1035, 1.35
    raise ValueError("El parámetro 'method' debe ser 'mean' o 'design'.")


def oblique_factor_overtopping(
    angle_deg: ArrayLike,
    *,
    structure_type: Literal["sloping", "vertical_short", "vertical_long"] = "sloping",
) -> FloatOrArray:
    """Calcula el factor reductor por oblicuidad ``gamma_beta`` para rebase (EurOtop, 2018).

    Parámetros
    ----------
    angle_deg : float o array_like
        Ángulo de incidencia respecto a la normal a la estructura [grados].
    structure_type : {"sloping", "vertical_short", "vertical_long"}, default="sloping"
        Aplica ``max(1 - 0.0033 * min(|beta|, 80), 0.736)`` para taludes y oleaje de
        cresta corta en muros, o ``max(1 - 0.0062 * min(|beta|, 45), 0.721)`` para oleaje
        de cresta larga en muros verticales.
    """
    ang_arr = np.asarray(angle_deg, dtype=np.float64)
    if np.any(np.isnan(ang_arr)):
        raise ValueError("El ángulo 'angle_deg' no debe contener NaN.")
    is_scalar = ang_arr.ndim == 0
    abs_ang = np.abs(ang_arr)
    if np.any(abs_ang > 110.0):
        raise ValueError("El parámetro 'angle_deg' debe estar en el rango [-110, 110] grados.")

    if structure_type in ("sloping", "vertical_short"):
        gb = np.maximum(1.0 - 0.0033 * np.minimum(abs_ang, 80.0), 0.736)
    elif structure_type == "vertical_long":
        gb = np.maximum(1.0 - 0.0062 * np.minimum(abs_ang, 45.0), 0.721)
    else:
        raise ValueError(
            "El parámetro 'structure_type' debe ser 'sloping', 'vertical_short' o 'vertical_long'."
        )
    return _as_out(gb, is_scalar)


def _eval_eurotop_slope(
    Hm0: ArrayLike,
    Tm_10: ArrayLike,
    slope: ArrayLike,
    Rc: ArrayLike,
    *,
    gamma_b: ArrayLike,
    gamma_f: ArrayLike,
    angle_deg: ArrayLike,
    gamma_beta: ArrayLike | None,
    gamma_crest: ArrayLike,
    crest_name: str,
    method: Literal["mean", "design"],
    apply_breaking: bool,
    gravity: float,
) -> SlopeOvertoppingResult:
    """Núcleo compartido para ecuaciones de rebase en taludes y diques (EurOtop, 2018)."""
    a_brk, b_brk, a_non, b_non = _slope_coefficients(method)
    gb_expr: ArrayLike = (
        oblique_factor_overtopping(angle_deg, structure_type="sloping")
        if gamma_beta is None
        else gamma_beta
    )

    h_a, t_a, s_a, rc_a, g_b, g_f, g_beta, g_c = np.broadcast_arrays(
        np.asarray(Hm0, dtype=np.float64),
        np.asarray(Tm_10, dtype=np.float64),
        np.asarray(slope, dtype=np.float64),
        np.asarray(Rc, dtype=np.float64),
        np.asarray(gamma_b, dtype=np.float64),
        np.asarray(gamma_f, dtype=np.float64),
        np.asarray(gb_expr, dtype=np.float64),
        np.asarray(gamma_crest, dtype=np.float64),
    )
    is_scalar = h_a.ndim == 0

    _validate_positive("Rc", rc_a, allow_zero=True)
    _validate_positive("slope", s_a, allow_zero=False)
    _validate_positive("gamma_b", g_b, allow_zero=False)
    _validate_positive("gamma_f", g_f, allow_zero=False)
    _validate_positive("gamma_beta", g_beta, allow_zero=False)
    _validate_positive(crest_name, g_c, allow_zero=False)

    L0, steepness, xi = _deep_water_length_and_iribarren(h_a, t_a, s_a, gravity)

    denom_brk = np.maximum(xi * h_a * g_b * g_f * g_beta * g_c, 1e-30)
    arg_brk = np.where(h_a > 0.0, b_brk * rc_a / denom_brk, 0.0)
    q_brk_star = np.where(
        h_a > 0.0,
        (a_brk / np.sqrt(s_a)) * g_b * xi * np.exp(-np.power(np.maximum(arg_brk, 0.0), 1.3)),
        0.0,
    )

    denom_non = np.maximum(h_a * g_f * g_beta * g_c, 1e-30)
    arg_non = np.where(h_a > 0.0, b_non * rc_a / denom_non, 0.0)
    q_non_star = np.where(
        h_a > 0.0,
        a_non * np.exp(-np.power(np.maximum(arg_non, 0.0), 1.3)),
        0.0,
    )

    if apply_breaking:
        is_brk = (h_a > 0.0) & (q_brk_star <= q_non_star)
        q_star = np.minimum(q_brk_star, q_non_star)
    else:
        is_brk = np.zeros_like(h_a, dtype=bool)
        q_star = q_non_star

    q_m3 = np.where(h_a > 0.0, q_star * np.sqrt(gravity * np.power(h_a, 3)), 0.0)
    q_l = 1000.0 * q_m3

    return SlopeOvertoppingResult(
        L0=_as_out(L0, is_scalar),
        steepness=_as_out(steepness, is_scalar),
        iribarren=_as_out(xi, is_scalar),
        gamma_beta=_as_out(g_beta, is_scalar),
        q_breaking_dimless=_as_out(q_brk_star, is_scalar),
        q_non_breaking_dimless=_as_out(q_non_star, is_scalar),
        q_dimless=_as_out(q_star, is_scalar),
        q_m3_s_m=_as_out(q_m3, is_scalar),
        q_l_s_m=_as_out(q_l, is_scalar),
        is_breaking_limited=_as_bool_out(is_brk, is_scalar),
    )


def overtopping_slope(
    Hm0: ArrayLike,
    Tm_10: ArrayLike,
    slope: ArrayLike,
    Rc: ArrayLike,
    *,
    gamma_b: ArrayLike = 1.0,
    gamma_f: ArrayLike = 1.0,
    angle_deg: ArrayLike = 0.0,
    gamma_beta: ArrayLike | None = None,
    gamma_v: ArrayLike = 1.0,
    method: Literal["mean", "design"] = "mean",
    gravity: float = GRAVITY,
) -> SlopeOvertoppingResult:
    """Calcula el caudal medio de rebase sobre taludes y diques (EurOtop, 2018, Cap. 5).

    Implementa las ecuaciones (5.10)–(5.11) para ``method="mean"`` y las ecuaciones
    (5.12)–(5.13) con margen de diseño (+1σ) para ``method="design"``.

    Parámetros
    ----------
    Hm0 : float o array_like
        Altura significativa espectral al pie de la estructura [m].
    Tm_10 : float o array_like
        Período espectral energético :math:`T_{m-1,0}` [s].
    slope : float o array_like
        Pendiente exterior :math:`\\tan\\alpha` (p. ej. 1:3 -> 0.333) [-].
    Rc : float o array_like
        Francobordo de coronamiento sobre el nivel medio del mar en reposo [m] (>= 0).
    gamma_b, gamma_f, gamma_v : float o array_like, default=1.0
        Factores reductores por berma, rugosidad/permeabilidad y muro vertical de coronamiento.
    angle_deg : float o array_like, default=0.0
        Oblicuidad del oleaje respecto a la normal [grados].
    gamma_beta : float o array_like o None, default=None
        Permite imponer :math:`\\gamma_\\beta` directamente; si es ``None``, se calcula
        con :func:`oblique_factor_overtopping`.
    method : {"mean", "design"}, default="mean"
        Predicción media (``mean``) o de diseño/verificación (``design``).
    """
    return _eval_eurotop_slope(
        Hm0,
        Tm_10,
        slope,
        Rc,
        gamma_b=gamma_b,
        gamma_f=gamma_f,
        angle_deg=angle_deg,
        gamma_beta=gamma_beta,
        gamma_crest=gamma_v,
        crest_name="gamma_v",
        method=method,
        apply_breaking=True,
        gravity=gravity,
    )


def overtopping_rubble_mound(
    Hm0: ArrayLike,
    Tm_10: ArrayLike,
    slope: ArrayLike,
    Rc: ArrayLike,
    *,
    gamma_f: ArrayLike = 0.40,
    angle_deg: ArrayLike = 0.0,
    gamma_beta: ArrayLike | None = None,
    gamma_b: ArrayLike = 1.0,
    crown_wall_factor: ArrayLike = 1.0,
    method: Literal["mean", "design"] = "mean",
    apply_breaking_cap: bool = True,
    gravity: float = GRAVITY,
) -> SlopeOvertoppingResult:
    """Calcula el caudal medio de rebase en diques en talud de escollera (EurOtop, 2018, Cap. 6).

    Para diques en talud típicos (``cot(alpha) = 1.5–2.0``) emplea las ecuaciones (6.5) y (6.6)
    con factor de rugosidad/permeabilidad por defecto ``gamma_f = 0.40`` (coraza de roca en
    dos capas sobre núcleo permeable). Si ``apply_breaking_cap=True`` y el talud es más
    tendido con rotura en talud, aplica además la cota de ondas rompientes (Cap. 5).
    """
    return _eval_eurotop_slope(
        Hm0,
        Tm_10,
        slope,
        Rc,
        gamma_b=gamma_b,
        gamma_f=gamma_f,
        angle_deg=angle_deg,
        gamma_beta=gamma_beta,
        gamma_crest=crown_wall_factor,
        crest_name="crown_wall_factor",
        method=method,
        apply_breaking=apply_breaking_cap,
        gravity=gravity,
    )


def overtopping_vertical_wall(
    Hm0: ArrayLike,
    Tm_10: ArrayLike,
    h_toe: ArrayLike,
    Rc: ArrayLike,
    *,
    method: Literal["mean", "design"] = "mean",
    angle_deg: ArrayLike = 0.0,
    gamma_beta: ArrayLike | None = None,
    wave_crest_type: Literal["short", "long"] = "short",
    foreshore_influencing: bool = True,
    gravity: float = GRAVITY,
) -> VerticalWallOvertoppingResult:
    """Calcula el caudal medio de rebase en muros verticales (EurOtop, 2018, Cap. 7).

    Distingue entre régimen **no impulsivo** (:math:`d^* = h_{\\text{toe}}^2 / (H_{m0}\\,L_{m-1,0}) \\ge 0.23`
    o sin fondo somero influyente, Ec. 7.1) y régimen **impulsivo** (:math:`d^* < 0.23`, Ec. 7.7 para
    :math:`R_c / H_{m0} < 1.35` y Ec. 7.8 para :math:`R_c / H_{m0} \\ge 1.35`).

    Cuando ``method="design"``, añade una desviación estándar (+1σ) documentada en EurOtop (2018)
    a los coeficientes empíricos (p. ej. ``a=0.054, b=2.12`` en régimen no impulsivo;
    ``0.0155`` y ``0.0020`` en régimen impulsivo).
    """
    if method not in ("mean", "design"):
        raise ValueError("El parámetro 'method' debe ser 'mean' o 'design'.")
    if gravity <= 0.0:
        raise ValueError("La aceleración de gravedad 'gravity' debe ser > 0.")

    if gamma_beta is None:
        st_type: Literal["vertical_short", "vertical_long"] = (
            "vertical_short" if wave_crest_type == "short" else "vertical_long"
        )
        gb_expr: ArrayLike = oblique_factor_overtopping(angle_deg, structure_type=st_type)
    else:
        gb_expr = gamma_beta

    h_a, t_a, ht_a, rc_a, g_beta = np.broadcast_arrays(
        np.asarray(Hm0, dtype=np.float64),
        np.asarray(Tm_10, dtype=np.float64),
        np.asarray(h_toe, dtype=np.float64),
        np.asarray(Rc, dtype=np.float64),
        np.asarray(gb_expr, dtype=np.float64),
    )
    is_scalar = h_a.ndim == 0

    _validate_positive("Hm0", h_a, allow_zero=True)
    _validate_positive("Tm_10", t_a, allow_zero=False)
    _validate_positive("h_toe", ht_a, allow_zero=False)
    _validate_positive("Rc", rc_a, allow_zero=True)
    _validate_positive("gamma_beta", g_beta, allow_zero=False)

    L0 = gravity * (t_a**2) / (2.0 * PI_STANDARD)
    steepness = np.where(h_a > 0.0, h_a / L0, 0.0)
    d_star = np.where(h_a > 0.0, (ht_a**2) / np.maximum(h_a * L0, 1e-30), np.inf)
    is_imp = (h_a > 0.0) & bool(foreshore_influencing) & (d_star < 0.23)

    if method == "mean":
        a_non, b_non = 0.047, 2.35
        c_imp_low, c_imp_high = 0.011, 0.0014
    else:
        a_non, b_non = 0.054, 2.12
        c_imp_low, c_imp_high = 0.0155, 0.0020

    rel_rc = np.where(h_a > 0.0, rc_a / np.maximum(h_a * g_beta, 1e-30), 0.0)
    q_non = np.where(
        h_a > 0.0,
        a_non * np.exp(-np.power(np.maximum(b_non * rel_rc, 0.0), 1.3)),
        0.0,
    )

    imp_pref = np.where(
        (h_a > 0.0) & (steepness > 0.0),
        np.sqrt(np.maximum(h_a / np.maximum(ht_a * steepness, 1e-30), 0.0)),
        0.0,
    )
    rel_rc_raw = np.where(h_a > 0.0, rc_a / np.maximum(h_a * g_beta, 1e-30), 0.0)
    q_imp_low = c_imp_low * imp_pref * np.exp(-2.2 * rel_rc_raw)
    q_imp_high = c_imp_high * imp_pref * np.power(np.maximum(rel_rc_raw, 1.35), -3.0)
    q_imp = np.where(rel_rc_raw < 1.35, q_imp_low, q_imp_high)

    q_star = np.where(is_imp, q_imp, q_non)
    q_m3 = np.where(h_a > 0.0, q_star * np.sqrt(gravity * np.power(h_a, 3)), 0.0)
    q_l = 1000.0 * q_m3

    return VerticalWallOvertoppingResult(
        L0=_as_out(L0, is_scalar),
        steepness=_as_out(steepness, is_scalar),
        impulsiveness_d_star=_as_out(d_star, is_scalar),
        is_impulsive=_as_bool_out(is_imp, is_scalar),
        gamma_beta=_as_out(g_beta, is_scalar),
        q_dimless=_as_out(q_star, is_scalar),
        q_m3_s_m=_as_out(q_m3, is_scalar),
        q_l_s_m=_as_out(q_l, is_scalar),
    )


def individual_overtopping_volumes(
    q: ArrayLike,
    Tm: ArrayLike,
    Ru2_percent: ArrayLike,
    Rc: ArrayLike,
    *,
    storm_duration_s: ArrayLike = 3600.0,
    shape_b: ArrayLike | None = None,
    Hm0: ArrayLike | None = None,
    Tm_10: ArrayLike | None = None,
    structure_type: Literal["sloping", "vertical_non_impulsive", "vertical_impulsive"] = "sloping",
    gravity: float = GRAVITY,
) -> OvertoppingVolumesResult:
    """Estima volúmenes individuales de rebase mediante distribución de Weibull (EurOtop, 2018).

    Calcula la fracción de olas que rebasan (:math:`P_{ow}`), número de olas rebasantes
    (:math:`N_{ow}`), volumen medio (:math:`\\bar{V}`), parámetros de Weibull (:math:`a, b`),
    volumen máximo esperado en la tormenta (:math:`V_{\\max}`) y volumen superado por el 1%
    de las olas rebasantes (:math:`V_{1\\%}`).
    """
    if gravity <= 0.0:
        raise ValueError("La aceleración de gravedad 'gravity' debe ser > 0.")

    q_a, tm_a, ru_a, rc_a, dur_a = np.broadcast_arrays(
        np.asarray(q, dtype=np.float64),
        np.asarray(Tm, dtype=np.float64),
        np.asarray(Ru2_percent, dtype=np.float64),
        np.asarray(Rc, dtype=np.float64),
        np.asarray(storm_duration_s, dtype=np.float64),
    )
    is_scalar = q_a.ndim == 0

    _validate_positive("q", q_a, allow_zero=True)
    _validate_positive("Tm", tm_a, allow_zero=False)
    _validate_positive("Ru2_percent", ru_a, allow_zero=False)
    _validate_positive("Rc", rc_a, allow_zero=True)
    _validate_positive("storm_duration_s", dur_a, allow_zero=False)

    # P_ow = exp(ln(0.02) * (Rc / Ru2%)^2)
    p_ow = np.exp(math.log(0.02) * np.power(rc_a / ru_a, 2))
    p_ow = np.where(rc_a == 0.0, 1.0, p_ow)

    n_waves = dur_a / tm_a
    n_ow = p_ow * n_waves

    if shape_b is not None:
        b_arr = np.broadcast_to(np.asarray(shape_b, dtype=np.float64), q_a.shape).astype(np.float64)
        _validate_positive("shape_b", b_arr, allow_zero=False)
    elif structure_type == "vertical_impulsive":
        b_arr = np.full(q_a.shape, 0.85, dtype=np.float64)
    elif structure_type == "sloping" and (Hm0 is not None) and (Tm_10 is not None):
        h_a = np.broadcast_to(np.asarray(Hm0, dtype=np.float64), q_a.shape).astype(np.float64)
        t10_a = np.broadcast_to(np.asarray(Tm_10, dtype=np.float64), q_a.shape).astype(np.float64)
        _validate_positive("Hm0", h_a, allow_zero=False)
        _validate_positive("Tm_10", t10_a, allow_zero=False)
        b_arr = 0.73 + 55.0 * np.power(np.maximum(q_a / (gravity * h_a * t10_a), 0.0), 0.8)
    else:
        b_arr = np.full(q_a.shape, 0.75, dtype=np.float64)

    v_mean = np.where((q_a > 0.0) & (p_ow > 1e-12), q_a * tm_a / np.maximum(p_ow, 1e-12), 0.0)
    a_arr = v_mean / gamma_func(1.0 + 1.0 / b_arr)
    v_max = np.where(
        (n_ow > 1.0) & (a_arr > 0.0),
        a_arr * np.power(np.maximum(np.log(np.maximum(n_ow, 1.0)), 0.0), 1.0 / b_arr),
        v_mean,
    )
    v_1pct = a_arr * np.power(-math.log(0.01), 1.0 / b_arr)

    return OvertoppingVolumesResult(
        P_ow=_as_out(p_ow, is_scalar),
        N_waves=_as_out(n_waves, is_scalar),
        N_overtopping=_as_out(n_ow, is_scalar),
        V_mean=_as_out(v_mean, is_scalar),
        weibull_a=_as_out(a_arr, is_scalar),
        weibull_b=_as_out(b_arr, is_scalar),
        V_max=_as_out(v_max, is_scalar),
        V_1percent=_as_out(v_1pct, is_scalar),
    )


def solve_required_freeboard(
    q_allowable: ArrayLike,
    Hm0: ArrayLike,
    Tm_10: ArrayLike,
    slope: ArrayLike,
    *,
    gamma_b: ArrayLike = 1.0,
    gamma_f: ArrayLike = 1.0,
    angle_deg: ArrayLike = 0.0,
    gamma_beta: ArrayLike | None = None,
    gamma_v: ArrayLike = 1.0,
    method: Literal["mean", "design"] = "design",
    unit: Literal["m3_s_m", "l_s_m"] = "l_s_m",
    gravity: float = GRAVITY,
) -> FloatOrArray:
    """Calcula analíticamente el francobordo mínimo ``Rc`` [m] para no superar ``q_allowable``.

    Invierte de forma exacta el mínimo entre la rama de ondas rompientes y la rama de
    ondas no rompientes de EurOtop (2018). Si ``q_allowable`` supera el rebase máximo a
    francobordo nulo (``Rc = 0``), devuelve ``0.0``.
    """
    a_brk, b_brk, a_non, b_non = _slope_coefficients(method)
    if gamma_beta is None:
        gb_expr: ArrayLike = oblique_factor_overtopping(angle_deg, structure_type="sloping")
    else:
        gb_expr = gamma_beta

    q_a, h_a, t_a, s_a, g_b, g_f, g_beta, g_v = np.broadcast_arrays(
        np.asarray(q_allowable, dtype=np.float64),
        np.asarray(Hm0, dtype=np.float64),
        np.asarray(Tm_10, dtype=np.float64),
        np.asarray(slope, dtype=np.float64),
        np.asarray(gamma_b, dtype=np.float64),
        np.asarray(gamma_f, dtype=np.float64),
        np.asarray(gb_expr, dtype=np.float64),
        np.asarray(gamma_v, dtype=np.float64),
    )
    is_scalar = q_a.ndim == 0

    _validate_positive("q_allowable", q_a, allow_zero=False)
    _validate_positive("Hm0", h_a, allow_zero=False)
    _validate_positive("slope", s_a, allow_zero=False)
    _validate_positive("gamma_b", g_b, allow_zero=False)
    _validate_positive("gamma_f", g_f, allow_zero=False)
    _validate_positive("gamma_beta", g_beta, allow_zero=False)
    _validate_positive("gamma_v", g_v, allow_zero=False)

    q_m3 = q_a / 1000.0 if unit == "l_s_m" else q_a
    if unit not in ("l_s_m", "m3_s_m"):
        raise ValueError("El parámetro 'unit' debe ser 'l_s_m' o 'm3_s_m'.")

    _, _, xi = _deep_water_length_and_iribarren(h_a, t_a, s_a, gravity)
    q_star_target = q_m3 / np.sqrt(gravity * np.power(h_a, 3))

    pref_brk = (a_brk / np.sqrt(s_a)) * g_b * xi
    ratio_brk = q_star_target / np.maximum(pref_brk, 1e-30)
    rc_brk = np.where(
        ratio_brk >= 1.0,
        0.0,
        (xi * h_a * g_b * g_f * g_beta * g_v / b_brk)
        * np.power(np.maximum(-np.log(np.minimum(ratio_brk, 1.0)), 0.0), 1.0 / 1.3),
    )

    ratio_non = q_star_target / a_non
    rc_non = np.where(
        ratio_non >= 1.0,
        0.0,
        (h_a * g_f * g_beta * g_v / b_non)
        * np.power(np.maximum(-np.log(np.minimum(ratio_non, 1.0)), 0.0), 1.0 / 1.3),
    )

    rc_req = np.minimum(rc_brk, rc_non)
    return _as_out(rc_req, is_scalar)


def classify_overtopping_hazard(
    q_l_s_m: float,
    *,
    V_max_l_m: float | None = None,
    category: str = "pedestrians_aware",
) -> OvertoppingHazardResult:
    """Evalúa si el rebase cumple los criterios tolerables de EurOtop (2018, Cap. 3)."""
    if category not in EUROTOP_TOLERABLE_LIMITS:
        valid = ", ".join(sorted(EUROTOP_TOLERABLE_LIMITS))
        raise ValueError(f"Categoría desconocida '{category}'. Opciones válidas: {valid}.")
    if q_l_s_m < 0.0:
        raise ValueError("El caudal 'q_l_s_m' debe ser >= 0.")
    if V_max_l_m is not None and V_max_l_m < 0.0:
        raise ValueError("El volumen máximo 'V_max_l_m' debe ser >= 0.")

    spec = EUROTOP_TOLERABLE_LIMITS[category]
    lim_q = float(spec["limit_q_l_s_m"])
    lim_v = float(spec["limit_V_max_l_m"])
    ok_q = q_l_s_m <= lim_q
    ok_v = True if V_max_l_m is None else (V_max_l_m <= lim_v)
    return OvertoppingHazardResult(
        q_l_s_m=float(q_l_s_m),
        V_max_l_m=None if V_max_l_m is None else float(V_max_l_m),
        category=category,
        is_acceptable=bool(ok_q and ok_v),
        limit_q_l_s_m=lim_q,
        limit_V_max_l_m=lim_v,
        description=str(spec["description"]),
    )


__all__ = [
    "EUROTOP_ROUGHNESS_FACTORS",
    "EUROTOP_TOLERABLE_LIMITS",
    "OvertoppingHazardResult",
    "OvertoppingVolumesResult",
    "SlopeOvertoppingResult",
    "VerticalWallOvertoppingResult",
    "classify_overtopping_hazard",
    "individual_overtopping_volumes",
    "oblique_factor_overtopping",
    "overtopping_rubble_mound",
    "overtopping_slope",
    "overtopping_vertical_wall",
    "solve_required_freeboard",
]
