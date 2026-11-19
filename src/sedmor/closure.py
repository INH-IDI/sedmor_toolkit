"""Profundidad de cierre (*depth of closure*) y derivados morfodinámicos costeros.

Implementa de forma vectorizada y autónoma las formulaciones científicas abiertas más
empleadas en ingeniería costera para definir el límite mar adentro del perfil activo de
playa y sus aplicaciones directas:

- **Profundidad de cierre interna (:math:`d_l`) y externa (:math:`d_i`)**:
  Hallermeier (1978, 1981), Birkemeier (1985), Houston (1996) y estimación estadística
  desde series temporales de oleaje para distintos horizontes temporales (Nicholls et al., 1998).
- **Perfil de equilibrio de Bruun / Dean (1954, 1977, 1987)**: parámetro de escala :math:`A`,
  ancho activo :math:`W_* = (d_c / A)^{3/2}`, pendiente media y volumen de equilibrio.
- **Regla de Bruun (1962) y extensión con *overwash* de Rosati, Dean y Walton (2013)**:
  retroceso de la línea de costa ante ascenso del nivel del mar.
- **Alimentación artificial de playas (*beach nourishment*, Dean, 1991, 2002)**:
  cálculo de volumen necesario y avance de línea de costa para arenas nativas, más gruesas
  (perfil intersecante) o más finas (perfil no intersecante / sumergido).
- **Prisma litoral activo**: equivalencia volumen–desplazamiento de línea de costa
  :math:`\\Delta V = (B + d_c)\\,\\Delta y` y cobertura de obras transversales respecto a :math:`d_c`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

from .constants import GRAVITY, KINEMATIC_VISCOSITY_20C, RHO_SEA_WATER, RHO_SEDIMENT
from .littoral import _settling_velocity_van_rijn
from .runup import _as_out, _validate_positive

FloatOrArray = NDArray[np.float64] | float


@dataclass(frozen=True)
class ClosureDepthResult:
    """Resumen comparativo de estimaciones de profundidad de cierre."""

    He_12h_m: FloatOrArray
    hallermeier_dl_m: FloatOrArray
    birkemeier_dl_m: FloatOrArray
    birkemeier_linear_dl_m: FloatOrArray
    hallermeier_di_outer_m: FloatOrArray
    houston_dl_m: FloatOrArray


@dataclass(frozen=True)
class DeanProfileResult:
    """Perfil de equilibrio de Dean ``h(x) = A * x**(2/3)`` hasta la profundidad de cierre."""

    A_param: float
    closure_depth_m: float
    active_width_m: float
    profile_volume_m3_m: float
    mean_slope: float
    d50_m: float | None
    ws_m_s: float | None
    x_m: NDArray[np.float64]
    depth_m: NDArray[np.float64]


@dataclass(frozen=True)
class BruunRuleResult:
    """Resultado de retroceso costero por ascenso del nivel del mar (Bruun / Rosati et al.)."""

    sea_level_rise_m: FloatOrArray
    closure_depth_m: FloatOrArray
    berm_height_m: FloatOrArray
    active_width_m: FloatOrArray
    active_height_m: FloatOrArray
    overwash_volume_m3_m: FloatOrArray
    recession_m: FloatOrArray


@dataclass(frozen=True)
class BeachNourishmentResult:
    """Resultado de diseño de alimentación artificial de playa según Dean (1991, 2002)."""

    closure_depth_m: float
    berm_height_m: float
    A_native: float
    A_fill: float
    native_width_m: float
    shoreline_advance_m: float
    volume_m3_m: float
    profile_type: Literal["native_match", "intersecting", "non_intersecting", "submerged"]
    intersection_depth_m: float | None
    critical_submerged_volume_m3_m: float
    overfill_ratio_to_native: float


def depth_of_closure_hallermeier(
    He: ArrayLike,
    Te: ArrayLike,
    *,
    gravity: float = GRAVITY,
) -> FloatOrArray:
    """Calcula la profundidad de cierre interna ``d_l`` según Hallermeier (1978, 1981).

    Implementa :math:`d_l = 2.28\\,H_e - 68.5\\,(H_e^2 / (g\\,T_e^2))`, donde :math:`H_e` es la
    altura significativa excedida 12 horas al año (cuantil ~0.137%) y :math:`T_e` su período.
    """
    if gravity <= 0.0:
        raise ValueError("La aceleración de gravedad 'gravity' debe ser > 0.")

    he_a, te_a = np.broadcast_arrays(
        np.asarray(He, dtype=np.float64),
        np.asarray(Te, dtype=np.float64),
    )
    is_scalar = he_a.ndim == 0
    _validate_positive("He", he_a, allow_zero=True)
    _validate_positive("Te", te_a, allow_zero=False)

    dl = np.maximum(2.28 * he_a - 68.5 * (he_a**2) / (gravity * (te_a**2)), 0.0)
    return _as_out(dl, is_scalar)


def depth_of_closure_birkemeier(
    He: ArrayLike,
    Te: ArrayLike | None = None,
    *,
    linear: bool = False,
    gravity: float = GRAVITY,
) -> FloatOrArray:
    """Calcula la profundidad de cierre ``d_l`` según Birkemeier (1985).

    Con ``linear=False`` (por defecto) aplica :math:`d_l = 1.75\\,H_e - 57.9\\,(H_e^2/(g\\,T_e^2))`;
    con ``linear=True`` (o cuando se omite ``Te``) aplica la simplificación :math:`d_l = 1.57\\,H_e`.
    """
    if gravity <= 0.0:
        raise ValueError("La aceleración de gravedad 'gravity' debe ser > 0.")

    he_in = np.asarray(He, dtype=np.float64)
    _validate_positive("He", he_in, allow_zero=True)

    if linear or Te is None:
        if not linear and Te is None:
            raise ValueError("Debe proporcionar 'Te' o activar 'linear=True'.")
        is_scalar = he_in.ndim == 0
        return _as_out(1.57 * he_in, is_scalar)

    he_a, te_a = np.broadcast_arrays(he_in, np.asarray(Te, dtype=np.float64))
    is_scalar = he_a.ndim == 0
    _validate_positive("Te", te_a, allow_zero=False)
    dl = np.maximum(1.75 * he_a - 57.9 * (he_a**2) / (gravity * (te_a**2)), 0.0)
    return _as_out(dl, is_scalar)


def outer_closure_depth_hallermeier(
    H_mean: ArrayLike,
    sigma_H: ArrayLike,
    T_mean: ArrayLike,
    d50: ArrayLike,
    *,
    gravity: float = GRAVITY,
) -> FloatOrArray:
    """Calcula el límite exterior de transporte significativo ``d_i`` (Hallermeier, 1981).

    Implementa :math:`d_i = (\\bar{H}_s - 0.3\\,\\sigma_H)\\,\\bar{T}_s\\,\\sqrt{g / (5000\\,D_{50})}`.
    """
    if gravity <= 0.0:
        raise ValueError("La aceleración de gravedad 'gravity' debe ser > 0.")

    hm_a, sg_a, tm_a, d_a = np.broadcast_arrays(
        np.asarray(H_mean, dtype=np.float64),
        np.asarray(sigma_H, dtype=np.float64),
        np.asarray(T_mean, dtype=np.float64),
        np.asarray(d50, dtype=np.float64),
    )
    is_scalar = hm_a.ndim == 0
    _validate_positive("H_mean", hm_a, allow_zero=True)
    _validate_positive("sigma_H", sg_a, allow_zero=True)
    _validate_positive("T_mean", tm_a, allow_zero=False)
    _validate_positive("d50", d_a, allow_zero=False)

    h_eff = np.maximum(hm_a - 0.3 * sg_a, 0.0)
    di = h_eff * tm_a * np.sqrt(gravity / (5000.0 * d_a))
    return _as_out(di, is_scalar)


def depth_of_closure_annual_stats(
    H_mean: ArrayLike,
    sigma_H: ArrayLike,
    *,
    T_mean: ArrayLike | None = None,
    T_e: ArrayLike | None = None,
    d50: ArrayLike | None = None,
    houston_calibration: Literal["birkemeier", "hallermeier"] = "birkemeier",
    gravity: float = GRAVITY,
) -> ClosureDepthResult:
    """Estima profundidades de cierre desde estadísticas anuales (:math:`\\bar{H}_s, \\sigma_H`).

    Utiliza la aproximación de Hallermeier (1981) :math:`H_e \\approx \\bar{H}_s + 5.6\\,\\sigma_H`
    junto con Houston (1996: ``6.75 * H_mean`` calibrado a Birkemeier, u ``8.9 * H_mean``
    equivalente a Hallermeier). Si se omiten ``T_e``/``T_mean`` o ``d50``, los campos
    no evaluables devuelven la aproximación lineal o ``nan``.
    """
    if houston_calibration not in ("birkemeier", "hallermeier"):
        raise ValueError("'houston_calibration' debe ser 'birkemeier' o 'hallermeier'.")
    if gravity <= 0.0:
        raise ValueError("La aceleración de gravedad 'gravity' debe ser > 0.")

    hm_a, sg_a = np.broadcast_arrays(
        np.asarray(H_mean, dtype=np.float64),
        np.asarray(sigma_H, dtype=np.float64),
    )
    _validate_positive("H_mean", hm_a, allow_zero=True)
    _validate_positive("sigma_H", sg_a, allow_zero=True)

    he = np.where(hm_a + sg_a > 0.0, hm_a + 5.6 * sg_a, 0.0)
    birk_lin = 1.57 * he
    houston_coef = 6.75 if houston_calibration == "birkemeier" else 8.9
    houston_dl = houston_coef * hm_a

    if T_e is not None:
        te_arr = np.asarray(T_e, dtype=np.float64)
        hall_dl = depth_of_closure_hallermeier(he, te_arr, gravity=gravity)
        birk_dl = depth_of_closure_birkemeier(he, te_arr, linear=False, gravity=gravity)
    else:
        hall_dl_arr = np.where(hm_a + sg_a > 0.0, 2.0 * hm_a + 11.0 * sg_a, 0.0)
        hall_dl = _as_out(hall_dl_arr, hm_a.ndim == 0)
        birk_dl = _as_out(birk_lin, hm_a.ndim == 0)

    if T_mean is not None and d50 is not None:
        di_out = outer_closure_depth_hallermeier(hm_a, sg_a, T_mean, d50, gravity=gravity)
    else:
        nan_shape = np.broadcast_shapes(hm_a.shape, np.asarray(birk_lin).shape)
        di_out = float("nan") if nan_shape == () else np.full(nan_shape, np.nan, dtype=np.float64)

    is_scalar = np.asarray(birk_lin).ndim == 0
    return ClosureDepthResult(
        He_12h_m=_as_out(he, is_scalar),
        hallermeier_dl_m=hall_dl,
        birkemeier_dl_m=birk_dl,
        birkemeier_linear_dl_m=_as_out(birk_lin, is_scalar),
        hallermeier_di_outer_m=di_out,
        houston_dl_m=_as_out(houston_dl, is_scalar),
    )


def closure_depth_from_wave_series(
    Hs_series: ArrayLike,
    T_series: ArrayLike,
    *,
    exceedance_hours_per_year: float = 12.0,
    years: float = 1.0,
    d50: float | None = None,
    gravity: float = GRAVITY,
) -> dict[str, float]:
    """Calcula la profundidad de cierre a partir de una serie temporal de oleaje ``(Hs, T)``.

    Para horizontes de diseño de ``years`` años (p. ej. 1, 10, 50 años según Nicholls et al.,
    1998), obtiene la altura excedida ``exceedance_hours_per_year`` horas en un año
    equivalente dividida por ``years`` (probabilidad :math:`p = h_{\\text{exc}} / (8760\\,\\text{años})`)
    y evalúa Hallermeier (1981) y Birkemeier (1985).
    """
    if exceedance_hours_per_year <= 0.0 or exceedance_hours_per_year >= 8760.0:
        raise ValueError("'exceedance_hours_per_year' debe estar en (0, 8760).")
    if years <= 0.0:
        raise ValueError("El horizonte 'years' debe ser > 0.")

    hs = np.asarray(Hs_series, dtype=np.float64).ravel()
    ts = np.asarray(T_series, dtype=np.float64).ravel()
    if hs.size < 2 or hs.shape != ts.shape:
        raise ValueError("'Hs_series' y 'T_series' deben tener igual tamaño (>= 2).")
    _validate_positive("Hs_series", hs, allow_zero=True)
    _validate_positive("T_series", ts, allow_zero=False)

    prob = min(exceedance_hours_per_year / (8760.0 * years), 0.999999)
    he = float(np.quantile(hs, 1.0 - prob))
    mask = hs >= he
    te = float(np.median(ts[mask])) if np.any(mask) else float(ts[int(np.argmax(hs))])

    h_mean = float(np.mean(hs))
    sigma_h = float(np.std(hs, ddof=1))
    t_mean = float(np.mean(ts))

    cmp = compare_closure_depth_methods(
        he, te, H_mean=h_mean, sigma_H=sigma_h, T_mean=t_mean, d50=d50, gravity=gravity
    )
    return {
        "exceedance_probability": float(prob),
        "He_m": he,
        "Te_s": te,
        "H_mean_m": h_mean,
        "sigma_H_m": sigma_h,
        "hallermeier_dl_m": float(cmp.hallermeier_dl_m),
        "birkemeier_dl_m": float(cmp.birkemeier_dl_m),
        "birkemeier_linear_dl_m": float(cmp.birkemeier_linear_dl_m),
        "hallermeier_di_outer_m": float(cmp.hallermeier_di_outer_m),
    }


def compare_closure_depth_methods(
    He: ArrayLike,
    Te: ArrayLike,
    *,
    H_mean: ArrayLike | None = None,
    sigma_H: ArrayLike | None = None,
    T_mean: ArrayLike | None = None,
    d50: ArrayLike | None = None,
    gravity: float = GRAVITY,
) -> ClosureDepthResult:
    """Compara Hallermeier (1981), Birkemeier (1985) y Houston (1996) en una sola llamada."""
    he_a = np.asarray(He, dtype=np.float64)
    is_scalar = he_a.ndim == 0

    hall_dl = depth_of_closure_hallermeier(He, Te, gravity=gravity)
    birk_dl = depth_of_closure_birkemeier(He, Te, linear=False, gravity=gravity)
    birk_lin = depth_of_closure_birkemeier(He, linear=True, gravity=gravity)

    if H_mean is not None:
        hm_a = np.asarray(H_mean, dtype=np.float64)
        _validate_positive("H_mean", hm_a, allow_zero=True)
        houston_dl = _as_out(6.75 * hm_a, hm_a.ndim == 0)
    else:
        houston_dl = float("nan") if is_scalar else np.full_like(he_a, np.nan)

    if H_mean is not None and sigma_H is not None and T_mean is not None and d50 is not None:
        di_out = outer_closure_depth_hallermeier(H_mean, sigma_H, T_mean, d50, gravity=gravity)
    else:
        di_out = float("nan") if is_scalar else np.full_like(he_a, np.nan)

    return ClosureDepthResult(
        He_12h_m=_as_out(he_a, is_scalar),
        hallermeier_dl_m=hall_dl,
        birkemeier_dl_m=birk_dl,
        birkemeier_linear_dl_m=birk_lin,
        hallermeier_di_outer_m=di_out,
        houston_dl_m=houston_dl,
    )


def dean_scale_parameter(
    d50: ArrayLike | None = None,
    *,
    ws: ArrayLike | None = None,
    rho_s: float = RHO_SEDIMENT,
    rho_w: float = RHO_SEA_WATER,
    nu: float = KINEMATIC_VISCOSITY_20C,
    gravity: float = GRAVITY,
) -> FloatOrArray:
    """Calcula el parámetro de escala ``A`` [m^(1/3)] del perfil de equilibrio de Dean (1987).

    Implementa :math:`A = 0.067\\,(100\\,w_s)^{0.44}` (Dean, 1987; Kriebel & Dean, 1993),
    donde ``ws`` es la velocidad de caída [m/s]. Si se pasa ``d50`` [m] en lugar de ``ws``,
    calcula ``ws`` mediante la formulación de van Rijn (1993) incluida en :mod:`sedmor.littoral`.
    """
    if (d50 is None) == (ws is None):
        raise ValueError("Debe especificarse exactamente uno de 'd50' o 'ws'.")
    if gravity <= 0.0 or rho_s <= rho_w or rho_w <= 0.0 or nu <= 0.0:
        raise ValueError("Los parámetros físicos deben satisfacer g > 0, rho_s > rho_w > 0, nu > 0.")

    if ws is not None:
        ws_a = np.asarray(ws, dtype=np.float64)
        is_scalar = ws_a.ndim == 0
        _validate_positive("ws", ws_a, allow_zero=False)
    else:
        d_a = np.asarray(d50, dtype=np.float64)
        is_scalar = d_a.ndim == 0
        _validate_positive("d50", d_a, allow_zero=False)
        s_rel = (rho_s / rho_w) - 1.0
        ws_a = _settling_velocity_van_rijn(d_a, s_rel, gravity, nu)

    a_param = 0.067 * np.power(100.0 * ws_a, 0.44)
    return _as_out(a_param, is_scalar)


def equilibrium_profile_dean(
    closure_depth: float,
    *,
    A_param: float | None = None,
    d50: float | None = None,
    ws: float | None = None,
    n_points: int = 101,
    rho_s: float = RHO_SEDIMENT,
    rho_w: float = RHO_SEA_WATER,
    nu: float = KINEMATIC_VISCOSITY_20C,
    gravity: float = GRAVITY,
) -> DeanProfileResult:
    """Genera el perfil de equilibrio ``h(x) = A * x**(2/3)`` hasta la profundidad de cierre.

    Calcula el ancho activo :math:`W_* = (d_c / A)^{3/2}`, el volumen subacuático del perfil
    de equilibrio :math:`V_{\\text{eq}} = \\frac{3}{5}\\,d_c\\,W_*` y la pendiente media :math:`d_c / W_*`.
    """
    if closure_depth <= 0.0:
        raise ValueError("La profundidad de cierre 'closure_depth' debe ser > 0.")
    if n_points < 2:
        raise ValueError("'n_points' debe ser >= 2.")

    ws_out: float | None = None if ws is None else float(ws)
    d50_out: float | None = None if d50 is None else float(d50)

    if A_param is not None:
        if d50 is not None or ws is not None:
            raise ValueError("Si especifica 'A_param', omita 'd50' y 'ws'.")
        if A_param <= 0.0:
            raise ValueError("'A_param' debe ser > 0.")
        a_val = float(A_param)
    else:
        a_val = float(
            dean_scale_parameter(d50=d50, ws=ws, rho_s=rho_s, rho_w=rho_w, nu=nu, gravity=gravity)
        )
        if ws_out is None and d50 is not None:
            s_rel = (rho_s / rho_w) - 1.0
            ws_out = float(
                _settling_velocity_van_rijn(np.asarray(float(d50)), s_rel, gravity, nu)
            )

    w_star = (closure_depth / a_val) ** 1.5
    x_grid = np.linspace(0.0, w_star, n_points, dtype=np.float64)
    h_grid = a_val * np.power(x_grid, 2.0 / 3.0)
    vol_eq = 0.6 * closure_depth * w_star
    mean_slope = closure_depth / w_star

    return DeanProfileResult(
        A_param=a_val,
        closure_depth_m=float(closure_depth),
        active_width_m=float(w_star),
        profile_volume_m3_m=float(vol_eq),
        mean_slope=float(mean_slope),
        d50_m=d50_out,
        ws_m_s=ws_out,
        x_m=x_grid,
        depth_m=h_grid,
    )


def bruun_rule_recession(
    sea_level_rise: ArrayLike,
    closure_depth: ArrayLike,
    *,
    berm_height: ArrayLike = 2.0,
    active_width: ArrayLike | None = None,
    A_param: ArrayLike | None = None,
    d50: ArrayLike | None = None,
    overwash_volume: ArrayLike = 0.0,
) -> BruunRuleResult:
    """Calcula el retroceso de la línea de costa por ascenso del nivel del mar (Bruun, 1962).

    Implementa la regla clásica de Bruun y su extensión con pérdidas hacia tierra por
    sobrepaso o transporte eólico (Rosati, Dean & Walton, 2013):

    .. math::
        R = \\frac{S\\,W_* + V_D}{B + d_c}

    Si se omite ``active_width``, se deduce automáticamente desde el perfil de equilibrio de
    Dean :math:`W_* = (d_c / A)^{3/2}` usando ``A_param`` o ``d50``.
    """
    slr_a, dc_a, b_a, vd_a = np.broadcast_arrays(
        np.asarray(sea_level_rise, dtype=np.float64),
        np.asarray(closure_depth, dtype=np.float64),
        np.asarray(berm_height, dtype=np.float64),
        np.asarray(overwash_volume, dtype=np.float64),
    )
    _validate_positive("sea_level_rise", slr_a, allow_zero=True)
    _validate_positive("closure_depth", dc_a, allow_zero=False)
    _validate_positive("berm_height", b_a, allow_zero=True)
    _validate_positive("overwash_volume", vd_a, allow_zero=True)

    if active_width is not None:
        if A_param is not None or d50 is not None:
            raise ValueError("Si entrega 'active_width', omita 'A_param' y 'd50'.")
        w_in = np.asarray(active_width, dtype=np.float64)
        _validate_positive("active_width", w_in, allow_zero=False)
        slr_a, dc_a, b_a, vd_a, w_a = np.broadcast_arrays(slr_a, dc_a, b_a, vd_a, w_in)
    else:
        if A_param is not None:
            if d50 is not None:
                raise ValueError("Especifique 'A_param' o 'd50', no ambos.")
            a_arr = np.asarray(A_param, dtype=np.float64)
            _validate_positive("A_param", a_arr, allow_zero=False)
        elif d50 is not None:
            a_arr = np.asarray(dean_scale_parameter(d50=d50), dtype=np.float64)
        else:
            raise ValueError("Debe proporcionar 'active_width', 'A_param' o 'd50'.")
        slr_a, dc_a, b_a, vd_a, a_arr = np.broadcast_arrays(slr_a, dc_a, b_a, vd_a, a_arr)
        w_a = np.power(dc_a / a_arr, 1.5)

    is_scalar = slr_a.ndim == 0
    h_act = b_a + dc_a
    rec = (slr_a * w_a + vd_a) / h_act

    return BruunRuleResult(
        sea_level_rise_m=_as_out(slr_a, is_scalar),
        closure_depth_m=_as_out(dc_a, is_scalar),
        berm_height_m=_as_out(b_a, is_scalar),
        active_width_m=_as_out(w_a, is_scalar),
        active_height_m=_as_out(h_act, is_scalar),
        overwash_volume_m3_m=_as_out(vd_a, is_scalar),
        recession_m=_as_out(rec, is_scalar),
    )


def beach_nourishment_volume(
    shoreline_advance: float,
    closure_depth: float,
    berm_height: float,
    A_native: float,
    *,
    A_fill: float | None = None,
) -> BeachNourishmentResult:
    """Calcula el volumen de regeneración artificial de playa por metro lineal (Dean, 1991, 2002).

    Distingue automáticamente si el perfil relleno con arena de parámetro ``A_fill`` respecto
    al nativo ``A_native`` resulta en ajuste nativo (``A_fill == A_native``), **intersecante**
    (arena más gruesa :math:`A_F > A_N` dentro de :math:`d_c`, ahorrando volumen) o
    **no intersecante** (corte más allá de :math:`d_c` o arena más fina :math:`A_F < A_N`).
    """
    if shoreline_advance <= 0.0:
        raise ValueError("'shoreline_advance' debe ser > 0.")
    if closure_depth <= 0.0 or berm_height < 0.0 or A_native <= 0.0:
        raise ValueError("'closure_depth' y 'A_native' deben ser > 0 y 'berm_height' >= 0.")

    a_fill_val = A_native if A_fill is None else float(A_fill)
    if a_fill_val <= 0.0:
        raise ValueError("'A_fill' debe ser > 0.")

    w_n = (closure_depth / A_native) ** 1.5
    native_vol = (berm_height + closure_depth) * shoreline_advance
    ratio_an_af = (A_native / a_fill_val) ** 1.5

    if a_fill_val < A_native:
        v_crit = 0.4 * closure_depth * w_n * (ratio_an_af - 1.0)
    else:
        v_crit = 0.0

    if abs(a_fill_val - A_native) <= 1e-12 * A_native:
        p_type: Literal["native_match", "intersecting", "non_intersecting", "submerged"] = (
            "native_match"
        )
        vol = native_vol
        h_int: float | None = None
    elif (a_fill_val > A_native) and ((shoreline_advance / w_n) < (1.0 - ratio_an_af)):
        p_type = "intersecting"
        h_int = closure_depth * ((shoreline_advance / w_n) / (1.0 - ratio_an_af)) ** (2.0 / 3.0)
        vol = shoreline_advance * (berm_height + 0.6 * h_int)
    else:
        p_type = "non_intersecting"
        vol = (berm_height + closure_depth) * shoreline_advance + 0.4 * closure_depth * w_n * (
            ratio_an_af - 1.0
        )
        h_int = None

    return BeachNourishmentResult(
        closure_depth_m=float(closure_depth),
        berm_height_m=float(berm_height),
        A_native=float(A_native),
        A_fill=float(a_fill_val),
        native_width_m=float(w_n),
        shoreline_advance_m=float(shoreline_advance),
        volume_m3_m=float(vol),
        profile_type=p_type,
        intersection_depth_m=None if h_int is None else float(h_int),
        critical_submerged_volume_m3_m=float(v_crit),
        overfill_ratio_to_native=float(vol / native_vol),
    )


def beach_nourishment_advance(
    volume: float,
    closure_depth: float,
    berm_height: float,
    A_native: float,
    *,
    A_fill: float | None = None,
) -> BeachNourishmentResult:
    """Problema inverso de Dean (1991, 2002): avance de línea de costa para un volumen dado."""
    if volume <= 0.0:
        raise ValueError("'volume' debe ser > 0.")
    if closure_depth <= 0.0 or berm_height < 0.0 or A_native <= 0.0:
        raise ValueError("'closure_depth' y 'A_native' deben ser > 0 y 'berm_height' >= 0.")

    a_fill_val = A_native if A_fill is None else float(A_fill)
    if a_fill_val <= 0.0:
        raise ValueError("'A_fill' debe ser > 0.")

    w_n = (closure_depth / A_native) ** 1.5
    ratio_an_af = (A_native / a_fill_val) ** 1.5
    v_crit = 0.4 * closure_depth * w_n * max(ratio_an_af - 1.0, 0.0)

    if abs(a_fill_val - A_native) <= 1e-12 * A_native:
        dy = volume / (berm_height + closure_depth)
        return beach_nourishment_volume(dy, closure_depth, berm_height, A_native, A_fill=a_fill_val)

    if a_fill_val < A_native:
        if volume <= v_crit:
            return BeachNourishmentResult(
                closure_depth_m=float(closure_depth),
                berm_height_m=float(berm_height),
                A_native=float(A_native),
                A_fill=float(a_fill_val),
                native_width_m=float(w_n),
                shoreline_advance_m=0.0,
                volume_m3_m=float(volume),
                profile_type="submerged",
                intersection_depth_m=None,
                critical_submerged_volume_m3_m=float(v_crit),
                overfill_ratio_to_native=float("inf"),
            )
        dy = (volume - v_crit) / (berm_height + closure_depth)
        return beach_nourishment_volume(dy, closure_depth, berm_height, A_native, A_fill=a_fill_val)

    # Arena más gruesa (A_fill > A_native): comprobar límite entre intersecante y no intersecante
    dy_lim = w_n * (1.0 - ratio_an_af)
    vol_lim = dy_lim * (berm_height + 0.6 * closure_depth)
    if volume >= vol_lim:
        dy = (volume + 0.4 * closure_depth * w_n * (1.0 - ratio_an_af)) / (
            berm_height + closure_depth
        )
    else:
        coeff = 0.6 * ((A_native**-1.5 - a_fill_val**-1.5) ** (-2.0 / 3.0))

        def residual(dy_val: float) -> float:
            return berm_height * dy_val + coeff * (dy_val ** (5.0 / 3.0)) - volume

        dy = float(brentq(residual, 0.0, dy_lim))

    return beach_nourishment_volume(dy, closure_depth, berm_height, A_native, A_fill=a_fill_val)


def shoreline_change_from_volume(
    delta_volume: ArrayLike,
    closure_depth: ArrayLike,
    berm_height: ArrayLike,
) -> FloatOrArray:
    """Convierte cambio de volumen por metro lineal en desplazamiento de costa :math:`\\Delta y`."""
    dv_a, dc_a, b_a = np.broadcast_arrays(
        np.asarray(delta_volume, dtype=np.float64),
        np.asarray(closure_depth, dtype=np.float64),
        np.asarray(berm_height, dtype=np.float64),
    )
    is_scalar = dv_a.ndim == 0
    _validate_positive("closure_depth", dc_a, allow_zero=False)
    _validate_positive("berm_height", b_a, allow_zero=True)
    return _as_out(dv_a / (dc_a + b_a), is_scalar)


def volume_change_from_shoreline(
    shoreline_change: ArrayLike,
    closure_depth: ArrayLike,
    berm_height: ArrayLike,
) -> FloatOrArray:
    """Convierte desplazamiento de costa :math:`\\Delta y` en volumen activo :math:`\\Delta V`."""
    dy_a, dc_a, b_a = np.broadcast_arrays(
        np.asarray(shoreline_change, dtype=np.float64),
        np.asarray(closure_depth, dtype=np.float64),
        np.asarray(berm_height, dtype=np.float64),
    )
    is_scalar = dy_a.ndim == 0
    _validate_positive("closure_depth", dc_a, allow_zero=False)
    _validate_positive("berm_height", b_a, allow_zero=True)
    return _as_out(dy_a * (dc_a + b_a), is_scalar)


def structure_active_depth_ratio(
    closure_depth: ArrayLike,
    *,
    depth_at_tip: ArrayLike | None = None,
    structure_length: ArrayLike | None = None,
    A_param: ArrayLike | None = None,
) -> dict[str, FloatOrArray]:
    """Evalúa qué fracción de la zona activa y del ancho de cierre cubre un espigón/obra.

    Puede entregar ``depth_at_tip`` directamente, o bien ``structure_length`` y ``A_param``
    para evaluar la profundidad del morro sobre el perfil de equilibrio de Dean.
    """
    if (depth_at_tip is None) == (structure_length is None):
        raise ValueError("Especifique exactamente uno de 'depth_at_tip' o 'structure_length'.")

    dc_a = np.asarray(closure_depth, dtype=np.float64)
    _validate_positive("closure_depth", dc_a, allow_zero=False)

    if depth_at_tip is not None:
        ht_a, dc_b = np.broadcast_arrays(np.asarray(depth_at_tip, dtype=np.float64), dc_a)
        _validate_positive("depth_at_tip", ht_a, allow_zero=True)
    else:
        if A_param is None:
            raise ValueError("Debe proporcionar 'A_param' cuando especifica 'structure_length'.")
        len_a, a_a, dc_b = np.broadcast_arrays(
            np.asarray(structure_length, dtype=np.float64),
            np.asarray(A_param, dtype=np.float64),
            dc_a,
        )
        _validate_positive("structure_length", len_a, allow_zero=True)
        _validate_positive("A_param", a_a, allow_zero=False)
        ht_a = a_a * np.power(len_a, 2.0 / 3.0)

    is_scalar = ht_a.ndim == 0
    depth_ratio = np.minimum(ht_a / dc_b, 1.0)
    width_ratio = np.power(depth_ratio, 1.5)
    return {
        "depth_at_tip_m": _as_out(ht_a, is_scalar),
        "active_depth_ratio": _as_out(depth_ratio, is_scalar),
        "active_width_ratio": _as_out(width_ratio, is_scalar),
    }


__all__ = [
    "BeachNourishmentResult",
    "BruunRuleResult",
    "ClosureDepthResult",
    "DeanProfileResult",
    "beach_nourishment_advance",
    "beach_nourishment_volume",
    "bruun_rule_recession",
    "closure_depth_from_wave_series",
    "compare_closure_depth_methods",
    "dean_scale_parameter",
    "depth_of_closure_annual_stats",
    "depth_of_closure_birkemeier",
    "depth_of_closure_hallermeier",
    "equilibrium_profile_dean",
    "outer_closure_depth_hallermeier",
    "shoreline_change_from_volume",
    "structure_active_depth_ratio",
    "volume_change_from_shoreline",
]
