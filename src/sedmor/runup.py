"""Cálculo de ascenso del oleaje (*wave run-up*) y nivel total de inundación costera.

Implementa de forma vectorizada formulaciones científicas clásicas y modernas
publicadas en la literatura abierta para playas naturales y estructuras costeras
en talud:

- Número de Iribarren y clasificación del tipo de rotura (Battjes, 1974).
- Playas naturales de arena: Stockdon et al. (2006), Holman (1986),
  Mase (1989) y Nielsen y Hanslow (1991).
- Estructuras costeras y taludes rígidos/escollera: Hunt (1959) y manual
  EurOtop (van der Meer et al., 2018) con factores de berma, rugosidad y
  oblicuidad.
- Combinación con nivel del mar en reposo (*TWL — Total Water Level*).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import GRAVITY, PI_STANDARD


FloatOrArray = NDArray[np.float64] | float


@dataclass(frozen=True)
class IribarrenResult:
    """Parámetros adimensionales de pendiente y rotura en aguas profundas."""

    L0: FloatOrArray
    steepness: FloatOrArray
    iribarren: FloatOrArray
    breaker_type: NDArray[np.str_] | str


@dataclass(frozen=True)
class StockdonRunupResult:
    """Resultados desglosados del modelo de Stockdon et al. (2006)."""

    L0: FloatOrArray
    iribarren: FloatOrArray
    setup: FloatOrArray
    swash_incident: FloatOrArray
    swash_infragravity: FloatOrArray
    swash_total: FloatOrArray
    R2_percent: FloatOrArray
    is_dissipative: NDArray[np.bool_] | bool


@dataclass(frozen=True)
class BeachRunupResult:
    """Resultado genérico para modelos paramétricos de *run-up* en playa o talud."""

    L0: FloatOrArray
    iribarren: FloatOrArray
    relative_runup: FloatOrArray
    R2_percent: FloatOrArray


@dataclass(frozen=True)
class NielsenHanslowResult:
    """Resultado del modelo de *run-up* de Nielsen y Hanslow (1991)."""

    L0: FloatOrArray
    iribarren: FloatOrArray
    scale_LR: FloatOrArray
    R2_percent: FloatOrArray
    is_steep_branch: NDArray[np.bool_] | bool


@dataclass(frozen=True)
class EurOtopRunupResult:
    """Resultado del método EurOtop (2018) para diques, escolleras y defensas en talud."""

    L0: FloatOrArray
    iribarren: FloatOrArray
    gamma_beta: FloatOrArray
    R2_breaking: FloatOrArray
    R2_surging_cap: FloatOrArray
    R2_percent: FloatOrArray
    limited_by_cap: NDArray[np.bool_] | bool


@dataclass(frozen=True)
class TotalWaterLevelResult:
    """Nivel total de inundación costera (*Total Water Level*)."""

    swl: FloatOrArray
    runup_2percent: FloatOrArray
    twl: FloatOrArray


def _as_out(arr: np.ndarray, is_scalar: bool) -> FloatOrArray:
    """Devuelve float de Python si las entradas fueron escalares."""
    if is_scalar:
        return float(np.asarray(arr).item())
    return np.asarray(arr, dtype=np.float64)


def _as_bool_out(arr: np.ndarray, is_scalar: bool) -> NDArray[np.bool_] | bool:
    """Devuelve bool de Python si las entradas fueron escalares."""
    if is_scalar:
        return bool(np.asarray(arr).item())
    return np.asarray(arr, dtype=bool)


def _validate_positive(name: str, values: np.ndarray, *, allow_zero: bool = False) -> None:
    """Valida positividad física de las variables de entrada."""
    if np.any(np.isnan(values)):
        raise ValueError(f"El parámetro '{name}' no debe contener NaN.")
    if allow_zero:
        if np.any(values < 0.0):
            raise ValueError(f"El parámetro '{name}' debe ser mayor o igual que 0.")
    elif np.any(values <= 0.0):
        raise ValueError(f"El parámetro '{name}' debe ser estrictamente positivo (> 0).")


def _deep_water_length_and_iribarren(
    wave_height: NDArray[np.float64],
    period: NDArray[np.float64],
    slope: NDArray[np.float64],
    gravity: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Calcula L0, peralte H0/L0 y número de Iribarren en aguas profundas."""
    _validate_positive("wave_height", wave_height, allow_zero=True)
    _validate_positive("period", period, allow_zero=False)
    _validate_positive("slope", slope, allow_zero=True)
    if gravity <= 0.0:
        raise ValueError("La aceleración de gravedad 'gravity' debe ser > 0.")

    L0 = gravity * (period**2) / (2.0 * PI_STANDARD)
    steepness = np.where(L0 > 0.0, wave_height / L0, 0.0)
    iribarren = np.where(
        steepness > 0.0,
        slope / np.sqrt(np.maximum(steepness, 1e-30)),
        0.0,
    )
    return L0, steepness, iribarren


def iribarren_number(
    Hs0: ArrayLike,
    T: ArrayLike,
    slope: ArrayLike,
    *,
    gravity: float = GRAVITY,
) -> IribarrenResult:
    """Calcula la longitud de onda profunda ``L0``, el peralte y el número de Iribarren.

    Clasifica además el tipo de rotura según Battjes (1974):
    ``"spilling"`` (``xi0 < 0.5``), ``"plunging"`` (``0.5 <= xi0 < 3.3``) y
    ``"collapsing_surging"`` (``xi0 >= 3.3``).
    """
    h_a, t_a, b_a = np.broadcast_arrays(
        np.asarray(Hs0, dtype=np.float64),
        np.asarray(T, dtype=np.float64),
        np.asarray(slope, dtype=np.float64),
    )
    is_scalar = h_a.ndim == 0
    L0, steepness, xi0 = _deep_water_length_and_iribarren(h_a, t_a, b_a, gravity)

    breaker = np.where(
        xi0 < 0.5,
        "spilling",
        np.where(xi0 < 3.3, "plunging", "collapsing_surging"),
    )
    breaker_out: NDArray[np.str_] | str = (
        str(np.asarray(breaker).item()) if is_scalar else np.asarray(breaker, dtype=str)
    )
    return IribarrenResult(
        L0=_as_out(L0, is_scalar),
        steepness=_as_out(steepness, is_scalar),
        iribarren=_as_out(xi0, is_scalar),
        breaker_type=breaker_out,
    )


def runup_stockdon(
    Hs0: ArrayLike,
    T: ArrayLike,
    slope: ArrayLike,
    *,
    use_dissipative_branch: bool = True,
    gravity: float = GRAVITY,
) -> StockdonRunupResult:
    """Calcula el *setup*, *swash* y ascenso ``R2%`` según Stockdon et al. (2006).

    Referencia: Stockdon, H. F., Holman, R. A., Howd, P. A., & Sallenger Jr., A. H.
    (2006). Empirical parameterization of setup, swash, and runup. *Coastal
    Engineering*, 53(7), 573–588.

    Parámetros
    ----------
    Hs0 : float o array_like
        Altura significativa de ola en aguas profundas [m].
    T : float o array_like
        Período pico de ola [s].
    slope : float o array_like
        Pendiente de la cara de playa (:math:`\\beta_f = \\tan\\beta`) [-].
    use_dissipative_branch : bool, default=True
        Si ``True``, aplica la rama disipativa ``R2% = 0.043 * sqrt(H0 * L0)``
        cuando ``xi0 < 0.3`` (Ec. 18 de Stockdon et al., 2006). Si ``False``,
        evalúa de forma continua la Ec. (19) general.
    gravity : float, default=9.81
        Aceleración de la gravedad [m/s²].
    """
    h_a, t_a, b_a = np.broadcast_arrays(
        np.asarray(Hs0, dtype=np.float64),
        np.asarray(T, dtype=np.float64),
        np.asarray(slope, dtype=np.float64),
    )
    is_scalar = h_a.ndim == 0
    L0, _, xi0 = _deep_water_length_and_iribarren(h_a, t_a, b_a, gravity)

    sqrt_h0_l0 = np.sqrt(np.maximum(h_a * L0, 0.0))
    setup = 0.35 * b_a * sqrt_h0_l0
    swash_inc = 0.75 * b_a * sqrt_h0_l0
    swash_ig = 0.06 * sqrt_h0_l0
    swash_tot = np.sqrt(np.maximum(h_a * L0 * (0.563 * (b_a**2) + 0.004), 0.0))

    r2_general = 1.1 * (setup + 0.5 * swash_tot)
    r2_diss = 0.043 * sqrt_h0_l0

    is_diss = (xi0 < 0.3) & (h_a > 0.0) & bool(use_dissipative_branch)
    r2 = np.where(h_a <= 0.0, 0.0, np.where(is_diss, r2_diss, r2_general))

    return StockdonRunupResult(
        L0=_as_out(L0, is_scalar),
        iribarren=_as_out(xi0, is_scalar),
        setup=_as_out(setup, is_scalar),
        swash_incident=_as_out(swash_inc, is_scalar),
        swash_infragravity=_as_out(swash_ig, is_scalar),
        swash_total=_as_out(swash_tot, is_scalar),
        R2_percent=_as_out(r2, is_scalar),
        is_dissipative=_as_bool_out(is_diss, is_scalar),
    )


def runup_holman(
    Hs0: ArrayLike,
    T: ArrayLike,
    slope: ArrayLike,
    *,
    gravity: float = GRAVITY,
) -> BeachRunupResult:
    """Calcula el ascenso ``R2%`` según Holman (1986).

    Expresión empírica (Duck, NC): :math:`R_{2\\%} / H_{s0} = 0.83\\,\\xi_0 + 0.20`.
    """
    h_a, t_a, b_a = np.broadcast_arrays(
        np.asarray(Hs0, dtype=np.float64),
        np.asarray(T, dtype=np.float64),
        np.asarray(slope, dtype=np.float64),
    )
    is_scalar = h_a.ndim == 0
    L0, _, xi0 = _deep_water_length_and_iribarren(h_a, t_a, b_a, gravity)

    rel = np.where(h_a > 0.0, 0.83 * xi0 + 0.20, 0.0)
    r2 = h_a * rel
    return BeachRunupResult(
        L0=_as_out(L0, is_scalar),
        iribarren=_as_out(xi0, is_scalar),
        relative_runup=_as_out(rel, is_scalar),
        R2_percent=_as_out(r2, is_scalar),
    )


def runup_mase(
    Hs0: ArrayLike,
    T: ArrayLike,
    slope: ArrayLike,
    *,
    gravity: float = GRAVITY,
) -> BeachRunupResult:
    """Calcula el ascenso ``R2%`` para oleaje irregular sobre taludes planos según Mase (1989).

    Expresión: :math:`R_{2\\%} / H_{s0} = 1.86\\,\\xi_0^{0.71}`.
    """
    h_a, t_a, b_a = np.broadcast_arrays(
        np.asarray(Hs0, dtype=np.float64),
        np.asarray(T, dtype=np.float64),
        np.asarray(slope, dtype=np.float64),
    )
    is_scalar = h_a.ndim == 0
    L0, _, xi0 = _deep_water_length_and_iribarren(h_a, t_a, b_a, gravity)

    rel = np.where((h_a > 0.0) & (xi0 > 0.0), 1.86 * np.power(np.maximum(xi0, 0.0), 0.71), 0.0)
    r2 = h_a * rel
    return BeachRunupResult(
        L0=_as_out(L0, is_scalar),
        iribarren=_as_out(xi0, is_scalar),
        relative_runup=_as_out(rel, is_scalar),
        R2_percent=_as_out(r2, is_scalar),
    )


def runup_hunt(
    H0: ArrayLike,
    T: ArrayLike,
    slope: ArrayLike,
    *,
    max_relative_runup: float | None = None,
    gravity: float = GRAVITY,
) -> BeachRunupResult:
    """Calcula el ascenso clásico de rotura en taludes lisos según Hunt (1959).

    Expresión: :math:`R / H_0 = \\xi_0 = \\tan\\beta\\,\\sqrt{L_0 / H_0}`. Si se pasa
    ``max_relative_runup`` (por ejemplo ``2.3``), limita el ascenso relativo máximo.
    """
    h_a, t_a, b_a = np.broadcast_arrays(
        np.asarray(H0, dtype=np.float64),
        np.asarray(T, dtype=np.float64),
        np.asarray(slope, dtype=np.float64),
    )
    is_scalar = h_a.ndim == 0
    L0, _, xi0 = _deep_water_length_and_iribarren(h_a, t_a, b_a, gravity)

    rel = np.where(h_a > 0.0, xi0, 0.0)
    if max_relative_runup is not None:
        if max_relative_runup <= 0.0:
            raise ValueError("'max_relative_runup' debe ser > 0 cuando se especifica.")
        rel = np.minimum(rel, max_relative_runup)
    r2 = h_a * rel
    return BeachRunupResult(
        L0=_as_out(L0, is_scalar),
        iribarren=_as_out(xi0, is_scalar),
        relative_runup=_as_out(rel, is_scalar),
        R2_percent=_as_out(r2, is_scalar),
    )


def runup_nielsen_hanslow(
    Hrms0: ArrayLike,
    T: ArrayLike,
    slope: ArrayLike,
    *,
    use_revised_flat: bool = True,
    z100: ArrayLike = 0.0,
    gravity: float = GRAVITY,
) -> NielsenHanslowResult:
    """Calcula el ascenso ``R2%`` de campo según Nielsen y Hanslow (1991).

    Referencia: Nielsen, P., & Hanslow, D. J. (1991). Wave runup distributions on
    natural beaches. *Journal of Coastal Research*, 7(4), 1139–1152.

    Nota: Recibe la altura cuadrática media profunda ``Hrms0`` (para olas Rayleigh,
    :math:`H_{\\text{rms},0} \\approx H_{s0} / \\sqrt{2}`).
    """
    h_a, t_a, b_a, z_a = np.broadcast_arrays(
        np.asarray(Hrms0, dtype=np.float64),
        np.asarray(T, dtype=np.float64),
        np.asarray(slope, dtype=np.float64),
        np.asarray(z100, dtype=np.float64),
    )
    is_scalar = h_a.ndim == 0
    L0, _, xi0 = _deep_water_length_and_iribarren(h_a, t_a, b_a, gravity)

    flat_coef = 0.06 if use_revised_flat else 0.05
    sqrt_term = np.sqrt(np.maximum(h_a * L0, 0.0))
    is_steep = b_a >= 0.10
    lr = np.where(is_steep, 0.6 * b_a * sqrt_term, flat_coef * sqrt_term)
    r2 = np.where(h_a > 0.0, 1.98 * lr + z_a, z_a)

    return NielsenHanslowResult(
        L0=_as_out(L0, is_scalar),
        iribarren=_as_out(xi0, is_scalar),
        scale_LR=_as_out(lr, is_scalar),
        R2_percent=_as_out(r2, is_scalar),
        is_steep_branch=_as_bool_out(is_steep, is_scalar),
    )


def runup_eurotop(
    Hm0: ArrayLike,
    Tm_10: ArrayLike,
    slope: ArrayLike,
    *,
    gamma_b: ArrayLike = 1.0,
    gamma_f: ArrayLike = 1.0,
    angle_deg: ArrayLike = 0.0,
    gamma_beta: ArrayLike | None = None,
    method: Literal["mean", "design"] = "mean",
    gravity: float = GRAVITY,
) -> EurOtopRunupResult:
    """Calcula el ascenso ``Ru2%`` sobre diques, defensas y escolleras según EurOtop (2018).

    Implementa las ecuaciones (5.1)/(5.2) para ``method="mean"`` (c0=1.65, c1=4.0, c2=1.5)
    y (5.4)/(5.5) para ``method="design"`` (c0=1.75, c1=4.3, c2=1.6) del manual
    EurOtop (2018).

    Parámetros
    ----------
    Hm0 : float o array_like
        Altura espectral significativa al pie de la estructura [m].
    Tm_10 : float o array_like
        Período espectral energético :math:`T_{m-1,0}` [s].
    slope : float o array_like
        Pendiente del paramento :math:`\\tan\\alpha` (p. ej. 1/3 -> 0.333) [-].
    gamma_b : float o array_like, default=1.0
        Factor reductor por berma [-].
    gamma_f : float o array_like, default=1.0
        Factor reductor por rugosidad/permeabilidad (1.0 liso; ~0.4–0.6 escollera) [-].
    angle_deg : float o array_like, default=0.0
        Ángulo de incidencia respecto a la normal a la estructura [grados].
    gamma_beta : float o array_like o None, default=None
        Permite especificar directamente :math:`\\gamma_\\beta`. Si es ``None``, se
        calcula con ``max(1.0 - 0.0022 * abs(angle_deg), 0.824)`` (oleaje de cresta corta).
    method : {"mean", "design"}, default="mean"
        Enfoque estadístico medio o de diseño/verificación estructural.
    """
    if method == "mean":
        c0, c1, c2 = 1.65, 4.0, 1.5
    elif method == "design":
        c0, c1, c2 = 1.75, 4.3, 1.6
    else:
        raise ValueError("El parámetro 'method' debe ser 'mean' o 'design'.")

    if gamma_beta is None:
        ang_arr = np.abs(np.asarray(angle_deg, dtype=np.float64))
        gb_expr = np.maximum(1.0 - 0.0022 * np.minimum(ang_arr, 80.0), 0.824)
    else:
        gb_expr = np.asarray(gamma_beta, dtype=np.float64)

    h_a, t_a, s_a, g_b, g_f, g_beta = np.broadcast_arrays(
        np.asarray(Hm0, dtype=np.float64),
        np.asarray(Tm_10, dtype=np.float64),
        np.asarray(slope, dtype=np.float64),
        np.asarray(gamma_b, dtype=np.float64),
        np.asarray(gamma_f, dtype=np.float64),
        np.asarray(gb_expr, dtype=np.float64),
    )
    is_scalar = h_a.ndim == 0

    _validate_positive("gamma_b", g_b, allow_zero=False)
    _validate_positive("gamma_f", g_f, allow_zero=False)
    _validate_positive("gamma_beta", g_beta, allow_zero=False)

    L0, _, xi = _deep_water_length_and_iribarren(h_a, t_a, s_a, gravity)

    r_brk = h_a * c0 * g_b * g_f * g_beta * xi
    denom_sqrt = np.sqrt(np.maximum(g_b * xi, 1e-30))
    r_cap = np.where(
        (h_a > 0.0) & (xi > 0.0),
        h_a * g_f * g_beta * np.maximum(c1 - c2 / denom_sqrt, 0.0),
        0.0,
    )
    limited = (h_a > 0.0) & (r_cap < r_brk)
    r2 = np.where(h_a > 0.0, np.minimum(r_brk, r_cap), 0.0)

    return EurOtopRunupResult(
        L0=_as_out(L0, is_scalar),
        iribarren=_as_out(xi, is_scalar),
        gamma_beta=_as_out(g_beta, is_scalar),
        R2_breaking=_as_out(r_brk, is_scalar),
        R2_surging_cap=_as_out(r_cap, is_scalar),
        R2_percent=_as_out(r2, is_scalar),
        limited_by_cap=_as_bool_out(limited, is_scalar),
    )


def total_water_level(
    runup_2percent: ArrayLike,
    *,
    swl: ArrayLike = 0.0,
    tide: ArrayLike = 0.0,
    storm_surge: ArrayLike = 0.0,
) -> TotalWaterLevelResult:
    """Calcula la cota total de inundación (*Total Water Level*): ``TWL = SWL + R2%``.

    Si se pasan ``tide`` y/o ``storm_surge``, se suman a ``swl`` para conformar el nivel
    medio instantáneo total en reposo.
    """
    r_a, swl_a, tide_a, surge_a = np.broadcast_arrays(
        np.asarray(runup_2percent, dtype=np.float64),
        np.asarray(swl, dtype=np.float64),
        np.asarray(tide, dtype=np.float64),
        np.asarray(storm_surge, dtype=np.float64),
    )
    is_scalar = r_a.ndim == 0
    total_swl = swl_a + tide_a + surge_a
    twl = total_swl + r_a
    return TotalWaterLevelResult(
        swl=_as_out(total_swl, is_scalar),
        runup_2percent=_as_out(r_a, is_scalar),
        twl=_as_out(twl, is_scalar),
    )


def compare_runup_models(
    Hs0: ArrayLike,
    T: ArrayLike,
    slope: ArrayLike,
    *,
    angle_deg: ArrayLike = 0.0,
    gamma_f: ArrayLike = 1.0,
    swl: ArrayLike = 0.0,
    gravity: float = GRAVITY,
) -> dict[str, FloatOrArray]:
    """Compara rápidamente los 6 modelos de ``R2%`` y devuelve la cota TWL asociada.

    Para ``Nielsen y Hanslow (1991)`` convierte automáticamente a altura cuadrática
    media Rayleigh mediante :math:`H_{\\text{rms},0} = H_{s0} / \\sqrt{2}`.
    """
    st = runup_stockdon(Hs0, T, slope, gravity=gravity)
    ho = runup_holman(Hs0, T, slope, gravity=gravity)
    ma = runup_mase(Hs0, T, slope, gravity=gravity)
    nh = runup_nielsen_hanslow(np.asarray(Hs0, dtype=np.float64) / math.sqrt(2.0), T, slope, gravity=gravity)
    hu = runup_hunt(Hs0, T, slope, gravity=gravity)
    eu = runup_eurotop(Hs0, T, slope, gamma_f=gamma_f, angle_deg=angle_deg, gravity=gravity)

    swl_arr = np.asarray(swl, dtype=np.float64)
    is_scalar = np.asarray(st.R2_percent).ndim == 0 and swl_arr.ndim == 0
    return {
        "L0": st.L0,
        "iribarren": st.iribarren,
        "stockdon_R2": st.R2_percent,
        "holman_R2": ho.R2_percent,
        "mase_R2": ma.R2_percent,
        "nielsen_hanslow_R2": nh.R2_percent,
        "hunt_R": hu.R2_percent,
        "eurotop_R2": eu.R2_percent,
        "stockdon_TWL": _as_out(swl_arr + np.asarray(st.R2_percent), is_scalar),
    }


__all__ = [
    "BeachRunupResult",
    "EurOtopRunupResult",
    "IribarrenResult",
    "NielsenHanslowResult",
    "StockdonRunupResult",
    "TotalWaterLevelResult",
    "compare_runup_models",
    "iribarren_number",
    "runup_eurotop",
    "runup_holman",
    "runup_hunt",
    "runup_mase",
    "runup_nielsen_hanslow",
    "runup_stockdon",
    "total_water_level",
]
