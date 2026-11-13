"""Transporte de sedimentos fluviales e integracion de perfiles medidos.

Implementa las herramientas Excel:
- ``Computation Discharge and surface slope of compound river channels.xls``
  (transporte Engelund-Hansen e inicio de movimiento en fondo y taludes)
- ``Flow and transport capacity in river channel.xls``
- ``Measured Suspended transport.xls``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sedmor.constants import (
    GRAVITY,
    PI_STANDARD,
    RHO_FRESH_WATER,
    RHO_SEDIMENT,
)
from sedmor.rivers import CompoundChannelFlowResult, compound_channel_flow

_SECONDS_PER_YEAR = 365.0 * 86400.0

__all__ = [
    "CompoundChannelTransportResult",
    "MeasuredSuspendedTransportResult",
    "compound_channel_transport",
    "integrate_measured_suspended_transport",
]


@dataclass(frozen=True)
class CompoundChannelTransportResult:
    """Capacidad de flujo, transporte total (Engelund-Hansen) y estabilidad en canal compuesto."""

    flow: CompoundChannelFlowResult
    qs_vol_subsection: NDArray[np.float64]
    qs_mass_subsection: NDArray[np.float64]
    total_qs_vol: float
    total_qs_mass: float
    mean_concentration: float
    annual_solid_volume_m3: float
    critical_velocity: NDArray[np.float64]
    bed_shear_stress: NDArray[np.float64]
    critical_shear_stress: NDArray[np.float64]


@dataclass(frozen=True)
class MeasuredSuspendedTransportResult:
    """Integracion vertical de transporte en suspension a partir de mediciones."""

    z: NDArray[np.float64]
    velocity: NDArray[np.float64]
    concentration: NDArray[np.float64]
    flux_profile: NDArray[np.float64]
    cumulative_transport: NDArray[np.float64]
    total_suspended_transport: float
    depth_averaged_velocity: float
    flux_averaged_concentration: float


def compound_channel_transport(
    water_level: float,
    slope: float,
    bed_levels: Sequence[float] | ArrayLike,
    widths: Sequence[float] | ArrayLike,
    ks_values: Sequence[float] | ArrayLike,
    *,
    d50: float = 0.0003,
    rho_w: float = RHO_FRESH_WATER,
    rho_s: float = RHO_SEDIMENT,
    side_angles_deg: Sequence[float] | ArrayLike | None = None,
    critical_shields: float = 0.04,
    angle_of_repose_deg: float = 35.0,
    gravity: float = GRAVITY,
    strict_excel: bool = False,
) -> CompoundChannelTransportResult:
    """Calcula el caudal y la capacidad de transporte total de arena (Engelund-Hansen) en canal compuesto.

    Combina las hojas de calculo ``Computations Discharge`` y
    ``Flow and transport capacity in river channel``. Incluye ademas la velocidad critica
    de inicio de movimiento y el esfuerzo cortante critico corregido por talud transversal.
    """
    if d50 <= 0.0 or rho_s <= rho_w:
        raise ValueError("d50 debe ser positivo y rho_s > rho_w.")

    flow = compound_channel_flow(
        water_level=water_level,
        slope=slope,
        bed_levels=bed_levels,
        widths=widths,
        ks_values=ks_values,
        strict_excel=strict_excel,
    )

    hy = flow.depth
    by = flow.width
    cy = flow.chezy
    u = flow.velocity
    wet = hy > 0.0

    rel_density = 1.65 if strict_excel else (rho_s / rho_w - 1.0)
    mass_density_factor = 2650.0 if strict_excel else float(rho_s)

    qs_vol = np.zeros_like(hy)
    qs_vol[wet] = (
        0.05
        * by[wet]
        * u[wet] ** 5
        / (rel_density**2 * np.sqrt(gravity) * d50 * cy[wet] ** 3)
    )
    qs_mass = mass_density_factor * qs_vol

    total_qs_vol = float(np.sum(qs_vol))
    total_qs_mass = float(np.sum(qs_mass))
    mean_conc = total_qs_mass / flow.total_discharge if flow.total_discharge > 0.0 else 0.0
    annual_vol = total_qs_mass * _SECONDS_PER_YEAR / rho_s

    u_cr = np.zeros_like(hy)
    coef_u = 0.19 * d50**0.1 if d50 < 0.0005 else 8.5 * d50**0.6
    u_cr[wet] = coef_u * np.log10(12.0 * hy[wet] / (6.0 * d50))

    tau = rho_w * gravity * hy * slope

    if side_angles_deg is None:
        alpha_arr = np.zeros_like(hy)
    else:
        alpha_arr = np.asarray(side_angles_deg, dtype=np.float64)
        if alpha_arr.shape != hy.shape:
            raise ValueError("side_angles_deg debe tener la misma longitud que las subsecciones.")

    two_pi = 6.28 if strict_excel else 2.0 * PI_STANDARD
    alpha_rad = alpha_arr / 360.0 * two_pi
    phi_rad = angle_of_repose_deg / 360.0 * two_pi
    bank_factor = np.cos(alpha_rad) * np.sqrt(
        np.maximum(1.0 - (np.tan(alpha_rad) / np.tan(phi_rad)) ** 2, 0.0)
    )
    tau_cr = critical_shields * (rho_s - rho_w) * gravity * d50 * bank_factor

    return CompoundChannelTransportResult(
        flow=flow,
        qs_vol_subsection=qs_vol,
        qs_mass_subsection=qs_mass,
        total_qs_vol=total_qs_vol,
        total_qs_mass=total_qs_mass,
        mean_concentration=mean_conc,
        annual_solid_volume_m3=annual_vol,
        critical_velocity=u_cr,
        bed_shear_stress=tau,
        critical_shear_stress=tau_cr,
    )


def integrate_measured_suspended_transport(
    z: ArrayLike,
    velocity: ArrayLike,
    concentration: ArrayLike,
    *,
    bottom_sublayer_factor: float = 0.75,
) -> MeasuredSuspendedTransportResult:
    """Integra verticalmente perfiles medidos de velocidad y concentracion en suspension.

    Reproduce la planilla ``Measured Suspended transport.xls`` utilizando la regla
    del trapecio para ``i >= 2`` y el factor ``0.75 * q_s(z_1) * (z_1 - z_0)`` para la
    primera subcapa cercana al fondo no medida directamente.

    Parameters
    ----------
    z : ArrayLike
        Alturas sobre el fondo (m), ordenadas estrictamente de menor a mayor (iniciando en 0).
    velocity : ArrayLike
        Velocidad longitudinal medida ``u(z)`` (m/s).
    concentration : ArrayLike
        Concentracion de sedimento medida ``c(z)`` (kg/m3).
    bottom_sublayer_factor : float, default=0.75
        Factor geometrico para estimar la integral entre ``z[0]`` y ``z[1]``.
    """
    z_arr = np.asarray(z, dtype=np.float64)
    u_arr = np.asarray(velocity, dtype=np.float64)
    c_arr = np.asarray(concentration, dtype=np.float64)

    if z_arr.ndim != 1 or u_arr.shape != z_arr.shape or c_arr.shape != z_arr.shape:
        raise ValueError("z, velocity y concentration deben ser vectores 1D de igual tamano.")
    if len(z_arr) < 2:
        raise ValueError("Se requieren al menos 2 niveles verticales.")
    if np.any(np.diff(z_arr) <= 0.0):
        raise ValueError("Las elevaciones z deben ser estrictamente crecientes.")

    flux = u_arr * c_arr
    dz = np.diff(z_arr)
    layer_q = np.zeros_like(z_arr)
    layer_q[1] = bottom_sublayer_factor * flux[1] * dz[0]
    if len(z_arr) > 2:
        layer_q[2:] = 0.5 * (flux[1:-1] + flux[2:]) * dz[1:]

    cumulative = np.cumsum(layer_q)
    total_qs = float(cumulative[-1])

    depth_span = float(z_arr[-1] - z_arr[0])
    u_mean = float(0.5 * np.sum((u_arr[1:] + u_arr[:-1]) * dz) / depth_span)
    flux_conc = total_qs / (u_mean * depth_span) if u_mean * depth_span > 0.0 else 0.0

    return MeasuredSuspendedTransportResult(
        z=z_arr,
        velocity=u_arr,
        concentration=c_arr,
        flux_profile=flux,
        cumulative_transport=cumulative,
        total_suspended_transport=total_qs,
        depth_averaged_velocity=u_mean,
        flux_averaged_concentration=flux_conc,
    )
