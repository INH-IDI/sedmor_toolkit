"""Constantes fisicas y parametros por defecto usados en la libreria ``sedmor``.

Cuando ``strict_excel=True`` algunas herramientas de referencia de Leo C. van Rijn
utilizan aproximaciones numericas heredadas de las hojas Excel originales (por ejemplo
``pi = 3.14`` o ``g = 10.0`` en hojas especificas); cuando ``strict_excel=False`` (valor
por defecto) se emplean valores fisicos de mayor precision o las correcciones
documentadas correspondientes.
"""

from __future__ import annotations

import math

GRAVITY: float = 9.81
"""Aceleracion de la gravedad estandar [m/s^2]."""

RHO_FRESH_WATER: float = 1000.0
"""Densidad del agua dulce [kg/m^3]."""

RHO_SEA_WATER: float = 1025.0
"""Densidad tipica del agua de mar [kg/m^3]."""

RHO_SEDIMENT: float = 2650.0
"""Densidad mineral del cuarzo / sedimento no cohesivo [kg/m^3]."""

KINEMATIC_VISCOSITY_20C: float = 1.0e-6
"""Viscosidad cinematica del agua a ~20 C [m^2/s]."""

KINEMATIC_VISCOSITY_15C: float = 1.15e-6
"""Viscosidad cinematica del agua a ~15 C usada en varios calculos costeros [m^2/s]."""

VON_KARMAN: float = 0.4
"""Constante de von Karman [-]."""

PI_EXCEL: float = 3.14
"""Valor redondeado de pi empleado frecuentemente en las formulas Excel originales."""

TWO_PI_EXCEL: float = 6.28
"""Valor redondeado de 2*pi empleado en las hojas Excel originales."""

PI_STANDARD: float = math.pi
"""Valor de pi en doble precision."""
