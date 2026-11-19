"""``sedmor``: Librería educativa de hidrodinámica, transporte de sedimentos y morfodinámica.

Implementación científica independiente y autónoma en Python puro basada en la
literatura de ingeniería fluvial, estuarina y costera (incluyendo formulaciones
sistematizadas por L. C. van Rijn y otros autores clásicos), acompañada de
cuadernos de ejemplo en español. No contiene ni redistribuye archivos Excel de terceros.
"""

from __future__ import annotations

from sedmor import (
    armor,
    closure,
    coastal_transport,
    constants,
    currents,
    littoral,
    overtopping,
    propagation,
    river_transport,
    rivers,
    runup,
    scour,
    sedimentation,
    tidal_transport,
    tides,
    turbidity,
    waves,
)
from sedmor.armor import *  # noqa: F401,F403
from sedmor.closure import *  # noqa: F401,F403
from sedmor.coastal_transport import *  # noqa: F401,F403
from sedmor.constants import (  # noqa: F401
    GRAVITY,
    KINEMATIC_VISCOSITY_15C,
    KINEMATIC_VISCOSITY_20C,
    PI_EXCEL,
    PI_STANDARD,
    RHO_FRESH_WATER,
    RHO_SEA_WATER,
    RHO_SEDIMENT,
    TWO_PI_EXCEL,
    VON_KARMAN,
)
from sedmor.currents import *  # noqa: F401,F403
from sedmor.littoral import *  # noqa: F401,F403
from sedmor.overtopping import *  # noqa: F401,F403
from sedmor.propagation import *  # noqa: F401,F403
from sedmor.river_transport import *  # noqa: F401,F403
from sedmor.rivers import *  # noqa: F401,F403
from sedmor.runup import *  # noqa: F401,F403
from sedmor.scour import *  # noqa: F401,F403
from sedmor.sedimentation import *  # noqa: F401,F403
from sedmor.tidal_transport import *  # noqa: F401,F403
from sedmor.tides import *  # noqa: F401,F403
from sedmor.turbidity import *  # noqa: F401,F403
from sedmor.waves import *  # noqa: F401,F403

__version__ = "0.1.0"

_SUBMODULES = [
    "armor",
    "closure",
    "coastal_transport",
    "constants",
    "currents",
    "littoral",
    "overtopping",
    "propagation",
    "river_transport",
    "rivers",
    "runup",
    "scour",
    "sedimentation",
    "tidal_transport",
    "tides",
    "turbidity",
    "waves",
]

_CONSTANTS = [
    "GRAVITY",
    "KINEMATIC_VISCOSITY_15C",
    "KINEMATIC_VISCOSITY_20C",
    "PI_EXCEL",
    "PI_STANDARD",
    "RHO_FRESH_WATER",
    "RHO_SEA_WATER",
    "RHO_SEDIMENT",
    "TWO_PI_EXCEL",
    "VON_KARMAN",
]

__all__ = sorted(
    {
        "__version__",
        *_SUBMODULES,
        *_CONSTANTS,
        *waves.__all__,
        *propagation.__all__,
        *runup.__all__,
        *overtopping.__all__,
        *armor.__all__,
        *closure.__all__,
        *currents.__all__,
        *tides.__all__,
        *rivers.__all__,
        *river_transport.__all__,
        *coastal_transport.__all__,
        *tidal_transport.__all__,
        *littoral.__all__,
        *scour.__all__,
        *turbidity.__all__,
        *sedimentation.__all__,
    }
)
