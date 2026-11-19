# AGENTS.md — Guía de desarrollo y arquitectura para `sedmor`

Este repositorio contiene la librería educativa **`sedmor`** (`sedmor-0.1.0`), implementada de forma independiente en Python puro (NumPy y SciPy) a partir de formulaciones científicas abiertas de ingeniería fluvial, estuarina y costera, acompañada de **15 cuadernos Jupyter didácticos en español** en [`examples/`](examples/).

Repositorio remoto: `https://github.com/INH-IDI/sedmor_toolkit` (rama principal: `main`).

---

## 1. Alcance, independencia y reglas estrictas

1. **Independencia total y propiedad intelectual**:
   - `sedmor` es un desarrollo original en código abierto (licencia MIT sobre el código Python, pruebas y cuadernos).
   - **No** incluir, almacenar, empaquetar ni referenciar rutas a archivos `.xls`/`.xlsx` externos o macros propietarias de terceros; tanto la librería como [`tests/`](tests/) deben ser 100% autónomas.
   - Al añadir módulos nuevos (como [`runup.py`](src/sedmor/runup.py), [`overtopping.py`](src/sedmor/overtopping.py), [`armor.py`](src/sedmor/armor.py), [`closure.py`](src/sedmor/closure.py), [`propagation.py`](src/sedmor/propagation.py) o [`sediments.py`](src/sedmor/sediments.py)), basarlos exclusivamente en ecuaciones publicadas en literatura científica abierta y documentar las referencias bibliográficas en el docstring y en [`README.md`](README.md).
2. **Alcance funcional**:
   - Cobertura centrada en formulaciones analíticas, empíricas y semiempíricas 1D/0D en Python puro. No incorporar ejecutables binarios externos, código compilado heredado ni herramientas ajenas al alcance de esta librería.
   - No modificar archivos fuera de `sedmor_toolkit/`.
3. **Convención de idiomas**:
   - Nombres públicos de módulos, funciones, parámetros y `dataclass` en **inglés** (`snake_case` / `PascalCase`).
   - Docstrings, mensajes de error orientados al usuario, [`README.md`](README.md) y cuadernos en **español**.

---

## 2. Convención numérica `strict_excel: bool = False`

Las funciones calibradas frente a formulaciones tabuladas clásicas aceptan el argumento opcional por palabra clave **`strict_excel: bool = False`**:

- **`strict_excel=False` (por defecto)**: aplica física estándar (`math.pi` / `PI_STANDARD` de doble precisión, simetría trigonométrica en ángulos positivos/negativos y correcciones analíticas documentadas).
- **`strict_excel=True`**: aplica constantes decimales históricas (`3.14`, `6.28`, `3.13`, `0.3333`) únicamente para verificación numérica retrospectiva en [`tests/`](tests/), sin leer ni requerir archivos de hoja de cálculo externos.

> **Importante**: no alterar la lógica bajo `strict_excel=True` sin comprobar que las pruebas de regresión en [`tests/`](tests/) sigan pasando dentro de sus tolerancias.

---

## 3. Mapa de arquitectura (`src/sedmor/`)

[`src/sedmor/__init__.py`](src/sedmor/__init__.py) exporta `__version__ = "0.1.0"`, los **18 submódulos** y **245 símbolos públicos** en `__all__`:

| Archivo | Rol y contenido principal |
| :--- | :--- |
| [`constants.py`](src/sedmor/constants.py) | Constantes físicas (`GRAVITY`, `RHO_FRESH_WATER`, `RHO_SEA_WATER`, `RHO_SEDIMENT`, `VON_KARMAN`, viscosidades, `PI_EXCEL`, `PI_STANDARD`) |
| [`sediments.py`](src/sedmor/sediments.py) | Propiedades agua–sedimento, velocidad de caída (`fall_velocity_stokes`, `fall_velocity_rubey`, `fall_velocity_van_rijn`, `fall_velocity_soulsby`, `fall_velocity_ferguson_church`, `flocculated_settling_velocity`) y análisis granulométrico (`grain_size_curve_statistics`, `blend_grain_size_curves`, `check_filter_compatibility`) |
| [`waves.py`](src/sedmor/waves.py) | Relación de dispersión (`dispersion_relation`, Hunt, Nielsen), espectro JONSWAP, capa límite oscilatoria, decaimiento por fricción, Battjes & Janssen, Abreu, Isobe-Horikawa, Nielsen, *undertow* y *streaming* |
| [`propagation.py`](src/sedmor/propagation.py) | Propagación, refracción, difracción y dispersión con corriente (`dispersion_with_current`, `shoaling_refraction_snell`, `diffraction_semi_infinite_breakwater`, `diffraction_breakwater_gap`, `diffraction_random_directional`, `propagate_wave_profile`) |
| [`runup.py`](src/sedmor/runup.py) | Número de Iribarren y modelos de *run-up* (`runup_stockdon`, `runup_holman`, `runup_mase`, `runup_hunt`, `runup_nielsen_hanslow`, `runup_eurotop`), `total_water_level` (*TWL*) y `compare_runup_models` |
| [`overtopping.py`](src/sedmor/overtopping.py) | Rebase de oleaje EurOtop (2018) en taludes, escolleras y muros verticales (`overtopping_slope`, `overtopping_rubble_mound`, `overtopping_vertical_wall`, `individual_overtopping_volumes`, `solve_required_freeboard`, `classify_overtopping_hazard`) |
| [`armor.py`](src/sedmor/armor.py) | Dimensionamiento de roca de coraza y filtros (`hudson_armor_weight`, `van_der_meer_rock_armor`, `van_gent_shallow_rock_armor`, `isbash_stone_stability`, `rock_size_properties`, `classify_rock_grading`, `filter_and_underlayer_design`, `compare_armor_methods`) |
| [`closure.py`](src/sedmor/closure.py) | Profundidad de cierre y derivados (`depth_of_closure_hallermeier`, `depth_of_closure_birkemeier`, `outer_closure_depth_hallermeier`, `depth_of_closure_annual_stats`, `closure_depth_from_wave_series`, `equilibrium_profile_dean`, `bruun_rule_recession`, `beach_nourishment_volume`, `beach_nourishment_advance`) |
| [`rivers.py`](src/sedmor/rivers.py) | Canales trapezoidales, calado normal, flujo en sección compuesta, curva de remanso y resalto libre |
| [`currents.py`](src/sedmor/currents.py) | Asimetría mareal M2+M4, corrientes inducidas por viento y perfil estratificado por gradiente de densidad (`stratified_density_flow_profile`, `stratified_channel_sedimentation`) |
| [`tides.py`](src/sedmor/tides.py) | Propagación analítica y compuesta de mareas, reflexión prismática/convergente, cuatripolo, onda no lineal de Lamb, flujo costero de Soulsby e intrusión salina (van Rijn, Kuijper, transitoria) |
| [`river_transport.py`](src/sedmor/river_transport.py) | Transporte sólido en sección compuesta e integración de aforos en suspensión |
| [`coastal_transport.py`](src/sedmor/coastal_transport.py) | Transporte de arena bajo corriente + oleaje, promediado por clima de olas y balance anual |
| [`tidal_transport.py`](src/sedmor/tidal_transport.py) | Simuladores mareales de mezclas cohesivas (`tidal_mud_sand_transport`, `tidal_sand_mud_transport`) |
| [`turbidity.py`](src/sedmor/turbidity.py) | Corrientes de turbidez 1D cohesivas (`turbidity_current_mud`) y arenosas (`turbidity_current_sand`) |
| [`scour.py`](src/sedmor/scour.py) | 8 métodos de socavación local (pilares de puente, aguas abajo, espigones fluviales, pie de muros/escolleras, morros, pilotes y tuberías submarinas) |
| [`sedimentation.py`](src/sedmor/sedimentation.py) | Sedimentación en canales/fosas, dársenas de arena y fango, tubos de corriente 1D estacionarios/mareales y embalses multi-sección (`reservoir_sedimentation`, `ReservoirSectionInput`) |
| [`littoral.py`](src/sedmor/littoral.py) | Deriva litoral (Kamphuis, CERC, van Rijn), transporte anual direccional y erosión de dunas (van Rijn, Kriebel & Dean, Hallermeier & Rhodes) |

Si se añaden símbolos públicos nuevos en cualquier submódulo:
1. Exportarlos en `__all__` del submódulo y en [`src/sedmor/__init__.py`](src/sedmor/__init__.py).
2. Agregar pruebas unitarias dedicadas en [`tests/`](tests/).
3. Documentarlos junto con sus citas bibliográficas en [`README.md`](README.md) y en un cuaderno en [`examples/`](examples/).

---

## 4. Convenciones de código y estilo

- Usar `from __future__ import annotations`, anotaciones de tipos explícitas y `@dataclass(frozen=True)` o `@dataclass` para resultados estructurados de múltiples campos.
- Vectorizar entradas mediante `np.broadcast_arrays` y devolver escalar Python cuando todas las entradas son escalares.
- Reutilizar los auxiliares existentes (`constants.py`, helpers internos de dispersión en `waves.py`, perfiles estratificados de `currents.py` reexportados en `sedimentation.py`) antes de duplicar fórmulas.
- Conservar todos los docstrings y comentarios no relacionados con los cambios realizados.

---

## 5. Pruebas, cuadernos y empaquetado

### Suite de pruebas (`pytest`)

Ejecutar siempre en un entorno limpio (sin instalaciones heredadas previas en `site-packages`) o anteponiendo `PYTHONPATH=src`:

```bash
PYTHONPATH=src python3 -m pytest -v
```

La suite actual comprende **96 pruebas** repartidas en 14 archivos en [`tests/`](tests/) (`test_armor.py`, `test_closure.py`, `test_currents_tides.py`, `test_littoral.py`, `test_overtopping.py`, `test_propagation.py`, `test_rivers_transport.py`, `test_runup.py`, `test_scour.py`, `test_sedimentation.py`, `test_sediments.py`, `test_tidal_transport.py`, `test_turbidity.py`, `test_waves.py`).

### Cuadernos de ejemplo (`examples/01..15_*.ipynb`)

Si se modifican APIs públicas o cuadernos, re-ejecutarlos sin definir `MPLBACKEND` (para incrustar las figuras PNG generadas con `matplotlib`) y usando `PYTHONPATH` absoluto hacia `src/` (dado que `nbconvert` ejecuta con cwd en `examples/`):

```bash
env -u MPLBACKEND PYTHONPATH=/home/model2/Documents/Hulp/sedmor_toolkit/src python3 -m jupyter nbconvert \
  --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=python3 examples/*.ipynb
```

### Empaquetado y control de Git

- Verificar construcción del *wheel* evitando cualquier carpeta residual `build/`:
  ```bash
  python3 -m pip wheel --no-deps --no-build-isolation -w /tmp/sedmor_dist .
  ```
- Mantener sin seguimiento en Git los directorios transitorios (`__pycache__/`, `.pytest_cache/`, `build/`, `dist/`, `*.egg-info/`), ya cubiertos por [`.gitignore`](.gitignore).
