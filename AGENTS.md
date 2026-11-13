# AGENTS.md — Guía de desarrollo y arquitectura para `sedmor`

Este repositorio contiene la librería educativa **`sedmor`** (`sedmor-0.1.0`), implementada en Python puro con NumPy y SciPy a partir de las **57 planillas de cálculo Excel** analíticas y semiempíricas del *MODELLING PACKAGE 2013* (Leo C. van Rijn, 2013), junto con **9 cuadernos Jupyter didácticos en español** en [`examples/`](examples/).

Repositorio remoto: `https://github.com/INH-IDI/sedmor_toolkit` (rama principal: `main`).

---

## 1. Alcance y reglas estrictas

1. **Solo modelos basados en las 57 planillas Excel**:
   - No añadir ejecutables externos, código compilado heredado ni modelos numéricos fuera de las formulaciones implementadas en [`src/sedmor/`](src/sedmor/).
   - Mantener los docstrings, pruebas unitarias, cuadernos y documentación libre de referencias a herramientas o modelos externos excluidos del alcance de esta librería.
2. **Autonomía total de código y pruebas**:
   - Las funciones y la suite en [`tests/`](tests/) deben ejecutarse sin acceso a archivos `.xls`/`.xlsx` externos ni modificaciones sobre carpetas fuera de `sedmor_toolkit/`.
3. **Convención de idiomas**:
   - Nombres públicos de módulos, funciones, parámetros y `dataclass` en **inglés** (`snake_case` / `PascalCase`).
   - Docstrings, mensajes de error orientados al usuario, `README.md` y cuadernos en **español**.

---

## 2. Convención `strict_excel: bool = False`

Todas las funciones calibradas frente a hojas de cálculo aceptan el argumento opcional por palabra clave **`strict_excel: bool = False`**:

- **`strict_excel=False` (por defecto)**: aplica física estándar (uso de `math.pi` / `PI_STANDARD` de doble precisión, simetría trigonométrica en ángulos positivos/negativos y correcciones documentadas a erratas puntuales detectadas en celdas Excel).
- **`strict_excel=True`**: reproduce fielmente las constantes redondeadas (`3.14`, `6.28`, `3.13`, `0.3333`) y peculiaridades originales de las hojas de cálculo para validar paridad exacta frente a los valores cacheados en `tests/`.

> **Importante**: no alterar la lógica bajo `strict_excel=True` sin comprobar que las 62 pruebas de regresión en [`tests/`](tests/) sigan pasando dentro de sus tolerancias.

---

## 3. Mapa de arquitectura (`src/sedmor/`)

[`src/sedmor/__init__.py`](src/sedmor/__init__.py) exporta `__version__ = "0.1.0"`, los **12 submódulos** y **146 símbolos públicos** en `__all__`:

| Archivo | Rol y contenido principal |
| :--- | :--- |
| [`constants.py`](src/sedmor/constants.py) | Constantes físicas (`GRAVITY`, `RHO_FRESH_WATER`, `RHO_SEA_WATER`, `RHO_SEDIMENT`, `VON_KARMAN`, viscosidades, `PI_EXCEL`, `PI_STANDARD`) |
| [`waves.py`](src/sedmor/waves.py) | Relación de dispersión (`dispersion_relation`, Hunt, Nielsen), espectro JONSWAP, capa límite oscilatoria, decaimiento por fricción, Battjes & Janssen, Abreu, Isobe-Horikawa, Nielsen, *undertow* y *streaming* |
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

Si se añaden símbolos públicos nuevos en cualquier submódulo, actualizar también su exportación en [`src/sedmor/__init__.py`](src/sedmor/__init__.py) y documentarlos en [`README.md`](README.md).

---

## 4. Convenciones de código y estilo

- Usar `from __future__ import annotations`, anotaciones de tipos explícitas y `@dataclass` para resultados estructurados de múltiples campos.
- Vectorizar entradas mediante `np.broadcast_arrays` y respetar la devolución escalar frente a arreglos según el patrón existente en cada módulo.
- Reutilizar los auxiliares existentes (`constants.py`, helpers internos de dispersión en `waves.py`, perfiles estratificados de `currents.py` reexportados en `sedimentation.py`) antes de duplicar fórmulas.
- Conservar todos los docstrings y comentarios no relacionados con los cambios realizados.

---

## 5. Pruebas, cuadernos y empaquetado

### Suite de pruebas (`pytest`)

Ejecutar siempre en un entorno limpio (sin instalaciones heredadas previas en `site-packages`) o anteponiendo `PYTHONPATH=src`:

```bash
PYTHONPATH=src python3 -m pytest -v
```

La suite actual comprende **62 pruebas** repartidas en 8 archivos en [`tests/`](tests/) (`test_currents_tides.py`, `test_littoral.py`, `test_rivers_transport.py`, `test_scour.py`, `test_sedimentation.py`, `test_tidal_transport.py`, `test_turbidity.py`, `test_waves.py`).

### Cuadernos de ejemplo (`examples/01..09_*.ipynb`)

Si se modifican APIs públicas o cuadernos, re-ejecutarlos sin definir `MPLBACKEND` (para incrustar las figuras PNG generadas con `matplotlib`) y comprobar 0 errores de celda:

```bash
env -u MPLBACKEND PYTHONPATH=src python3 -m jupyter nbconvert \
  --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=python3 examples/*.ipynb
```

### Empaquetado y control de Git

- Verificar construcción del *wheel* desde fuera de `sedmor_toolkit/` o evitando cualquier carpeta residual `build/`:
  ```bash
  python3 -m pip wheel --no-deps --no-build-isolation -w /tmp/sedmor_dist .
  ```
- Mantener sin seguimiento en Git los directorios transitorios (`__pycache__/`, `.pytest_cache/`, `build/`, `dist/`, `*.egg-info/`), ya cubiertos por [`.gitignore`](.gitignore).
