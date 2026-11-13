# sedmor — Librería de Hidrodinámica, Transporte de Sedimentos y Morfología

`sedmor` es una librería científica en Python, ligera y fácil de instalar, diseñada para cursos e ingeniería de **hidráulica fluvial, estuarina y costera**. Implementa de forma vectorizada, modular y validada las **57 planillas de cálculo Excel** del *MODELLING PACKAGE 2013* del Prof. Leo C. van Rijn (2013), acompañadas de **9 cuadernos Jupyter didácticos en español** listos para ejecutar.

---

## Instalación rápida

Requisitos mínimos: **Python >= 3.9**, `numpy >= 1.24` y `scipy >= 1.10`.

```bash
# Opción A: Instalar directamente desde GitHub
pip install "sedmor[examples] @ git+https://github.com/INH-IDI/sedmor_toolkit.git"

# Opción B: Clonar el repositorio (recomendado para trabajar con los cuadernos)
git clone https://github.com/INH-IDI/sedmor_toolkit.git
cd sedmor_toolkit

# Instalación estándar (librería núcleo)
pip install -e .

# Instalación recomendada para estudiantes (incluye matplotlib y Jupyter)
pip install -e ".[examples]"

# Instalación para desarrollo y ejecución de pruebas (incluye pytest, openpyxl y build)
pip install -e ".[all]"
```

Para instalar directamente sin modo editable:

```bash
pip install .
```

---

## Ejemplo mínimo en Python

```python
import sedmor as sm

# 1. Relación de dispersión lineal de oleaje (h = 10 m, T = 8 s)
onda = sm.waves.dispersion_relation(h=10.0, T=8.0)
print(f"Longitud de onda L = {onda['L']:.2f} m | kh = {onda['kh']:.3f}")

# 2. Transporte de arena bajo corriente + oleaje (van Rijn, 2007)
tr = sm.coastal_transport.sand_transport_current_waves(
    depth=8.0, velocity=0.6, wave_height=2.0, wave_period=7.0, d50=0.00022, d90=0.00044
)
print(f"Transporte fondo qb = {tr.qb:.4f} | suspensión qs = {tr.qs:.4f} | total qt = {tr.qt:.4f} kg/(m·s)")

# 3. Deriva litoral y erosión de dunas por tormenta
lit = sm.littoral.longshore_and_dune_erosion(
    depth_offshore=25.0, d50=0.00025, wave_height_offshore=4.0,
    wave_period=11.0, wave_angle_offshore=25.0, ssl=3.5, dune_height=5.0, duration_hours=12.0
)
print(f"Deriva van Rijn = {lit.qt_van_rijn_kg_s:.1f} kg/s | Erosión de duna = {lit.vr_volume_t:.1f} m³/m")
```

También es posible importar símbolos directamente desde la raíz (`from sedmor import dispersion_relation, channel_sedimentation, turbidity_current_mud`) o desde cada submódulo (`import sedmor.waves`).

---

## Arquitectura de la librería (`src/sedmor/`)

La librería está organizada en **12 submódulos** temáticos:

| Submódulo | Descripción | Funciones y clases principales |
| :--- | :--- | :--- |
| `sedmor.constants` | Constantes físicas y matemáticas | `GRAVITY`, `RHO_FRESH_WATER`, `RHO_SEA_WATER`, `RHO_SEDIMENT`, `VON_KARMAN`, `KINEMATIC_VISCOSITY_20C` |
| `sedmor.waves` | Ondas lineales, capa límite, rotura y asimetría | `dispersion_relation`, `wavelength_hunt`, `wavelength_nielsen`, `jonswap_spectrum`, `orbital_velocity_profile`, `wave_boundary_layer`, `wave_decay_friction`, `battjes_janssen_profile`, `undertow_velocity`, `eulerian_streaming`, `abreu_skewness_asymmetry`, `isobe_horikawa_skewness`, `nielsen_acceleration_transport` |
| `sedmor.rivers` | Hidráulica de canales y curvas de remanso | `trapezoidal_channel_discharge`, `solve_trapezoidal_depth`, `compound_channel_flow`, `compound_channel_backwater`, `free_overfall_brink_depth` |
| `sedmor.currents` | Asimetría mareal, viento y flujo estratificado | `tidal_asymmetry_m2_m4`, `wind_induced_depth_averaged_velocity`, `wind_induced_velocity_profile`, `stratified_density_flow_profile`, `stratified_channel_sedimentation` |
| `sedmor.tides` | Propagación de marea e intrusión salina | `tidal_analytical_no_reflection`, `tidal_compound_channel`, `tidal_reflection_prismatic`, `tidal_reflection_converging`, `tidal_fourpole`, `tidal_wave_nonlinear_lamb`, `tidal_progressive_wave`, `soulsby_coastal_flow`, `salinity_van_rijn_steady`, `tidal_salinity_kuijper`, `salinity_time_dependent` |
| `sedmor.river_transport` | Transporte fluvial y aforos en suspensión | `compound_channel_transport`, `integrate_measured_suspended_transport` |
| `sedmor.coastal_transport` | Transporte costero bajo corriente y oleaje | `sand_transport_current_waves`, `wave_averaged_sand_transport`, `annual_sand_transport` |
| `sedmor.tidal_transport` | Mezclas cohesivas arena–fango en ciclo mareal | `tidal_mud_sand_transport`, `tidal_sand_mud_transport` |
| `sedmor.turbidity` | Corrientes de turbidez submarinas | `turbidity_current_mud`, `turbidity_current_sand` |
| `sedmor.scour` | Socavación en estructuras hidráulicas y marinas | `scour_bridge_pier`, `scour_downstream_structure`, `scour_river_groyne`, `scour_wall_and_rubblemound_toe`, `scour_structure_tip_waves`, `scour_structure_tip_current_waves`, `scour_vertical_pile_waves`, `scour_horizontal_pipe` |
| `sedmor.sedimentation` | Azolvamiento de canales, puertos, zanjas y embalses | `channel_sedimentation`, `harbour_sand_sedimentation`, `harbour_mud_sedimentation`, `streamtube_sedimentation`, `streamtube_tidal_sedimentation`, `reservoir_sedimentation`, `ReservoirSectionInput` |
| `sedmor.littoral` | Deriva litoral anual y erosión de dunas | `longshore_and_dune_erosion`, `annual_longshore_transport`, `dune_erosion_van_rijn`, `dune_erosion_kriebel_dean`, `dune_erosion_hallermeier_rhodes`, `wrap_wave_angle_to_normal` |

---

## Cuadernos de ejemplo (`examples/`)

En la carpeta [`examples/`](examples/) se incluyen **9 cuadernos Jupyter ejecutados** en español con fundamento teórico, simulaciones numéricas y gráficos comparativos:

| Nº | Cuaderno | Submódulos cubiertos |
| :--- | :--- | :--- |
| 01 | [`01_Hidrodinamica_Olas.ipynb`](examples/01_Hidrodinamica_Olas.ipynb) | `sedmor.waves` |
| 02 | [`02_Hidrodinamica_Rios_Mareas_y_Viento.ipynb`](examples/02_Hidrodinamica_Rios_Mareas_y_Viento.ipynb) | `sedmor.rivers`, `sedmor.currents`, `sedmor.tides` |
| 03 | [`03_Transporte_de_Arena_Fluvial_y_Costero.ipynb`](examples/03_Transporte_de_Arena_Fluvial_y_Costero.ipynb) | `sedmor.river_transport`, `sedmor.coastal_transport` |
| 04 | [`04_Transporte_de_Fangos_y_Mezclas.ipynb`](examples/04_Transporte_de_Fangos_y_Mezclas.ipynb) | `sedmor.tidal_transport` |
| 05 | [`05_Corrientes_de_Turbidez.ipynb`](examples/05_Corrientes_de_Turbidez.ipynb) | `sedmor.turbidity` |
| 06 | [`06_Socavacion_en_Estructuras.ipynb`](examples/06_Socavacion_en_Estructuras.ipynb) | `sedmor.scour` |
| 07 | [`07_Sedimentacion_Canales_Fosas_y_Puertos.ipynb`](examples/07_Sedimentacion_Canales_Fosas_y_Puertos.ipynb) | `sedmor.sedimentation` |
| 08 | [`08_Sedimentacion_en_Embalses.ipynb`](examples/08_Sedimentacion_en_Embalses.ipynb) | `sedmor.sedimentation` |
| 09 | [`09_Morfologia_Litoral_y_Costera.ipynb`](examples/09_Morfologia_Litoral_y_Costera.ipynb) | `sedmor.littoral` |

Para abrirlos localmente con JupyterLab o Jupyter Notebook:

```bash
jupyter lab examples/
```

---

## Modo físico estándar vs. reproducción exacta Excel (`strict_excel`)

Todas las funciones relevantes aceptan el argumento opcional **`strict_excel: bool = False`**:

- **`strict_excel=False` (valor por defecto, recomendado para ingeniería y docencia)**: utiliza `math.pi` de doble precisión, signos trigonométricos consistentes en ángulos simétricos y corrige erratas detectadas en celdas específicas de las hojas originales (como constantes redondeadas a `3.14` o factores omitidos).
- **`strict_excel=True`**: reproduce exactamente las fórmulas, constantes redondeadas (p. ej. `3.14`, `6.28`, `3.13` para $\sqrt{g}$) y particularidades de las hojas Excel de referencia, permitiendo validar los resultados frente a los valores cacheados de las 57 planillas con discrepancias relativas típicamente inferiores a $10^{-11}$.

---

## Pruebas automatizadas

La suite de pruebas en [`tests/`](tests/) es completamente autónoma y no requiere los archivos `.xls`/`.xlsx` originales:

```bash
pytest -v
```

---

## Licencia y créditos

- Implementación en Python bajo licencia **MIT** (ver [`LICENSE`](LICENSE)).
- Basado en las formulaciones y planillas de cálculo educativas desarrolladas por **Leo C. van Rijn (2013)** (*Modelling Package 2013*).
