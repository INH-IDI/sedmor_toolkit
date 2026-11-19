# sedmor — Librería de Hidrodinámica, Transporte de Sedimentos, Obras Costeras y Morfología

`sedmor` es una librería científica abierta en Python, ligera y fácil de instalar, diseñada para docencia e ingeniería en **hidráulica fluvial, estuarina y costera**. Implementa de forma vectorizada, modular y validada formulaciones clásicas y modernas publicadas en la literatura técnica de mecánica de fluidos, sedimentos y obras marítimas (en **17 submódulos**), acompañadas de **14 cuadernos Jupyter didácticos en español** listos para ejecutar.

> **Aviso importante de independencia**: `sedmor` es una implementación original e independiente escrita íntegramente en Python. **Este repositorio NO contiene, publica, empaqueta ni redistribuye archivos Excel (`.xls`/`.xlsx`), macros ni materiales propietarios de terceros**, y su funcionamiento es 100% autónomo.

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

# Instalación para desarrollo y ejecución de pruebas (incluye pytest y build)
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

# 1. Relación de dispersión lineal y propagación con corriente (h = 10 m, T = 8 s, U = -1.2 m/s)
onda = sm.waves.dispersion_relation(h=10.0, T=8.0)
prop = sm.propagation.shoaling_refraction_snell(H0=2.5, T=8.0, theta0_deg=25.0, h=10.0, U=1.2, current_dir_deg=180.0)
print(f"L (U=0) = {onda['L']:.2f} m | L (U=-1.2) = {prop.L:.2f} m | H = {prop.H:.2f} m | θ = {prop.theta_deg:.1f}°")

# 2. Ascenso (run-up R2%), rebase (overtopping) y coraza de roca
ru = sm.runup.runup_stockdon(Hs0=3.5, T=11.0, slope=0.06)
twl = sm.runup.total_water_level(ru.R2_percent, tide=1.1, storm_surge=0.6)
reb = sm.overtopping.overtopping_rubble_mound(Hm0=3.5, Tm_10=10.0, slope=0.50, Rc=4.0, method="design")
roca = sm.armor.van_der_meer_rock_armor(Hs=3.5, Tm=9.0, slope=0.50, P=0.4, S=2.0, approach="design")
print(f"R2% = {ru.R2_percent:.2f} m | TWL = {twl.twl:.2f} m | q = {reb.q_l_s_m:.2f} L/(s·m) | Roca M50 = {roca.mass_tonnes:.2f} t")

# 3. Transporte de arena bajo corriente + oleaje (van Rijn, 2007)
tr = sm.coastal_transport.sand_transport_current_waves(
    depth=8.0, velocity=0.6, wave_height=2.0, wave_period=7.0, d50=0.00022, d90=0.00044
)
print(f"Transporte fondo qb = {tr.qb:.4f} | suspensión qs = {tr.qs:.4f} | total qt = {tr.qt:.4f} kg/(m·s)")

# 4. Deriva litoral y profundidad de cierre / Regla de Bruun
lit = sm.littoral.longshore_and_dune_erosion(
    depth_offshore=25.0, d50=0.00025, wave_height_offshore=4.0,
    wave_period=11.0, wave_angle_offshore=25.0, ssl=3.5, dune_height=5.0, duration_hours=12.0
)
dc = sm.closure.depth_of_closure_birkemeier(He=4.2, Te=11.0)
br = sm.closure.bruun_rule_recession(sea_level_rise=0.5, closure_depth=dc, berm_height=2.0, d50=0.00025)
print(f"Deriva van Rijn = {lit.qt_van_rijn_kg_s:.1f} kg/s | dc (Birkemeier) = {dc:.2f} m | Bruun R = {br.recession_m:.1f} m")
```

También es posible importar símbolos directamente desde la raíz (`from sedmor import dispersion_relation, dispersion_with_current, runup_stockdon, overtopping_rubble_mound, hudson_armor_weight, depth_of_closure_birkemeier`) o desde cada submódulo (`import sedmor.propagation, sedmor.runup, sedmor.overtopping, sedmor.armor, sedmor.closure`).

---

## Arquitectura de la librería (`src/sedmor/`)

La librería está organizada en **17 submódulos** temáticos:

| Submódulo | Descripción | Funciones y clases principales |
| :--- | :--- | :--- |
| `sedmor.constants` | Constantes físicas y matemáticas | `GRAVITY`, `RHO_FRESH_WATER`, `RHO_SEA_WATER`, `RHO_SEDIMENT`, `VON_KARMAN`, `KINEMATIC_VISCOSITY_20C` |
| `sedmor.waves` | Ondas lineales, capa límite, rotura y asimetría | `dispersion_relation`, `wavelength_hunt`, `wavelength_nielsen`, `jonswap_spectrum`, `orbital_velocity_profile`, `wave_boundary_layer`, `wave_decay_friction`, `battjes_janssen_profile`, `undertow_velocity`, `eulerian_streaming`, `abreu_skewness_asymmetry`, `isobe_horikawa_skewness`, `nielsen_acceleration_transport` |
| `sedmor.propagation` | Propagación, refracción, difracción y corrientes | `dispersion_with_current`, `shoaling_refraction_snell`, `diffraction_semi_infinite_breakwater`, `diffraction_breakwater_gap`, `diffraction_random_directional`, `propagate_wave_profile` |
| `sedmor.runup` | Ascenso de oleaje (*run-up*) y nivel total (*TWL*) | `iribarren_number`, `runup_stockdon`, `runup_holman`, `runup_mase`, `runup_hunt`, `runup_nielsen_hanslow`, `runup_eurotop`, `total_water_level`, `compare_runup_models` |
| `sedmor.overtopping` | Rebase de oleaje (*overtopping*, EurOtop 2018) | `overtopping_slope`, `overtopping_rubble_mound`, `overtopping_vertical_wall`, `individual_overtopping_volumes`, `solve_required_freeboard`, `oblique_factor_overtopping`, `classify_overtopping_hazard` |
| `sedmor.armor` | Peso de roca, estabilidad de escolleras y filtros | `hudson_armor_weight`, `van_der_meer_rock_armor`, `van_gent_shallow_rock_armor`, `isbash_stone_stability`, `rock_size_properties`, `classify_rock_grading`, `filter_and_underlayer_design`, `compare_armor_methods` |
| `sedmor.closure` | Profundidad de cierre, equilibrio y derivados | `depth_of_closure_hallermeier`, `depth_of_closure_birkemeier`, `outer_closure_depth_hallermeier`, `depth_of_closure_annual_stats`, `closure_depth_from_wave_series`, `equilibrium_profile_dean`, `bruun_rule_recession`, `beach_nourishment_volume`, `beach_nourishment_advance`, `structure_active_depth_ratio` |
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

En la carpeta [`examples/`](examples/) se incluyen **14 cuadernos Jupyter ejecutados** en español con fundamento teórico, simulaciones numéricas y gráficos comparativos:

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
| 10 | [`10_Runup_y_Cota_Total_Inundacion.ipynb`](examples/10_Runup_y_Cota_Total_Inundacion.ipynb) | `sedmor.runup` |
| 11 | [`11_Rebase_de_Oleaje_EurOtop.ipynb`](examples/11_Rebase_de_Oleaje_EurOtop.ipynb) | `sedmor.overtopping`, `sedmor.runup` |
| 12 | [`12_Peso_de_Roca_y_Estabilidad_de_Escolleras.ipynb`](examples/12_Peso_de_Roca_y_Estabilidad_de_Escolleras.ipynb) | `sedmor.armor` |
| 13 | [`13_Profundidad_de_Cierre_y_Perfil_de_Equilibrio.ipynb`](examples/13_Profundidad_de_Cierre_y_Perfil_de_Equilibrio.ipynb) | `sedmor.closure` |
| 14 | [`14_Propagacion_Refraccion_Difraccion_y_Corrientes.ipynb`](examples/14_Propagacion_Refraccion_Difraccion_y_Corrientes.ipynb) | `sedmor.propagation` |

Para abrirlos localmente con JupyterLab o Jupyter Notebook:

```bash
jupyter lab examples/
```

---

## Convención numérica `strict_excel`

Algunas funciones heredan el parámetro opcional **`strict_excel: bool = False`**:

- **`strict_excel=False` (valor por defecto, recomendado)**: utiliza `math.pi` de doble precisión, signos trigonométricos consistentes en ángulos simétricos y correcciones analíticas documentadas.
- **`strict_excel=True`**: aplica constantes decimales históricas (p. ej. `3.14`, `6.28`, `3.13` para $\sqrt{g}$) exclusivamente para verificación numérica retrospectiva frente a cálculos tabulados clásicos, sin leer ni requerir ningún archivo externo.

---

## Pruebas automatizadas

La suite de pruebas en [`tests/`](tests/) es 100% autónoma:

```bash
pytest -v
```

---

## Independencia de planillas externas, licencia y propiedad intelectual

- **Software independiente**: `sedmor` es un desarrollo original en código abierto bajo licencia **MIT** (ver [`LICENSE`](LICENSE)). La licencia cubre exclusivamente el código Python, las pruebas y los cuadernos didácticos de este repositorio.
- **No publicación de archivos Excel**: este proyecto **no incluye, distribuye, enlaza ni depende** de archivos `.xls`/`.xlsx`, hojas de cálculo originales ni macros propietarias de terceros. Todas las rutinas se basan exclusivamente en ecuaciones matemáticas y modelos empíricos públicos publicados en la literatura científica y libros de texto de ingeniería hidráulica y costera.
- **Sin afiliación ni respaldo**: la cita académica de autores e investigadores originales reconoce su autoría intelectual sobre las ecuaciones científicas publicadas y no implica afiliación, aprobación ni patrocinio oficial respecto a esta implementación en Python.

---

## Referencias bibliográficas y cómo citar

### Cómo citar esta librería

Si utilizas `sedmor` en docencia, informes técnicos o publicaciones, se sugiere citar tanto el software como a los autores originales de las formulaciones científicas empleadas:

```bibtex
@software{sedmor_inh_2026,
  author       = {{Instituto Nacional de Hidráulica (INH-IDI) y contribuidores}},
  title        = {sedmor: Librería educativa en Python para hidrodinámica, transporte de sedimentos, obras costeras y morfología},
  year         = {2026},
  version      = {0.1.0},
  url          = {https://github.com/INH-IDI/sedmor_toolkit},
  license      = {MIT}
}
```

### Fuentes científicas principales implementadas

* **Obras generales de hidráulica, transporte y morfología**:
  * van Rijn, L. C. (1993). *Principles of Sediment Transport in Rivers, Estuaries and Coastal Seas*. Aqua Publications.
  * van Rijn, L. C. (2005 / 2013). *Principles of Sedimentation and Erosion Engineering in Rivers, Estuaries and Coastal Seas* (y compendios académicos asociados en [leovanrijn-sediment.com](https://www.leovanrijn-sediment.com)).
  * van Rijn, L. C. (2007). *Unified View of Sediment Transport by Currents and Waves. I–IV*. Journal of Hydraulic Engineering, ASCE, 133(6–7).
  * Soulsby, R. L. (1997). *Dynamics of Marine Sands: A Manual for Practical Applications*. Thomas Telford.
* **Ondas, capa límite y asimetría**:
  * Battjes, J. A., & Janssen, J. P. F. M. (1978). *Energy loss and set-up due to breaking of random waves*. Proc. 16th Int. Conf. Coastal Engineering (ICCE), ASCE, 569–587.
  * Hunt, J. N. (1979). *Direct solution of wave dispersion equation*. J. Waterway, Port, Coastal and Ocean Div., ASCE, 105(4), 457–459.
  * Isobe, M., & Horikawa, K. (1982). *Study on water particle velocities of shoaling and breaking waves*. Coastal Engineering in Japan, 25, 109–123.
  * Nielsen, P. (1992). *Coastal Bottom Boundary Layers and Sediment Transport*. World Scientific.
  * Nielsen, P. (2006). *Sheet flow sediment transport under waves with acceleration skewness and boundary layer streaming*. Coastal Engineering, 53(9), 749–758.
  * Abreu, T., Silva, P. A., Sancho, F., & Temperville, A. (2010). *Analytical approximate wave form for asymmetric waves*. Coastal Engineering, 57(7), 656–667.
* **Propagación, refracción, difracción e interacción ola–corriente (`sedmor.propagation`)**:
  * Sommerfeld, A. (1896). *Mathematische Theorie der Diffraction*. Mathematische Annalen, 47, 317–374.
  * Penney, W. G., & Price, A. T. (1952). *The diffraction theory of sea waves and the shelter afforded by breakwaters*. Phil. Trans. Royal Society London A, 244(882), 236–253.
  * Goda, Y., Takayama, T., & Suzuki, Y. (1978). *Diffraction diagrams for directional random waves*. Proc. 16th Int. Conf. Coastal Engineering (ICCE), ASCE, 628–650; y Goda, Y. (2000). *Random Seas and Design of Maritime Structures* (2ª ed.). World Scientific.
  * Bretherton, F. P., & Garrett, C. J. R. (1968). *Wavetrains in inhomogeneous moving media*. Proc. Royal Society London A, 302, 529–554.
  * Peregrine, D. H. (1976). *Interaction of water waves and currents*. Advances in Applied Mechanics, 16, 9–117; y Jonsson, I. G. (1990). *Wave–current interactions*. En *The Sea*, Vol. 9, Wiley.
* **Ascenso (*run-up*) y rebase (*overtopping*) de oleaje (`sedmor.runup`, `sedmor.overtopping`)**:
  * Hunt, I. A. (1959). *Design of seawalls and breakwaters*. J. Waterways and Harbors Div., ASCE, 85(3), 123–152.
  * Battjes, J. A. (1974). *Surf similarity*. Proc. 14th Int. Conf. Coastal Engineering (ICCE), ASCE, 466–480.
  * Holman, R. A. (1986). *Extreme value statistics for wave run-up on a natural beach*. Coastal Engineering, 9(6), 527–544.
  * Mase, H. (1989). *Random wave runup height on gentle slope*. J. Waterway, Port, Coastal and Ocean Engineering, ASCE, 115(5), 649–661.
  * Nielsen, P., & Hanslow, D. J. (1991). *Wave runup distributions on natural beaches*. Journal of Coastal Research, 7(4), 1139–1152.
  * Stockdon, H. F., Holman, R. A., Howd, P. A., & Sallenger Jr., A. H. (2006). *Empirical parameterization of setup, swash, and runup*. Coastal Engineering, 53(7), 573–588.
  * van der Meer, J. W., Allsop, N. W. H., Bruce, T., De Rouck, J., Kortenhaus, A., Pullen, T., Schüttrumpf, H., Troch, P., & Zanuttigh, B. (2018). *EurOtop — Manual on wave overtopping of sea defences and related structures* (2ª ed.).
* **Escolleras, peso de roca y diseño de filtros (`sedmor.armor`)**:
  * Hudson, R. Y. (1959 / 1974). *Laboratory investigation of rubble-mound breakwaters* y USACE *Shore Protection Manual* (1984) / *Coastal Engineering Manual* (CEM, 2006).
  * van der Meer, J. W. (1988). *Rock slopes and gravel beaches under wave attack* (Tesis doctoral, TU Delft / Delft Hydraulics Publication 396).
  * van Gent, M. R. A., Smale, A. J., & Kuiper, C. (2003 / 2004). *Stability of rock slopes with shallow foreshores*. Proc. Coastal Structures 2003, ASCE, 100–112.
  * CIRIA, CUR & CETMEF (2007). *The Rock Manual: The use of rock in hydraulic engineering* (2ª ed., C683).
  * Isbash, S. V. (1936). *Construction of dams by depositing rock in running water*. Proc. 2nd Congress on Large Dams, Washington D.C.
* **Profundidad de cierre, perfiles de equilibrio y regeneración (`sedmor.closure`)**:
  * Hallermeier, R. J. (1978 / 1981). *A profile zonation for seasonal sand beaches from wave climate*. Coastal Engineering, 4, 253–277.
  * Birkemeier, W. A. (1985). *Field data on seaward limit of profile change*. J. Waterway, Port, Coastal and Ocean Engineering, ASCE, 111(3), 598–602.
  * Houston, J. R. (1996). *Simplified engineering graphs for depth of closure*. Coastal Engineering Technical Note CETN-II-34, USACE.
  * Nicholls, R. J., Birkemeier, W. A., & Lee, G. (1998). *Evaluation of depth of closure using data from Duck, NC, USA*. Marine Geology, 148, 179–201.
  * Bruun, P. (1962). *Sea-level rise as a cause of shore erosion*. J. Waterways and Harbors Div., ASCE, 88(1), 117–130.
  * Rosati, J. D., Dean, R. G., & Walton, T. L. (2013). *The modified Bruun Rule extended for landward transport*. Marine Geology, 340, 71–81.
  * Dean, R. G. (1987 / 1991). *Equilibrium beach profiles: characteristics and applications*. Journal of Coastal Research, 7(1), 53–84; y *Beach Nourishment: Theory and Practice* (World Scientific, 2002).
* **Mareas, intrusión salina, socavación, sedimentación y morfología litoral**:
  * Kuijper, K., & van Rijn, L. C. (2011). *Analytical and numerical analysis of tides and salinities in estuaries; part II: salinity distributions in prismatic and convergent tidal channels*. Ocean Dynamics, 61, 1743–1765.
  * Eysink, W. D., & Vermaas, H. (1983). *Computational methods to estimate the sedimentation in dredged channels and harbour basins in estuarine environments*. Delft Hydraulics.
  * Sumer, B. M., & Fredsøe, J. (2002). *The Mechanics of Scour in the Marine Environment*. World Scientific.
  * Kamphuis, J. W. (1991). *Alongshore sediment transport rate*. J. Waterway, Port, Coastal and Ocean Engineering, ASCE, 117(6), 624–640.
  * Kriebel, D. L., & Dean, R. G. (1993). *Convolution method for time-dependent beach-profile response*. J. Waterway, Port, Coastal and Ocean Engineering, ASCE, 119(2), 204–226.
  * Hallermeier, R. J., & Rhodes, P. E. (1988). *Generic treatment of dune erosion for 100-year event*. Proc. 21st Int. Conf. Coastal Engineering, ASCE, 1197–1211.
  * USACE / CERC (1984). *Shore Protection Manual*. U.S. Army Engineer Waterways Experiment Station.
