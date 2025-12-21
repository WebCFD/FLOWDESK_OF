# mainPipeline.py - Guía de Uso

## Descripción

Script principal que ejecuta el pipeline completo: JSON → Geometría → Malla → CFD → Resultados

## Argumentos de Línea de Comandos

### `--mode` (Modo de ejecución)

Especifica qué parte del pipeline ejecutar:

- **`stl-only`**: Genera solo la geometría STL (no ejecuta cfMesh)
- **`mesh-only`**: Ejecuta cfMesh para generar la malla (sin simulación CFD)
- **`full`**: Ejecuta cfMesh + simulación CFD completa (por defecto)

### `--json` (Archivo JSON de entrada)

Especifica el archivo JSON a procesar:

- **Valor por defecto**: `MySim_FlowDeskModel.json`
- **Ruta**: Relativa a `PYTHON_STEPS/`
- **Ejemplo**: `3floors_FlowDeskModel.json`

## Ejemplos de Uso

### 1. Usar JSON por defecto, modo mesh-only

```bash
cd /home/plprm/FLOWDESK_OF && python3.12 PYTHON_STEPS/mainPipeline.py --mode mesh-only
```

Ejecuta:
- ✅ STEP 01: Genera geometría STL
- ✅ STEP 02: Genera script de mallado
- ✅ STEP 03: Ejecuta cfMesh (genera malla)
- ❌ NO ejecuta simulación CFD

### 2. Usar JSON personalizado, modo mesh-only

```bash
cd /home/plprm/FLOWDESK_OF && python3.12 PYTHON_STEPS/mainPipeline.py --mode mesh-only --json 3floors_FlowDeskModel.json
```

Ejecuta el pipeline con el archivo `PYTHON_STEPS/3floors_FlowDeskModel.json`

### 3. Usar JSON personalizado, modo stl-only

```bash
cd /home/plprm/FLOWDESK_OF && python3.12 PYTHON_STEPS/mainPipeline.py --mode stl-only --json nuevo.json
```

Ejecuta:
- ✅ STEP 01: Genera geometría STL
- ✅ STEP 02: Genera script de mallado
- ❌ NO ejecuta cfMesh
- ❌ NO ejecuta simulación CFD

### 4. Modo full (completo) con JSON personalizado

```bash
cd /home/plprm/FLOWDESK_OF && python3.12 PYTHON_STEPS/mainPipeline.py --mode full --json 3floors_FlowDeskModel.json
```

Ejecuta el pipeline completo (STEP 01 + STEP 02 + STEP 03 + cfMesh + CFD)

### 5. Modo stl-only (solo geometría)

```bash
cd /home/plprm/FLOWDESK_OF && python3.12 PYTHON_STEPS/mainPipeline.py --mode stl-only
```

Genera solo la geometría STL sin ejecutar cfMesh ni CFD

## Archivos JSON Disponibles

Los siguientes archivos JSON están disponibles en `PYTHON_STEPS/`:

- `MySim_FlowDeskModel.json` - Caso de prueba por defecto
- `3floors_FlowDeskModel.json` - Caso con 3 pisos
- `no_windows.MySim_FlowDeskModel.json` - Caso sin ventanas/puertas
- `testJRM2025_NO_ENTRIES.json` - Caso de prueba sin entradas

## Outputs Generados

El pipeline genera outputs en `PYTHON_STEPS/outputs/`:

```
outputs/
├── output_01_YYYYMMDD_HHMMSS/
│   ├── geometry_YYYYMMDD_HHMMSS.stl
│   ├── geometry_YYYYMMDD_HHMMSS.vtk
│   ├── boundary_conditions_YYYYMMDD_HHMMSS.csv
│   ├── boundary_conditions_YYYYMMDD_HHMMSS.json
│   └── summary_YYYYMMDD_HHMMSS.json
├── output_02_YYYYMMDD_HHMMSS/
│   ├── mesh_script_YYYYMMDD_HHMMSS.sh
│   ├── commands_preview_YYYYMMDD_HHMMSS.txt
│   └── summary_YYYYMMDD_HHMMSS.json
├── output_03_YYYYMMDD_HHMMSS/
│   ├── system/
│   ├── constant/
│   ├── 0.orig/
│   └── summary_YYYYMMDD_HHMMSS.json
└── pipeline_main_YYYYMMDD_HHMMSS.txt
```

## Validación de Geometría

El pipeline incluye validación automática de geometría con 14 validaciones críticas:

1. ✅ Perímetro de pared cerrado
2. ✅ Ventana en plano de pared
3. ✅ Ventana coplanar con pared
4. ✅ Ventana más pequeña que pared
5. ✅ Ventana completamente contenida
6. ✅ Ventanas no solapadas
7. ✅ Ventana con área mínima
8. ✅ Pared sin auto-intersecciones
9. ✅ Ventana sin auto-intersecciones
10. ✅ Coordenadas válidas (no NaN)
11. ✅ Dimensiones positivas
12. ✅ Pared con mínimo de puntos
13. ✅ Ventana dentro del rango Z
14. ✅ Tolerancia numérica

Para desactivar validaciones, editar `src/components/geo/geometry_validator.py`:

```python
ENABLE_VALIDATION = False  # Cambiar a False para desactivar
```

## Logs

Todos los logs se guardan en:
- **Consola**: Salida en tiempo real
- **Archivo**: `PYTHON_STEPS/outputs/pipeline_main_YYYYMMDD_HHMMSS.txt`

## Troubleshooting

### Error: "No se encontró el archivo JSON"

Verificar que el archivo JSON existe en `PYTHON_STEPS/`:

```bash
ls -la /home/plprm/FLOWDESK_OF/PYTHON_STEPS/*.json
```

### Error: "OpenFOAM environment NOT loaded"

En modo `mesh-only` o `full`, se requiere OpenFOAM. Cargar manualmente:

```bash
source /home/plprm/OpenFOAM/OpenFOAM-v2012/etc/bashrc
```

### Error: "cfMesh NO generó polyMesh"

Revisar los logs en `PYTHON_STEPS/outputs/pipeline_main_*.txt` para ver el error específico.

## Notas

- El argumento `--json` es relativo a `PYTHON_STEPS/`
- Si no se especifica `--json`, usa `MySim_FlowDeskModel.json` por defecto
- Si no se especifica `--mode`, usa `full` por defecto
- Los outputs se guardan con timestamp para evitar sobrescrituras
- La validación de geometría se ejecuta automáticamente en STEP 01
