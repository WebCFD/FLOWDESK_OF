# Migración a OpenFOAM v2412

## Resumen

Se ha actualizado el proyecto FLOWDESK_OF para usar **OpenFOAM v2412** en lugar de OpenFOAM v2012.

## Cambios Realizados

### 1. Rutas de OpenFOAM Actualizadas

**Antes (v2012):**
```bash
/home/plprm/OpenFOAM/OpenFOAM-v2012/etc/bashrc
```

**Ahora (v2412):**
```bash
/usr/lib/openfoam/openfoam2412/etc/bashrc
```

### 2. Archivos Modificados

#### Scripts Python
- ✅ **`PYTHON_STEPS/mainPipeline.py`**
  - Actualizada función `setup_openfoam_environment()` con nuevas rutas
  - Prioridad: `/usr/lib/openfoam/openfoam2412/etc/bashrc`
  - Ejecuta Allrun con nueva ruta en modo mesh-only

#### Configuración IDE
- ✅ **`.vscode/c_cpp_properties.json`**
  - Actualizadas todas las rutas `includePath` de v2012 a v2412
  - Paths actualizados:
    - `/usr/lib/openfoam/openfoam2412/src/OpenFOAM/lnInclude`
    - `/usr/lib/openfoam/openfoam2412/src/finiteVolume/lnInclude`
    - Y otros 5 paths más

#### Documentación
- ✅ **`PYTHON_STEPS/MAINPIPELINE_USAGE.md`**
  - Actualizada sección Troubleshooting con nueva ruta
  - Comando correcto: `source /usr/lib/openfoam/openfoam2412/etc/bashrc`

### 3. Script de Migración

Se creó **`update_openfoam_v2412.sh`** para actualizar casos existentes:

```bash
./update_openfoam_v2412.sh
```

**Funcionalidad:**
- Busca todos los archivos `Allrun` en `cases/*/sim/`
- Reemplaza rutas antiguas por nuevas
- Crea backups con extensión `.backup_v2012`
- Genera reporte de cambios

### 4. Verificación de Compatibilidad

✅ **cfMesh disponible en v2412:**
```bash
$ which cartesianMesh
/usr/lib/openfoam/openfoam2412/platforms/linux64GccDPInt32Opt/bin/cartesianMesh
```

✅ **Pipeline probado exitosamente:**
```bash
cd /home/plprm/FLOWDESK_OF
python PYTHON_STEPS/mainPipeline.py --mode mesh-only --json MySim_FlowDeskModel.json
```

**Resultado:** ✅ Malla generada correctamente en `constant/polyMesh/`

## Cómo Usar

### Para Nuevos Casos

El pipeline automáticamente usa OpenFOAM v2412:

```bash
cd /home/plprm/FLOWDESK_OF
python PYTHON_STEPS/mainPipeline.py --mode mesh-only
```

### Para Casos Existentes

Si tienes casos antiguos con rutas v2012, ejecuta:

```bash
cd /home/plprm/FLOWDESK_OF
./update_openfoam_v2412.sh
```

### Cargar OpenFOAM Manualmente

Si necesitas cargar OpenFOAM en la terminal:

```bash
source /usr/lib/openfoam/openfoam2412/etc/bashrc
```

O usando el alias:
```bash
of
```

## Verificar Instalación

Para verificar que OpenFOAM v2412 está correctamente instalado:

```bash
# Verificar variable de entorno
echo $WM_PROJECT_DIR
# Debería mostrar: /usr/lib/openfoam/openfoam2412

# Verificar versión
echo $WM_PROJECT_VERSION
# Debería mostrar: v2412

# Verificar que cfMesh está disponible
which cartesianMesh
# Debería mostrar: /usr/lib/openfoam/openfoam2412/.../cartesianMesh
```

## Revertir a v2012 (si es necesario)

Para revertir casos individuales:

```bash
# Restaurar backup de Allrun
mv cases/MySim/sim/Allrun.backup_v2012 cases/MySim/sim/Allrun
```

Para revertir cambios en el código:

```bash
# Usar git para revertir
git checkout HEAD -- PYTHON_STEPS/mainPipeline.py
git checkout HEAD -- .vscode/c_cpp_properties.json
git checkout HEAD -- PYTHON_STEPS/MAINPIPELINE_USAGE.md
```

## Archivos No Modificados

Los siguientes archivos **NO** requieren cambios:
- `step01_json2geo.py` - No usa OpenFOAM
- `step02_geo2mesh.py` - Solo genera scripts
- `step03_mesh2cfd.py` - Funciona con cualquier versión
- Archivos en `data/settings/` - Templates genéricos
- Archivos en `src/components/` - Lógica independiente de versión

## Beneficios de v2412

- ✅ Versión más reciente (diciembre 2024)
- ✅ Mejoras de rendimiento y estabilidad
- ✅ Compatible con cfMesh
- ✅ Instalación estándar en `/usr/lib/`
- ✅ Mejor soporte y documentación

## Solución de Problemas

### Error: "No such file or directory: bashrc"

**Solución:** Verificar que OpenFOAM v2412 está instalado:
```bash
ls -la /usr/lib/openfoam/openfoam2412/etc/bashrc
```

Si no existe, instalar OpenFOAM v2412.

### Error: "cartesianMesh: command not found"

**Solución:** Cargar el entorno de OpenFOAM:
```bash
source /usr/lib/openfoam/openfoam2412/etc/bashrc
```

### Pipeline usa rutas antiguas

**Solución:** Ejecutar el script de actualización:
```bash
./update_openfoam_v2412.sh
```

## Fecha de Migración

**21 de diciembre de 2025**

## Contacto

Para problemas relacionados con la migración, revisar los logs en:
- `PYTHON_STEPS/outputs/pipeline_main_*.txt`
