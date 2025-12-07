# Geometry Validator - Guía de Uso

## Descripción

Módulo robusto para validar geometrías de paredes, ventanas y puertas con 14 validaciones críticas.

## Características

✅ **14 validaciones críticas** para detectar errores geométricos
✅ **Fácil de desactivar** - Una sola línea de código
✅ **Logs claros** - Cada validación registra exactamente qué falló
✅ **Código legible** - Cada función es simple y clara
✅ **Modular** - Fácil agregar nuevas validaciones
✅ **Flexible** - Severidad configurable (ERROR, WARNING, INFO)

## Desactivar Validaciones

Para desactivar TODAS las validaciones, cambiar en `geometry_validator.py`:

```python
# Cambiar esto:
ENABLE_VALIDATION = True

# A esto:
ENABLE_VALIDATION = False
```

## Las 14 Validaciones

| # | Validación | Severidad | Descripción |
|---|-----------|-----------|-------------|
| 1 | `validate_wall_closed()` | 🔴 ERROR | Perímetro de la pared cerrado |
| 2 | `validate_entry_in_plane()` | 🔴 ERROR | Ventana está en el plano de la pared |
| 3 | `validate_entry_coplanar()` | 🔴 ERROR | Ventana es coplanar con la pared |
| 4 | `validate_entry_smaller_than_wall()` | 🔴 ERROR | Ventana más pequeña que la pared |
| 5 | `validate_entry_contained()` | 🔴 ERROR | Ventana completamente contenida |
| 6 | `validate_entries_no_overlap()` | 🟠 WARNING | Ventanas no solapadas |
| 7 | `validate_entry_min_area()` | 🟠 WARNING | Ventana con área mínima |
| 8 | `validate_wall_valid()` | 🔴 ERROR | Pared sin auto-intersecciones |
| 9 | `validate_entry_valid()` | 🔴 ERROR | Ventana sin auto-intersecciones |
| 10 | `validate_coordinates_valid()` | 🔴 ERROR | Coordenadas válidas (no NaN) |
| 11 | `validate_dimensions_positive()` | 🔴 ERROR | Dimensiones positivas |
| 12 | `validate_wall_min_points()` | 🔴 ERROR | Pared con mínimo de puntos |
| 13 | `validate_entry_in_height_range()` | 🔴 ERROR | Ventana dentro del rango Z |
| 14 | `validate_numerical_tolerance()` | 🟡 INFO | Tolerancia numérica |

## Ejemplos de Uso

### Ejemplo 1: Validación simple con log

```python
from src.components.geo.geometry_validator import validate_wall_closed, log_validation
import logging

logger = logging.getLogger(__name__)

# Validar pared
result = validate_wall_closed(wall_polygon, "wall_01")
if not log_validation(result, logger):
    print("La pared no es válida")
```

**Output:**
```
✓ Wall 'wall_01' is closed and valid
```

### Ejemplo 2: Validación con excepción

```python
from src.components.geo.geometry_validator import validate_and_log

# Validar y lanzar excepción si hay error
result = validate_wall_closed(wall_polygon, "wall_01")
validate_and_log(result, logger, raise_on_error=True)
```

### Ejemplo 3: Validar ventana en plano

```python
from src.components.geo.geometry_validator import validate_entry_in_plane

# Validar que la ventana está en el plano de la pared
result = validate_entry_in_plane(
    entry_points_3d=window_points,
    wall_plane_point=wall_origin,
    wall_plane_normal=wall_normal,
    entry_id="window_01"
)

log_validation(result, logger)
```

### Ejemplo 4: Validar múltiples ventanas

```python
from src.components.geo.geometry_validator import (
    validate_entry_valid,
    validate_entry_contained,
    validate_entries_no_overlap,
    log_validation
)

# Validar cada ventana
for entry_id, entry_polygon in entries_dict.items():
    # Validar que no tiene auto-intersecciones
    result = validate_entry_valid(entry_polygon, entry_id)
    if not log_validation(result, logger):
        continue
    
    # Validar que está contenida en la pared
    result = validate_entry_contained(entry_polygon, wall_polygon, entry_id)
    if not log_validation(result, logger):
        continue

# Validar que no se solapan
result = validate_entries_no_overlap(entries_dict, "wall_01")
log_validation(result, logger)
```

### Ejemplo 5: Desactivar validaciones temporalmente

```python
from src.components.geo import geometry_validator

# Desactivar validaciones
geometry_validator.ENABLE_VALIDATION = False

# Hacer operaciones sin validación
result = validate_wall_closed(wall_polygon, "wall_01")
# Output: ValidationResult(passed=True, severity='INFO')

# Reactivar validaciones
geometry_validator.ENABLE_VALIDATION = True
```

## Configuración

Cambiar tolerancias en `geometry_validator.py`:

```python
VALIDATION_CONFIG = {
    'tolerance': 1e-6,           # Tolerancia numérica (metros)
    'min_area': 0.001,           # Área mínima (m²)
    'min_points': 3,             # Mínimo de puntos para un polígono
}
```

## Integración en create_volumes.py

```python
from src.components.geo.geometry_validator import (
    validate_wall_closed,
    validate_entry_in_plane,
    validate_entry_coplanar,
    validate_entry_contained,
    validate_and_log
)

def create_wall(patch_df, data, height, base_height):
    # ... código existente ...
    
    # Validar pared
    result = validate_wall_closed(wall_polygon, data['id'])
    validate_and_log(result, logger, raise_on_error=True)
    
    # Validar ventanas
    for entry_id, entry_polygon in entries_dict.items():
        result = validate_entry_in_plane(
            entry_points_3d, p0, udir, entry_id
        )
        validate_and_log(result, logger, raise_on_error=True)
    
    # ... resto del código ...
```

## Logs Esperados

### Validación exitosa:
```
✓ Wall 'wall_01' is closed and valid
✓ Entry 'window_01' is in wall plane
✓ Entry 'window_01' is coplanar with wall
✓ Entry 'window_01' is contained in wall
```

### Validación fallida:
```
❌ Wall 'wall_01' is not closed or has self-intersections
⚠️  Entry 'window_01' area (0.0001m²) < minimum (0.001m²)
❌ Entry 'window_02' is not in wall plane (max distance: 1.23e-05m)
```

## Notas

- Las validaciones se pueden desactivar globalmente con `ENABLE_VALIDATION = False`
- Cada validación es independiente y puede usarse por separado
- Los logs incluyen información específica del error para debugging
- La severidad (ERROR/WARNING/INFO) indica la importancia del problema
- Las funciones retornan `ValidationResult` para máxima flexibilidad
