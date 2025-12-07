"""
Geometry Validation Module

Módulo robusto para validar geometrías de paredes, ventanas y puertas.
Proporciona 14 validaciones críticas con logs claros y fácil desactivación.

Uso:
    from src.components.geo.geometry_validator import validate_wall_closed, log_validation
    
    result = validate_wall_closed(wall_polygon, wall_id)
    if not log_validation(result, logger):
        raise ValueError(result.message)

Desactivar todas las validaciones:
    Cambiar ENABLE_VALIDATION = False en este archivo
"""

import numpy as np
import logging
from typing import Tuple, Dict, List

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURACIÓN GLOBAL
# ============================================================================

# Cambiar a False para desactivar TODAS las validaciones
ENABLE_VALIDATION = True

VALIDATION_CONFIG = {
    'tolerance': 1e-6,           # Tolerancia numérica para comparaciones
    'min_area': 0.001,           # Área mínima en m²
    'min_points': 3,             # Mínimo de puntos para un polígono
}


# ============================================================================
# CLASE RESULTADO DE VALIDACIÓN
# ============================================================================

class ValidationResult:
    """Resultado de una validación con severidad y mensaje"""
    
    def __init__(self, passed: bool, severity: str, message: str):
        """
        Args:
            passed: True si la validación pasó
            severity: 'ERROR', 'WARNING', o 'INFO'
            message: Mensaje descriptivo
        """
        self.passed = passed
        self.severity = severity
        self.message = message
    
    def __repr__(self):
        return f"ValidationResult(passed={self.passed}, severity={self.severity})"


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def distance_point_to_plane(point: np.ndarray, plane_point: np.ndarray, 
                           plane_normal: np.ndarray) -> float:
    """Calcula la distancia de un punto a un plano"""
    return abs(np.dot(plane_normal, point - plane_point))


def calculate_polygon_normal(points: np.ndarray) -> np.ndarray:
    """Calcula el normal de un polígono 3D usando los primeros 3 puntos"""
    if len(points) < 3:
        return np.array([0, 0, 1])
    
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    
    if norm > 1e-10:
        return normal / norm
    return np.array([0, 0, 1])


# ============================================================================
# VALIDACIÓN 1: Perímetro de la pared cerrado
# ============================================================================

def validate_wall_closed(wall_polygon, wall_id: str) -> ValidationResult:
    """
    Validación 1: Verifica que el perímetro de la pared está cerrado
    y no tiene auto-intersecciones.
    
    Args:
        wall_polygon: Polígono Shapely de la pared
        wall_id: ID de la pared para logs
    
    Returns:
        ValidationResult
    """
    if not ENABLE_VALIDATION:
        return ValidationResult(True, 'INFO', 'Validation disabled')
    
    if not wall_polygon.is_valid:
        return ValidationResult(False, 'ERROR',
            f"Wall '{wall_id}' is not closed or has self-intersections")
    
    return ValidationResult(True, 'INFO', f"Wall '{wall_id}' is closed and valid")


# ============================================================================
# VALIDACIÓN 2: Ventana está en el plano de la pared
# ============================================================================

def validate_entry_in_plane(entry_points_3d: np.ndarray, wall_plane_point: np.ndarray,
                           wall_plane_normal: np.ndarray, entry_id: str,
                           tolerance: float = None) -> ValidationResult:
    """
    Validación 2: Verifica que la ventana/puerta está en el plano de la pared.
    
    Args:
        entry_points_3d: Puntos 3D de la ventana
        wall_plane_point: Un punto del plano de la pared
        wall_plane_normal: Normal del plano de la pared
        entry_id: ID de la ventana para logs
        tolerance: Tolerancia numérica (usa VALIDATION_CONFIG si None)
    
    Returns:
        ValidationResult
    """
    if not ENABLE_VALIDATION:
        return ValidationResult(True, 'INFO', 'Validation disabled')
    
    tolerance = tolerance or VALIDATION_CONFIG['tolerance']
    
    # Calcular distancia de cada punto al plano
    distances = [distance_point_to_plane(p, wall_plane_point, wall_plane_normal) 
                 for p in entry_points_3d]
    max_distance = max(distances) if distances else 0
    
    if max_distance > tolerance:
        return ValidationResult(False, 'ERROR',
            f"Entry '{entry_id}' is not in wall plane (max distance: {max_distance:.2e}m)")
    
    return ValidationResult(True, 'INFO', f"Entry '{entry_id}' is in wall plane")


# ============================================================================
# VALIDACIÓN 3: Ventana es coplanar con la pared
# ============================================================================

def validate_entry_coplanar(entry_points_3d: np.ndarray, wall_normal: np.ndarray,
                           entry_id: str, tolerance: float = None) -> ValidationResult:
    """
    Validación 3: Verifica que la ventana es coplanar con la pared
    (no está rotada respecto a la pared).
    
    Args:
        entry_points_3d: Puntos 3D de la ventana
        wall_normal: Normal del plano de la pared
        entry_id: ID de la ventana para logs
        tolerance: Tolerancia numérica (usa VALIDATION_CONFIG si None)
    
    Returns:
        ValidationResult
    """
    if not ENABLE_VALIDATION:
        return ValidationResult(True, 'INFO', 'Validation disabled')
    
    tolerance = tolerance or VALIDATION_CONFIG['tolerance']
    
    # Calcular normal de la ventana
    entry_normal = calculate_polygon_normal(entry_points_3d)
    
    # Comparar con normal de la pared (dot product debe ser ≈ 1 o -1)
    dot_product = abs(np.dot(entry_normal, wall_normal))
    
    if dot_product < (1 - tolerance):
        return ValidationResult(False, 'ERROR',
            f"Entry '{entry_id}' is not coplanar with wall (dot product: {dot_product:.6f})")
    
    return ValidationResult(True, 'INFO', f"Entry '{entry_id}' is coplanar with wall")


# ============================================================================
# VALIDACIÓN 4: Ventana más pequeña que la pared
# ============================================================================

def validate_entry_smaller_than_wall(entry_area: float, wall_area: float,
                                    entry_id: str) -> ValidationResult:
    """
    Validación 4: Verifica que el área de la ventana es menor que la pared.
    
    Args:
        entry_area: Área de la ventana (m²)
        wall_area: Área de la pared (m²)
        entry_id: ID de la ventana para logs
    
    Returns:
        ValidationResult
    """
    if not ENABLE_VALIDATION:
        return ValidationResult(True, 'INFO', 'Validation disabled')
    
    if entry_area > wall_area:
        return ValidationResult(False, 'ERROR',
            f"Entry '{entry_id}' area ({entry_area:.3f}m²) > wall area ({wall_area:.3f}m²)")
    
    return ValidationResult(True, 'INFO', f"Entry '{entry_id}' area is smaller than wall")


# ============================================================================
# VALIDACIÓN 5: Ventana completamente contenida en pared
# ============================================================================

def validate_entry_contained(entry_polygon, wall_polygon, entry_id: str) -> ValidationResult:
    """
    Validación 5: Verifica que la ventana está completamente contenida en la pared.
    
    Args:
        entry_polygon: Polígono Shapely de la ventana
        wall_polygon: Polígono Shapely de la pared
        entry_id: ID de la ventana para logs
    
    Returns:
        ValidationResult
    """
    if not ENABLE_VALIDATION:
        return ValidationResult(True, 'INFO', 'Validation disabled')
    
    if not wall_polygon.contains(entry_polygon):
        return ValidationResult(False, 'ERROR',
            f"Entry '{entry_id}' is not completely contained in wall")
    
    return ValidationResult(True, 'INFO', f"Entry '{entry_id}' is contained in wall")


# ============================================================================
# VALIDACIÓN 6: Ventanas no solapadas
# ============================================================================

def validate_entries_no_overlap(entries_dict: Dict[str, object], wall_id: str) -> ValidationResult:
    """
    Validación 6: Verifica que las ventanas no se solapan entre sí.
    
    Args:
        entries_dict: Diccionario {entry_id: entry_polygon}
        wall_id: ID de la pared para logs
    
    Returns:
        ValidationResult
    """
    if not ENABLE_VALIDATION:
        return ValidationResult(True, 'INFO', 'Validation disabled')
    
    entry_ids = list(entries_dict.keys())
    for i, id1 in enumerate(entry_ids):
        for id2 in entry_ids[i+1:]:
            if entries_dict[id1].intersects(entries_dict[id2]):
                return ValidationResult(False, 'WARNING',
                    f"Entries '{id1}' and '{id2}' overlap in wall '{wall_id}'")
    
    return ValidationResult(True, 'INFO', f"No overlapping entries in wall '{wall_id}'")


# ============================================================================
# VALIDACIÓN 7: Ventana con área mínima
# ============================================================================

def validate_entry_min_area(entry_area: float, entry_id: str,
                           min_area: float = None) -> ValidationResult:
    """
    Validación 7: Verifica que la ventana tiene un área mínima.
    
    Args:
        entry_area: Área de la ventana (m²)
        entry_id: ID de la ventana para logs
        min_area: Área mínima (usa VALIDATION_CONFIG si None)
    
    Returns:
        ValidationResult
    """
    if not ENABLE_VALIDATION:
        return ValidationResult(True, 'INFO', 'Validation disabled')
    
    min_area = min_area or VALIDATION_CONFIG['min_area']
    
    if entry_area < min_area:
        return ValidationResult(False, 'WARNING',
            f"Entry '{entry_id}' area ({entry_area:.6f}m²) < minimum ({min_area}m²)")
    
    return ValidationResult(True, 'INFO', f"Entry '{entry_id}' area is above minimum")


# ============================================================================
# VALIDACIÓN 8: Pared sin auto-intersecciones
# ============================================================================

def validate_wall_valid(wall_polygon, wall_id: str) -> ValidationResult:
    """
    Validación 8: Verifica que la pared no tiene auto-intersecciones.
    
    Args:
        wall_polygon: Polígono Shapely de la pared
        wall_id: ID de la pared para logs
    
    Returns:
        ValidationResult
    """
    if not ENABLE_VALIDATION:
        return ValidationResult(True, 'INFO', 'Validation disabled')
    
    if not wall_polygon.is_valid:
        return ValidationResult(False, 'ERROR',
            f"Wall '{wall_id}' has self-intersections or is invalid")
    
    return ValidationResult(True, 'INFO', f"Wall '{wall_id}' is valid")


# ============================================================================
# VALIDACIÓN 9: Ventana sin auto-intersecciones
# ============================================================================

def validate_entry_valid(entry_polygon, entry_id: str) -> ValidationResult:
    """
    Validación 9: Verifica que la ventana no tiene auto-intersecciones.
    
    Args:
        entry_polygon: Polígono Shapely de la ventana
        entry_id: ID de la ventana para logs
    
    Returns:
        ValidationResult
    """
    if not ENABLE_VALIDATION:
        return ValidationResult(True, 'INFO', 'Validation disabled')
    
    if not entry_polygon.is_valid:
        return ValidationResult(False, 'ERROR',
            f"Entry '{entry_id}' has self-intersections or is invalid")
    
    return ValidationResult(True, 'INFO', f"Entry '{entry_id}' is valid")


# ============================================================================
# VALIDACIÓN 10: Coordenadas válidas (no NaN)
# ============================================================================

def validate_coordinates_valid(coords: np.ndarray, entity_id: str) -> ValidationResult:
    """
    Validación 10: Verifica que las coordenadas no contienen NaN o Inf.
    
    Args:
        coords: Array de coordenadas
        entity_id: ID de la entidad para logs
    
    Returns:
        ValidationResult
    """
    if not ENABLE_VALIDATION:
        return ValidationResult(True, 'INFO', 'Validation disabled')
    
    if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
        return ValidationResult(False, 'ERROR',
            f"Entity '{entity_id}' has invalid coordinates (NaN or Inf)")
    
    return ValidationResult(True, 'INFO', f"Entity '{entity_id}' coordinates are valid")


# ============================================================================
# VALIDACIÓN 11: Dimensiones positivas
# ============================================================================

def validate_dimensions_positive(width: float, height: float, entry_id: str) -> ValidationResult:
    """
    Validación 11: Verifica que las dimensiones son positivas.
    
    Args:
        width: Ancho de la ventana (m)
        height: Alto de la ventana (m)
        entry_id: ID de la ventana para logs
    
    Returns:
        ValidationResult
    """
    if not ENABLE_VALIDATION:
        return ValidationResult(True, 'INFO', 'Validation disabled')
    
    if width <= 0 or height <= 0:
        return ValidationResult(False, 'ERROR',
            f"Entry '{entry_id}' has invalid dimensions (width={width}m, height={height}m)")
    
    return ValidationResult(True, 'INFO', f"Entry '{entry_id}' dimensions are positive")


# ============================================================================
# VALIDACIÓN 12: Pared con mínimo de puntos
# ============================================================================

def validate_wall_min_points(wall_points: np.ndarray, wall_id: str,
                            min_points: int = None) -> ValidationResult:
    """
    Validación 12: Verifica que la pared tiene mínimo de puntos.
    
    Args:
        wall_points: Array de puntos de la pared
        wall_id: ID de la pared para logs
        min_points: Mínimo de puntos (usa VALIDATION_CONFIG si None)
    
    Returns:
        ValidationResult
    """
    if not ENABLE_VALIDATION:
        return ValidationResult(True, 'INFO', 'Validation disabled')
    
    min_points = min_points or VALIDATION_CONFIG['min_points']
    
    if len(wall_points) < min_points:
        return ValidationResult(False, 'ERROR',
            f"Wall '{wall_id}' has {len(wall_points)} points (minimum: {min_points})")
    
    return ValidationResult(True, 'INFO', f"Wall '{wall_id}' has enough points")


# ============================================================================
# VALIDACIÓN 13: Ventana dentro del rango Z (altura)
# ============================================================================

def validate_entry_in_height_range(entry_z: float, wall_z_min: float, wall_z_max: float,
                                  entry_id: str) -> ValidationResult:
    """
    Validación 13: Verifica que la ventana está dentro del rango Z de la pared.
    
    Args:
        entry_z: Coordenada Z de la ventana (m)
        wall_z_min: Z mínimo de la pared (m)
        wall_z_max: Z máximo de la pared (m)
        entry_id: ID de la ventana para logs
    
    Returns:
        ValidationResult
    """
    if not ENABLE_VALIDATION:
        return ValidationResult(True, 'INFO', 'Validation disabled')
    
    if entry_z < wall_z_min or entry_z > wall_z_max:
        return ValidationResult(False, 'ERROR',
            f"Entry '{entry_id}' Z={entry_z}m outside wall range [{wall_z_min}, {wall_z_max}]m")
    
    return ValidationResult(True, 'INFO', f"Entry '{entry_id}' is within height range")


# ============================================================================
# VALIDACIÓN 14: Tolerancia numérica
# ============================================================================

def validate_numerical_tolerance(value: float, tolerance: float = None) -> ValidationResult:
    """
    Validación 14: Verifica que el valor está dentro de la tolerancia numérica.
    
    Args:
        value: Valor a verificar
        tolerance: Tolerancia (usa VALIDATION_CONFIG si None)
    
    Returns:
        ValidationResult
    """
    if not ENABLE_VALIDATION:
        return ValidationResult(True, 'INFO', 'Validation disabled')
    
    tolerance = tolerance or VALIDATION_CONFIG['tolerance']
    
    if abs(value) < tolerance:
        return ValidationResult(True, 'INFO', f"Value {value:.2e} is within tolerance")
    
    return ValidationResult(True, 'INFO', f"Value {value:.2e} is outside tolerance")


# ============================================================================
# LOGGER HELPER
# ============================================================================

def log_validation(result: ValidationResult, logger_obj=None) -> bool:
    """
    Registra el resultado de una validación en los logs.
    
    Args:
        result: ValidationResult a registrar
        logger_obj: Logger a usar (usa el logger del módulo si None)
    
    Returns:
        True si la validación pasó, False si falló
    """
    log = logger_obj or logger
    
    if result.severity == 'ERROR':
        log.error(f"❌ {result.message}")
    elif result.severity == 'WARNING':
        log.warning(f"⚠️  {result.message}")
    else:
        log.info(f"✓ {result.message}")
    
    return result.passed


def validate_and_log(result: ValidationResult, logger_obj=None, raise_on_error: bool = True) -> bool:
    """
    Registra el resultado y opcionalmente lanza excepción si hay error.
    
    Args:
        result: ValidationResult a registrar
        logger_obj: Logger a usar
        raise_on_error: Si True, lanza ValueError en caso de ERROR
    
    Returns:
        True si la validación pasó
    
    Raises:
        ValueError: Si raise_on_error=True y result.severity='ERROR'
    """
    passed = log_validation(result, logger_obj)
    
    if not passed and raise_on_error and result.severity == 'ERROR':
        raise ValueError(result.message)
    
    return passed
