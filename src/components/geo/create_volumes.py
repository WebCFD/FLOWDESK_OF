import os
import shapely
import numpy as np
import pandas as pd
import pyvista as pv
import logging

from typing import List, Tuple, Dict, Any
from src.components.tools.performance import (
    optimize_mesh_memory,
    PerformanceMonitor
)

logger = logging.getLogger(__name__)

VOLUMES_TOLERANCE = 1e-5
DEFAULT_TEMPERATURE = 20

# Furniture positioning constants
FURNITURE_FLOOR_PENETRATION = -0.000  # m, negative offset to ensure floor intersection

# Flow value mapping for different boundary condition types
FLOW_VALUES = {
    # Velocity boundary conditions (m/s)
    'velocity': {
        'low': 1.0,
        'medium': 2.5,
        'high': 5.0
    },
    # Mass flow boundary conditions (m³/h)
    'massFlow': {
        'low': 150.0,
        'medium': 250.0,
        'high': 400.0
    },
    # Pressure boundary conditions (Pa)
    'pressure': {
        'low': 0.3,
        'medium': 2.5,
        'high': 5.0
    }
}

# Legacy flow levels for backward compatibility (velocity in m/s)
FLOW_LEVELS = {
    'low':      0.5,
    'medium':   1.0,
    'high':     2.0,
}


def get_flow_value(flow_intensity: str, flow_type: str, custom_value: float = None) -> float:
    """
    Get flow value based on intensity level and flow type.
    
    Args:
        flow_intensity: Intensity level ('low', 'medium', 'high', 'custom')
        flow_type: Type of flow ('velocity', 'massFlow', 'pressure')
        custom_value: Custom value when flow_intensity is 'custom'
    
    Returns:
        Flow value in appropriate units (m/s for velocity, m³/h for massFlow, Pa for pressure)
    """
    if flow_intensity == 'custom':
        if custom_value is None:
            logger.warning(f"Custom flow intensity specified but no custom_value provided. Using medium default.")
            return FLOW_VALUES.get(flow_type, FLOW_VALUES['velocity'])['medium']
        return custom_value
    
    # Get flow type values, default to velocity if unknown type
    flow_type_values = FLOW_VALUES.get(flow_type, FLOW_VALUES['velocity'])
    return flow_type_values.get(flow_intensity, flow_type_values['medium'])


def angles_to_direction_vector(vertical_angle: float, horizontal_angle: float, wall_normal: np.ndarray) -> np.ndarray:
    """
    Convert orientation angles to 3D direction vector for air flow.
    
    Applies vertical and horizontal angle rotations to the wall normal vector
    to compute the actual flow direction for HVAC vents/grilles.
    
    Args:
        vertical_angle: Vertical tilt angle in degrees (-45 to +45)
                       Positive = upward tilt, Negative = downward tilt
        horizontal_angle: Horizontal rotation angle in degrees (-45 to +45)
                         Positive = clockwise, Negative = counterclockwise
        wall_normal: Base normal vector of the wall [nx, ny, nz]
    
    Returns:
        Normalized 3D direction vector for the air flow
    """
    # Convert angles from degrees to radians
    v_rad = np.deg2rad(vertical_angle)
    h_rad = np.deg2rad(horizontal_angle)
    
    # Ensure wall_normal is a numpy array and normalized
    normal = np.array(wall_normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    
    # Find perpendicular vectors to create local coordinate system
    # Choose an arbitrary vector not parallel to normal
    if abs(normal[2]) < 0.9:
        up = np.array([0, 0, 1])
    else:
        up = np.array([1, 0, 0])
    
    # Create orthogonal basis vectors
    # right vector (tangent to wall, horizontal direction)
    right = np.cross(normal, up)
    right = right / np.linalg.norm(right)
    
    # up vector (tangent to wall, vertical direction)
    up_wall = np.cross(right, normal)
    up_wall = up_wall / np.linalg.norm(up_wall)
    
    # Apply rotations using rotation matrices
    # Step 1: Rotate around the 'right' axis for vertical angle (pitch)
    cos_v = np.cos(v_rad)
    sin_v = np.sin(v_rad)
    direction = cos_v * normal + sin_v * up_wall
    
    # Step 2: Rotate around the 'normal' axis for horizontal angle (yaw)
    # Use Rodrigues' rotation formula to preserve normal component
    cos_h = np.cos(h_rad)
    sin_h = np.sin(h_rad)
    direction = (direction * cos_h + 
                 np.cross(normal, direction) * sin_h + 
                 normal * np.dot(normal, direction) * (1 - cos_h))
    
    # Normalize the final direction vector
    direction = direction / np.linalg.norm(direction)
    
    return direction


ELEMENTS_MESHES = {
    # ID: (NAME, FOLDER PATH)
     1: ('h_open_arms',                         'HUMANS'),
     2: ('h_standing',                          'HUMANS'),
     3: ('h_sitting',                           'HUMANS'),
    11: ('table_round',                         'TABLES'),
    12: ('table_rect',                          'TABLES'),
    21: ('chair_basic',                         'CHAIRS'),
    22: ('chair_office',                        'UNKNOWN'),         # CHECK WITH JUAN
    41: ('h_open_arms_NOMOUTH',                 'BREATHING'),
    42: ('h_standing_NOMOUTH',                  'BREATHING'),
    43: ('h_sitting_NOMOUTH',                   'BREATHING'),
    51: ('MOUTH_h_open_arms',                   'BREATHING'),
    52: ('MOUTH_h_standing',                    'BREATHING'),
    53: ('MOUTH_h_sitting',                     'BREATHING'),
   111: ('set_chair_basic_N4',                  'SETS'),
   112: ('set_h_sitting_N4',                    'SETS'),
   121: ('set_4chairs_table_rectangular',       'JRM_Designs'),    # CHECK WITH JUAN
   122: ('set_4people_table_rectangular',       'JRM_Designs'),    # CHECK WITH JUAN
   131: ('set_6chairs_Long_table_rectangular',  'JRM_Designs'),    # CHECK WITH JUAN
   132: ('set_6people_Long_table_rectangular',  'JRM_Designs'),    # CHECK WITH JUAN
   143: ('set_h_sitting_N4_NOMOUTH',            'SETS'),
   153: ('set_h_sitting_N4_Mouth',              'SETS'),
}

# Object type mapping for furniture IDs: "object_{FLOOR}_{TYPE}_{NUMBER}"
OBJECT_TYPE_MAPPING = {
    'person':    2,   # h_standing (HUMANS)
    'block':     None, # Crear cubo programático
    'table':     12,  # table_round (TABLES)
    'armchair':  21,  # chair_basic (CHAIRS)
}


def create_cube_mesh(width=1.0, height=1.0, depth=1.0):
    """
    Create a programmatic cube mesh with specified dimensions.
    
    Args:
        width: X dimension (meters)
        height: Z dimension (meters)
        depth: Y dimension (meters)
    
    Returns:
        Triangulated and cleaned PyVista PolyData cube mesh
    """
    cube = pv.Cube(x_length=width, y_length=depth, z_length=height)
    return cube.triangulate().clean()


def create_furniture_mesh(patch_df: pd.DataFrame, data: Dict[str, Any]) -> Tuple[pd.DataFrame, pv.PolyData]:
    """
    Create furniture mesh from JSON format.
    
    Supports object IDs in format: "object_{FLOOR}_{TYPE}_{NUMBER}"
    - person: h_standing (HUMANS)
    - block: programmatic cube
    - table: table_round (TABLES)
    - armchair: chair_basic (CHAIRS)
    """
    # Sanitize patch ID for OpenFOAM compatibility
    patch_id = data['id'].replace(' ', '_')
    
    # Add patch info
    new_patch = get_wall_bc_dict(patch_id)
    patch_df = pd.concat([patch_df, pd.DataFrame([new_patch])], ignore_index=True)
    patch_idx = patch_df.index[patch_df['id'] == patch_id][0].astype(np.int16)

    # Extract position, rotation, and scale
    translation = np.array([data['position']['x'], data['position']['y'], data['position']['z']])
    rotation = data['rotation']['z']
    scale = data.get('scale', {'x': 1, 'y': 1, 'z': 1})
    
    # Parse object type from ID: "object_0F_table_1" → "table"
    id_parts = data['id'].split('_')
    object_type = id_parts[2] if len(id_parts) >= 3 else 'unknown'
    
    logger.info(f"    * Creating furniture: {data['id']} (type: {object_type})")
    
    # Determine mesh source based on type
    if object_type == 'block':
        # Create programmatic cube
        logger.info(f"      → Creating programmatic cube with scale: {scale}")
        obj_mesh = create_cube_mesh(width=scale['x'], height=scale['z'], depth=scale['y'])
        
        # Adjust Z position: JSON z=0 refers to bottom face, but pv.Cube() is centered at origin
        # Offset by half the height so bottom face sits at JSON's z position
        # Apply floor penetration offset to ensure intersection with floor
        z_offset = scale['z'] / 2 + FURNITURE_FLOOR_PENETRATION
        adjusted_translation = np.array([
            translation[0],
            translation[1],
            translation[2] + z_offset
        ])
        
        logger.info(f"      → Adjusting Z position: original={translation[2]:.3f}m, offset=+{z_offset:.3f}m (includes {FURNITURE_FLOOR_PENETRATION:.3f}m floor penetration), final={adjusted_translation[2]:.3f}m")
        
        obj_mesh.translate(xyz=adjusted_translation, inplace=True)
        obj_mesh.rotate_z(rotation, inplace=True)
    else:
        # Try to load from STL library, fallback to programmatic mesh if not found
        mesh_id = OBJECT_TYPE_MAPPING.get(object_type, 11)  # Default to table_round
        
        if mesh_id is None:
            logger.warning(f"      ⚠️  Unknown object type '{object_type}', defaulting to table_round")
            mesh_id = 11
        
        mesh_path_info = ELEMENTS_MESHES[mesh_id]
        mesh_path = os.path.join(os.getcwd(), 'data', 'CAD_database', mesh_path_info[1], mesh_path_info[0] + '.stl')
        
        logger.info(f"      → Loading STL: {mesh_path_info[0]} from {mesh_path_info[1]}")
        
        if not os.path.exists(mesh_path):
            logger.error(f"      ❌ STL file not found: {mesh_path}")
            raise FileNotFoundError(f"Furniture STL not found: {mesh_path}")
        
        obj_mesh = pv.read(mesh_path)
        
        # Apply floor penetration offset to ensure intersection with floor
        adjusted_translation = np.array([
            translation[0],
            translation[1],
            translation[2] + FURNITURE_FLOOR_PENETRATION
        ])
        
        logger.info(f"      → Applying floor penetration: Z offset={FURNITURE_FLOOR_PENETRATION:.3f}m")
        
        # Apply transformations
        obj_mesh.rotate_z(rotation, inplace=True)
        obj_mesh.translate(xyz=adjusted_translation, inplace=True)
        obj_mesh.scale([scale['x'], scale['y'], scale['z']], inplace=True)
    
    # Clean and prepare mesh
    obj_mesh = obj_mesh.clean().triangulate().extract_surface()
    obj_mesh.compute_normals(inplace=True, auto_orient_normals=True, consistent_normals=True, split_vertices=True, point_normals=False)
    obj_mesh.cell_data['patch_id'] = [patch_idx] * obj_mesh.n_cells
    
    logger.info(f"      ✓ Furniture mesh created: {obj_mesh.n_points} points, {obj_mesh.n_cells} cells")
    
    return patch_df, optimize_mesh_memory(obj_mesh)


def add_new_bc(bc_names, new_name):
    if(not new_name in bc_names):
        bc_names.append(new_name)
    bc_idx = bc_names.index(new_name)
    return bc_names, bc_idx


def get_entry_bc_dict(data):
    """Create boundary condition dictionary for air entries (FDM_iter2.json format)."""
    simulation = data['simulation']
    
    # Sanitize patch ID for OpenFOAM compatibility (replace spaces with underscores)
    patch_id = data['id'].replace(' ', '_')
    
    new_patch = {
        'id': patch_id,
        'open': simulation['state'] == 'open'
    }
    
    if not new_patch['open']:
        # Closed entry behaves like a wall
        return {
            'id': patch_id,
            'type': 'wall',
            'T': simulation.get('temperature', DEFAULT_TEMPERATURE)
        }
    
    # Determine entry type: windows/doors always use pressure BCs, vents allow user choice
    entry_type = data.get('type', 'window')  # 'window', 'door', 'vent'
    air_direction = simulation.get('airDirection', 'inflow')
    flow_intensity = simulation.get('flowIntensity', 'medium')
    custom_value = simulation.get('customValue', None)
    
    # For windows/doors: ALWAYS use pressure BCs (no flowType in JSON)
    if entry_type in ['window', 'door']:
        # Get pressure differential (ΔP) based on flow intensity
        delta_p = get_flow_value(flow_intensity, 'pressure', custom_value)
        
        if air_direction == 'inflow':
            # Pressure inlet: p = p_internal + ΔP (higher pressure pushes air in)
            new_patch['type'] = 'pressure_inlet'
            new_patch['pressure'] = delta_p  # Positive pressure differential
        else:  # outflow
            # Pressure outlet: p = p_internal - ΔP (lower pressure pulls air out)
            new_patch['type'] = 'pressure_outlet'
            new_patch['pressure'] = -delta_p  # Negative pressure differential
        
        new_patch['U'] = np.nan  # Not used for pressure BCs
    
    # For vents: User specifies flowType (velocity, massFlow, or pressure)
    else:
        flow_type_raw = simulation.get('flowType', 'velocity')
        
        # Normalize flow_type to handle case variations (massflow → massFlow)
        flow_type_map = {
            'massflow': 'massFlow',
            'massFlow': 'massFlow',
            'velocity': 'velocity',
            'pressure': 'pressure'
        }
        flow_type = flow_type_map.get(flow_type_raw, 'velocity')
        
        if air_direction == 'inflow':
            # Handle different flow types for inlet
            if flow_type == 'massFlow':
                new_patch['type'] = 'mass_flow_inlet'
                new_patch['massFlow'] = get_flow_value(flow_intensity, 'massFlow', custom_value)
                new_patch['U'] = np.nan
            elif flow_type == 'pressure':
                # Pressure inlet with positive differential
                new_patch['type'] = 'pressure_inlet'
                delta_p = get_flow_value(flow_intensity, 'pressure', custom_value)
                new_patch['pressure'] = delta_p
                new_patch['U'] = np.nan
            else:  # velocity
                new_patch['type'] = 'velocity_inlet'
                new_patch['U'] = get_flow_value(flow_intensity, 'velocity', custom_value)
                
        else:  # outflow
            if flow_type == 'pressure':
                # Pressure outlet with negative differential
                new_patch['type'] = 'pressure_outlet'
                delta_p = get_flow_value(flow_intensity, 'pressure', custom_value)
                new_patch['pressure'] = -delta_p
            else:
                # Velocity/massFlow outlets use atmospheric pressure
                new_patch['type'] = 'pressure_outlet'
                new_patch['pressure'] = 0
            
            new_patch['U'] = np.nan
    
    new_patch['T'] = simulation.get('temperature', DEFAULT_TEMPERATURE)
    
    # Extract wall normal vector
    wall_normal = np.array([
        data['position']['normal']['x'],
        data['position']['normal']['y'],
        data['position']['normal']['z']
    ])
    
    # Extract orientation angles from simulation data
    air_orientation = simulation.get('airOrientation', None)
    if air_orientation and air_orientation.get('verticalAngle') is not None and air_orientation.get('horizontalAngle') is not None:
        # Use orientation angles to compute directional flow vector
        vertical_angle = air_orientation.get('verticalAngle', 0.0)
        horizontal_angle = air_orientation.get('horizontalAngle', 0.0)
        
        # Convert angles to directional vector
        direction_vector = angles_to_direction_vector(vertical_angle, horizontal_angle, wall_normal)
        
        # Store directional components
        new_patch['nx'] = direction_vector[0]
        new_patch['ny'] = direction_vector[1]
        new_patch['nz'] = direction_vector[2]
    else:
        # No orientation angles specified, use wall normal directly
        new_patch['nx'] = wall_normal[0]
        new_patch['ny'] = wall_normal[1]
        new_patch['nz'] = wall_normal[2]
    
    return new_patch


def get_wall_bc_dict(id, temperature=DEFAULT_TEMPERATURE):
    # Sanitize patch ID for OpenFOAM compatibility
    new_patch = dict()
    new_patch['id'] = id.replace(' ', '_')
    new_patch['type'] = 'wall'
    new_patch['T'] = temperature
    return new_patch


def from_3d_to_wall2d(points_3d, p_origin, udir, vdir):
    """
    Transforms a batch of 3D points from a 3D coordinate system to a 2D wall coordinate system.

    Parameters:
    points (np.ndarray): A NumPy array of 3D points. Can be a 1D array for a single point
    p_origin (np.ndarray): A 1D NumPy array representing the origin of the wall in 3D space.
    udir (np.ndarray): A 1D NumPy array representing the u-direction (x-axis) of the wall.
    vdir (np.ndarray): A 1D NumPy array representing the v-direction (y-axis) of the wall.

    Returns:
    np.ndarray: A NumPy array of 2D points in the wall coordinate system.

    Raises:
    BaseException: If any 3D point is not on the wall surface (i.e., its z_wall component is not zero).
    """
    if points_3d.ndim == 1:
        points_3d = points_3d[np.newaxis, :]

    if points_3d.shape[1] != 3:
        raise ValueError("Each point in 'points_3d' must be a 3D point")
    
    p_origin = p_origin[np.newaxis, :]
    x_2d = np.dot(points_3d - p_origin, udir)[:, np.newaxis]
    y_2d = np.dot(points_3d - p_origin, vdir)[:, np.newaxis]
    z_2d = (points_3d - x_2d * udir - y_2d * vdir)[..., 2] - p_origin[:, 2]

    if not np.allclose(z_2d, 0):
        raise BaseException('At least one 3D point is not on the wall surface')
    
    return np.hstack([x_2d, y_2d])


def from_wall2d_to_3d(points_2d, p_origin, udir, vdir):
    # DEBUG: Log para entender el problema
    logger.info(f"    [DEBUG] from_wall2d_to_3d:")
    logger.info(f"      - Input shape: {points_2d.shape}")
    logger.info(f"      - Input dtype: {points_2d.dtype}")
    if len(points_2d) > 0:
        logger.info(f"      - First point: {points_2d[0]}")
    
    if points_2d.ndim == 1:
        points_2d = points_2d[np.newaxis, :]

    if points_2d.shape[1] != 2:
        logger.error(f"      ❌ ERROR: Expected 2 columns but got {points_2d.shape[1]}")
        logger.error(f"      - Full shape: {points_2d.shape}")
        raise ValueError("Each point in 'points_2d' must be a 2D point")

    return p_origin + np.outer(points_2d[:, 0], udir) + np.outer(points_2d[:, 1], vdir)


def create_single_entry(entry_data: Dict, base_height: float, p0: np.ndarray, udir: np.ndarray, vdir: np.ndarray) -> Tuple[Dict, shapely.Polygon]:
    """Process a single entry for FDM_iter2.json format."""
    # Convert 3D position to 2D wall coordinates
    centre_3d = np.array([
        entry_data['position']['x'], 
        entry_data['position']['y'], 
        entry_data['position']['z'] + base_height
    ])
    centre_2d = from_3d_to_wall2d(centre_3d, p0, udir, vdir)

    # Create rectangular polygon
    dimensions = entry_data['dimensions']
    width, height = dimensions['width'], dimensions['height']
    polygon = shapely.box(-width/2, -height/2, +width/2, +height/2)
    polygon = shapely.affinity.translate(polygon, xoff=centre_2d[0, 0], yoff=centre_2d[0, 1])
    
    return get_entry_bc_dict(entry_data), polygon


def create_entries(patch_df, entries_data, base_height, p0, udir, vdir):
    """Create air entries for FDM_iter2.json format."""
    entries_dict = {}
    for entry_data in entries_data:
        new_patch, polygon = create_single_entry(entry_data, base_height, p0, udir, vdir)
        patch_df = pd.concat([patch_df, pd.DataFrame([new_patch])], ignore_index=True)
        
        # ✅ FIX: Use sanitized ID (same as in patch_df) to avoid lookup errors
        # get_entry_bc_dict() sanitizes IDs by replacing spaces with underscores
        sanitized_id = entry_data['id'].replace(' ', '_')
        entries_dict[sanitized_id] = polygon
    
    return patch_df, entries_dict


def create_wall(patch_df: pd.DataFrame, data: Dict[str, Any], height: float, base_height: float) -> Tuple[pd.DataFrame, pv.PolyData]:
    """Optimized wall creation with parallel processing for entries."""
    p0 = np.array([data['start']['x'], data['start']['y'], base_height])
    p1 = np.array([data['end']['x'], data['end']['y'], base_height])
    udir = p1 - p0
    udir = udir / np.linalg.norm(udir)
    vdir = np.array([0, 0, 1])

    p0_top = p0 + height*vdir
    p1_top = p1 + height*vdir
    wall_points_3d = np.vstack([p0, p0_top, p1_top, p1, p0])
    wall_points_2d = from_3d_to_wall2d(wall_points_3d, p0, udir, vdir)
    wall_polygon = shapely.Polygon(wall_points_2d)

    new_patch = get_wall_bc_dict(data['id'], temperature=data['temp'])
    patch_df = pd.concat([patch_df, pd.DataFrame([new_patch])], ignore_index=True)
    patch_df, entries_dict = create_entries(patch_df, data['airEntries'], base_height, p0, udir, vdir)

    wall_meshes = []
    for entry_id, entry_polygon in entries_dict.items():
        wall_polygon = wall_polygon.difference(entry_polygon)
        entry_mesh = create_mesh_from_polygon(patch_df, entry_id, entry_polygon, p0, udir, vdir)
        wall_meshes.append(entry_mesh)

    wall_mesh = create_mesh_from_polygon(patch_df, data['id'], wall_polygon, p0, udir, vdir)
    wall_meshes.append(wall_mesh)
    
    # Merge meshes efficiently
    result_mesh = pv.merge(wall_meshes)
    return patch_df, result_mesh


def create_bound_surface(str_type: str, patch_df: pd.DataFrame, polygon: pv.PolyData, level_name: str, data: Dict[str, Any]) -> Tuple[pd.DataFrame, pv.PolyData]:
    floor_id = f"{str_type}_{level_name}F"
    new_patch = get_wall_bc_dict(floor_id, temperature=data['temp'])
    patch_df = pd.concat([patch_df, pd.DataFrame([new_patch])], ignore_index=True)

    z_floor = np.mean(polygon.points[:,2])
    p0 = np.array([0, 0, z_floor])
    udir = np.array([1, 0, 0])
    vdir = np.array([0, 1, 0])

    # Get the ordered vertex indices directly from faces
    floor_polygon = create_polygon_from_mesh(polygon)

    patch_df, entries_dict = create_entries(patch_df, data['airEntries'], 0, p0, udir, vdir)

    wall_meshes = []
    
    # CORRECT FIX: Usar unary_union para combinar todas las ventanas/puertas
    # Esto evita auto-intersecciones que ocurren con múltiples diferencias sucesivas
    if entries_dict:
        # Combinar todos los polígonos de ventanas/puertas en uno solo
        all_entries_union = shapely.unary_union(list(entries_dict.values()))
        
        # Hacer UNA sola operación de diferencia
        floor_polygon = floor_polygon.difference(all_entries_union)
    
    # Crear meshes para cada ventana/puerta
    for entry_id, entry_polygon in entries_dict.items():
        entry_mesh = create_mesh_from_polygon(patch_df, entry_id, entry_polygon, p0, udir, vdir)
        wall_meshes.append(entry_mesh)

    wall_mesh = create_mesh_from_polygon(patch_df, floor_id, floor_polygon, p0, udir, vdir)
    wall_meshes.append(wall_mesh)
    
    # Merge meshes efficiently
    result_mesh = pv.merge(wall_meshes)
    return patch_df, result_mesh


def create_polygon_from_mesh(mesh, target_z=None):
    """
    Extract a 2D polygon from a mesh boundary, filtering by Z coordinate.
    
    ROBUST FIX: Filtra puntos por su coordenada Z para evitar duplicados
    cuando el mesh contiene puntos de múltiples alturas (ej: floor y ceiling
    de las paredes que van de Z=0 a Z=height).
    
    Args:
        mesh: PyVista PolyData mesh
        target_z: Altura Z objetivo para filtrar puntos. Si es None, usa la mediana.
    
    Returns:
        Polígono 2D válido
    """
    # Detectar el Z objetivo si no se proporciona
    if target_z is None:
        z_values = mesh.points[:, 2]
        target_z = np.median(z_values)
        logger.info(f"    [DEBUG] target_z detectado: {target_z:.4f}")
    
    # Filtrar puntos con el Z objetivo (tolerancia pequeña)
    tolerance = 0.01
    mask = np.abs(mesh.points[:, 2] - target_z) < tolerance
    filtered_points = mesh.points[mask]
    
    logger.info(f"    [DEBUG] Puntos totales: {len(mesh.points)}, Puntos filtrados (Z≈{target_z:.4f}): {len(filtered_points)}")
    
    if len(filtered_points) < 3:
        logger.error(f"Insuficientes puntos después de filtrar por Z. Retornando polígono vacío.")
        return shapely.Polygon()
    
    try:
        # Extraer edges del mesh original pero usar solo los puntos filtrados
        edges = mesh.cells.reshape(-1,3)[:,1:]
        u, v = edges.T
        
        # Mapear índices originales a índices filtrados
        original_to_filtered = {}
        filtered_idx = 0
        for orig_idx in range(len(mesh.points)):
            if mask[orig_idx]:
                original_to_filtered[orig_idx] = filtered_idx
                filtered_idx += 1
        
        # Filtrar edges para que solo contengan puntos filtrados
        valid_edges = []
        for edge_u, edge_v in zip(u, v):
            if edge_u in original_to_filtered and edge_v in original_to_filtered:
                valid_edges.append((original_to_filtered[edge_u], original_to_filtered[edge_v]))
        
        if len(valid_edges) < 3:
            logger.warning(f"Insuficientes edges válidos. Usando convex_hull...")
            raise ValueError("Insuficientes edges")
        
        # Construir polígono desde edges filtrados
        adj = {}
        for u_idx, v_idx in valid_edges:
            adj[u_idx] = v_idx
        
        v = valid_edges[0][0]
        vert_idxs = [v]
        for _ in range(len(valid_edges)):
            v = adj[v]
            vert_idxs.append(v)
        
        polygon = shapely.force_2d(shapely.Polygon(filtered_points[vert_idxs]))
        
        # Validar que el polígono extraído es válido y tiene área significativa
        if polygon.is_valid and polygon.area > 1e-10:
            logger.info(f"    [DEBUG] Polígono extraído exitosamente: área={polygon.area:.6f}")
            return polygon
        else:
            logger.warning(f"Polígono extraído inválido o con área negligible ({polygon.area:.6f}). Usando convex_hull...")
            raise ValueError("Polígono inválido")
    
    except Exception as e:
        # Fallback: usar convex_hull de los puntos filtrados 2D
        logger.warning(f"Error extrayendo polígono: {e}. Usando convex_hull como fallback...")
        points_2d = filtered_points[:, :2]
        multipoint = shapely.MultiPoint(points_2d)
        polygon = shapely.convex_hull(multipoint)
        
        if polygon.is_valid and polygon.area > 1e-10:
            logger.info(f"    [DEBUG] Convex_hull exitoso: área={polygon.area:.6f}")
            return polygon
        else:
            logger.error(f"Convex_hull también falló. Retornando polígono vacío.")
            return shapely.Polygon()


def create_mesh_from_polygon(patch_df, entry_id, entry_polygon, p0, udir, vdir):
    points_2d = []
    faces = []
    point_index = {}
    next_index = 0

    # ROBUST FIX 1: Validar y reparar polígono inválido
    if not entry_polygon.is_valid:
        print(f"[WARNING] Polígono {entry_id} inválido. Reparando con make_valid()...")
        entry_polygon = shapely.make_valid(entry_polygon)
    
    # ROBUST FIX 2: Manejar polígonos vacíos o con área negligible
    if entry_polygon.is_empty or entry_polygon.area < 1e-10:
        print(f"[WARNING] Polígono {entry_id} vacío o con área negligible. Creando mesh vacío.")
        # Retornar mesh vacío
        points_2d = np.array([]).reshape(0, 2)
        faces = np.array([], dtype=int)
        points_3d = from_wall2d_to_3d(points_2d, p0, udir, vdir)
        mesh = pv.PolyData(points_3d, faces)
        patch_idx = patch_df.index[patch_df['id'] == entry_id][0].astype(np.int16)
        mesh.cell_data['patch_id'] = []
        return mesh
    
    triangles  = shapely.constrained_delaunay_triangles(entry_polygon)
    
    for triangle in triangles.geoms:
        coords = list(triangle.exterior.coords)[:-1]  # omit duplicate closing point
        face = [3]  # number of points in the triangle
        for x, y in coords:
            key = (x, y)
            if key not in point_index:
                point_index[key] = next_index
                points_2d.append([x, y])
                next_index += 1
            face.append(point_index[key])
        faces.extend(face)

    points_2d = np.array(points_2d)
    points_3d = from_wall2d_to_3d(points_2d, p0, udir, vdir)
    faces = np.array(faces)

    # Create PyVista PolyData
    mesh = pv.PolyData(points_3d, faces)
    patch_idx = patch_df.index[patch_df['id'] == entry_id][0].astype(np.int16)
    mesh.cell_data['patch_id'] = [patch_idx] * mesh.n_cells
    return mesh


def create_floor_mesh(patch_df: pd.DataFrame, level_name: str, level_data: Dict[str, Any], base_height: float = 0) -> Tuple[pd.DataFrame, pv.PolyData]:
    """Optimized floor mesh creation with parallel wall processing."""
    height = level_data["height"]

    # Create all walls and merge them
    wall_meshes = []
    for wall in level_data["walls"]:
        patch_df, wall_mesh = create_wall(patch_df, wall, height, base_height)
        wall_meshes.append(wall_mesh)
    walls_mesh = pv.merge(wall_meshes)
    walls_mesh.clean(tolerance=1e-3, absolute=True, inplace=True)

    # Process boundaries floor and ceiling
    wall_boundaries = walls_mesh.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False
    )
    del wall_boundaries.cell_data['patch_id']
    polygon_meshes = wall_boundaries.split_bodies()

    # 2. Compute the centroid (center of mass) of each part
    centroids = [part.center for part in polygon_meshes]
    sorted_polygons = [p for _, p in sorted(zip(centroids, polygon_meshes), key=lambda cp: cp[0][2])]
    
    # Validate that we have exactly 2 boundaries (floor and ceiling)
    if len(sorted_polygons) < 2:
        raise ValueError(
            f"Room does not form a closed volume. Expected 2 boundaries (floor + ceiling), "
            f"but found {len(sorted_polygons)}. This usually means:\n"
            f"  1. Walls are not connected at corners\n"
            f"  2. There are gaps between walls\n"
            f"  3. The room layout is incomplete\n"
            f"Please check that all walls connect properly to form a closed rectangular room."
        )
    
    polygon_floor = sorted_polygons[0]
    polygon_ceil = sorted_polygons[1]

    # Handle ceiling and floor_surf air entries
    if "floor" in level_data:
        patch_df, mesh_floor = create_bound_surface("floor", patch_df, polygon_floor, level_name, level_data["floor"])

    if "ceiling" in level_data:
        patch_df, mesh_ceil = create_bound_surface("ceil", patch_df, polygon_ceil, level_name, level_data["ceiling"])

    # Merge room meshes efficiently
    floor_mesh = pv.merge([walls_mesh, mesh_floor, mesh_ceil])
    floor_mesh.triangulate(inplace=True)
    floor_mesh.compute_normals(inplace=True, auto_orient_normals=True, consistent_normals=True, split_vertices=True, point_normals=False)
    
    # Optimize memory usage
    floor_mesh = optimize_mesh_memory(floor_mesh)
    
    return patch_df, floor_mesh


def create_stair_mesh(patch_df: pd.DataFrame, data: Dict[str, Any], current_base: float, floor_deck: float) -> Tuple[pd.DataFrame, pv.PolyData]:
    """Optimized stair mesh creation."""
    new_patch = get_wall_bc_dict(data['id'])
    patch_df = pd.concat([patch_df, pd.DataFrame([new_patch])], ignore_index=True)

    points = []
    connectivity = []
    for i, line in enumerate(data['lines']):
        # TODO: TELL JUAN THAT NO LINE ID IS NEEDED IN STAIR ELEMENT
        p0 = np.array([line['start']['x'], line['start']['y'], current_base - VOLUMES_TOLERANCE])
        p1 = np.array([line['end']['x'], line['end']['y'], current_base - VOLUMES_TOLERANCE])
        if len(points) == 0:
            points = np.vstack((p0, p1))
        else:
            points = np.vstack((points, p0))
            points = np.vstack((points, p1))
        connectivity.extend([2, 2*i, 2*i + 1])

    poly_mesh = pv.PolyData()
    poly_mesh.points = points
    poly_mesh.lines = connectivity
    poly_mesh.clean(inplace=True)
    poly_filled = poly_mesh.triangulate_contours()

    patch_idx = patch_df.index[patch_df['id'] == data['id']][0].astype(np.int16)

    stair_mesh = poly_filled.extrude([0, 0, floor_deck + 2*VOLUMES_TOLERANCE], capping=True)
    stair_mesh.compute_normals(inplace=True, auto_orient_normals=True, consistent_normals=True, split_vertices=True, point_normals=False)
    stair_mesh.triangulate(inplace=True)
    stair_mesh.cell_data['patch_id'] = [patch_idx] * stair_mesh.n_cells
    
    # Optimize memory usage
    stair_mesh = optimize_mesh_memory(stair_mesh)
    
    return patch_df, stair_mesh


def create_volumes(building_config_data: Dict[str, Any], valid_floors: List[str] = None) -> Tuple[List[pv.PolyData], List[pv.PolyData], pd.DataFrame]:
    """
    Create 3D room geometry and furniture from building configuration data.
    
    This function processes each floor level to create:
    - Room geometry (walls, floors, ceilings) 
    - Stair connections between floors
    - Furniture objects within each room
    
    Args:
        building_config_data: JSON data containing building floor definitions
        valid_floors: List of valid floor names to process (from JSON validation).
                     If None, processes all floors (backward compatibility).
        
    Returns:
        Tuple of (room_geometry_meshes, furniture_meshes, boundary_conditions_df)
    """
    performance_monitor = PerformanceMonitor()
    performance_monitor.start()
    
    # If valid_floors not provided, process all floors (backward compatibility)
    if valid_floors is None:
        valid_floors = sorted(building_config_data['levels'].keys(), key=lambda x: int(x))
        logger.info(f"    * No valid_floors provided, processing all {len(valid_floors)} floors")
    
    # Track current floor elevation for proper stacking
    current_floor_elevation = 0.0

    # Initialize data structures for geometry creation
    boundary_conditions_df = pd.DataFrame()
    room_geometry_meshes = []
    furniture_meshes = []
    
    # Track stair tubes from previous floor for subtraction
    previous_stair_tubes = []

    logger.info(f"    * Processing {len(valid_floors)} valid floor levels")
    logger.info(f"    * Using LAYER-BASED ARCHITECTURE for robust geometry creation\n")

    for idx, floor_name in enumerate(valid_floors):
        floor_config = building_config_data["levels"][floor_name]
        logger.info(f"    * Creating geometry for floor #{floor_name}")
        performance_monitor.update_memory()
        
        floor_deck_thickness = floor_config["deck"]
        floor_height = floor_config["height"]
        
        # Detect if this is the top floor
        is_top_floor = (idx == len(valid_floors) - 1)
        
        # CREATE ROOM GEOMETRY using layer-based architecture
        # This replaces create_floor_mesh() and integrates stair creation
        from src.components.geo.create_volumes_layered import create_floor_mesh_layered
        
        boundary_conditions_df, room_mesh, current_stair_tubes = create_floor_mesh_layered(
            boundary_conditions_df, floor_name, floor_config, 
            base_height=current_floor_elevation,
            previous_stair_tubes=previous_stair_tubes,
            is_top_floor=is_top_floor
        )
        room_geometry_meshes.append(room_mesh)
        
        # Save stair tubes for next floor
        previous_stair_tubes = current_stair_tubes

        # ADD FURNITURE OBJECTS
        for furniture_config in floor_config.get("furniture", []):
            boundary_conditions_df, furniture_mesh = create_furniture_mesh(
                boundary_conditions_df, furniture_config
            )
            furniture_meshes.append(furniture_mesh)
        
        # Update elevation for next floor
        current_floor_elevation += floor_height + floor_deck_thickness
        performance_monitor.update_memory()

    # Log performance summary
    performance_summary = performance_monitor.get_summary()
    logger.info(f"    * Room geometry creation completed: {performance_summary['total_time']:.2f}s, peak memory: {performance_summary['peak_memory_mb']:.1f}MB")
    
    return room_geometry_meshes, furniture_meshes, boundary_conditions_df
