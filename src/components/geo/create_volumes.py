import os
import shapely
import numpy as np
import pandas as pd
import pyvista as pv
import logging
from pathlib import Path

from typing import List, Tuple, Dict, Any
from src.components.tools.performance import (
    optimize_mesh_memory,
    PerformanceMonitor
)

logger = logging.getLogger(__name__)

# PROJECT_ROOT: Absolute path to project root (3 levels up from this file)
# This ensures correct paths regardless of execution directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

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
    # DIAGNOSTIC TEST: All set to 0 to test if ΔP is causing the crash
    'pressure': {
        'low': 0.3,     # Pa - Low pressure differential
        'medium': 5,    # Pa - Medium pressure differential  
        'high': 25      # Pa - High pressure differential
    }
}

# Legacy flow levels for backward compatibility (velocity in m/s)
FLOW_LEVELS = {
    'low':      0.5,
    'medium':   1.0,
    'high':     2.0,
}

# ── Central normalisation map: accepts any front-end casing ──────────────────
_FLOW_TYPE_NORMALIZE: Dict[str, str] = {
    'velocity':  'velocity',
    'massflow':  'massFlow',
    'massFlow':  'massFlow',
    'pressure':  'pressure',
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


def _build_vent_bc(
    patch_id: str,
    temperature: float,
    air_direction: str,
    flow_type_raw: str,
    flow_intensity: str,
    custom_value: float = None,
) -> dict:
    """
    Single source of truth for open-vent BC dicts.

    Supports flowType: velocity | massFlow | pressure.
    Supports airDirection: inflow | outflow | equilibrium.
    Used by both wall/floor/ceiling airEntries (get_entry_bc_dict)
    and face-based furniture vents (create_face_based_mesh).
    """
    flow_type = _FLOW_TYPE_NORMALIZE.get(flow_type_raw, 'pressure')
    bc = {'id': patch_id.replace(' ', '_'), 'T': temperature, 'open': True}

    if air_direction == 'inflow':
        if flow_type == 'massFlow':
            bc['type']     = 'mass_flow_inlet'
            bc['massFlow'] = get_flow_value(flow_intensity, 'massFlow', custom_value)
            bc['U']        = np.nan
        elif flow_type == 'velocity':
            bc['type'] = 'velocity_inlet'
            bc['U']    = get_flow_value(flow_intensity, 'velocity', custom_value)
        else:  # pressure
            bc['type']     = 'pressure_inlet'
            bc['pressure'] = get_flow_value(flow_intensity, 'pressure', custom_value)
            bc['U']        = np.nan
    elif air_direction == 'equilibrium':
        # Neutral boundary: 0 Pa gauge, allows bidirectional flow driven by internal dynamics.
        # Equivalent to pressure_outlet at 0 Pa: pressureDirectedInletOutletVelocity handles
        # both inflow (constrained to inletDirection) and outflow (zeroGradient) automatically.
        bc['type']     = 'pressure_outlet'
        bc['pressure'] = 0.0
        bc['U']        = np.nan
    else:  # outflow → always pressure_outlet as reference boundary
        bc['type']     = 'pressure_outlet'
        bc['pressure'] = (
            -get_flow_value(flow_intensity, 'pressure', custom_value)
            if flow_type == 'pressure' else 0.0
        )
        bc['U'] = np.nan

    return bc


def angles_to_direction_vector(vertical_angle: float, horizontal_angle: float, wall_normal: np.ndarray) -> np.ndarray:
    """
    Convert orientation angles to 3D direction vector for air flow.

    Local coordinate system at the vent face (wall_normal = inward direction):
        forward  = wall_normal (inward, perpendicular to wall)
        up_wall  = vertical tangent of the wall surface (≈ global Z for vertical walls)
        right    = cross(forward, up_wall)  (horizontal tangent)

    Angle conventions:
        vertical_angle   > 0 → downward deflection (chorro baja)
                         < 0 → upward  deflection (chorro sube)
        horizontal_angle > 0 → leftward  deflection (chorro gira a la izquierda)
                         < 0 → rightward deflection (chorro gira a la derecha)

    Args:
        vertical_angle:   Vertical  tilt  in degrees (–45 … +45)
        horizontal_angle: Horizontal yaw   in degrees (–45 … +45)
        wall_normal:      Inward normal vector [nx, ny, nz] (toward room interior)

    Returns:
        Normalized 3D direction vector for the air flow
    """
    v_rad = np.deg2rad(vertical_angle)
    h_rad = np.deg2rad(horizontal_angle)

    # Normalise inward normal
    normal = np.array(wall_normal, dtype=float)
    normal = normal / np.linalg.norm(normal)

    # Build local orthogonal frame at the vent face
    world_up = np.array([0, 0, 1]) if abs(normal[2]) < 0.9 else np.array([1, 0, 0])
    right    = np.cross(normal, world_up);  right    /= np.linalg.norm(right)
    up_wall  = np.cross(right,  normal);    up_wall  /= np.linalg.norm(up_wall)

    # ── Step 1: Vertical pitch (rotate around 'right' axis) ─────────────────
    # v > 0 → down  →  subtract up_wall component
    # v < 0 → up    →  add      up_wall component
    cos_v = np.cos(v_rad)
    sin_v = np.sin(v_rad)
    direction = cos_v * normal - sin_v * up_wall

    # ── Step 2: Horizontal yaw (rotate around 'up_wall' axis) ───────────────
    # Rodrigues' formula around up_wall (≈ Z for vertical walls):
    #   cross(up_wall, forward) = right-like vector → h > 0 adds a component
    #   that points LEFT (positive convention).
    cos_h = np.cos(h_rad)
    sin_h = np.sin(h_rad)
    direction = (direction * cos_h
                 + np.cross(up_wall, direction) * sin_h
                 + up_wall * np.dot(up_wall, direction) * (1 - cos_h))

    return direction / np.linalg.norm(direction)


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
# None = no STL needed (programmatic or face-based)
OBJECT_TYPE_MAPPING = {
    'person':      2,    # h_standing (HUMANS)
    'block':       None, # box from position + dimensions (new format) or scale (legacy)
    'table':       12,   # table_round (TABLES)
    'armchair':    21,   # chair_basic (CHAIRS)
    'rack':        None, # face-based → create_face_based_mesh
    'topVentBox':  None, # face-based → create_face_based_mesh
    'sideVentBox': None, # face-based → create_face_based_mesh
}

# Face-based object types (have explicit face vertices + per-face BCs)
FACE_BASED_TYPES = {'rack', 'topVentBox', 'sideVentBox'}


def create_cube_mesh(width=1.0, height=1.0, depth=1.0):
    """Create a programmatic cube mesh with specified dimensions."""
    cube = pv.Cube(x_length=width, y_length=depth, z_length=height)
    return cube.triangulate().clean()


def create_block_mesh(patch_df: pd.DataFrame, data: Dict[str, Any]) -> Tuple[pd.DataFrame, pv.PolyData]:
    """
    Create a box mesh for 'block' type furniture using the new JSON format.

    New format: position + dimensions{width, height, depth} + simulationProperties
    Legacy format: position + scale{x, y, z}  (still supported for backward compatibility)

    Physical convention:
        - position.{x,y,z}: corner at bottom face (z = floor level)
        - dimensions.width  → X dimension
        - dimensions.depth  → Y dimension
        - dimensions.height → Z dimension (vertical)

    All 6 faces share a single patch_id (uniform wall BC).
    """
    patch_id = data['id'].replace(' ', '_')

    # --- Boundary condition (all faces are wall) ---
    sim = data.get('simulationProperties', {})
    new_patch = get_wall_bc_dict(
        patch_id,
        temperature=sim.get('temperature', DEFAULT_TEMPERATURE),
        emissivity=sim.get('emissivity', 0.9),
        material=sim.get('material', 'default')
    )
    patch_df = pd.concat([patch_df, pd.DataFrame([new_patch])], ignore_index=True)
    patch_idx = patch_df.index[patch_df['id'] == patch_id][0].astype(np.int16)

    # --- Dimensions: prefer new 'dimensions' field, fall back to legacy 'scale' ---
    if 'dimensions' in data:
        dims = data['dimensions']
        w = dims['width']
        d = dims['depth']
        h = dims['height']
    else:
        # Legacy: scale → {x, y, z}
        scale = data.get('scale', {'x': 1.0, 'y': 1.0, 'z': 1.0})
        w, d, h = scale['x'], scale['y'], scale['z']

    # --- Position ---
    pos = data['position']
    cx, cy, cz = pos['x'], pos['y'], pos['z']

    # --- Rotation (radians → degrees for PyVista) ---
    rot = data.get('rotation', {})
    rx_rad = rot.get('x', 0.0)
    ry_rad = rot.get('y', 0.0)
    rz_rad = rot.get('z', 0.0)
    rx_deg = np.degrees(rx_rad)
    ry_deg = np.degrees(ry_rad)
    rz_deg = np.degrees(rz_rad)

    # Ceiling-mounted detection: rotation.x ≈ π means object is upside-down
    # position.z = TOP face (ceiling attachment) → center goes DOWN from cz
    ceiling_mounted = abs(rx_rad) > (np.pi / 2)
    if ceiling_mounted:
        center_z = cz - h / 2
        logger.info(f"      ↑ ceiling-mounted block (rx={rx_rad:.4f} rad): center_z={center_z:.3f}")
    else:
        center_z = cz + h / 2 + FURNITURE_FLOOR_PENETRATION

    # pv.Cube is centred at origin → build at (cx, cy, center_z) then rotate
    cube = pv.Cube(x_length=w, y_length=d, z_length=h)
    cube.translate([cx, cy, center_z], inplace=True)

    # Apply rotations around the block center (cx, cy, center_z)
    center_pt = [cx, cy, center_z]
    if rx_deg:
        cube.rotate_x(rx_deg, point=center_pt, inplace=True)
    if ry_deg:
        cube.rotate_y(ry_deg, point=center_pt, inplace=True)
    if rz_deg:
        cube.rotate_z(rz_deg, point=center_pt, inplace=True)

    cube = cube.triangulate().clean().extract_surface()
    cube.compute_normals(
        inplace=True, auto_orient_normals=True,
        consistent_normals=True, split_vertices=True, point_normals=False
    )
    cube.cell_data['patch_id'] = [patch_idx] * cube.n_cells

    logger.info(f"      ✓ Block mesh: {w:.2f}×{d:.2f}×{h:.2f}m, {cube.n_cells} cells")
    return patch_df, optimize_mesh_memory(cube)


def _make_quad_mesh(vertices: list) -> pv.PolyData:
    """Create a quad face from 4 ordered vertices (2 triangles)."""
    pts = np.array(vertices, dtype=float)  # (4, 3)
    faces = np.array([3, 0, 1, 2,  3, 0, 2, 3])
    return pv.PolyData(pts, faces)


def create_face_based_mesh(patch_df: pd.DataFrame, data: Dict[str, Any]) -> Tuple[pd.DataFrame, pv.PolyData]:
    """
    Create mesh for face-based furniture: rack, topVentBox, sideVentBox.

    Each face in data['faces'] has explicit vertices and a 'role' that determines
    the OpenFOAM BC type:

        wall    → wall BC with temperature, emissivity, material
        inlet   → pressure_outlet (rack sucks cold air FROM room)
        outlet  → mass_flow_inlet (rack blows hot air INTO room)
        vent    → pressure_inlet / pressure_outlet based on airDirection

    Each face gets its own patch_id: "{obj_id}_{face_name}"
    """
    obj_id = data['id'].replace(' ', '_')
    faces_data = data['faces']
    face_meshes = []

    for face_name, fd in faces_data.items():
        role        = fd['role']
        temperature = fd.get('temperature', DEFAULT_TEMPERATURE)
        face_pid    = f"{obj_id}_{face_name}"

        # --- Build BC dict based on role ---
        if role == 'wall':
            new_patch = get_wall_bc_dict(
                face_pid,
                temperature=temperature,
                emissivity=fd.get('emissivity', 0.9),
                material=fd.get('material', 'default')
            )

        elif role == 'inlet':
            # Rack cold-air intake: air LEAVES the room domain through this face.
            # flowRateInletVelocity with negative volumetricFlowRate = outflow from domain.
            # T: zeroGradient — takes temperature from the interior flow.
            new_patch = {
                'id':       face_pid,
                'type':     'rack_inlet',
                'T':        temperature,
                'massFlow': fd.get('airFlow', 0),   # m³/h — used for U BC
                'open':     True,
            }

        elif role == 'outlet':
            # Rack hot-air exhaust: hot air ENTERS the room domain from the rack.
            # flowRateInletVelocity with positive volumetricFlowRate = inflow into domain.
            # T: codedFixedValue that reads average T of corresponding inlet patch and adds ΔT.
            # Find the inlet face name in this rack (first face with role='inlet')
            inlet_face_name = next(
                (name for name, fdata in faces_data.items() if fdata.get('role') == 'inlet'),
                'front'  # fallback if no explicit inlet face
            )
            inlet_patch_id = f"{obj_id}_{inlet_face_name}"
            new_patch = {
                'id':           face_pid,
                'type':         'rack_outlet',
                'T':            temperature,
                'massFlow':     fd.get('airFlow', 0),                  # m³/h
                'thermalPower': fd.get('thermalPower_kW', 0) * 1000.0, # W
                'inlet_id':     inlet_patch_id,                        # ref to corresponding inlet
                'open':         True,
            }

        elif role == 'vent':
            # Closed vent face → wall BC (same as closed airEntry)
            if fd.get('state', 'open') == 'closed':
                new_patch = get_wall_bc_dict(
                    face_pid,
                    temperature=temperature,
                    emissivity=fd.get('emissivity', 0.9),
                    material=fd.get('material', 'default')
                )
                logger.info(f"      • face {face_name} (vent/closed) → wall BC")
            else:
                # Open vent – delegate to shared helper
                new_patch = _build_vent_bc(
                    patch_id       = face_pid,
                    temperature    = temperature,
                    air_direction  = fd.get('airDirection', 'inflow'),
                    flow_type_raw  = fd.get('flowType', 'pressure'),
                    flow_intensity = fd.get('flowIntensity', 'low'),
                    custom_value   = fd.get('customIntensityValue', None),
                )
        else:
            logger.warning(f"Unknown face role '{role}' in {face_pid}, treating as wall")
            new_patch = get_wall_bc_dict(face_pid, temperature=temperature)

        # Add to DataFrame and get index
        patch_df  = pd.concat([patch_df, pd.DataFrame([new_patch])], ignore_index=True)
        patch_idx = patch_df.index[patch_df['id'] == face_pid][0].astype(np.int16)

        # Build quad mesh for this face and tag it
        face_mesh = _make_quad_mesh(fd['vertices'])
        face_mesh.cell_data['patch_id'] = [patch_idx, patch_idx]
        face_meshes.append(face_mesh)

        logger.info(f"      • face {face_name} ({role}) → patch '{face_pid}'")

    # Merge all faces into a single PolyData
    result = pv.merge(face_meshes)
    result = result.triangulate().extract_surface()
    result.compute_normals(
        inplace=True, auto_orient_normals=True,
        consistent_normals=True, split_vertices=True, point_normals=False
    )
    logger.info(f"      ✓ Face-based mesh '{obj_id}': {result.n_cells} cells, {len(faces_data)} faces")
    return patch_df, optimize_mesh_memory(result)


def create_furniture_mesh(patch_df: pd.DataFrame, data: Dict[str, Any]) -> Tuple[pd.DataFrame, pv.PolyData]:
    """
    Dispatcher: routes furniture to the appropriate mesh builder based on JSON structure.

    Routing rules (checked in order):
      1. data has 'faces' key  → create_face_based_mesh  (rack, topVentBox, sideVentBox)
      2. object type is 'block' (from ID)  → create_block_mesh  (box geometry)
      3. otherwise  → STL-based legacy path (person, table, armchair)
    """
    # Parse object type from ID: "object_0F_rack_1" → "rack"
    id_parts    = data['id'].split('_')
    object_type = id_parts[2] if len(id_parts) >= 3 else 'unknown'

    logger.info(f"    * Creating furniture: {data['id']} (type: {object_type})")

    # Route 1: face-based objects
    if 'faces' in data or object_type in FACE_BASED_TYPES:
        return create_face_based_mesh(patch_df, data)

    # Route 2: block with position + dimensions (new) or position + scale (legacy)
    if object_type == 'block':
        return create_block_mesh(patch_df, data)

    # Route 3: STL-based legacy objects (person, table, armchair, …)
    patch_id = data['id'].replace(' ', '_')
    sim = data.get('simulationProperties', {})
    new_patch = get_wall_bc_dict(
        patch_id,
        temperature=sim.get('temperature', DEFAULT_TEMPERATURE),
        emissivity=sim.get('emissivity', 0.9),
        material=sim.get('material', 'default')
    )
    patch_df  = pd.concat([patch_df, pd.DataFrame([new_patch])], ignore_index=True)
    patch_idx = patch_df.index[patch_df['id'] == patch_id][0].astype(np.int16)

    pos = data['position']
    translation = np.array([pos['x'], pos['y'], pos['z']])

    # Rotation: JSON stores radians, PyVista expects degrees
    rot = data.get('rotation', {})
    rz_deg = np.degrees(rot.get('z', 0.0))

    mesh_id = OBJECT_TYPE_MAPPING.get(object_type)
    if mesh_id is None:
        logger.warning(f"      ⚠️  Unknown type '{object_type}', defaulting to table_round (id=11)")
        mesh_id = 11

    mesh_path_info = ELEMENTS_MESHES[mesh_id]
    mesh_path = PROJECT_ROOT / 'data' / 'CAD_database' / mesh_path_info[1] / f"{mesh_path_info[0]}.stl"
    logger.info(f"      → Loading STL: {mesh_path_info[0]} from {mesh_path_info[1]}")

    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Furniture STL not found: {mesh_path}")

    obj_mesh = pv.read(mesh_path)

    # Step 1: Normalize – translate STL base to origin (z_min → 0)
    bounds = obj_mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
    obj_mesh.translate(
        [-((bounds[0] + bounds[1]) / 2),   # centre X
         -((bounds[2] + bounds[3]) / 2),   # centre Y
         -bounds[4]],                        # base Z → 0
        inplace=True
    )

    # Step 2: Scale to target dimensions
    if 'dimensions' in data:
        dims = data['dimensions']
        target_w = dims['width']
        target_d = dims['depth']
        target_h = dims['height']
        b = obj_mesh.bounds
        stl_w = b[1] - b[0]  # X extent after normalisation
        stl_d = b[3] - b[2]  # Y extent
        stl_h = b[5] - b[4]  # Z extent
        sx = target_w / stl_w if stl_w > 1e-9 else 1.0
        sy = target_d / stl_d if stl_d > 1e-9 else 1.0
        sz = target_h / stl_h if stl_h > 1e-9 else 1.0
        obj_mesh.scale([sx, sy, sz], inplace=True)
        logger.info(f"      → Scaled STL to {target_w:.2f}×{target_d:.2f}×{target_h:.2f}m")
    else:
        # Legacy: explicit scale dict or default 1:1:1
        scale = data.get('scale', {'x': 1.0, 'y': 1.0, 'z': 1.0})
        obj_mesh.scale([scale['x'], scale['y'], scale['z']], inplace=True)

    # Step 3: Rotate around Z (degrees), then translate to final position
    if rz_deg:
        obj_mesh.rotate_z(rz_deg, inplace=True)
    obj_mesh.translate(
        xyz=[translation[0], translation[1], translation[2] + FURNITURE_FLOOR_PENETRATION],
        inplace=True
    )

    obj_mesh = obj_mesh.clean().triangulate().extract_surface()
    obj_mesh.compute_normals(
        inplace=True, auto_orient_normals=True,
        consistent_normals=True, split_vertices=True, point_normals=False
    )
    obj_mesh.cell_data['patch_id'] = [patch_idx] * obj_mesh.n_cells
    logger.info(f"      ✓ STL mesh created: {obj_mesh.n_points} pts, {obj_mesh.n_cells} cells")
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
        # Closed entry behaves like a wall - use emissivity/material from simulation if available
        return {
            'id': patch_id,
            'type': 'wall',
            'T': simulation.get('temperature', DEFAULT_TEMPERATURE),
            'emissivity': simulation.get('emissivity', 0.9),
            'material': simulation.get('material', 'default')
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
        elif air_direction == 'equilibrium':
            # Neutral boundary: 0 Pa gauge, lets internal dynamics determine flow direction.
            # pressureDirectedInletOutletVelocity handles bidirectional flow automatically.
            new_patch['type'] = 'pressure_outlet'
            new_patch['pressure'] = 0.0
        else:  # outflow
            # Pressure outlet: p = p_internal - ΔP (lower pressure pulls air out)
            new_patch['type'] = 'pressure_outlet'
            new_patch['pressure'] = -delta_p  # Negative pressure differential
        
        new_patch['U'] = np.nan  # Not used for pressure BCs
    
    # For vents: User specifies flowType (velocity, massFlow, or pressure)
    else:
        new_patch.update(_build_vent_bc(
            patch_id       = patch_id,
            temperature    = simulation.get('temperature', DEFAULT_TEMPERATURE),
            air_direction  = air_direction,
            flow_type_raw  = simulation.get('flowType', 'velocity'),
            flow_intensity = flow_intensity,
            custom_value   = custom_value,
        ))
    
    new_patch['T'] = simulation.get('temperature', DEFAULT_TEMPERATURE)

    # Compute geometric area of the entry [m²] from dimensions
    # Used in hvac.py to convert massFlow [m³/h] → velocity [m/s] with prescribed direction
    dims_for_area = data.get('dimensions', {})
    if dims_for_area.get('shape') == 'circular':
        import math
        new_patch['area'] = math.pi * (dims_for_area['diameter'] / 2.0) ** 2
    else:
        w = dims_for_area.get('width', 1.0)
        h = dims_for_area.get('height', 1.0)
        new_patch['area'] = w * h  # [m²]

    # Face normal (from JSON position.normal) — OUTWARD (pointing toward exterior)
    wall_normal = np.array([
        data['position']['normal']['x'],
        data['position']['normal']['y'],
        data['position']['normal']['z']
    ])

    # nx/ny/nz = JSON outward normal placeholders.
    # finalise_geometry.py overwrites them with mesh-computed INWARD normals (single source of truth).
    new_patch['nx']      = wall_normal[0]
    new_patch['ny']      = wall_normal[1]
    new_patch['nz']      = wall_normal[2]
    # json_nx/ny/nz = raw JSON outward normal kept for coherence check in finalise_geometry.py
    new_patch['json_nx'] = wall_normal[0]
    new_patch['json_ny'] = wall_normal[1]
    new_patch['json_nz'] = wall_normal[2]

    # ── Flow direction — required field for all open entries ─────────────────
    # simulation.flowDirection {x, y, z} must be exported by the frontend and
    # must match the green arrows drawn on the canvas.
    # Raises ValueError immediately if missing — no silent fallback.
    flow_dir = simulation.get('flowDirection', None)
    if (not flow_dir
            or flow_dir.get('x') is None
            or flow_dir.get('y') is None
            or flow_dir.get('z') is None):
        raise ValueError(
            f"Open entry '{patch_id}' is missing simulation.flowDirection. "
            f"All open airEntries must export flowDirection {{x, y, z}} from the frontend."
        )
    new_patch['fd_x'] = float(flow_dir['x'])
    new_patch['fd_y'] = float(flow_dir['y'])
    new_patch['fd_z'] = float(flow_dir['z'])

    # fluid_nx/ny/nz: NaN placeholders — written by finalise_geometry.py
    new_patch['fluid_nx'] = np.nan
    new_patch['fluid_ny'] = np.nan
    new_patch['fluid_nz'] = np.nan

    return new_patch


def get_wall_bc_dict(id, temperature=DEFAULT_TEMPERATURE, emissivity=0.9, material='default'):
    """Create boundary condition dictionary for wall patches.
    
    Args:
        id: Patch identifier (will be sanitized for OpenFOAM)
        temperature: Wall surface temperature in °C
        emissivity: Surface emissivity for radiation model (0-1), default 0.9
        material: Material name for reference, default 'default'
    """
    # Sanitize patch ID for OpenFOAM compatibility
    new_patch = dict()
    new_patch['id'] = id.replace(' ', '_')
    new_patch['type'] = 'wall'
    new_patch['T'] = temperature
    new_patch['emissivity'] = emissivity
    new_patch['material'] = material
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

    # Create polygon based on shape (circular or rectangular)
    dimensions = entry_data['dimensions']
    shape = dimensions.get('shape', 'rectangular')
    rotation_deg = entry_data.get('position', {}).get('rotation', 0.0)  # in-plane rotation [degrees]

    if shape == 'circular':
        radius = dimensions['diameter'] / 2.0
        polygon = shapely.Point(0, 0).buffer(radius, resolution=32)
    else:  # rectangular (default)
        width = dimensions['width']
        height = dimensions['height']
        polygon = shapely.box(-width/2, -height/2, +width/2, +height/2)
        if rotation_deg != 0.0:
            polygon = shapely.affinity.rotate(polygon, rotation_deg, origin=(0, 0))

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

    new_patch = get_wall_bc_dict(
        data['id'],
        temperature=data['temp'],
        emissivity=data.get('emissivity', 0.9),
        material=data.get('material', 'default')
    )
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
    new_patch = get_wall_bc_dict(
        floor_id,
        temperature=data['temp'],
        emissivity=data.get('emissivity', 0.9),
        material=data.get('material', 'default')
    )
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
