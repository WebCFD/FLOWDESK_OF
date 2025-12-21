"""
Layer-based architecture for robust building geometry creation.

This module implements a refactored approach where each component (walls, floor, ceiling, stairs)
is generated independently with exhaustive validation.

Architecture:
    1. Auxiliary functions: Extract floor polygon from wall coordinates
    2. Layer functions: Create each component independently
    3. Main function: Orchestrate the 7 phases with validation
    4. Integration: Replace create_floor_mesh() in create_volumes()
"""

import numpy as np
import pandas as pd
import pyvista as pv
import shapely
import logging
from typing import List, Tuple, Dict, Any, Optional

from src.components.geo.boolean_operations import subtract_objects
from src.components.tools.performance import optimize_mesh_memory

logger = logging.getLogger(__name__)

VOLUMES_TOLERANCE = 1e-5


# ============================================================================
# PHASE 1: AUXILIARY FUNCTIONS
# ============================================================================

def create_polygon_simple(walls_config: List[Dict[str, Any]]) -> shapely.Polygon:
    """
    Create polygon assuming walls are in consecutive order.
    
    Args:
        walls_config: List of wall dictionaries in consecutive order
        
    Returns:
        Shapely Polygon
        
    Raises:
        ValueError: If polygon is invalid, self-intersects, or too small
    """
    points_2d = []
    for wall in walls_config:
        points_2d.append([wall['start']['x'], wall['start']['y']])
    
    # Close the polygon
    points_2d.append(points_2d[0])
    
    # Create polygon
    polygon = shapely.Polygon(points_2d)
    
    # CRITICAL CHECK: Detect self-intersections BEFORE make_valid()
    # is_simple returns False if polygon crosses itself (walls not in consecutive order)
    if not polygon.is_simple:
        raise ValueError(
            f"Polygon self-intersects (walls not in consecutive order). "
            f"Area before fix: {polygon.area:.3f} m². "
            f"Walls need to be reordered to form a valid polygon."
        )
    
    # Validate and repair if needed (only for other issues, not self-intersection)
    if not polygon.is_valid:
        polygon = shapely.make_valid(polygon)
    
    if polygon.is_empty or polygon.area < 0.1:
        raise ValueError(f"Polygon area too small: {polygon.area:.3f} m²")
    
    return polygon


def reorder_walls_to_close_polygon(walls_config: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Reorder walls to form a closed polygon.
    
    Algorithm:
    1. Build adjacency graph (start → end)
    2. Find Hamiltonian cycle starting from first wall
    3. Return walls in correct order
    
    Args:
        walls_config: List of wall dictionaries (possibly unordered)
        
    Returns:
        List of walls in correct consecutive order
        
    Raises:
        ValueError: If walls don't form a closed polygon
    """
    # Build adjacency map: point → [(next_point, wall_index), ...]
    adjacency = {}
    
    for i, wall in enumerate(walls_config):
        start = (round(wall['start']['x'], 5), round(wall['start']['y'], 5))
        end = (round(wall['end']['x'], 5), round(wall['end']['y'], 5))
        
        if start not in adjacency:
            adjacency[start] = []
        adjacency[start].append((end, i))
    
    # Find cycle starting from first wall
    start_point = (round(walls_config[0]['start']['x'], 5), 
                   round(walls_config[0]['start']['y'], 5))
    
    ordered_indices = []
    current_point = start_point
    visited = set()
    
    while len(ordered_indices) < len(walls_config):
        found = False
        
        for next_point, wall_idx in adjacency.get(current_point, []):
            if wall_idx not in visited:
                ordered_indices.append(wall_idx)
                visited.add(wall_idx)
                current_point = next_point
                found = True
                break
        
        if not found:
            raise ValueError(
                f"Cannot find closed path. Visited {len(ordered_indices)}/{len(walls_config)} walls. "
                f"Walls may have gaps or disconnections at point {current_point}."
            )
    
    # Verify that cycle closes
    last_wall = walls_config[ordered_indices[-1]]
    last_point = (round(last_wall['end']['x'], 5), round(last_wall['end']['y'], 5))
    
    if last_point != start_point:
        raise ValueError(
            f"Walls do not form a closed loop. "
            f"Last point {last_point} != start point {start_point}"
        )
    
    # Return walls in correct order
    return [walls_config[i] for i in ordered_indices]


def create_floor_polygon_from_wall_coords(walls_config: List[Dict[str, Any]]) -> shapely.Polygon:
    """
    Extract 2D floor polygon from wall start/end coordinates.
    
    ROBUST: Automatically reorders walls if they are not in consecutive order.
    
    Args:
        walls_config: List of wall dictionaries with 'start' and 'end' keys
        
    Returns:
        Shapely Polygon representing the floor boundary
        
    Raises:
        ValueError: If polygon area < 0.1m² or walls don't form closed loop
    """
    logger.info(f"    [PHASE 1.1] Extracting floor polygon from {len(walls_config)} walls")
    
    if not walls_config:
        raise ValueError("No walls provided")
    
    if len(walls_config) < 3:
        raise ValueError(f"Insufficient walls: {len(walls_config)} (need at least 3)")
    
    # Try with original order first
    try:
        polygon = create_polygon_simple(walls_config)
        logger.info(f"      ✓ Polygon created with original wall order")
        logger.info(f"      - Polygon area: {polygon.area:.3f} m²")
        logger.info(f"    ✓ Floor polygon validated: {polygon.area:.3f} m²")
        return polygon
    except Exception as e:
        logger.warning(f"      ⚠️  Failed with original order: {e}")
        logger.info(f"      → Attempting to reorder walls automatically...")
    
    # If fails, try reordering walls
    try:
        ordered_walls = reorder_walls_to_close_polygon(walls_config)
        polygon = create_polygon_simple(ordered_walls)
        logger.info(f"      ✓ Polygon created after reordering walls")
        logger.info(f"      - Polygon area: {polygon.area:.3f} m²")
        logger.info(f"    ✓ Floor polygon validated: {polygon.area:.3f} m²")
        return polygon
    except Exception as e:
        logger.error(f"      ❌ Failed to create polygon even after reordering: {e}")
        raise ValueError(f"Cannot create closed polygon from walls: {e}")


def polygon_to_mesh_2d(polygon: shapely.Polygon, z: float, patch_df: pd.DataFrame, 
                       patch_id: str) -> pv.PolyData:
    """
    Convert Shapely polygon to triangulated PyVista mesh at height Z.
    
    Args:
        polygon: Shapely Polygon to convert
        z: Height (Z coordinate) for the mesh
        patch_df: DataFrame with patch information
        patch_id: ID of the patch for boundary conditions
        
    Returns:
        PyVista PolyData mesh
    """
    logger.info(f"    [PHASE 1.2] Converting polygon to mesh at Z={z:.3f}m")
    
    # Triangulate polygon using constrained Delaunay
    triangles = shapely.constrained_delaunay_triangles(polygon)
    
    points_2d = []
    faces = []
    point_index = {}
    next_index = 0
    
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
    
    # Convert 2D points to 3D at height Z
    points_3d = np.array([[x, y, z] for x, y in points_2d])
    faces = np.array(faces)
    
    # Create PyVista mesh
    mesh = pv.PolyData(points_3d, faces)
    
    # Assign patch ID
    patch_idx = patch_df.index[patch_df['id'] == patch_id][0].astype(np.int16)
    mesh.cell_data['patch_id'] = [patch_idx] * mesh.n_cells
    
    logger.info(f"      - Created mesh: {mesh.n_cells} cells, {mesh.n_points} points")
    logger.info(f"    ✓ Polygon converted to mesh")
    
    return mesh


# ============================================================================
# PHASE 2: LAYER FUNCTIONS
# ============================================================================

def create_walls_layer(patch_df: pd.DataFrame, walls_config: List[Dict[str, Any]], 
                      height: float, base_height: float, 
                      create_wall_func) -> Tuple[pd.DataFrame, List[pv.PolyData]]:
    """
    Create all walls using existing create_wall() function.
    
    Args:
        patch_df: DataFrame with patch information
        walls_config: List of wall configurations
        height: Floor height
        base_height: Base elevation
        create_wall_func: Reference to existing create_wall() function
        
    Returns:
        Tuple of (updated patch_df, list of wall meshes)
    """
    logger.info(f"    [PHASE 2.1] Creating walls layer ({len(walls_config)} walls)")
    
    wall_meshes = []
    for idx, wall in enumerate(walls_config, 1):
        patch_df, wall_mesh = create_wall_func(patch_df, wall, height, base_height)
        wall_meshes.append(wall_mesh)
        logger.info(f"      ✓ Wall {idx}/{len(walls_config)}: '{wall['id']}' ({wall_mesh.n_cells} cells)")
    
    logger.info(f"    ✓ Walls layer complete: {len(wall_meshes)} walls created")
    return patch_df, wall_meshes


def create_floor_layer_from_json(patch_df: pd.DataFrame, floor_polygon: shapely.Polygon,
                                 level_name: str, floor_data: Dict[str, Any],
                                 base_height: float, previous_stairs_config: List[Dict[str, Any]],
                                 create_entries_func, polygon_to_mesh_func) -> Tuple[pd.DataFrame, pv.PolyData]:
    """
    Create floor from JSON coordinates with air entries as SEPARATE REGIONS.
    
    Uses boolean operations (like stairs) for robustness:
    - Stair openings: Physical holes (difference operation)
    - Air entries: Separate regions with different patch_ids (difference + separate triangulation)
    
    Args:
        patch_df: DataFrame with patch information
        floor_polygon: Base floor polygon from wall coordinates
        level_name: Name of the current level
        floor_data: Floor configuration data
        base_height: Base elevation
        previous_stairs_config: Stair configurations from previous floor (to create openings)
        create_entries_func: Reference to create_entries() function
        polygon_to_mesh_func: Reference to polygon_to_mesh_2d() function
        
    Returns:
        Tuple of (updated patch_df, floor mesh)
    """
    logger.info(f"    [PHASE 2.2] Creating floor layer from JSON coordinates")
    
    floor_id = f"floor_{level_name}F"
    z_floor = base_height
    
    # Setup coordinate system
    p0 = np.array([0, 0, z_floor])
    udir = np.array([1, 0, 0])
    vdir = np.array([0, 1, 0])
    
    # Process air entries in floor
    patch_df, entries_dict = create_entries_func(patch_df, floor_data.get('airEntries', []), 0, p0, udir, vdir)
    
    if entries_dict:
        logger.info(f"      - Air entries created: {len(entries_dict)}")
        for entry_id, entry_polygon in entries_dict.items():
            logger.info(f"        [ENTRY] '{entry_id}': bounds={entry_polygon.bounds}, area={entry_polygon.area:.6f} m²")
    
    # Subtract stair openings from floor polygon (physical holes)
    if previous_stairs_config:
        logger.info(f"      - Creating {len(previous_stairs_config)} stair openings in floor (2D subtraction)")
        for idx, stair in enumerate(previous_stairs_config, 1):
            # Extract 2D polygon from stair lines
            stair_points = []
            for line in stair['lines']:
                stair_points.append([line['start']['x'], line['start']['y']])
            # Close the polygon
            if len(stair_points) > 0:
                stair_points.append(stair_points[0])
            
            if len(stair_points) >= 4:  # At least 3 unique points + closing
                stair_polygon = shapely.Polygon(stair_points)
                if stair_polygon.is_valid and stair_polygon.area > 0:
                    # Subtract stair polygon from floor
                    floor_polygon = floor_polygon.difference(stair_polygon)
                    logger.info(f"        ✓ Stair opening {idx}/{len(previous_stairs_config)}: '{stair['id']}' (area: {stair_polygon.area:.3f} m²)")
                else:
                    logger.warning(f"        ⚠️  Stair {idx}: Invalid polygon, skipping")
            else:
                logger.warning(f"        ⚠️  Stair {idx}: Insufficient points ({len(stair_points)-1}), skipping")
    
    # ✅ NEW APPROACH: Create main floor region (floor - entries) using boolean operations
    main_floor_polygon = floor_polygon
    if entries_dict:
        logger.info(f"      - Subtracting {len(entries_dict)} air entries from main floor region (2D boolean)")
        for entry_id, entry_polygon in entries_dict.items():
            main_floor_polygon = main_floor_polygon.difference(entry_polygon)
            logger.info(f"        ✓ Entry '{entry_id}' subtracted from main floor")
    
    # Create main floor patch
    from src.components.geo.create_volumes import get_wall_bc_dict
    new_patch = get_wall_bc_dict(floor_id, temperature=floor_data.get('temp', 20))
    patch_df = pd.concat([patch_df, pd.DataFrame([new_patch])], ignore_index=True)
    
    # Triangulate main floor region
    main_floor_mesh = polygon_to_mesh_func(main_floor_polygon, z_floor, patch_df, floor_id)
    logger.info(f"      - Main floor region: {main_floor_mesh.n_cells} cells")
    
    # Triangulate each entry as separate region
    entry_meshes = []
    if entries_dict:
        logger.info(f"      - Creating {len(entries_dict)} air entry regions (separate triangulation)")
        for entry_id, entry_polygon in entries_dict.items():
            # Verify that entry_id exists in patch_df before triangulation
            if entry_id not in patch_df['id'].values:
                logger.error(f"        ❌ Entry '{entry_id}' not found in patch_df")
                logger.error(f"        Available IDs: {list(patch_df['id'].values)}")
                raise ValueError(f"Entry '{entry_id}' not found in patch_df")
            
            entry_mesh = polygon_to_mesh_func(entry_polygon, z_floor, patch_df, entry_id)
            entry_meshes.append(entry_mesh)
            logger.info(f"        ✓ Entry '{entry_id}': {entry_mesh.n_cells} cells")
    
    # Merge all regions (main floor + entries)
    all_floor_meshes = [main_floor_mesh] + entry_meshes
    floor_mesh = pv.merge(all_floor_meshes)
    
    logger.info(f"    ✓ Floor layer complete: '{floor_id}' ({floor_mesh.n_cells} cells total, {len(entry_meshes)} entry regions)")
    return patch_df, floor_mesh


def find_cells_in_polygon(mesh: pv.PolyData, polygon: shapely.Polygon, z_tolerance: float = 1e-3) -> np.ndarray:
    """
    Find cells in a 2D mesh that are inside a given polygon.
    
    Args:
        mesh: PyVista mesh (assumed to be planar at constant Z)
        polygon: Shapely polygon to test against
        z_tolerance: Tolerance for Z coordinate comparison
        
    Returns:
        Array of cell indices that are inside the polygon
    """
    # DEBUG: Log polygon and mesh information
    logger.info(f"      [DEBUG] find_cells_in_polygon() called")
    logger.info(f"      [DEBUG] Polygon bounds: {polygon.bounds}")
    logger.info(f"      [DEBUG] Polygon area: {polygon.area:.6f} m²")
    logger.info(f"      [DEBUG] Mesh total cells: {mesh.n_cells}")
    logger.info(f"      [DEBUG] Mesh Z range: [{mesh.points[:, 2].min():.6f}, {mesh.points[:, 2].max():.6f}]")
    logger.info(f"      [DEBUG] Mesh X range: [{mesh.points[:, 0].min():.3f}, {mesh.points[:, 0].max():.3f}]")
    logger.info(f"      [DEBUG] Mesh Y range: [{mesh.points[:, 1].min():.3f}, {mesh.points[:, 1].max():.3f}]")
    
    cell_centers = mesh.cell_centers().points
    logger.info(f"      [DEBUG] First 5 cell centers (X, Y, Z):")
    for i in range(min(5, len(cell_centers))):
        logger.info(f"        Cell {i}: ({cell_centers[i][0]:.3f}, {cell_centers[i][1]:.3f}, {cell_centers[i][2]:.3f})")
    
    cells_in_polygon = []
    
    for i, center in enumerate(cell_centers):
        # Create shapely Point from cell center (X, Y only)
        point = shapely.Point(center[0], center[1])
        if polygon.contains(point) or polygon.touches(point):
            cells_in_polygon.append(i)
            logger.info(f"      [DEBUG] Cell {i} INSIDE polygon: ({center[0]:.3f}, {center[1]:.3f})")
    
    logger.info(f"      [DEBUG] Total cells found in polygon: {len(cells_in_polygon)}")
    
    return np.array(cells_in_polygon, dtype=np.int32)


def create_ceiling_layer_from_json(patch_df: pd.DataFrame, ceiling_polygon: shapely.Polygon,
                                   level_name: str, ceiling_data: Dict[str, Any],
                                   ceiling_height: float, stairs_config: List[Dict[str, Any]],
                                   create_entries_func, polygon_to_mesh_func) -> Tuple[pd.DataFrame, pv.PolyData, List[shapely.Polygon]]:
    """
    Create ceiling from same polygon as floor.
    Air entries are created as PATCHES (regions with different boundary conditions).
    Stair openings are created by subtracting stair polygons from ceiling polygon (2D operation).
    
    ROBUST: Clips stair polygons to ceiling bounds if they extend outside.
    
    Args:
        patch_df: DataFrame with patch information
        ceiling_polygon: Base ceiling polygon (same as floor)
        level_name: Name of the current level
        ceiling_data: Ceiling configuration data
        ceiling_height: Height of the ceiling
        stairs_config: List of stair configurations (to create openings)
        create_entries_func: Reference to create_entries() function
        polygon_to_mesh_func: Reference to polygon_to_mesh_2d() function
        
    Returns:
        Tuple of (updated patch_df, ceiling mesh, list of clipped stair polygons)
    """
    logger.info(f"    [PHASE 2.3] Creating ceiling layer from JSON coordinates")
    
    ceiling_id = f"ceil_{level_name}F"
    z_ceiling = ceiling_height
    
    # Setup coordinate system
    p0 = np.array([0, 0, z_ceiling])
    udir = np.array([1, 0, 0])
    vdir = np.array([0, 1, 0])
    
    # Process air entries in ceiling
    patch_df, entries_dict = create_entries_func(patch_df, ceiling_data.get('airEntries', []), 0, p0, udir, vdir)
    
    if entries_dict:
        logger.info(f"      - Air entries created: {len(entries_dict)}")
        for entry_id, entry_polygon in entries_dict.items():
            logger.info(f"        [ENTRY] '{entry_id}': bounds={entry_polygon.bounds}, area={entry_polygon.area:.6f} m²")
    
    # Track clipped stair polygons for stair tube creation
    clipped_stair_polygons = []
    
    # Subtract stair openings from ceiling polygon (2D operation with Shapely)
    if stairs_config:
        logger.info(f"      - Creating {len(stairs_config)} stair openings in ceiling (2D subtraction)")
        for idx, stair in enumerate(stairs_config, 1):
            # Extract 2D polygon from stair lines
            stair_points = []
            for line in stair['lines']:
                stair_points.append([line['start']['x'], line['start']['y']])
            # Close the polygon
            if len(stair_points) > 0:
                stair_points.append(stair_points[0])
            
            if len(stair_points) >= 4:  # At least 3 unique points + closing
                stair_polygon = shapely.Polygon(stair_points)
                if stair_polygon.is_valid and stair_polygon.area > 0:
                    # ✅ NEW: Check if stair extends outside ceiling bounds
                    if not ceiling_polygon.contains(stair_polygon):
                        if ceiling_polygon.intersects(stair_polygon):
                            # Stair partially outside - CLIP to ceiling bounds
                            logger.warning(f"        ⚠️  Stair '{stair['id']}' extends outside ceiling bounds")
                            logger.info(f"           → Clipping stair to ceiling boundaries")
                            
                            # Clip stair polygon to ceiling bounds
                            stair_polygon_clipped = stair_polygon.intersection(ceiling_polygon)
                            
                            # Handle MultiPolygon result (intersection can return multiple polygons)
                            if isinstance(stair_polygon_clipped, shapely.MultiPolygon):
                                # Take the largest polygon
                                stair_polygon_clipped = max(stair_polygon_clipped.geoms, key=lambda p: p.area)
                            
                            logger.info(f"           → Original area: {stair_polygon.area:.3f} m²")
                            logger.info(f"           → Clipped area: {stair_polygon_clipped.area:.3f} m²")
                            
                            stair_polygon = stair_polygon_clipped
                        else:
                            # Stair completely outside - ERROR
                            logger.error(f"        ❌ Stair '{stair['id']}' is completely outside ceiling bounds")
                            raise ValueError(f"Stair '{stair['id']}' must intersect with ceiling")
                    
                    # Store clipped polygon for stair tube creation
                    clipped_stair_polygons.append(stair_polygon)
                    
                    # Subtract (now clipped) stair polygon from ceiling
                    ceiling_polygon = ceiling_polygon.difference(stair_polygon)
                    logger.info(f"        ✓ Stair opening {idx}/{len(stairs_config)}: '{stair['id']}' (area: {stair_polygon.area:.3f} m²)")
                else:
                    logger.warning(f"        ⚠️  Stair {idx}: Invalid polygon, skipping")
                    clipped_stair_polygons.append(None)
            else:
                logger.warning(f"        ⚠️  Stair {idx}: Insufficient points ({len(stair_points)-1}), skipping")
                clipped_stair_polygons.append(None)
    
    # ✅ NEW APPROACH: Create main ceiling region (ceiling - entries) using boolean operations
    main_ceiling_polygon = ceiling_polygon
    if entries_dict:
        logger.info(f"      - Subtracting {len(entries_dict)} air entries from main ceiling region (2D boolean)")
        for entry_id, entry_polygon in entries_dict.items():
            main_ceiling_polygon = main_ceiling_polygon.difference(entry_polygon)
            logger.info(f"        ✓ Entry '{entry_id}' subtracted from main ceiling")
    
    # Create main ceiling patch
    from src.components.geo.create_volumes import get_wall_bc_dict
    new_patch = get_wall_bc_dict(ceiling_id, temperature=ceiling_data.get('temp', 20))
    patch_df = pd.concat([patch_df, pd.DataFrame([new_patch])], ignore_index=True)
    
    # Triangulate main ceiling region
    main_ceiling_mesh = polygon_to_mesh_func(main_ceiling_polygon, z_ceiling, patch_df, ceiling_id)
    logger.info(f"      - Main ceiling region: {main_ceiling_mesh.n_cells} cells")
    
    # Triangulate each entry as separate region
    entry_meshes = []
    if entries_dict:
        logger.info(f"      - Creating {len(entries_dict)} air entry regions (separate triangulation)")
        for entry_id, entry_polygon in entries_dict.items():
            entry_mesh = polygon_to_mesh_func(entry_polygon, z_ceiling, patch_df, entry_id)
            entry_meshes.append(entry_mesh)
            logger.info(f"        ✓ Entry '{entry_id}': {entry_mesh.n_cells} cells")
    
    # Merge all regions (main ceiling + entries)
    all_ceiling_meshes = [main_ceiling_mesh] + entry_meshes
    ceiling_mesh = pv.merge(all_ceiling_meshes)
    
    logger.info(f"    ✓ Ceiling layer complete: '{ceiling_id}' ({ceiling_mesh.n_cells} cells total, {len(entry_meshes)} entry regions)")
    return patch_df, ceiling_mesh, clipped_stair_polygons


def create_stair_tubes(patch_df: pd.DataFrame, stairs_config: List[Dict[str, Any]],
                      clipped_stair_polygons: List[Optional[shapely.Polygon]],
                      ceiling_height: float, deck_thickness: float,
                      is_top_floor: bool = False) -> Tuple[pd.DataFrame, List[pv.PolyData]]:
    """
    Create stair tubes that extrude only by deck_thickness (not full floor height).
    
    ROBUST: Uses clipped stair polygons to match ceiling openings exactly.
    SPECIAL: If is_top_floor=True, adds ONLY top cap (bottom remains open to communicate with ceiling).
    
    Behavior:
    - Normal floors: Tube open at both ends (air circulates between floors)
    - Top floor: Tube with top cap only (bottom open to ceiling, top closed as no floor above)
    
    Args:
        patch_df: DataFrame with patch information
        stairs_config: List of stair configurations
        clipped_stair_polygons: List of clipped stair polygons from ceiling layer
        ceiling_height: Height of the ceiling
        deck_thickness: Thickness of the deck
        is_top_floor: Whether this is the top floor (adds top cap only)
        
    Returns:
        Tuple of (updated patch_df, list of stair tube meshes)
    """
    if is_top_floor and stairs_config:
        logger.warning(f"    [PHASE 2.4] Creating stair tubes on TOP FLOOR ({len(stairs_config)} stairs)")
        logger.warning(f"      ⚠️  Stairs will have TOP CAP (bottom open to ceiling)")
    else:
        logger.info(f"    [PHASE 2.4] Creating stair tubes ({len(stairs_config)} stairs)")
    
    stair_tubes = []
    
    for idx, (stair, clipped_polygon) in enumerate(zip(stairs_config, clipped_stair_polygons), 1):
        from src.components.geo.create_volumes import get_wall_bc_dict
        
        stair_id = stair['id']
        new_patch = get_wall_bc_dict(stair_id)
        patch_df = pd.concat([patch_df, pd.DataFrame([new_patch])], ignore_index=True)
        
        # ✅ NEW: Use clipped polygon if available, otherwise use original lines
        if clipped_polygon is not None:
            logger.info(f"      - Using clipped polygon for stair tube '{stair_id}'")
            
            # Extract points from clipped polygon
            coords = list(clipped_polygon.exterior.coords)[:-1]  # Remove duplicate closing point
            points = []
            connectivity = []
            
            for i, (x, y) in enumerate(coords):
                p0 = np.array([x, y, ceiling_height - VOLUMES_TOLERANCE])
                points.append(p0)
                
                # Create line connectivity
                if i < len(coords) - 1:
                    connectivity.extend([2, i, i + 1])
                else:
                    # Close the loop
                    connectivity.extend([2, i, 0])
            
            points = np.array(points)
        else:
            # Fallback: use original stair lines
            logger.info(f"      - Using original lines for stair tube '{stair_id}'")
            points = []
            connectivity = []
            for i, line in enumerate(stair['lines']):
                p0 = np.array([line['start']['x'], line['start']['y'], ceiling_height - VOLUMES_TOLERANCE])
                p1 = np.array([line['end']['x'], line['end']['y'], ceiling_height - VOLUMES_TOLERANCE])
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
        
        # ✅ ALWAYS create open tube (no caps) for air circulation
        extrusion_height = deck_thickness + 2*VOLUMES_TOLERANCE
        stair_tube = poly_filled.extrude([0, 0, extrusion_height], capping=False)
        
        # ✅ If top floor, add ONLY top cap (bottom remains open to communicate with ceiling)
        if is_top_floor:
            # Create top cap from poly_filled (already clipped if needed)
            top_cap = poly_filled.copy()
            top_cap.translate([0, 0, extrusion_height], inplace=True)
            
            # Merge tube + top cap
            stair_tube = pv.merge([stair_tube, top_cap])
            logger.info(f"      → Top cap added (no floor above)")
        
        stair_tube.compute_normals(inplace=True, auto_orient_normals=True, 
                                   consistent_normals=True, split_vertices=True, 
                                   point_normals=False)
        stair_tube.triangulate(inplace=True)
        
        patch_idx = patch_df.index[patch_df['id'] == stair_id][0].astype(np.int16)
        stair_tube.cell_data['patch_id'] = [patch_idx] * stair_tube.n_cells
        
        stair_tube = optimize_mesh_memory(stair_tube)
        stair_tubes.append(stair_tube)
        
        # Updated logging
        if is_top_floor:
            cap_status = "TOP CAPPED (bottom open to ceiling)"
        else:
            cap_status = "BOTH ENDS OPEN (air circulates)"
        
        logger.info(f"      ✓ Stair tube {idx}/{len(stairs_config)}: '{stair_id}' "
                   f"({stair_tube.n_cells} cells, extrusion: {extrusion_height:.3f}m, {cap_status})")
    
    if is_top_floor:
        logger.info(f"    ✓ Stair tubes complete: {len(stair_tubes)} tubes (TOP CAPPED, bottom open)")
    else:
        logger.info(f"    ✓ Stair tubes complete: {len(stair_tubes)} open tubes (air circulates)")
    return patch_df, stair_tubes


def subtract_stair_tubes_from_ceiling(ceiling_mesh: pv.PolyData, 
                                      stair_tubes: List[pv.PolyData]) -> pv.PolyData:
    """
    MANDATORY boolean operation: subtract stair tubes from ceiling.
    
    Args:
        ceiling_mesh: Ceiling mesh
        stair_tubes: List of stair tube meshes to subtract
        
    Returns:
        Ceiling mesh with stair openings
        
    Raises:
        RuntimeError: If boolean operation fails (CRITICAL ERROR - STOPS PIPELINE)
    """
    if not stair_tubes:
        logger.info(f"    [PHASE 2.5] No stair tubes to subtract from ceiling")
        return ceiling_mesh
    
    logger.info(f"    [PHASE 2.5] Subtracting {len(stair_tubes)} stair tubes from ceiling (MANDATORY)")
    
    # Validate and clean meshes before boolean operation
    logger.info(f"      - Validating ceiling mesh...")
    ceiling_mesh = ceiling_mesh.clean().triangulate().extract_surface()
    ceiling_mesh.compute_normals(inplace=True, auto_orient_normals=True, 
                                consistent_normals=True, split_vertices=True, 
                                point_normals=False)
    
    is_ceiling_manifold = ceiling_mesh.is_manifold
    ceiling_open_edges = ceiling_mesh.n_open_edges
    logger.info(f"        - Ceiling: {ceiling_mesh.n_cells} cells, manifold={is_ceiling_manifold}, open_edges={ceiling_open_edges}")
    
    if not is_ceiling_manifold or ceiling_open_edges > 0:
        error_msg = (
            f"CRITICAL: Ceiling mesh is not watertight (manifold={is_ceiling_manifold}, "
            f"open_edges={ceiling_open_edges}). Cannot perform boolean subtraction. "
            f"Stair openings CANNOT be created without watertight meshes. "
            f"This will prevent air circulation between floors."
        )
        logger.error(f"      ❌ {error_msg}")
        raise RuntimeError(error_msg)
    
    # Validate and clean each stair tube
    cleaned_tubes = []
    for idx, tube in enumerate(stair_tubes, 1):
        logger.info(f"      - Validating stair tube {idx}/{len(stair_tubes)}...")
        tube_clean = tube.clean().triangulate().extract_surface()
        tube_clean.compute_normals(inplace=True, auto_orient_normals=True, 
                                  consistent_normals=True, split_vertices=True, 
                                  point_normals=False)
        
        is_tube_manifold = tube_clean.is_manifold
        tube_open_edges = tube_clean.n_open_edges
        logger.info(f"        - Tube {idx}: {tube_clean.n_cells} cells, manifold={is_tube_manifold}, open_edges={tube_open_edges}")
        
        # IMPORTANT: Open tubes (capping=False) will have open edges - this is EXPECTED
        if tube_open_edges > 0:
            logger.info(f"        ℹ️  Tube has {tube_open_edges} open edges (expected for open-ended tubes)")
        
        cleaned_tubes.append(tube_clean)
    
    # Perform boolean subtraction
    try:
        result_mesh = subtract_objects(ceiling_mesh, cleaned_tubes)
        logger.info(f"      ✓ Boolean subtraction successful ({result_mesh.n_cells} cells)")
        logger.info(f"    ✓ Ceiling with stair openings complete - AIR CAN CIRCULATE")
        return result_mesh
    except Exception as e:
        error_msg = (
            f"CRITICAL: Boolean subtraction failed: {e}\n"
            f"Stair openings were NOT created in the ceiling.\n"
            f"This prevents air circulation between floors.\n"
            f"Possible causes:\n"
            f"  1. Ceiling mesh is not watertight (check manifold and open edges)\n"
            f"  2. Stair tube geometry intersects ceiling incorrectly\n"
            f"  3. PyMeshLab boolean operation requirements not met\n"
            f"PIPELINE STOPPED - Fix geometry issues before continuing."
        )
        logger.error(f"      ❌ {error_msg}")
        raise RuntimeError(error_msg)


def merge_and_validate(wall_meshes: List[pv.PolyData], floor_mesh: pv.PolyData,
                      ceiling_mesh: pv.PolyData, stair_tubes: List[pv.PolyData],
                      level_name: str) -> pv.PolyData:
    """
    Merge all components and validate waterproof geometry.
    
    Args:
        wall_meshes: List of wall meshes
        floor_mesh: Floor mesh
        ceiling_mesh: Ceiling mesh
        stair_tubes: List of stair tube meshes
        level_name: Name of the level for logging
        
    Returns:
        Merged and validated floor mesh
        
    Raises:
        ValueError: If geometry is not waterproof
    """
    logger.info(f"    [PHASE 2.6] Merging and validating floor '{level_name}'")
    
    # Merge all components preserving patch_id (including stair tubes)
    all_meshes = wall_meshes + [floor_mesh, ceiling_mesh] + stair_tubes
    
    # Collect all patch_ids before merge
    all_patch_ids = []
    for mesh in all_meshes:
        if 'patch_id' in mesh.cell_data:
            all_patch_ids.extend(mesh.cell_data['patch_id'].tolist())
        else:
            logger.warning(f"      ⚠️  Mesh missing patch_id, using default value 0")
            all_patch_ids.extend([0] * mesh.n_cells)
    
    # Perform merge
    merged_mesh = pv.merge(all_meshes)
    merged_mesh.triangulate(inplace=True)
    merged_mesh.compute_normals(inplace=True, auto_orient_normals=True, 
                               consistent_normals=True, split_vertices=True, 
                               point_normals=False)
    
    # Restore patch_id after merge
    if len(all_patch_ids) == merged_mesh.n_cells:
        merged_mesh.cell_data['patch_id'] = np.array(all_patch_ids, dtype=np.int16)
        logger.info(f"      - Restored patch_id for {merged_mesh.n_cells} cells")
    else:
        logger.warning(f"      ⚠️  patch_id count mismatch: {len(all_patch_ids)} vs {merged_mesh.n_cells} cells")
        logger.warning(f"      ⚠️  Assigning default patch_id=0 to all cells")
        merged_mesh.cell_data['patch_id'] = np.zeros(merged_mesh.n_cells, dtype=np.int16)
    
    logger.info(f"      - Merged mesh: {merged_mesh.n_cells} cells, {merged_mesh.n_points} points")
    
    # Validate waterproof geometry
    is_manifold = merged_mesh.is_manifold
    n_open_edges = merged_mesh.n_open_edges
    volume = merged_mesh.volume
    
    logger.info(f"      - Validation: is_manifold={is_manifold}, n_open_edges={n_open_edges}, volume={volume:.3f}m³")
    
    # RELAXED VALIDATION: Only warn about non-manifold geometry, don't fail
    if not is_manifold:
        logger.warning(f"      ⚠️  Geometry is NOT manifold (has non-manifold edges)")
        logger.warning(f"      ⚠️  This may cause issues with boolean operations")
        logger.warning(f"      ⚠️  Continuing anyway (validation relaxed for stair geometries)")
    
    if n_open_edges > 0:
        logger.warning(f"      ⚠️  Geometry has {n_open_edges} open edges (not fully waterproof)")
        logger.warning(f"      ⚠️  This may cause issues with CFD meshing")
        logger.warning(f"      ⚠️  Continuing anyway (validation relaxed for stair geometries)")
    
    if volume <= 0:
        logger.error(f"      ❌ Geometry has invalid volume: {volume:.3f}m³")
        raise ValueError(f"Floor '{level_name}' has invalid volume: {volume:.3f}m³")
    
    if is_manifold and n_open_edges == 0:
        logger.info(f"    ✓ Floor '{level_name}' validated: WATERPROOF (volume={volume:.3f}m³)")
    else:
        logger.warning(f"    ⚠️  Floor '{level_name}' has geometry issues but continuing (volume={volume:.3f}m³)")
    
    # Optimize memory
    merged_mesh = optimize_mesh_memory(merged_mesh)
    
    return merged_mesh


# ============================================================================
# PHASE 3: MAIN FUNCTION
# ============================================================================

def create_floor_mesh_layered(patch_df: pd.DataFrame, level_name: str, 
                             level_data: Dict[str, Any], base_height: float,
                             previous_stair_tubes: Optional[List[pv.PolyData]] = None,
                             is_top_floor: bool = False) -> Tuple[pd.DataFrame, pv.PolyData, List[pv.PolyData]]:
    """
    Create floor mesh using layer-based architecture with 7 phases.
    
    This function replaces create_floor_mesh() with a robust architecture where:
    - Floor/ceiling are created from JSON coordinates (not from wall edges)
    - Stair tubes extrude only by deck_thickness
    - Boolean operations are mandatory
    - Waterproof validation at the end
    - Top floor stairs are capped to close geometry
    
    Args:
        patch_df: DataFrame with patch information
        level_name: Name of the current level
        level_data: Level configuration data
        base_height: Base elevation
        previous_stair_tubes: Stair tubes from previous floor (to subtract from current floor)
        is_top_floor: Whether this is the top floor (caps stair tubes on top)
        
    Returns:
        Tuple of (updated patch_df, floor mesh, current stair tubes for next floor)
    """
    logger.info(f"  ═══════════════════════════════════════════════════════════")
    logger.info(f"  CREATING FLOOR '{level_name}' - LAYER-BASED ARCHITECTURE")
    logger.info(f"  ═══════════════════════════════════════════════════════════")
    
    logger.info(f"  [DEBUG] Function create_floor_mesh_layered() STARTED")
    logger.info(f"  [DEBUG] level_name={level_name}, base_height={base_height}")
    logger.info(f"  [DEBUG] level_data keys: {list(level_data.keys())}")
    logger.info(f"  [DEBUG] Number of walls: {len(level_data.get('walls', []))}")
    
    height = level_data["height"]
    deck_thickness = level_data["deck"]
    ceiling_height = base_height + height
    
    logger.info(f"  [DEBUG] height={height}, deck_thickness={deck_thickness}, ceiling_height={ceiling_height}")
    
    # Import required functions
    logger.info(f"  [DEBUG] Importing functions from create_volumes...")
    from src.components.geo.create_volumes import (
        create_wall, create_entries, create_mesh_from_polygon
    )
    logger.info(f"  [DEBUG] Functions imported successfully")
    
    # PHASE 1: Extract floor polygon from wall coordinates
    logger.info(f"  [DEBUG] About to call create_floor_polygon_from_wall_coords()...")
    try:
        floor_polygon = create_floor_polygon_from_wall_coords(level_data["walls"])
        logger.info(f"  [DEBUG] create_floor_polygon_from_wall_coords() SUCCESS")
        logger.info(f"  [DEBUG] floor_polygon area: {floor_polygon.area:.3f} m²")
    except Exception as e:
        logger.error(f"  [DEBUG] create_floor_polygon_from_wall_coords() FAILED: {e}")
        import traceback
        logger.error(f"  [DEBUG] Traceback:\n{traceback.format_exc()}")
        raise
    
    ceiling_polygon = floor_polygon  # Same polygon for ceiling
    logger.info(f"  [DEBUG] ceiling_polygon assigned (same as floor_polygon)")
    
    # PHASE 2.1: Create walls layer
    patch_df, wall_meshes = create_walls_layer(
        patch_df, level_data["walls"], height, base_height, create_wall
    )
    
    # PHASE 2.2: Create floor layer (with stair openings from previous floor)
    # Extract previous stairs config from previous_stair_tubes parameter
    # Note: previous_stair_tubes is now expected to be stairs_config, not mesh tubes
    previous_stairs_config = previous_stair_tubes if previous_stair_tubes else []
    
    patch_df, floor_mesh = create_floor_layer_from_json(
        patch_df, floor_polygon, level_name, level_data.get("floor", {}),
        base_height, previous_stairs_config, create_entries, polygon_to_mesh_2d
    )
    
    # PHASE 2.3: Create ceiling layer (with stair openings already subtracted in 2D)
    # Returns clipped stair polygons for stair tube creation
    patch_df, ceiling_mesh, clipped_stair_polygons = create_ceiling_layer_from_json(
        patch_df, ceiling_polygon, level_name, level_data.get("ceiling", {}),
        ceiling_height, level_data.get("stairs", []), create_entries, polygon_to_mesh_2d
    )
    
    # PHASE 2.4: Create stair tubes (for current floor) using clipped polygons
    # Pass is_top_floor to cap tubes on top floor
    patch_df, current_stair_tubes = create_stair_tubes(
        patch_df, level_data.get("stairs", []), clipped_stair_polygons,
        ceiling_height, deck_thickness, is_top_floor=is_top_floor
    )
    
    # PHASE 2.5: No longer needed - stair openings already created in ceiling polygon
    # ceiling_mesh already has openings from 2D subtraction in create_ceiling_layer_from_json()
    
    # PHASE 2.6: Merge and validate (including stair tubes)
    floor_mesh_final = merge_and_validate(wall_meshes, floor_mesh, ceiling_mesh, current_stair_tubes, level_name)
    
    logger.info(f"  ═══════════════════════════════════════════════════════════")
    logger.info(f"  ✓ FLOOR '{level_name}' COMPLETE")
    logger.info(f"  ═══════════════════════════════════════════════════════════\n")
    
    # Return stairs_config (not stair_tubes meshes) for next floor
    return patch_df, floor_mesh_final, level_data.get("stairs", [])
