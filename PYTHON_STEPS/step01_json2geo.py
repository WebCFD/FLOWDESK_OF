import os
import json
import logging
import pandas as pd
import pyvista as pv

from typing import Tuple, Dict, Any
from src.components.tools.clear_case import clear_case_all
from src.components.geo.create_volumes import create_volumes
from src.components.geo.boolean_operations import perform_boolean_operations
from src.components.geo.finalise_geometry import finalise_geometry
from src.components.geo.export_result import export_geo
from src.components.geo.geometry_validator import log_validation
from src.components.geo.json_validator import validate_building_json, JSONValidationError
from pipeline_exceptions import GeometryStepError

logger = logging.getLogger(__name__)


def run(json_payload: Dict[str, Any], case_name: str) -> Tuple[pv.PolyData, pd.DataFrame]:
    """
    Convert JSON building data to 3D geometry with parallel processing and memory management.
    
    This function orchestrates the complete geometry creation pipeline:
    1. Validates JSON structure and floor sequence
    2. Loads building configuration from JSON
    3. Creates room volumes (walls, floors, ceilings)
    4. Adds furniture objects
    5. Performs boolean operations to combine geometry
    6. Applies boundary conditions and exports results
    
    Args:
        json_payload: Building configuration JSON data
        case_name: Name of the case for output files
        
    Returns:
        Tuple of (final_geometry_mesh, boundary_conditions_dataframe)
    """
    
    logger.info("\n=========== RUNNING JSON TO GEOMETRY CONVERSION ===========")

    # Step 0: Validate JSON structure
    logger.info("0 - Validating JSON structure")
    try:
        validation_results = validate_building_json(json_payload)
    except Exception as e:
        raise GeometryStepError(
            f"JSON validation failed: {str(e)}",
            {
                'case_name': case_name,
                'error_type': 'json_validation',
                'suggestion': 'Check JSON structure and fix validation errors'
            }
        )
    
    # Check if validation passed
    if not validation_results['valid']:
        error_messages = [f"{err['location']}: {err['message']}" for err in validation_results['errors']]
        raise JSONValidationError(
            f"JSON validation failed with {len(validation_results['errors'])} errors",
            validation_results['errors']
        )
    
    # Extract valid floors from validation results
    valid_floors = validation_results['valid_floors']
    logger.info(f"   → Processing {len(valid_floors)} valid floors: {', '.join(valid_floors)}\n")

    # Step 1: Initialize case directory
    logger.info(f"1 - Initializing case directory: {case_name}")
    clear_case_all(case_name)

    # Step 2: Create room geometry (walls, floors, ceilings) and furniture
    logger.info("2 - Creating room geometry and furniture")
    try:
        room_geometry_meshes, furniture_meshes, boundary_conditions_df = create_volumes(json_payload, valid_floors)
    except Exception as e:
        raise GeometryStepError(
            f"Failed to create room geometry: {str(e)}",
            {
                'case_name': case_name,
                'error_type': 'volume_creation',
                'suggestion': 'Check if room polygon is closed and has valid coordinates'
            }
        )
    
    # Validate that we have geometry to work with
    if not room_geometry_meshes or len(room_geometry_meshes) == 0:
        raise GeometryStepError(
            "No room geometry was created",
            {
                'case_name': case_name,
                'error_type': 'empty_geometry',
                'suggestion': 'Ensure the JSON contains valid room definitions with closed polygons'
            }
        )
    
    # Step 2.5: Validate geometry (basic checks)
    logger.info("2.5 - Validating geometry")
    logger.info("-" * 70)
    
    # Validar que cada room tiene puntos y celdas
    for i, room_mesh in enumerate(room_geometry_meshes):
        room_id = f"room_{i}"
        n_points = room_mesh.n_points
        n_cells = room_mesh.n_cells
        
        if n_points > 0 and n_cells > 0:
            logger.info(f"✓ {room_id}: {n_points} points, {n_cells} cells")
        else:
            logger.warning(f"⚠️  {room_id}: Invalid geometry (points={n_points}, cells={n_cells})")
    
    # Validar que cada furniture tiene puntos y celdas
    for i, furniture_mesh in enumerate(furniture_meshes):
        furniture_id = f"furniture_{i}"
        n_points = furniture_mesh.n_points
        n_cells = furniture_mesh.n_cells
        
        if n_points > 0 and n_cells > 0:
            logger.info(f"✓ {furniture_id}: {n_points} points, {n_cells} cells")
        else:
            logger.warning(f"⚠️  {furniture_id}: Invalid geometry (points={n_points}, cells={n_cells})")
    
    logger.info("-" * 70)

    # Step 3: Combine room geometry and subtract furniture using boolean operations
    logger.info("3 - Combining room geometry and applying furniture subtractions")
    try:
        final_geometry_mesh = perform_boolean_operations(room_geometry_meshes, furniture_meshes)
    except Exception as e:
        logger.warning(f"Boolean operations failed: {e}")
        logger.warning("Falling back to simple mesh merging (furniture will be added, not subtracted)")
        final_geometry_mesh = room_geometry_meshes[0] if room_geometry_meshes else pv.PolyData()
        # Merge remaining room meshes
        for room_mesh in room_geometry_meshes[1:]:
            final_geometry_mesh = final_geometry_mesh.merge(room_mesh)
        # Add furniture meshes as separate entities (no subtraction)
        for furniture_mesh in furniture_meshes:
            final_geometry_mesh = final_geometry_mesh.merge(furniture_mesh)

    # Step 4: Apply boundary conditions and finalize geometry
    logger.info("4 - Applying boundary conditions and finalizing geometry")
    final_geometry_mesh, boundary_conditions_df = finalise_geometry(final_geometry_mesh, boundary_conditions_df)

    # Step 5: Export geometry and boundary condition data
    logger.info("5 - Exporting final geometry and boundary condition information")
    export_geo(case_name, final_geometry_mesh, boundary_conditions_df)

    logger.info(f"✅ Geometry creation pipeline completed successfully:")
    return final_geometry_mesh, boundary_conditions_df


if __name__ == "__main__":
    case_name = "FDM_iter2"
    input_json = os.path.join(os.getcwd(), "input", case_name + ".json")
    with open(input_json) as f:
        payload = json.load(f)
    final_geometry_mesh, boundary_conditions_df = run(payload, case_name)
