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
from pipeline_exceptions import GeometryStepError

logger = logging.getLogger(__name__)


def run(json_payload: Dict[str, Any], case_name: str) -> Tuple[pv.PolyData, pd.DataFrame]:
    """
    Convert JSON building data to 3D geometry with parallel processing and memory management.
    
    This function orchestrates the complete geometry creation pipeline:
    1. Loads building configuration from JSON
    2. Creates room volumes (walls, floors, ceilings)
    3. Adds furniture objects
    4. Performs boolean operations to combine geometry
    5. Applies boundary conditions and exports results
    
    Args:
        input_json: Path to input JSON file containing building configuration
        
    Returns:
        Tuple of (final_geometry_mesh, boundary_conditions_dataframe)
    """
    
    logger.info("\n=========== RUNNING JSON TO GEOMETRY CONVERSION ===========")

    # Step 1: Initialize case directory
    logger.info(f"1 - Initializing case directory: {case_name}")
    clear_case_all(case_name)

    # Step 2: Create room geometry (walls, floors, ceilings) and furniture
    logger.info("2 - Creating room geometry and furniture")
    try:
        room_geometry_meshes, furniture_meshes, boundary_conditions_df = create_volumes(json_payload)
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