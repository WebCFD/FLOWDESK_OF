"""
HVAC Professional Mesh Preparation
===================================

Prepares snappyHexMesh workflow with HVAC professional configuration.
"""

import os
import logging
import pyvista as pv
import pandas as pd

from src.components.mesh.snappy import (
    export_to_stl,
    split_polydata_by_cell_data,
    create_blockMeshDict,
    validate_mesh_patches
)
from src.components.mesh.hvac_pro import create_hvac_pro_snappyHexMeshDict
from src.components.tools.populate_template_file import replace_in_file

logger = logging.getLogger(__name__)


def create_surfaceFeatureExtractDict(template_path, sim_path, stl_filename):
    """Create surfaceFeatureExtractDict from template"""
    input_path = os.path.join(template_path, "system", "surfaceFeatureExtractDict") 
    output_path = os.path.join(sim_path, "system", "surfaceFeatureExtractDict") 
    str_replace_dict = dict()
    str_replace_dict["$STL_FILENAME"] = stl_filename
    replace_in_file(input_path, output_path, str_replace_dict)


def prepare_hvac_pro(geo_mesh: pv.PolyData, sim_path: str, geo_df: pd.DataFrame, 
                     stl_filename: str = "geometry.stl", quality_level: int = 2) -> list:
    """
    Prepare HVAC Professional mesh configuration and scripts.
    
    This uses the professional HVAC configuration designed from scratch
    with physics-based refinement levels and optimized boundary layers.
    
    Args:
        geo_mesh: Geometry mesh to be meshed
        sim_path: Path to simulation directory
        geo_df: DataFrame containing boundary condition information
        stl_filename: Name of the STL file to export
        quality_level: 1 (coarse ~50k), 2 (medium ~500k), 3 (fine ~5M cells)
        
    Returns:
        List of script commands for mesh generation
    """
    from src.components.mesh.hvac_pro import MeshQualityLevel
    
    config = MeshQualityLevel.get_config(quality_level)
    
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"HVAC MESH PREPARATION - Quality Level {quality_level}")
    logger.info("=" * 80)
    logger.info(f"  Quality: {config['name']}")
    logger.info(f"  Description: {config['description']}")
    logger.info(f"  Geometry: {geo_mesh.n_cells} cells")
    logger.info(f"  Boundary conditions: {len(geo_df)} patches")
    logger.info("=" * 80)
    logger.info("")
    
    # Split geometry mesh by boundary condition patches
    logger.info("  * Splitting geometry mesh by boundary condition patches")
    geo_mesh_dict = split_polydata_by_cell_data(geo_mesh, geo_df)
    logger.info(f"  * Split into {len(geo_mesh_dict)} patch meshes")
    
    # Export geometry to STL format
    logger.info(f"  * Exporting geometry to STL format: {stl_filename}")
    export_to_stl(geo_mesh_dict, sim_path, stl_filename)
    
    # Use HVAC Pro template path
    template_path = os.path.join(os.getcwd(), "data", "settings", "mesh", "hvac_pro")
    logger.info(f"  * Using HVAC Professional template: {template_path}")
    
    # Create configuration files
    logger.info("  * Creating surfaceFeatureExtractDict")
    create_surfaceFeatureExtractDict(template_path, sim_path, stl_filename)
    
    logger.info("  * Creating blockMeshDict")
    create_blockMeshDict(template_path, sim_path, geo_mesh)
    
    logger.info(f"  * Creating HVAC snappyHexMeshDict (quality level {quality_level})")
    create_hvac_pro_snappyHexMeshDict(template_path, sim_path, stl_filename, geo_mesh, geo_df, quality_level)
    
    # Get expected patches for validation
    expected_patches = geo_df["id"].tolist()
    expected_patches_str = ", ".join(expected_patches)
    
    # Generate mesh script commands
    script_commands = [
        '#!/bin/sh', 
        'cd "${0%/*}" || exit',
        '. ${WM_PROJECT_DIR:?}/bin/tools/RunFunctions',
        '',
        '# HVAC Professional Mesh Generation Pipeline',
        '# Optimized for thermal comfort and air flow simulations',
        '',
        'runApplication surfaceFeatureExtract',
        'runApplication blockMesh',
        'runApplication snappyHexMesh -overwrite',
        '',
        '# Validate mesh quality',
        'runApplication checkMesh -allGeometry -allTopology',
        '',
        f'echo "Expected patches: {expected_patches_str}"',
        'echo "Mesh generation complete"',
        '',
    ]
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("✓ HVAC mesh preparation completed successfully")
    logger.info(f"  Quality level: {quality_level} ({config['name']})")
    logger.info(f"  Mesh script: {len(script_commands)} commands")
    logger.info("=" * 80)
    logger.info("")
    
    return script_commands
