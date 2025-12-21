"""
Geometry to mesh conversion with parallel processing and memory management.
"""

import os
import logging
from typing import List, Optional
import pyvista as pv
import pandas as pd

from src.components.tools.load_geo import load_geo_files
from src.components.tools.performance import PerformanceMonitor
from pipeline_exceptions import MeshingStepError

logger = logging.getLogger(__name__)


def run(case_name: str, geo_mesh: pv.PolyData, geo_df: pd.DataFrame, type: str = "cfmesh", quality_level: Optional[int] = None) -> List[str]:
    """
    Convert 3D geometry to computational mesh with parallel processing and memory management.
    
    This function orchestrates the complete mesh generation pipeline:
    1. Sets up the mesh case directory structure
    2. Prepares mesh generation scripts based on selected meshing software
    3. Configures mesh parameters and boundary conditions
    4. Generates mesh generation commands for execution
    
    Args:
        case_name: Name of the simulation case
        geo_mesh: 3D geometry mesh from previous step
        geo_df: DataFrame containing boundary condition information
        type: Type of meshing software ("hvac_pro", "cfmesh", or "snappy")
        quality_level: Mesh quality level (1=coarse ~50k, 2=medium ~500k, 3=fine ~5M cells) - only for hvac_pro
        
    Returns:
        List of script commands for mesh generation
    """
    # Use configured default if not specified
    if quality_level is None:
        from mesher_config import get_default_quality_level
        quality_level = get_default_quality_level()
    
    performance_monitor = PerformanceMonitor()
    performance_monitor.start()
    
    logger.info("\n=========== RUNNING GEOMETRY TO MESH CONVERSION ===========")

    # Step 1: Set up case directory structure
    case_path = os.path.join(os.getcwd(), "cases", case_name)
    sim_path = os.path.join(case_path, "sim")
    logger.info(f"1 - Setting up case directory structure: {case_path}")
    performance_monitor.update_memory()

    # Step 2: Prepare mesh generation scripts
    logger.info(f"2 - Preparing {type} mesh generation scripts")
    performance_monitor.update_memory()
    
    try:
        if type == "cfmesh":
            from src.components.mesh.cfmesh import prepare_cfmesh
            script_commands = prepare_cfmesh(geo_mesh, sim_path, geo_df)
        elif type == "hvac_pro":
            from src.components.mesh.hvac_pro_prepare import prepare_hvac_pro
            script_commands = prepare_hvac_pro(geo_mesh, sim_path, geo_df, quality_level=quality_level)
        elif type == "snappy":
            from src.components.mesh.snappy import prepare_snappy
            script_commands = prepare_snappy(geo_mesh, sim_path, geo_df)
        else:
            raise MeshingStepError(
                f"Unknown meshing software: {type}",
                {
                    'case_name': case_name,
                    'meshing_type': type,
                    'suggestion': 'Use "hvac_pro", "cfmesh", or "snappy"'
                }
            )
    except MeshingStepError:
        # Re-raise MeshingStepError without wrapping
        raise
    except Exception as e:
        raise MeshingStepError(
            f"Mesh generation failed: {str(e)}",
            {
                'case_name': case_name,
                'meshing_type': type,
                'suggestion': 'Check if geometry is valid and mesh parameters are correct'
            }
        )
    
    performance_monitor.update_memory()
    
    # Log performance summary
    performance_summary = performance_monitor.get_summary()
    logger.info(f"Total processing time: {performance_summary['total_time']:.2f}s")
    logger.info(f"Peak memory usage: {performance_summary['peak_memory_mb']:.1f}MB")
    logger.info(f"✅ Mesh preparation pipeline completed successfully:")
    return script_commands


if __name__ == "__main__":
    from mesher_config import get_default_mesher, get_default_quality_level
    
    case_name = "FDM_iter2"
    geo_mesh, geo_df = load_geo_files(case_name)
    mesher_type = get_default_mesher()
    quality_level = get_default_quality_level()
    print(f"Using mesher: {mesher_type}")
    if mesher_type == "hvac_pro":
        print(f"Quality level: {quality_level}")
    result = run(case_name=case_name, geo_mesh=geo_mesh, geo_df=geo_df, type=mesher_type, quality_level=quality_level)