import os
import logging

logger = logging.getLogger(__name__)


def export_geo(case_name, vtk_mesh, patch_df):
    """
    Export final geometry and boundary condition data to case directory.
    
    Args:
        case_name: Name of the simulation case
        vtk_mesh: Final geometry mesh
        patch_df: DataFrame containing boundary condition information
    """
    logger.info(f"    * Exporting geometry and boundary conditions for case: {case_name}")
    
    # Create case geometry directory
    case_geo_path = os.path.join(os.getcwd(), 'cases', case_name, 'geo')
    logger.info(f"    * Creating geometry directory: {case_geo_path}")
    os.makedirs(case_geo_path, exist_ok=True)

    # Export boundary condition information
    patch_file = os.path.join(case_geo_path, "patch_info.csv")
    logger.info(f"    * Exporting boundary condition data to: {patch_file}")
    patch_df.to_csv(patch_file, index=False)
    logger.info(f"    * Exported {len(patch_df)} boundary condition patches")

    # Export geometry mesh
    vtm_path = os.path.join(case_geo_path, 'geometry.vtk')
    logger.info(f"    * Exporting geometry mesh to: {vtm_path}")
    logger.info(f"    * Mesh contains {vtk_mesh.n_cells} cells and {vtk_mesh.n_points} points")
    vtk_mesh.save(vtm_path)
    
    logger.info("    * Geometry export completed successfully")