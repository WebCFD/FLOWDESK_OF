import numpy as np
import pyvista as pv
import pandas as pd
import logging

from typing import Tuple

logger = logging.getLogger(__name__)


def apply_boundary_conditions_to_geometry(geometry_mesh: pv.PolyData, boundary_conditions_df: pd.DataFrame) -> Tuple[pv.PolyData, pd.DataFrame]:
    """
    Apply boundary conditions and compute surface normals for CFD simulation.
    
    This function:
    1. Initializes cell data arrays for velocity, temperature, and boundary condition types
    2. Computes surface normals for each cell
    3. Applies appropriate boundary conditions based on patch types
    4. Calculates velocity vectors for inlet/outlet boundaries
    
    Args:
        geometry_mesh: 3D geometry mesh with patch_id cell data
        boundary_conditions_df: DataFrame containing boundary condition definitions
        
    Returns:
        Tuple of (geometry_mesh_with_bc, updated_boundary_conditions_df)
    """
    logger.info(f"    * Applying boundary conditions to {geometry_mesh.n_cells} cells across {len(boundary_conditions_df)} patches")
    
    # Initialize cell data arrays for CFD simulation
    logger.info("    * Initializing cell data arrays for CFD simulation")
    geometry_mesh.cell_data['U'] = [np.array([0, 0, 0]).astype(np.float32)] * geometry_mesh.n_cells
    geometry_mesh.cell_data['T'] = [np.nan] * geometry_mesh.n_cells
    geometry_mesh.cell_data['BC_type'] = [0] * geometry_mesh.n_cells

    # Compute surface normals for boundary condition calculations
    logger.info("    * Computing surface normals for boundary condition calculations")
    geometry_mesh.compute_normals(
        inplace=True, 
        auto_orient_normals=True, 
        consistent_normals=True, 
        split_vertices=True, 
        point_normals=False
    )
    
    # Apply boundary conditions to each patch
    logger.info(f"    * Processing {len(boundary_conditions_df)} boundary condition patches")
    for patch_idx, boundary_condition in boundary_conditions_df.iterrows():
        patch_cells = np.where(geometry_mesh.cell_data['patch_id'] == patch_idx)[0]
        logger.info(f"    * Processing patch {patch_idx} ({boundary_condition['type']}) with {len(patch_cells)} cells")

        if boundary_condition['type'] == 'wall':
            # Wall boundary: set temperature, zero velocity
            logger.info(f"    * Setting wall boundary conditions for patch {patch_idx}")
            geometry_mesh.cell_data['T'][patch_cells] = [boundary_condition['T']] * len(patch_cells)
            
        elif boundary_condition['type'] == 'velocity_inlet':
            # Velocity inlet: calculate velocity vector from normal direction
            logger.info(f"    * Setting velocity inlet boundary conditions for patch {patch_idx}")
            surface_normals = geometry_mesh.cell_data['Normals'][patch_cells]
            nx = -surface_normals[:, 0]  # Inward normal direction
            ny = -surface_normals[:, 1]
            nz = -surface_normals[:, 2]
            
            # Store average normal direction in boundary conditions
            boundary_conditions_df.loc[patch_idx, 'nx'] = np.mean(nx)       
            boundary_conditions_df.loc[patch_idx, 'ny'] = np.mean(ny)
            boundary_conditions_df.loc[patch_idx, 'nz'] = np.mean(nz)
            
            # Apply velocity and temperature
            geometry_mesh.cell_data['BC_type'][patch_cells] = 1  # Velocity inlet type
            velocity_magnitude = boundary_condition['U']
            geometry_mesh.cell_data['U'][patch_cells] = velocity_magnitude * np.column_stack([nx, ny, nz])
            geometry_mesh.cell_data['T'][patch_cells] = [boundary_condition['T']] * len(patch_cells)
            
        elif boundary_condition['type'] == 'pressure_inlet':
            # Pressure inlet: pressure-driven flow, temperature specified
            logger.info(f"    * Setting pressure inlet boundary conditions for patch {patch_idx}")
            surface_normals = geometry_mesh.cell_data['Normals'][patch_cells]
            nx = -surface_normals[:, 0]
            ny = -surface_normals[:, 1]
            nz = -surface_normals[:, 2]
            
            # Store average normal direction in boundary conditions
            boundary_conditions_df.loc[patch_idx, 'nx'] = np.mean(nx)       
            boundary_conditions_df.loc[patch_idx, 'ny'] = np.mean(ny)
            boundary_conditions_df.loc[patch_idx, 'nz'] = np.mean(nz)
            
            geometry_mesh.cell_data['BC_type'][patch_cells] = 3  # Pressure inlet type
            geometry_mesh.cell_data['U'][patch_cells] = np.multiply(np.nan, np.column_stack([nx, ny, nz]))
            geometry_mesh.cell_data['T'][patch_cells] = [boundary_condition['T']] * len(patch_cells)
            
        elif boundary_condition['type'] == 'pressure_outlet':
            # Pressure outlet: pressure-driven outflow, temperature specified
            logger.info(f"    * Setting pressure outlet boundary conditions for patch {patch_idx}")
            surface_normals = geometry_mesh.cell_data['Normals'][patch_cells]
            nx = -surface_normals[:, 0]
            ny = -surface_normals[:, 1]
            nz = -surface_normals[:, 2]
            
            # Store average normal direction in boundary conditions
            boundary_conditions_df.loc[patch_idx, 'nx'] = np.mean(nx)       
            boundary_conditions_df.loc[patch_idx, 'ny'] = np.mean(ny)
            boundary_conditions_df.loc[patch_idx, 'nz'] = np.mean(nz)
            
            geometry_mesh.cell_data['BC_type'][patch_cells] = 2  # Pressure outlet type
            geometry_mesh.cell_data['U'][patch_cells] = np.multiply(np.nan, np.column_stack([nx, ny, nz]))
            geometry_mesh.cell_data['T'][patch_cells] = [boundary_condition['T']] * len(patch_cells)
        else:
            logger.error(f"    * Unknown boundary condition type: {boundary_condition['type']}")
            raise BaseException(f'Unknown boundary condition type: {boundary_condition["type"]}')

    # Clean up temporary data
    logger.info("    * Cleaning up temporary data arrays")
    del geometry_mesh.cell_data['Normals']
    del geometry_mesh.point_data['pyvistaOriginalPointIds']
    
    logger.info("    * Boundary conditions applied successfully")
    return geometry_mesh, boundary_conditions_df
 

def finalise_geometry(geometry_mesh: pv.PolyData, boundary_conditions_df: pd.DataFrame) -> Tuple[pv.PolyData, pd.DataFrame]:
    """
    Finalize geometry by applying boundary conditions and preparing for CFD simulation.
    
    Args:
        geometry_mesh: 3D geometry mesh
        boundary_conditions_df: DataFrame containing boundary condition definitions
        
    Returns:
        Tuple of (finalized_geometry_mesh, updated_boundary_conditions_df)
    """
    logger.info("    * Finalizing geometry with boundary conditions")
    geometry_mesh, boundary_conditions_df = apply_boundary_conditions_to_geometry(geometry_mesh, boundary_conditions_df)
    logger.info("    * Geometry finalization completed successfully")
    return geometry_mesh, boundary_conditions_df
