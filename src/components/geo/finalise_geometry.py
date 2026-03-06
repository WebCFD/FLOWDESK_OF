import numpy as np
import pyvista as pv
import pandas as pd
import logging

from typing import Tuple

logger = logging.getLogger(__name__)


def _compute_fluid_direction(
    inward_mean: np.ndarray,
    row: pd.Series,
) -> np.ndarray:
    """
    Compute fluid flow direction from mesh inward normal + optional airOrientation rotation.
    Also performs a coherence check against the JSON outward normal (warns if angle > 1°).

    Convention:
        - inward_mean   : mean inward normal computed from PyVista mesh surface normals
                          (already negated: = -surface_normals_mean)
        - row           : boundary condition row that MAY contain:
                          json_nx/ny/nz   → JSON outward normal (for coherence check)
                          airOrientation_v/h → deflection angles in degrees (NaN if absent)

    Returns:
        fluid_direction: np.ndarray shape (3,), normalized inward flow direction
                         (with airOrientation applied if available)
    """
    from src.components.geo.create_volumes import angles_to_direction_vector

    # ── 1. Coherence check: mesh inward vs -(JSON outward normal) ───────────
    if all(c in row.index for c in ('json_nx', 'json_ny', 'json_nz')):
        jn = np.array([row['json_nx'], row['json_ny'], row['json_nz']], dtype=float)
        json_inward = -jn  # JSON gives outward → negate for inward

        n_jnorm = np.linalg.norm(json_inward)
        n_mnorm = np.linalg.norm(inward_mean)

        if n_jnorm > 1e-9 and n_mnorm > 1e-9:
            n_j = json_inward / n_jnorm
            n_m = inward_mean / n_mnorm
            dot = float(np.clip(np.dot(n_j, n_m), -1.0, 1.0))
            angle_deg = float(np.degrees(np.arccos(dot)))
            if angle_deg > 1.0:
                logger.warning(
                    f"    ⚠️  Normal mismatch '{row['id']}': "
                    f"mesh inward={np.round(n_m, 3).tolist()} "
                    f"vs JSON inward={np.round(n_j, 3).tolist()} "
                    f"(Δangle={angle_deg:.2f}°) — using mesh normal"
                )

    # ── 2. Compute fluid direction: mesh inward + airOrientation ────────────
    v = row.get('airOrientation_v', np.nan)
    h = row.get('airOrientation_h', np.nan)

    n_norm = np.linalg.norm(inward_mean)
    base = inward_mean / n_norm if n_norm > 1e-9 else inward_mean

    if pd.notna(v) and pd.notna(h):
        fluid_dir = angles_to_direction_vector(float(v), float(h), base)
    else:
        fluid_dir = base

    return fluid_dir


def apply_boundary_conditions_to_geometry(geometry_mesh: pv.PolyData, boundary_conditions_df: pd.DataFrame) -> Tuple[pv.PolyData, pd.DataFrame]:
    """
    Apply boundary conditions and compute surface normals for CFD simulation.
    
    This function:
    1. Initializes cell data arrays for velocity, temperature, and boundary condition types
    2. Computes surface normals for each cell
    3. Applies appropriate boundary conditions based on patch types
    4. Calculates velocity vectors for inlet/outlet boundaries
    5. Sets nx/ny/nz (inward from mesh) and fluid_nx/ny/nz (inward + airOrientation)
       as the single source of truth for all flow direction data.
    
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
            
            # Store mesh-computed inward normal (single source of truth)
            boundary_conditions_df.loc[patch_idx, 'nx'] = np.mean(nx)       
            boundary_conditions_df.loc[patch_idx, 'ny'] = np.mean(ny)
            boundary_conditions_df.loc[patch_idx, 'nz'] = np.mean(nz)

            # Compute fluid flow direction: inward + airOrientation rotation
            inward_mean = np.array([np.mean(nx), np.mean(ny), np.mean(nz)])
            fluid_dir = _compute_fluid_direction(inward_mean, boundary_condition)
            boundary_conditions_df.loc[patch_idx, 'fluid_nx'] = fluid_dir[0]
            boundary_conditions_df.loc[patch_idx, 'fluid_ny'] = fluid_dir[1]
            boundary_conditions_df.loc[patch_idx, 'fluid_nz'] = fluid_dir[2]
            
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
            
            # Store mesh-computed inward normal (single source of truth)
            boundary_conditions_df.loc[patch_idx, 'nx'] = np.mean(nx)       
            boundary_conditions_df.loc[patch_idx, 'ny'] = np.mean(ny)
            boundary_conditions_df.loc[patch_idx, 'nz'] = np.mean(nz)

            # Compute fluid flow direction: inward + airOrientation rotation
            inward_mean = np.array([np.mean(nx), np.mean(ny), np.mean(nz)])
            fluid_dir = _compute_fluid_direction(inward_mean, boundary_condition)
            boundary_conditions_df.loc[patch_idx, 'fluid_nx'] = fluid_dir[0]
            boundary_conditions_df.loc[patch_idx, 'fluid_ny'] = fluid_dir[1]
            boundary_conditions_df.loc[patch_idx, 'fluid_nz'] = fluid_dir[2]
            
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
            
            # Store mesh-computed inward normal (single source of truth)
            boundary_conditions_df.loc[patch_idx, 'nx'] = np.mean(nx)       
            boundary_conditions_df.loc[patch_idx, 'ny'] = np.mean(ny)
            boundary_conditions_df.loc[patch_idx, 'nz'] = np.mean(nz)

            # Compute fluid flow direction: inward + airOrientation rotation
            inward_mean = np.array([np.mean(nx), np.mean(ny), np.mean(nz)])
            fluid_dir = _compute_fluid_direction(inward_mean, boundary_condition)
            boundary_conditions_df.loc[patch_idx, 'fluid_nx'] = fluid_dir[0]
            boundary_conditions_df.loc[patch_idx, 'fluid_ny'] = fluid_dir[1]
            boundary_conditions_df.loc[patch_idx, 'fluid_nz'] = fluid_dir[2]
            
            geometry_mesh.cell_data['BC_type'][patch_cells] = 2  # Pressure outlet type
            geometry_mesh.cell_data['U'][patch_cells] = np.multiply(np.nan, np.column_stack([nx, ny, nz]))
            geometry_mesh.cell_data['T'][patch_cells] = [boundary_condition['T']] * len(patch_cells)

        elif boundary_condition['type'] == 'mass_flow_inlet':
            # Mass flow inlet: flow rate specified, temperature fixed, direction from normals
            logger.info(f"    * Setting mass_flow_inlet boundary conditions for patch {patch_idx}")
            surface_normals = geometry_mesh.cell_data['Normals'][patch_cells]
            nx = -surface_normals[:, 0]
            ny = -surface_normals[:, 1]
            nz = -surface_normals[:, 2]

            # Store mesh-computed inward normal (single source of truth)
            boundary_conditions_df.loc[patch_idx, 'nx'] = np.mean(nx)
            boundary_conditions_df.loc[patch_idx, 'ny'] = np.mean(ny)
            boundary_conditions_df.loc[patch_idx, 'nz'] = np.mean(nz)

            # Compute fluid flow direction: inward + airOrientation rotation
            inward_mean = np.array([np.mean(nx), np.mean(ny), np.mean(nz)])
            fluid_dir = _compute_fluid_direction(inward_mean, boundary_condition)
            boundary_conditions_df.loc[patch_idx, 'fluid_nx'] = fluid_dir[0]
            boundary_conditions_df.loc[patch_idx, 'fluid_ny'] = fluid_dir[1]
            boundary_conditions_df.loc[patch_idx, 'fluid_nz'] = fluid_dir[2]

            geometry_mesh.cell_data['BC_type'][patch_cells] = 4  # mass_flow_inlet
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
