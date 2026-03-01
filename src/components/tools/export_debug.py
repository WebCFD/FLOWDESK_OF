import os
import logging
import pyvista as pv


def load_foam_results_transient(sim_path: str):
    """
    Load OpenFOAM results for transient analysis (t=0 and t=final).
    
    Returns:
        tuple: (mesh_t0, mesh_tf, multiblock_t0, multiblock_tf, time_values)
            - mesh_t0: Internal mesh at t=0 (initial state)
            - mesh_tf: Internal mesh at t=final (final state)
            - multiblock_t0: MultiBlock at t=0 with patches
            - multiblock_tf: MultiBlock at t=final with patches
            - time_values: Array of available timesteps
    """
    logger = logging.getLogger(__name__)
    
    foam_path = os.path.join(sim_path, "results.foam")
    
    # Check if results.foam exists, create if needed
    if not os.path.exists(foam_path):
        logger.info(f"    * Creating results.foam at: {foam_path}")
        open(foam_path, 'a').close()
    
    logger.info(f"    * Loading transient results from: {foam_path}")
    
    try:
        reader = pv.get_reader(foam_path)
        logger.info(f"    * Reader created successfully")
        
        # Check if time values exist
        if not hasattr(reader, 'time_values') or len(reader.time_values) == 0:
            logger.warning(f"    * No time values found - cannot load transient data")
            return None, None, None, None, None
        
        time_values = reader.time_values
        logger.info(f"    * Available time values: {time_values}")
        
        if len(time_values) < 2:
            logger.warning(f"    * Less than 2 timesteps - cannot compute transient metrics")
            return None, None, None, None, time_values
        
        # Load t=0 (first timestep)
        logger.info(f"    * Loading t=0 (initial): {time_values[0]}")
        reader.set_active_time_value(time_values[0])
        mesh_raw_t0 = reader.read()
        
        multiblock_t0 = None
        if isinstance(mesh_raw_t0, pv.MultiBlock):
            multiblock_t0 = mesh_raw_t0
            mesh_t0 = mesh_raw_t0.combine()
        else:
            mesh_t0 = mesh_raw_t0
        
        logger.info(f"    * t=0 mesh loaded: {mesh_t0.n_cells} cells")
        
        # Load t=final (last timestep)
        logger.info(f"    * Loading t=final: {time_values[-1]}")
        reader.set_active_time_value(time_values[-1])
        mesh_raw_tf = reader.read()
        
        multiblock_tf = None
        if isinstance(mesh_raw_tf, pv.MultiBlock):
            multiblock_tf = mesh_raw_tf
            mesh_tf = mesh_raw_tf.combine()
        else:
            mesh_tf = mesh_raw_tf
        
        logger.info(f"    * t=final mesh loaded: {mesh_tf.n_cells} cells")
        logger.info(f"    * Transient data loaded successfully (Δt = {time_values[-1] - time_values[0]:.1f}s)")
        
        return mesh_t0, mesh_tf, multiblock_t0, multiblock_tf, time_values
    
    except Exception as e:
        logger.error(f"    * ERROR loading transient results: {str(e)}")
        import traceback
        logger.error(f"    * Traceback: {traceback.format_exc()}")
        raise


def load_foam_results(sim_path: str):
    """
    Load OpenFOAM results and separate internal volume from surfaces.

    Returns:
        tuple: (internal_mesh, surfaces_mesh, multiblock)
            - internal_mesh: Combined mesh with all cells
            - surfaces_mesh: Boundary surfaces
            - multiblock: Original MultiBlock with separate patches (or None)
    """
    logger = logging.getLogger(__name__)
    
    foam_path = os.path.join(sim_path, "results.foam")
    
    # Check if results.foam exists, create if needed
    if not os.path.exists(foam_path):
        logger.info(f"    * Creating results.foam at: {foam_path}")
        open(foam_path, 'a').close()
    
    logger.info(f"    * Loading results from: {foam_path}")
    
    try:
        reader = pv.get_reader(foam_path)
        logger.info(f"    * Reader created successfully")
        
        # Check if time values exist
        if not hasattr(reader, 'time_values') or len(reader.time_values) == 0:
            logger.warning(f"    * No time values found, using default read")
            mesh_raw = reader.read()
        else:
            logger.info(f"    * Available time values: {reader.time_values}")
            reader.set_active_time_value(reader.time_values[-1])
            logger.info(f"    * Set active time to: {reader.time_values[-1]}")
            # Read complete mesh
            mesh_raw = reader.read()
        
        logger.info(f"    * Mesh loaded successfully")

        # Keep original MultiBlock for patch filtering
        multiblock = None
        if isinstance(mesh_raw, pv.MultiBlock):
            logger.info(f"    * MultiBlock detected with {len(mesh_raw)} blocks")
            multiblock = mesh_raw  # Keep original
            # Combine all blocks into a single UnstructuredGrid
            mesh = mesh_raw.combine()
        else:
            mesh = mesh_raw

        # Use complete mesh without filtering cell types
        # This includes POLYHEDRON cells from snappyHexMesh
        internal_mesh = mesh

        # Surface mesh: boundary faces
        surfaces_mesh = mesh.extract_surface()

        logger.info(f"    * Extraction complete - internal cells: {internal_mesh.n_cells}, surface cells: {surfaces_mesh.n_cells}")
        
        return internal_mesh, surfaces_mesh, multiblock
    
    except Exception as e:
        logger.error(f"    * ERROR loading results: {str(e)}")
        import traceback
        logger.error(f"    * Traceback: {traceback.format_exc()}")
        raise
