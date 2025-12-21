import logging
import numpy as np
import pyvista as pv
import pymeshlab as pyml

from typing import List
from src.components.tools.performance import (
    optimize_mesh_memory,
    PerformanceMonitor
)

logger = logging.getLogger(__name__)

def integers_to_rgb_matrix(values):
    values = np.asarray(values, dtype=np.uint64)
    if np.any(values < 0) or np.any(values >= np.power(256, 3)):
        raise ValueError("All values must be in the range [0, 16,777,215]")
    byte1 = (values >> 16) & 0xFF
    byte2 = (values >> 8) & 0xFF
    byte3 = values & 0xFF
    byte4 = np.full_like(byte3, 1, dtype=np.float64)
    return np.stack([byte1/255, byte2/255, byte3/255, byte4], axis=1)


def rgba_matrix_to_integers(matrix):
    matrix = 255*np.asarray(matrix, dtype=np.float64)
    if matrix.shape[1] not in (3, 4):
        raise ValueError("Input must have shape Nx3 (RGB) or Nx4 (RGBA)")
    r = matrix[:, 0].astype(np.uint32)
    g = matrix[:, 1].astype(np.uint32)
    b = matrix[:, 2].astype(np.uint32)
    return (r << 16) + (g << 8) + b


def polydata2meshlab(vtk_polydata):
    pv_mesh = pv.wrap(vtk_polydata)
    points = pv_mesh.points
    faces = pv_mesh.faces.reshape((-1, 4))[:, 1:4]
    cell_index = integers_to_rgb_matrix(pv_mesh.cell_data['patch_id'])
    return pyml.Mesh(vertex_matrix=points, face_matrix=faces, f_color_matrix=cell_index)


def meshlab2polydata(mesh):
    raw_points = mesh.vertex_matrix()
    raw_faces = mesh.face_matrix()
    raw_indices = mesh.face_color_matrix()

    points = raw_points.astype(np.float64)
    faces = np.hstack([np.full((raw_faces.shape[0], 1), 3), raw_faces]).astype(np.int64).flatten()
    indices = rgba_matrix_to_integers(raw_indices)

    polydata = pv.PolyData(points, faces)
    polydata.cell_data['patch_id'] = indices
    return polydata


def merge_volumes_sequential(volume_meshes: List[pv.PolyData]) -> pv.PolyData:
    """Sequential volume merging using PyMeshLab."""
    ms = pyml.MeshSet()

    # Add first mesh
    new_mesh = polydata2meshlab(volume_meshes[0])
    ms.add_mesh(new_mesh, "0")
    
    # Merge remaining meshes one by one
    for idx, volume_mesh in enumerate(volume_meshes[1:], 1):
        new_mesh = polydata2meshlab(volume_mesh)
        ms.add_mesh(new_mesh, str(idx))
        
        # Perform boolean union
        last_idx = ms.mesh_number() - 2
        ms.generate_boolean_union(
            first_mesh=last_idx, 
            second_mesh=idx, 
            transfer_face_color=True
        )

    # Finalize the result
    ms.set_current_mesh(new_curr_id=ms.mesh_number() - 1)
    ms.meshing_merge_close_vertices()
    united_mesh = ms.mesh(ms.mesh_number() - 1)
    result = meshlab2polydata(united_mesh)
    return optimize_mesh_memory(result)


def subtract_objects(volume_mesh: pv.PolyData, object_meshes: List[pv.PolyData]) -> pv.PolyData:
    """Sequential object subtraction using PyMeshLab."""
    ms = pyml.MeshSet()

    # Add volume mesh
    volume_mesh_lab = polydata2meshlab(volume_mesh)
    ms.add_mesh(volume_mesh_lab, 'volume')

    # Add object meshes
    for idx, object_mesh in enumerate(object_meshes):
        object_mesh_lab = polydata2meshlab(object_mesh)
        ms.add_mesh(object_mesh_lab, f'object_{idx}')
    
    # Perform boolean differences
    if len(object_meshes) > 0:
        ms.generate_boolean_difference(
            first_mesh=0, 
            second_mesh=1, 
            transfer_face_color=True
        )
        
        for idx in range(2, len(object_meshes) + 1):
            last_idx = ms.mesh_number() - 1
            ms.generate_boolean_difference(
                first_mesh=last_idx, 
                second_mesh=idx, 
                transfer_face_color=True
            )

    result_mesh = ms.mesh(ms.mesh_number() - 1)
    result = meshlab2polydata(result_mesh)
    return optimize_mesh_memory(result)


def perform_boolean_operations(room_geometry_meshes: List[pv.PolyData], furniture_meshes: List[pv.PolyData]) -> pv.PolyData:
    """
    Combine room geometry and subtract furniture using boolean operations.
    
    This function performs the following operations:
    1. Merges all room geometry meshes (walls, floors, ceilings) into a single volume
    2. Subtracts furniture meshes from the room volume to create air spaces
    
    Args:
        room_geometry_meshes: List of room geometry meshes to merge (walls, floors, ceilings)
        furniture_meshes: List of furniture meshes to subtract from room volume
        
    Returns:
        Final combined geometry mesh after all boolean operations
    """
    performance_monitor = PerformanceMonitor()
    performance_monitor.start()
    
    logger.info(f"    * Performing boolean operations on {len(room_geometry_meshes)} room geometries and {len(furniture_meshes)} furniture objects")
    
    # Step 1: Merge all room geometry into a single volume
    if len(room_geometry_meshes) == 0:
        logger.error("    * No room geometry meshes provided for boolean operations")
        raise BaseException("No room geometry meshes provided")
    elif len(room_geometry_meshes) == 1:
        logger.info("    * Single room geometry mesh found, skipping merge step")
        combined_room_volume = room_geometry_meshes[0]
    else:
        logger.info(f"    * Merging {len(room_geometry_meshes)} room geometry meshes into single volume")
        combined_room_volume = pv.merge(room_geometry_meshes)
        logger.info(f"    * Merged volume contains {combined_room_volume.n_cells} cells and {combined_room_volume.n_points} points")
        performance_monitor.update_memory()

    # Step 2: Subtract furniture from room volume to create air spaces
    if len(furniture_meshes) > 0:
        logger.info(f"    * Subtracting {len(furniture_meshes)} furniture meshes from room volume")
        combined_room_volume = subtract_objects(combined_room_volume, furniture_meshes)
        logger.info(f"    * After furniture subtraction: {combined_room_volume.n_cells} cells and {combined_room_volume.n_points} points")
        performance_monitor.update_memory()
    else:
        logger.info("    * No furniture meshes to subtract")
    
    # Log performance summary
    performance_summary = performance_monitor.get_summary()
    logger.info(f"    * Boolean operations completed: {performance_summary['total_time']:.2f}s, peak memory: {performance_summary['peak_memory_mb']:.1f}MB")
    
    return combined_room_volume