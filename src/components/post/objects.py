import os
import json
import logging
import numpy as np
import pyvista as pv

from src.components.tools.export_debug import load_foam_results


logger = logging.getLogger(__name__)


def calculate_comfort_metrics(slice_mesh):
    """
    Calculate comfort metrics from a slice mesh with PMV and PPD fields.
    
    Args:
        slice_mesh: PyVista mesh slice with PMV and PPD point data
        
    Returns:
        dict: Comfort metrics (comfort %, PMV stats, PPD stats)
    """
    # Extract PMV and PPD arrays
    pmv = slice_mesh.point_data.get('PMV', None)
    ppd = slice_mesh.point_data.get('PPD', None)
    
    if pmv is None or ppd is None:
        logger.warning("PMV or PPD fields not found in slice mesh")
        return {
            'error': 'PMV/PPD fields not found',
            'comfort_area_pct': 0.0,
            'pmv_mean': 0.0,
            'pmv_std': 0.0,
            'pmv_min': 0.0,
            'pmv_max': 0.0,
            'ppd_mean': 0.0,
            'ppd_max': 0.0
        }
    
    # Filter valid values (PMV != -1000 sentinel value from calculate_comfort.py)
    INVALID_VALUE = -1000.0
    valid_mask = (pmv != INVALID_VALUE) & (ppd != INVALID_VALUE)
    
    if valid_mask.sum() == 0:
        logger.warning("No valid PMV/PPD values in slice")
        return {
            'error': 'No valid values',
            'comfort_area_pct': 0.0,
            'pmv_mean': 0.0,
            'pmv_std': 0.0,
            'pmv_min': 0.0,
            'pmv_max': 0.0,
            'ppd_mean': 0.0,
            'ppd_max': 0.0
        }
    
    valid_pmv = pmv[valid_mask]
    valid_ppd = ppd[valid_mask]
    
    # Calculate % area in comfort zone (ISO 7730: -0.5 < PMV < 0.5)
    comfort_mask = (valid_pmv >= -0.5) & (valid_pmv <= 0.5)
    comfort_pct = 100.0 * comfort_mask.sum() / len(valid_pmv)
    
    # Calculate PMV statistics
    pmv_mean = float(valid_pmv.mean())
    pmv_std = float(valid_pmv.std())
    pmv_min = float(valid_pmv.min())
    pmv_max = float(valid_pmv.max())
    
    # Calculate PPD statistics
    ppd_mean = float(valid_ppd.mean())
    ppd_max = float(valid_ppd.max())
    
    return {
        'comfort_area_pct': float(comfort_pct),
        'pmv_mean': pmv_mean,
        'pmv_std': pmv_std,
        'pmv_min': pmv_min,
        'pmv_max': pmv_max,
        'ppd_mean': ppd_mean,
        'ppd_max': ppd_max,
        'n_points': int(len(valid_pmv))
    }


def render_isometric_png(slice_mesh, name, z_height, post_path, variable='PMV'):
    """
    Render isometric PNG image of a slice mesh.
    
    Args:
        slice_mesh: PyVista mesh slice
        name: Plane name (seated, standing, head)
        z_height: Height of plane [m]
        post_path: Path to post-processing directory
        variable: Variable to render ('PMV' or 'PPD')
    """
    # Configure colormap and limits
    if variable == 'PMV':
        cmap = 'coolwarm'
        clim = (-3, 3)
        label = 'PMV'
    elif variable == 'PPD':
        cmap = 'inferno_r'
        clim = (0, 100)
        label = 'PPD [%]'
    else:
        logger.warning(f"Unknown variable {variable}, using default")
        cmap = 'viridis'
        clim = None
        label = variable
    
    # Create plotter
    plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
    
    # Add mesh with scalar field
    plotter.add_mesh(
        slice_mesh,
        scalars=variable,
        cmap=cmap,
        clim=clim,
        show_edges=False,
        scalar_bar_args={
            'title': label,
            'title_font_size': 20,
            'label_font_size': 16,
            'shadow': True,
            'n_labels': 5,
            'italic': False,
            'fmt': '%.1f',
            'font_family': 'arial'
        }
    )
    
    # Set isometric camera view
    plotter.camera_position = 'iso'
    plotter.camera.zoom(1.2)
    
    # Add title
    title = f'{variable} at {z_height}m ({name.capitalize()})'
    plotter.add_text(title, position='upper_edge', font_size=14, color='black')
    
    # Save PNG
    images_dir = os.path.join(post_path, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    png_path = os.path.join(images_dir, f'comfort_plane_{name}_{z_height}m_{variable}.png')
    plotter.screenshot(png_path, transparent_background=False)
    plotter.close()
    
    logger.info(f"       Saved PNG: {os.path.basename(png_path)}")


def analyze_comfort_planes(sim_path, post_path):
    """
    Analyze PMV/PPD thermal comfort in 3 horizontal planes.
    
    Generates:
    - VTK files of slices for each plane
    - PNG images (isometric view) for PMV and PPD
    - JSON file with comfort metrics
    
    Args:
        sim_path: Path to simulation directory (with VTK/ folder)
        post_path: Path to post-processing output directory
        
    Returns:
        dict: Comfort metrics for all planes
    """
    logger.info("    * Analyzing thermal comfort in horizontal planes")
    
    # Load 3D mesh with PMV/PPD fields
    logger.info("    * Loading CFD results from VTK")
    internal_mesh, _ = load_foam_results(sim_path)
    
    logger.info(f"    * Loaded mesh with {internal_mesh.n_cells:,} cells")
    
    # Check if PMV and PPD fields exist
    if 'PMV' not in internal_mesh.point_data or 'PPD' not in internal_mesh.point_data:
        logger.error("PMV/PPD fields not found in mesh!")
        logger.error("Available fields: " + str(list(internal_mesh.point_data.keys())))
        raise ValueError("PMV/PPD fields not found. Run calculate_comfort.py first.")
    
    # Define analysis planes (heights in meters)
    planes = {
        'seated':  0.6,   # Seated person (ankle height)
        'standing': 1.1,  # Standing person (waist height)
        'head':    1.7    # Head height
    }
    
    logger.info(f"    * Analyzing {len(planes)} horizontal planes:")
    for name, z in planes.items():
        logger.info(f"       - {name.capitalize()}: z = {z}m")
    
    # Create output directories
    vtk_dir = os.path.join(post_path, 'vtk')
    os.makedirs(vtk_dir, exist_ok=True)
    os.makedirs(os.path.join(post_path, 'images'), exist_ok=True)
    
    # Process each plane
    results = {}
    
    for name, z_height in planes.items():
        logger.info(f"    * Processing plane: {name} (z={z_height}m)")
        
        # Create slice
        with pv.vtk_verbosity('off'):
            slice_mesh = internal_mesh.slice(normal='z', origin=(0, 0, z_height))
        
        logger.info(f"       Slice has {slice_mesh.n_points:,} points")
        
        # Calculate comfort metrics
        metrics = calculate_comfort_metrics(slice_mesh)
        metrics['height_m'] = z_height
        metrics['plane_name'] = name
        
        logger.info(f"       Comfort area: {metrics['comfort_area_pct']:.1f}%")
        logger.info(f"       PMV: mean={metrics['pmv_mean']:.2f}, range=[{metrics['pmv_min']:.2f}, {metrics['pmv_max']:.2f}]")
        logger.info(f"       PPD: mean={metrics['ppd_mean']:.1f}%, max={metrics['ppd_max']:.1f}%")
        
        results[name] = metrics
        
        # Save VTK slice
        vtk_path = os.path.join(vtk_dir, f'comfort_plane_{name}_{z_height}m.vtk')
        slice_mesh.save(vtk_path)
        logger.info(f"       Saved VTK: {os.path.basename(vtk_path)}")
        
        # Render PMV image
        render_isometric_png(slice_mesh, name, z_height, post_path, variable='PMV')
        
        # Render PPD image
        render_isometric_png(slice_mesh, name, z_height, post_path, variable='PPD')
    
    # Save metrics to JSON
    metrics_path = os.path.join(post_path, 'comfort_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"    * Saved comfort metrics: {os.path.basename(metrics_path)}")
    logger.info("    * Thermal comfort analysis completed successfully")
    
    return results
