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


def render_isometric_png(slice_mesh, name, z_height, post_path, variable='PMV', show_contours=True):
    """
    Render isometric PNG image of a slice mesh.
    
    Args:
        slice_mesh: PyVista mesh slice
        name: Plane name (seated, standing, head)
        z_height: Height of plane [m]
        post_path: Path to post-processing directory
        variable: Variable to render ('PMV' or 'PPD')
        show_contours: Whether to show isocontour lines with labels (default: True for PMV)
    """
    # Configure colormap and limits
    if variable == 'PMV':
        cmap = 'coolwarm'
        clim = (-3, 3)
        label = 'PMV'
        # PMV isocontour levels (ASHRAE 55 zones)
        contour_levels = [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]
    elif variable == 'PPD':
        cmap = 'inferno_r'
        clim = (0, 100)
        label = 'PPD [%]'
        # PPD isocontour levels (every 10%)
        contour_levels = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        show_contours = True  # Enable contours for PPD
    else:
        logger.warning(f"Unknown variable {variable}, using default")
        cmap = 'viridis'
        clim = None
        label = variable
        show_contours = False
    
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
            'title_font_size': 38,      # Reduced 52.5% total from 80 (igual flow_report)
            'label_font_size': 64,
            'shadow': True,
            'n_labels': 5,
            'italic': False,
            'fmt': '%.1f',
            'font_family': 'arial',
            'vertical': True,           # Vertical orientation only
            'position_x': 0.85,         # Right side
            'position_y': 0.25,         # Centered vertically
            'height': 0.5,              # 50% of viewport height
            'width': 0.12               # Proportional width
        }
    )
    
    # Add isocontour lines with labels
    if show_contours and variable in slice_mesh.point_data:
        try:
            # Generate contours at specified levels
            contours = slice_mesh.contour(isosurfaces=contour_levels, scalars=variable)
            
            if contours.n_points > 0:
                # Add contour lines (black, thicker)
                plotter.add_mesh(
                    contours,
                    color='black',
                    line_width=3,
                    render_lines_as_tubes=False,
                    lighting=False
                )
                
                # Add labels at contour line points (subsample to avoid clutter)
                # Sample one point per contour level
                for level in contour_levels:
                    # Get points where value ≈ level (within tolerance)
                    level_contour = slice_mesh.contour(isosurfaces=[level], scalars=variable)
                    
                    if level_contour.n_points > 0:
                        # Sample one point from the middle of the contour
                        mid_idx = level_contour.n_points // 2
                        point = level_contour.points[mid_idx]
                        
                        # Create label
                        if variable == 'PMV':
                            label_text = f'{level:+.1f}'  # +1.0, -0.5, etc.
                        else:
                            label_text = f'{level:.0f}%'
                        
                        # Add label at that point
                        plotter.add_point_labels(
                            [point],
                            [label_text],
                            font_size=48,  # 3x más grande para mejor legibilidad
                            text_color='black',
                            font_family='arial',
                            fill_shape=True,
                            shape='rounded_rect',
                            shape_color='white',
                            shape_opacity=0.8,
                            point_size=0,  # Don't show point marker
                            render_points_as_spheres=False,
                            always_visible=False
                        )
                
                logger.info(f"       Added {len(contour_levels)} isocontour lines with labels")
        except Exception as e:
            logger.warning(f"       Failed to add contours: {e}")
    
    # Set isometric camera view
    plotter.camera_position = 'iso'
    plotter.camera.zoom(1.2)
    
    # Add title with units
    if variable == 'PMV':
        title = f'PMV (dimensionless) at {z_height}m ({name.capitalize()})'
    elif variable == 'PPD':
        title = f'PPD [%] at {z_height}m ({name.capitalize()})'
    else:
        title = f'{variable} at {z_height}m ({name.capitalize()})'
    
    plotter.add_text(title, position='upper_edge', font_size=16, color='black')
    
    # Add axes widget (XYZ)
    plotter.show_axes()
    
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
    internal_mesh, _, _ = load_foam_results(sim_path)
    
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
        
        # Create slice (suppress VTK warnings if available)
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                slice_mesh = internal_mesh.slice(normal='z', origin=(0, 0, z_height))
        except:
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
    
    # Convert numpy types to Python natives for JSON serialization
    def convert_numpy_types(obj):
        """Recursively convert numpy types to Python native types."""
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    # Save metrics to JSON
    metrics_path = os.path.join(post_path, 'comfort_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(convert_numpy_types(results), f, indent=2)
    
    logger.info(f"    * Saved comfort metrics: {os.path.basename(metrics_path)}")
    logger.info("    * Thermal comfort analysis completed successfully")
    
    return results


def generate_html_report(results, post_path, case_name='CFD_Case'):
    """
    Generate HTML report with thermal comfort analysis results.
    
    Creates a professional HTML report with:
    - Executive summary
    - Comfort metrics for each plane
    - PMV/PPD images side by side
    - Interpretation according to ISO 7730
    
    Args:
        results: dict with comfort metrics from analyze_comfort_planes()
        post_path: Path to post-processing directory
        case_name: Name of the simulation case
    """
    logger.info("    * Generating HTML comfort report")
    
    # Get timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate global comfort (average of all planes)
    global_comfort = np.mean([r['comfort_area_pct'] for r in results.values()])
    
    # Build HTML content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Thermal Comfort Report - {case_name}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 300;
        }}
        
        .header .subtitle {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .executive-summary {{
            background: #f8f9fa;
            padding: 30px;
            border-bottom: 3px solid #e9ecef;
        }}
        
        .executive-summary h2 {{
            margin-bottom: 15px;
            color: #495057;
        }}
        
        .summary-metric {{
            display: inline-block;
            padding: 15px 30px;
            margin: 10px;
            background: white;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        
        .summary-metric .label {{
            font-size: 0.9em;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .summary-metric .value {{
            font-size: 2em;
            font-weight: bold;
            margin-top: 5px;
        }}
        
        .comfort-high {{ color: #28a745; }}
        .comfort-medium {{ color: #ffc107; }}
        .comfort-low {{ color: #dc3545; }}
        
        .plane-section {{
            padding: 40px;
            border-bottom: 1px solid #e9ecef;
        }}
        
        .plane-section:last-child {{
            border-bottom: none;
        }}
        
        .plane-header {{
            margin-bottom: 25px;
        }}
        
        .plane-header h2 {{
            color: #495057;
            font-size: 1.8em;
            margin-bottom: 5px;
        }}
        
        .plane-header .height-info {{
            color: #6c757d;
            font-size: 1.1em;
        }}
        
        .images-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .image-box {{
            background: #f8f9fa;
            border-radius: 6px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .image-box img {{
            width: 100%;
            display: block;
        }}
        
        .image-box .caption {{
            padding: 15px;
            text-align: center;
            font-weight: 600;
            background: white;
            border-top: 2px solid #e9ecef;
        }}
        
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            background: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border-radius: 6px;
            overflow: hidden;
        }}
        
        .metrics-table th,
        .metrics-table td {{
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #e9ecef;
        }}
        
        .metrics-table th {{
            background: #f8f9fa;
            font-weight: 600;
            color: #495057;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 0.5px;
        }}
        
        .metrics-table tr:last-child td {{
            border-bottom: none;
        }}
        
        .metrics-table .metric-value {{
            font-weight: 600;
            font-size: 1.1em;
        }}
        
        .status-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .status-excellent {{
            background: #d4edda;
            color: #155724;
        }}
        
        .status-good {{
            background: #fff3cd;
            color: #856404;
        }}
        
        .status-poor {{
            background: #f8d7da;
            color: #721c24;
        }}
        
        .comparison-section {{
            background: #f8f9fa;
            padding: 40px;
        }}
        
        .comparison-section h2 {{
            color: #495057;
            margin-bottom: 25px;
        }}
        
        .comparison-chart {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-top: 20px;
        }}
        
        .chart-bar {{
            background: white;
            border-radius: 6px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        
        .chart-bar .bar-label {{
            font-weight: 600;
            margin-bottom: 15px;
            color: #495057;
        }}
        
        .chart-bar .bar {{
            height: 200px;
            background: #e9ecef;
            border-radius: 4px;
            position: relative;
            overflow: hidden;
        }}
        
        .chart-bar .bar-fill {{
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
            transition: height 0.3s ease;
        }}
        
        .chart-bar .bar-value {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 1.5em;
            font-weight: bold;
            color: white;
            text-shadow: 0 1px 2px rgba(0,0,0,0.3);
        }}
        
        .footer {{
            background: #343a40;
            color: #adb5bd;
            padding: 30px;
            text-align: center;
        }}
        
        .footer .info-row {{
            margin: 5px 0;
        }}
        
        .footer .label {{
            color: #6c757d;
        }}
        
        .footer .value {{
            color: #f8f9fa;
            font-weight: 600;
        }}
        
        @media (max-width: 768px) {{
            .images-container {{
                grid-template-columns: 1fr;
            }}
            
            .comparison-chart {{
                grid-template-columns: 1fr;
            }}
            
            .header h1 {{
                font-size: 1.8em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>Thermal Comfort Analysis Report</h1>
            <div class="subtitle">{case_name}</div>
            <div class="subtitle">Generated: {timestamp}</div>
        </div>
        
        <!-- Executive Summary -->
        <div class="executive-summary">
            <h2>Executive Summary</h2>
            <div class="summary-metric">
                <div class="label">Global Comfort</div>
                <div class="value {'comfort-high' if global_comfort >= 80 else 'comfort-medium' if global_comfort >= 60 else 'comfort-low'}">
                    {global_comfort:.1f}%
                </div>
                <div style="font-size: 0.75em; color: #6c757d; margin-top: 8px; line-height: 1.4;">
                    Average comfort area with PMV in neutral zone (-0.5 to +0.5)<br>
                    Across 3 analysis planes: Seated (0.6m), Standing (1.1m), Head (1.7m)<br>
                    ISO 7730 target: ≥80%
                </div>
            </div>
        </div>
"""
    
    # Add section for each plane
    plane_order = ['seated', 'standing', 'head']
    plane_titles = {
        'seated': 'Seated Level (Ankle Height)',
        'standing': 'Standing Level (Waist Height)',
        'head': 'Head Level'
    }
    
    for plane_name in plane_order:
        if plane_name not in results:
            continue
            
        metrics = results[plane_name]
        z_height = metrics['height_m']
        comfort_pct = metrics['comfort_area_pct']
        
        # Determine status
        if comfort_pct >= 80:
            status_class = 'status-excellent'
            status_text = 'Excellent'
        elif comfort_pct >= 60:
            status_class = 'status-good'
            status_text = 'Acceptable'
        else:
            status_class = 'status-poor'
            status_text = 'Poor'
        
        html_content += f"""
        <!-- Plane: {plane_name} -->
        <div class="plane-section">
            <div class="plane-header">
                <h2>{plane_titles[plane_name]}</h2>
                <div class="height-info">Height: {z_height} m</div>
                <span class="{status_class} status-badge">{status_text} Comfort</span>
            </div>
            
            <div class="images-container">
                <div class="image-box">
                    <img src="images/comfort_plane_{plane_name}_{z_height}m_PMV.png" alt="PMV at {z_height}m">
                    <div class="caption">PMV (Predicted Mean Vote)</div>
                </div>
                <div class="image-box">
                    <img src="images/comfort_plane_{plane_name}_{z_height}m_PPD.png" alt="PPD at {z_height}m">
                    <div class="caption">PPD (Predicted Percentage Dissatisfied)</div>
                </div>
            </div>
            
            <table class="metrics-table">
                <tr>
                    <th>Metric</th>
                    <th>Value Simulation</th>
                    <th>Target</th>
                </tr>
                <tr>
                    <td>Comfort area with |PMV| &lt; 0.5<br><span style="font-size: 0.9em; color: #6c757d;">Percentage in neutral zone</span></td>
                    <td class="metric-value {'comfort-high' if comfort_pct >= 80 else 'comfort-medium' if comfort_pct >= 60 else 'comfort-low'}">
                        {comfort_pct:.1f}%
                    </td>
                    <td>&ge; 80%<br><span style="font-size: 0.85em; color: #6c757d;">ISO 7730 Category B</span></td>
                </tr>
                <tr>
                    <td>PMV Mean<br><span style="font-size: 0.9em; color: #6c757d;">Plane average ± std deviation</span></td>
                    <td class="metric-value {'comfort-high' if abs(metrics['pmv_mean']) <= 0.5 else 'comfort-medium' if abs(metrics['pmv_mean']) <= 1.0 else 'comfort-low'}">
                        {metrics['pmv_mean']:.2f} &plusmn; {metrics['pmv_std']:.2f}
                    </td>
                    <td>-0.5 to +0.5<br><span style="font-size: 0.85em; color: #6c757d;">ISO 7730 Neutral zone</span></td>
                </tr>
                <tr>
                    <td>PMV Range<br><span style="font-size: 0.9em; color: #6c757d;">Min/max extremes</span></td>
                    <td class="metric-value">[{metrics['pmv_min']:.2f}, {metrics['pmv_max']:.2f}]</td>
                    <td>Within [-0.5, +0.5]<br><span style="font-size: 0.85em; color: #6c757d;">Optimal comfort range</span></td>
                </tr>
                <tr>
                    <td>PPD Mean<br><span style="font-size: 0.9em; color: #6c757d;">Plane average dissatisfaction</span></td>
                    <td class="metric-value {'comfort-high' if metrics['ppd_mean'] <= 10 else 'comfort-medium' if metrics['ppd_mean'] <= 20 else 'comfort-low'}">
                        {metrics['ppd_mean']:.1f}%
                    </td>
                    <td>&lt; 10%<br><span style="font-size: 0.85em; color: #6c757d;">ISO 7730 Category B</span></td>
                </tr>
                <tr>
                    <td>PPD Maximum<br><span style="font-size: 0.9em; color: #6c757d;">Peak dissatisfaction value</span></td>
                    <td class="metric-value {'comfort-high' if metrics['ppd_max'] <= 10 else 'comfort-medium' if metrics['ppd_max'] <= 20 else 'comfort-low'}">
                        {metrics['ppd_max']:.1f}%
                    </td>
                    <td>&lt; 10%<br><span style="font-size: 0.85em; color: #6c757d;">ISO 7730 Target</span></td>
                </tr>
            </table>
        </div>
"""
    
    # Add comparison section
    html_content += """
        <!-- Comparison Section -->
        <div class="comparison-section">
            <h2>Comfort Comparison by Height</h2>
            <div class="comparison-chart">
"""
    
    for plane_name in plane_order:
        if plane_name not in results:
            continue
        metrics = results[plane_name]
        comfort_pct = metrics['comfort_area_pct']
        z_height = metrics['height_m']
        
        html_content += f"""
                <div class="chart-bar">
                    <div class="bar-label">{plane_titles[plane_name]}<br>{z_height}m</div>
                    <div class="bar">
                        <div class="bar-fill" style="height: {comfort_pct}%;"></div>
                        <div class="bar-value">{comfort_pct:.1f}%</div>
                    </div>
                </div>
"""
    
    html_content += """
            </div>
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <div class="info-row">
                <span class="label">Analysis Standard:</span>
                <span class="value">ISO 7730:2005 - Ergonomics of the thermal environment</span>
            </div>
            <div class="info-row">
                <span class="label">Solver:</span>
                <span class="value">OpenFOAM buoyantBoussinesqPimpleFoam</span>
            </div>
            <div class="info-row">
                <span class="label">Comfort Model:</span>
                <span class="value">Fanger PMV/PPD (ISO 7730)</span>
            </div>
        </div>
    </div>
</body>
</html>
"""
    
    # Write HTML file
    html_path = os.path.join(post_path, 'comfort_report.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"    * HTML report generated: {os.path.basename(html_path)}")
    logger.info(f"    * Open in browser: file://{html_path}")
    
    return html_path


def calculate_advanced_flow_metrics(slice_mesh, has_k_field=False):
    """
    Calculate advanced flow metrics including Draft Risk (DR).
    
    Args:
        slice_mesh: PyVista mesh slice with T, U, and optionally k fields
        has_k_field: Whether turbulent kinetic energy 'k' is available
        
    Returns:
        dict: Advanced metrics (DR_mean, DR_max, DR_high_risk_pct, Tu_method)
    """
    # Extract fields
    T = slice_mesh.point_data.get('T', None)
    U = slice_mesh.point_data.get('U', None)
    k = slice_mesh.point_data.get('k', None) if has_k_field else None
    
    if T is None or U is None:
        return {
            'error': 'T/U fields not found',
            'DR_mean': 0.0,
            'DR_max': 0.0,
            'DR_high_risk_pct': 0.0,
            'Tu_method': 'N/A'
        }
    
    # Calculate U magnitude
    if len(U.shape) > 1 and U.shape[1] == 3:
        U_mag = np.linalg.norm(U, axis=1)
    else:
        U_mag = U
    
    # Convert T to Celsius if needed
    T_celsius = T.copy()
    if T_celsius.mean() > 100:
        T_celsius = T_celsius - 273.15
    
    # Calculate turbulence intensity (Tu)
    if k is not None and has_k_field:
        # Tu from k: Tu ≈ 100 × √(2k/3) / U_mean
        # Avoid division by zero
        Tu = 100.0 * np.sqrt(2.0 * k / 3.0) / (U_mag + 1e-10)
        Tu = np.clip(Tu, 0, 100)  # Reasonable range
        Tu_method = 'Calculated from k field'
        logger.info(f"       Tu calculated from k: mean={Tu.mean():.1f}%, range=[{Tu.min():.1f}, {Tu.max():.1f}]")
    else:
        # Fallback: assume Tu = 40% (typical for office spaces)
        Tu = np.ones_like(U_mag) * 40.0
        Tu_method = 'Assumed 40% (typical office - k field not available)'
        logger.warning(f"       Tu fallback: k field not found, using Tu=40% (typical office)")
    
    # Calculate Draft Risk (DR) according to ISO 7730
    # DR = (34 - T) * (U - 0.05)^0.62 * (0.37*U*Tu + 3.14)
    # Only valid for U > 0.05 m/s and T < 34°C
    
    # Mask for valid DR calculation
    valid_mask = (U_mag > 0.05) & (T_celsius < 34)
    
    if valid_mask.sum() == 0:
        return {
            'DR_mean': 0.0,
            'DR_max': 0.0,
            'DR_high_risk_pct': 0.0,
            'Tu_mean': Tu.mean(),
            'Tu_method': Tu_method,
            'n_points': 0
        }
    
    # Calculate DR
    DR = np.zeros_like(U_mag)
    DR[valid_mask] = (
        (34 - T_celsius[valid_mask]) * 
        (U_mag[valid_mask] - 0.05)**0.62 * 
        (0.37 * U_mag[valid_mask] * Tu[valid_mask] / 100.0 + 3.14)
    )
    
    # Clip to reasonable range [0, 100]
    DR = np.clip(DR, 0, 100)
    
    # Calculate DR statistics
    DR_mean = float(DR[valid_mask].mean())
    DR_max = float(DR[valid_mask].max())
    
    # % of area with high draft risk (DR > 20%)
    high_risk_mask = DR[valid_mask] > 20
    DR_high_risk_pct = 100.0 * high_risk_mask.sum() / valid_mask.sum()
    
    return {
        'DR_mean': DR_mean,
        'DR_max': DR_max,
        'DR_high_risk_pct': DR_high_risk_pct,
        'Tu_mean': float(Tu.mean()),
        'Tu_method': Tu_method,
        'n_points': int(valid_mask.sum())
    }


def calculate_flow_metrics(slice_mesh):
    """
    Calculate flow metrics (T, U) from a slice mesh.
    
    Args:
        slice_mesh: PyVista mesh slice with T and U fields
        
    Returns:
        dict: Flow metrics (T stats, U stats)
    """
    # Extract T and U arrays
    T = slice_mesh.point_data.get('T', None)
    U = slice_mesh.point_data.get('U', None)
    
    if T is None or U is None:
        logger.warning("T or U fields not found in slice mesh")
        return {
            'error': 'T/U fields not found',
            'T_mean': 0.0,
            'T_std': 0.0,
            'T_min': 0.0,
            'T_max': 0.0,
            'U_mean': 0.0,
            'U_max': 0.0
        }
    
    # Calculate U magnitude if U is vectorial
    if len(U.shape) > 1 and U.shape[1] == 3:
        U_mag = np.linalg.norm(U, axis=1)
    else:
        U_mag = U
    
    # Temperature statistics (convert K to °C if needed)
    T_values = T.copy()
    if T_values.mean() > 100:  # Likely in Kelvin
        T_values = T_values - 273.15
    
    T_mean = float(T_values.mean())
    T_std = float(T_values.std())
    T_min = float(T_values.min())
    T_max = float(T_values.max())
    
    # Velocity statistics
    U_mean = float(U_mag.mean())
    U_max = float(U_mag.max())
    U_std = float(U_mag.std())
    
    # Comfort zone compliance
    T_comfort = ((T_values >= 20) & (T_values <= 26)).sum()
    T_comfort_pct = 100.0 * T_comfort / len(T_values)
    
    U_comfort = (U_mag < 0.25).sum()
    U_comfort_pct = 100.0 * U_comfort / len(U_mag)
    
    return {
        'T_mean': T_mean,
        'T_std': T_std,
        'T_min': T_min,
        'T_max': T_max,
        'T_comfort_pct': T_comfort_pct,
        'U_mean': U_mean,
        'U_std': U_std,
        'U_max': U_max,
        'U_comfort_pct': U_comfort_pct,
        'n_points': int(len(T_values))
    }


def render_flow_png(slice_mesh, name, z_height, post_path, variable='T', surfaces_mesh=None):
    """
    Render PNG image of flow field (T or U vectors).
    
    Args:
        slice_mesh: PyVista mesh slice
        name: Plane name (seated, standing, head)
        z_height: Height of plane [m]
        post_path: Path to post-processing directory
        variable: Variable to render ('T' or 'U')
        surfaces_mesh: Boundary surfaces mesh (optional)
    """
    # Create plotter
    plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
    
    if variable == 'T':
        # Temperature as scalar colormap
        T = slice_mesh.point_data.get('T', None)
        if T is not None:
            # Convert K to °C if needed
            if T.mean() > 100:
                T_celsius = T - 273.15
                slice_mesh.point_data['T_celsius'] = T_celsius
                scalar_field = 'T_celsius'
                label = 'Temperature [°C]'
            else:
                scalar_field = 'T'
                label = 'Temperature [°C]'
            
            plotter.add_mesh(
                slice_mesh,
                scalars=scalar_field,
                cmap='coolwarm',
                clim=(18, 28),
                show_edges=False,
                scalar_bar_args={
                    'title': label,
                    'title_font_size': 38,      # Reduced 52.5% total from 80 (igual comfort)
                    'label_font_size': 64,
                    'shadow': True,
                    'n_labels': 5,
                    'italic': False,
                    'fmt': '%.1f',
                    'font_family': 'arial',
                    'vertical': True,
                    'position_x': 0.85,
                    'position_y': 0.25,
                    'height': 0.5,
                    'width': 0.12
                }
            )
            
            # Add temperature isocontour lines
            try:
                contour_levels = [18, 20, 22, 24, 26, 28]  # Every 2°C
                contours = slice_mesh.contour(isosurfaces=contour_levels, scalars=scalar_field)
                
                if contours.n_points > 0:
                    plotter.add_mesh(contours, color='black', line_width=3, render_lines_as_tubes=False, lighting=False)
                    
                    # Add labels
                    for level in contour_levels:
                        level_contour = slice_mesh.contour(isosurfaces=[level], scalars=scalar_field)
                        if level_contour.n_points > 0:
                            mid_idx = level_contour.n_points // 2
                            point = level_contour.points[mid_idx]
                            plotter.add_point_labels(
                                [point], [f'{level:.0f}°C'],
                                font_size=48, text_color='black', font_family='arial',
                                fill_shape=True, shape='rounded_rect', shape_color='white',
                                shape_opacity=0.8, point_size=0, render_points_as_spheres=False, always_visible=False
                            )
                    logger.info(f"       Added {len(contour_levels)} T isocontour lines with labels")
            except Exception as e:
                logger.warning(f"       Failed to add T contours: {e}")
    
    elif variable == 'U':
        # Velocity as colormap + vector arrows
        U = slice_mesh.point_data.get('U', None)
        if U is not None and len(U.shape) > 1:
            # Calculate magnitude
            U_mag = np.linalg.norm(U, axis=1)
            slice_mesh.point_data['U_magnitude'] = U_mag
            
            # Add colormap for magnitude
            plotter.add_mesh(
                slice_mesh,
                scalars='U_magnitude',
                cmap='viridis',
                clim=(0, 1.0),
                show_edges=False,
                scalar_bar_args={
                    'title': 'Velocity [m/s]',
                    'title_font_size': 38,      # Reduced 52.5% total from 80 (igual comfort/flow T)
                    'label_font_size': 64,
                    'shadow': True,
                    'n_labels': 5,
                    'italic': False,
                    'fmt': '%.2f',
                    'font_family': 'arial',
                    'vertical': True,
                    'position_x': 0.85,
                    'position_y': 0.25,
                    'height': 0.5,
                    'width': 0.12
                }
            )
            
            # Add arrow glyphs for direction (subsample to avoid clutter)
            subsample = 30  # 1 arrow every 30 points
            points = slice_mesh.points[::subsample]
            vectors = U[::subsample]
            magnitudes = U_mag[::subsample]
            
            # Only show arrows where velocity > threshold
            threshold = 0.05  # m/s
            mask = magnitudes > threshold
            
            if mask.sum() > 0:
                # Project vectors onto XY plane (set Z component to 0)
                vectors_xy = vectors[mask].copy()
                vectors_xy[:, 2] = 0  # Project to horizontal plane
                
                # Calculate magnitude in XY plane
                mag_xy = np.linalg.norm(vectors_xy, axis=1)
                
                # Only show arrows where XY magnitude > small threshold
                mask_xy = mag_xy > 0.01
                
                if mask_xy.sum() > 0:
                    # Normalize and scale to half size
                    directions_norm = vectors_xy[mask_xy] / mag_xy[mask_xy][:, np.newaxis]
                    directions_norm = directions_norm * 0.5  # Half size
                    
                    plotter.add_arrows(
                        cent=points[mask][mask_xy],
                        direction=directions_norm,  # Normalized XY directions, half size
                        color='white',
                        opacity=0.7
                    )
    
    # Add boundary patches (inlet/outlet) with field colormap
    if surfaces_mesh is not None and surfaces_mesh.n_cells > 0:
        if variable == 'T' and 'T' in surfaces_mesh.point_data:
            # Add patches with temperature colormap
            T_patch = surfaces_mesh.point_data['T']
            if T_patch.mean() > 100:
                T_patch_celsius = T_patch - 273.15
                surfaces_mesh.point_data['T_patch_celsius'] = T_patch_celsius
                patch_scalar = 'T_patch_celsius'
            else:
                patch_scalar = 'T'
            
            plotter.add_mesh(
                surfaces_mesh,
                scalars=patch_scalar,
                cmap='coolwarm',
                clim=(18, 28),
                show_edges=False,
                show_scalar_bar=False,  # No colorbar for patches
                opacity=0.9
            )
        elif variable == 'U' and 'U' in surfaces_mesh.point_data:
            # Add patches with velocity magnitude colormap
            U_patch = surfaces_mesh.point_data['U']
            if len(U_patch.shape) > 1 and U_patch.shape[1] == 3:
                U_mag_patch = np.linalg.norm(U_patch, axis=1)
                surfaces_mesh.point_data['U_magnitude_patch'] = U_mag_patch
                
                plotter.add_mesh(
                    surfaces_mesh,
                    scalars='U_magnitude_patch',
                    cmap='viridis',
                    clim=(0, 1.0),
                    show_edges=False,
                    show_scalar_bar=False,  # No colorbar for patches
                    opacity=0.9
                )
                
                # Add velocity vectors on patches (subsample)
                subsample_patch = 10  # 1 arrow every 10 points
                points_patch = surfaces_mesh.points[::subsample_patch]
                vectors_patch = U_patch[::subsample_patch]
                magnitudes_patch = U_mag_patch[::subsample_patch]
                
                # Only show arrows where velocity > threshold
                threshold = 0.05  # m/s
                mask_patch = magnitudes_patch > threshold
                
                if mask_patch.sum() > 0:
                    # Project vectors onto Z plane (set Z component to 0)
                    vectors_patch_z = vectors_patch[mask_patch].copy()
                    vectors_patch_z[:, 2] = 0  # Project to horizontal
                    
                    # Calculate magnitude in XY plane
                    mag_patch_z = np.linalg.norm(vectors_patch_z, axis=1)
                    
                    # Only show arrows where XY magnitude > small threshold
                    mask_patch_z = mag_patch_z > 0.01
                    
                    if mask_patch_z.sum() > 0:
                        # Normalize and scale to half size
                        directions_patch_norm = vectors_patch_z[mask_patch_z] / mag_patch_z[mask_patch_z][:, np.newaxis]
                        directions_patch_norm = directions_patch_norm * 0.5  # Half size
                        
                        plotter.add_arrows(
                            cent=points_patch[mask_patch][mask_patch_z],
                            direction=directions_patch_norm,
                            color='white',
                            opacity=0.7
                        )
    
    # Set isometric camera view
    plotter.camera_position = 'iso'
    plotter.camera.zoom(1.2)
    
    # Add title with units
    if variable == 'T':
        title = f'Temperature [°C] at {z_height}m ({name.capitalize()})'
    elif variable == 'U':
        title = f'Velocity [m/s] at {z_height}m ({name.capitalize()})'
    else:
        title = f'{variable} at {z_height}m ({name.capitalize()})'
    
    plotter.add_text(title, position='upper_edge', font_size=16, color='black')
    
    # Add axes widget (XYZ)
    plotter.show_axes()
    
    # Save PNG
    images_dir = os.path.join(post_path, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    png_path = os.path.join(images_dir, f'flow_plane_{name}_{z_height}m_{variable}.png')
    plotter.screenshot(png_path, transparent_background=False)
    plotter.close()
    
    logger.info(f"       Saved PNG: {os.path.basename(png_path)}")


def analyze_flow_planes(sim_path, post_path):
    """
    Analyze Temperature and Velocity in 3 horizontal planes.
    
    Generates:
    - VTK files of slices for each plane
    - PNG images for T (scalar) and U (vectors)
    - JSON file with flow metrics
    
    Args:
        sim_path: Path to simulation directory
        post_path: Path to post-processing output directory
        
    Returns:
        dict: Flow metrics for all planes
    """
    logger.info("    * Analyzing flow fields in horizontal planes")
    
    # Load 3D mesh with T/U fields, surfaces, and multiblock with patches
    logger.info("    * Loading CFD results")
    internal_mesh, surfaces_mesh, multiblock = load_foam_results(sim_path)
    
    logger.info(f"    * Loaded mesh with {internal_mesh.n_cells:,} cells")
    
    # Extract inlet/outlet patches from multiblock
    boundary_patches = None
    
    if multiblock is not None:
        logger.info(f"    * DEBUG: MultiBlock has {len(multiblock)} blocks: {list(multiblock.keys())}")
        
        # Check if 'boundary' block exists (contains patches)
        if 'boundary' in multiblock.keys():
            boundary_block = multiblock['boundary']
            logger.info(f"    * DEBUG: Found 'boundary' block (type={type(boundary_block).__name__})")
            
            if isinstance(boundary_block, pv.MultiBlock):
                logger.info(f"    * DEBUG: 'boundary' is MultiBlock with {len(boundary_block)} patches")
                logger.info(f"    * DEBUG: Patch names: {list(boundary_block.keys())}")
                
                patches_to_merge = []
                
                for i, patch_name in enumerate(boundary_block.keys()):
                    patch = boundary_block[i]
                    patch_name_lower = str(patch_name).lower()
                    
                    if patch is not None and hasattr(patch, 'n_cells'):
                        logger.info(f"       Patch {i}: '{patch_name}' ({patch.n_cells} cells)")
                        
                        # Include inlet, outlet, vent, window patches (exclude walls)
                        if 'inlet' in patch_name_lower or 'outlet' in patch_name_lower or 'vent' in patch_name_lower or 'window' in patch_name_lower:
                            logger.info(f"       ✓ MATCH: {patch_name} - ADDING")
                            patches_to_merge.append(patch)
                        else:
                            logger.info(f"       ✗ SKIP: {patch_name} (wall or other)")
                    else:
                        logger.info(f"       Patch {i}: '{patch_name}' (empty or invalid)")
                
                logger.info(f"    * DEBUG: Total patches to merge = {len(patches_to_merge)}")
                
                if patches_to_merge:
                    # Merge all inlet/outlet patches
                    boundary_patches = patches_to_merge[0]
                    for patch in patches_to_merge[1:]:
                        boundary_patches = boundary_patches.merge(patch)
                    logger.info(f"    * ✓ Merged {len(patches_to_merge)} boundary patches ({boundary_patches.n_cells} cells total)")
                else:
                    logger.warning(f"    * ✗ No inlet/outlet patches found!")
            else:
                logger.warning(f"    * ✗ 'boundary' is not MultiBlock")
        else:
            logger.warning(f"    * ✗ 'boundary' key not found in MultiBlock")
    else:
        logger.warning(f"    * ✗ multiblock is None")
    
    # Check if T and U fields exist
    if 'T' not in internal_mesh.point_data or 'U' not in internal_mesh.point_data:
        logger.error("T/U fields not found in mesh!")
        logger.error("Available fields: " + str(list(internal_mesh.point_data.keys())))
        raise ValueError("T/U fields not found.")
    
    # Check if k field exists for turbulence intensity calculation
    has_k_field = 'k' in internal_mesh.point_data
    if has_k_field:
        logger.info("    * Turbulent kinetic energy 'k' field found - will calculate Tu from k")
    else:
        logger.warning("    * 'k' field not found - will use Tu fallback (40% typical office)")
    
    # Define analysis planes (include ankle for vertical gradient)
    planes = {
        'ankle':   0.1,   # Ankle level (for vertical gradient ΔT)
        'seated':  0.6,   # Seated person (ankle height)
        'standing': 1.1,  # Standing person (waist height)
        'head':    1.7    # Head height
    }
    
    logger.info(f"    * Analyzing {len(planes)} horizontal planes")
    
    # Create output directories
    vtk_dir = os.path.join(post_path, 'vtk')
    os.makedirs(vtk_dir, exist_ok=True)
    os.makedirs(os.path.join(post_path, 'images'), exist_ok=True)
    
    # Process each plane
    results = {}
    
    for name, z_height in planes.items():
        logger.info(f"    * Processing plane: {name} (z={z_height}m)")
        
        # Create slice
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                slice_mesh = internal_mesh.slice(normal='z', origin=(0, 0, z_height))
        except:
            slice_mesh = internal_mesh.slice(normal='z', origin=(0, 0, z_height))
        
        logger.info(f"       Slice has {slice_mesh.n_points:,} points")
        
        # Calculate flow metrics
        metrics = calculate_flow_metrics(slice_mesh)
        metrics['height_m'] = z_height
        metrics['plane_name'] = name
        
        # Calculate advanced metrics (DR) for ankle plane
        if name == 'ankle':
            logger.info(f"       Calculating Draft Risk (DR) for ankle level")
            advanced_metrics = calculate_advanced_flow_metrics(slice_mesh, has_k_field=has_k_field)
            metrics.update(advanced_metrics)
            logger.info(f"       DR: mean={metrics['DR_mean']:.1f}%, max={metrics['DR_max']:.1f}%")
            logger.info(f"       DR high risk area (>20%): {metrics['DR_high_risk_pct']:.1f}%")
            logger.info(f"       Tu method: {metrics['Tu_method']}")
        
        logger.info(f"       T: mean={metrics['T_mean']:.1f}°C, range=[{metrics['T_min']:.1f}, {metrics['T_max']:.1f}]")
        logger.info(f"       U: mean={metrics['U_mean']:.2f}m/s, max={metrics['U_max']:.2f}m/s")
        
        results[name] = metrics
        
        # Save VTK slice
        vtk_path = os.path.join(vtk_dir, f'flow_plane_{name}_{z_height}m.vtk')
        slice_mesh.save(vtk_path)
        logger.info(f"       Saved VTK: {os.path.basename(vtk_path)}")
        
        # Render T image (with boundary patches)
        render_flow_png(slice_mesh, name, z_height, post_path, variable='T', surfaces_mesh=boundary_patches)
        
        # Render U image (vectors, with boundary patches)
        render_flow_png(slice_mesh, name, z_height, post_path, variable='U', surfaces_mesh=boundary_patches)
    
    # Calculate vertical temperature gradient ΔT (ankle-head)
    if 'ankle' in results and 'head' in results:
        T_ankle = results['ankle']['T_mean']
        T_head = results['head']['T_mean']
        delta_T_vertical = T_head - T_ankle
        
        logger.info(f"    * Vertical Temperature Gradient:")
        logger.info(f"       T(ankle z=0.1m) = {T_ankle:.2f}°C")
        logger.info(f"       T(head z=1.7m) = {T_head:.2f}°C")
        logger.info(f"       ΔT(head-ankle) = {delta_T_vertical:.2f}°C")
        
        # Add to results
        results['vertical_gradient'] = {
            'delta_T_head_ankle': delta_T_vertical,
            'T_ankle': T_ankle,
            'T_head': T_head,
            'compliant_ISO_7730_A': delta_T_vertical < 2.0,  # ISO 7730 Category A
            'compliant_ISO_7730_B': delta_T_vertical < 3.0,  # ISO 7730 Category B
            'compliant_ASHRAE_55': delta_T_vertical < 3.0    # ASHRAE 55
        }
    
    # Convert numpy types to Python natives for JSON serialization
    def convert_numpy_types(obj):
        """Recursively convert numpy types to Python native types."""
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    # Save metrics to JSON
    metrics_path = os.path.join(post_path, 'flow_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(convert_numpy_types(results), f, indent=2)
    
    logger.info(f"    * Saved flow metrics: {os.path.basename(metrics_path)}")
    logger.info("    * Flow analysis completed successfully")
    
    return results


def generate_flow_html_report(results, post_path, case_name='CFD_Case'):
    """
    Generate HTML report with flow analysis results (T, U).
    
    Args:
        results: dict with flow metrics from analyze_flow_planes()
        post_path: Path to post-processing directory
        case_name: Name of the simulation case
    """
    logger.info("    * Generating HTML flow report")
    
    # Get timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate global compliance (average T and U comfort)
    # Filter out 'vertical_gradient' entry (it's not a plane)
    plane_results = {k: v for k, v in results.items() if k != 'vertical_gradient'}
    global_T_comfort = np.mean([r['T_comfort_pct'] for r in plane_results.values()])
    global_U_comfort = np.mean([r['U_comfort_pct'] for r in plane_results.values()])
    
    # Build HTML content (similar structure to comfort_report)
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Flow Analysis Report - {case_name}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 300;
        }}
        
        .header .subtitle {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .executive-summary {{
            background: #f8f9fa;
            padding: 30px;
            border-bottom: 3px solid #e9ecef;
        }}
        
        .executive-summary h2 {{
            margin-bottom: 15px;
            color: #495057;
        }}
        
        .summary-metric {{
            display: inline-block;
            padding: 15px 30px;
            margin: 10px;
            background: white;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        
        .summary-metric .label {{
            font-size: 0.9em;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .summary-metric .value {{
            font-size: 2em;
            font-weight: bold;
            margin-top: 5px;
        }}
        
        .comfort-high {{ color: #28a745; }}
        .comfort-medium {{ color: #ffc107; }}
        .comfort-low {{ color: #dc3545; }}
        
        .plane-section {{
            padding: 40px;
            border-bottom: 1px solid #e9ecef;
        }}
        
        .plane-section:last-child {{
            border-bottom: none;
        }}
        
        .plane-header {{
            margin-bottom: 25px;
        }}
        
        .plane-header h2 {{
            color: #495057;
            font-size: 1.8em;
            margin-bottom: 5px;
        }}
        
        .plane-header .height-info {{
            color: #6c757d;
            font-size: 1.1em;
        }}
        
        .images-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .image-box {{
            background: #f8f9fa;
            border-radius: 6px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .image-box img {{
            width: 100%;
            display: block;
        }}
        
        .image-box .caption {{
            padding: 15px;
            text-align: center;
            font-weight: 600;
            background: white;
            border-top: 2px solid #e9ecef;
        }}
        
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            background: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border-radius: 6px;
            overflow: hidden;
        }}
        
        .metrics-table th,
        .metrics-table td {{
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #e9ecef;
        }}
        
        .metrics-table th {{
            background: #f8f9fa;
            font-weight: 600;
            color: #495057;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 0.5px;
        }}
        
        .metrics-table tr:last-child td {{
            border-bottom: none;
        }}
        
        .metrics-table .metric-value {{
            font-weight: 600;
            font-size: 1.1em;
        }}
        
        .status-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .status-excellent {{
            background: #d4edda;
            color: #155724;
        }}
        
        .status-good {{
            background: #fff3cd;
            color: #856404;
        }}
        
        .status-poor {{
            background: #f8d7da;
            color: #721c24;
        }}
        
        .footer {{
            background: #343a40;
            color: #adb5bd;
            padding: 30px;
            text-align: center;
        }}
        
        .footer .info-row {{
            margin: 5px 0;
        }}
        
        .footer .label {{
            color: #6c757d;
        }}
        
        .footer .value {{
            color: #f8f9fa;
            font-weight: 600;
        }}
        
        @media (max-width: 768px) {{
            .images-container {{
                grid-template-columns: 1fr;
            }}
            
            .header h1 {{
                font-size: 1.8em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>Flow Analysis Report</h1>
            <div class="subtitle">{case_name}</div>
            <div class="subtitle">Generated: {timestamp}</div>
        </div>
        
        <!-- Executive Summary -->
        <div class="executive-summary">
            <h2>Executive Summary</h2>
            <div class="summary-metric">
                <div class="label">T Compliance (20-26°C)</div>
                <div class="value {'comfort-high' if global_T_comfort >= 80 else 'comfort-medium' if global_T_comfort >= 60 else 'comfort-low'}">
                    {global_T_comfort:.1f}%
                </div>
                <div style="font-size: 0.75em; color: #6c757d; margin-top: 8px; line-height: 1.4;">
                    Average across 3 analysis planes (0.6m, 1.1m, 1.7m)<br>
                    ASHRAE 55 thermal comfort range
                </div>
            </div>
            <div class="summary-metric">
                <div class="label">U Compliance (&lt;0.25 m/s)</div>
                <div class="value {'comfort-high' if global_U_comfort >= 80 else 'comfort-medium' if global_U_comfort >= 60 else 'comfort-low'}">
                    {global_U_comfort:.1f}%
                </div>
                <div style="font-size: 0.75em; color: #6c757d; margin-top: 8px; line-height: 1.4;">
                    Average across 3 analysis planes (0.6m, 1.1m, 1.7m)<br>
                    ISO 7730 velocity limit
                </div>
            </div>
        </div>
"""
    
    # Add section for each plane
    plane_order = ['seated', 'standing', 'head']
    plane_titles = {
        'seated': 'Seated Level (Ankle Height)',
        'standing': 'Standing Level (Waist Height)',
        'head': 'Head Level'
    }
    
    for plane_name in plane_order:
        if plane_name not in results:
            continue
            
        metrics = results[plane_name]
        z_height = metrics['height_m']
        
        html_content += f"""
        <!-- Plane: {plane_name} -->
        <div class="plane-section">
            <div class="plane-header">
                <h2>{plane_titles[plane_name]}</h2>
                <div class="height-info">Height: {z_height} m</div>
            </div>
            
            <div class="images-container">
                <div class="image-box">
                    <img src="images/flow_plane_{plane_name}_{z_height}m_T.png" alt="Temperature at {z_height}m">
                    <div class="caption">Temperature Field</div>
                </div>
                <div class="image-box">
                    <img src="images/flow_plane_{plane_name}_{z_height}m_U.png" alt="Velocity at {z_height}m">
                    <div class="caption">Velocity Field (vectors)</div>
                </div>
            </div>
            
            <table class="metrics-table">
                <tr>
                    <th>Metric</th>
                    <th>Value Simulation</th>
                    <th>Target</th>
                </tr>
                <tr>
                    <td>Temperature Mean<br><span style="font-size: 0.9em; color: #6c757d;">Plane average</span></td>
                    <td class="metric-value {'comfort-high' if 20 <= metrics['T_mean'] <= 26 else 'comfort-medium' if 18 <= metrics['T_mean'] <= 28 else 'comfort-low'}">
                        {metrics['T_mean']:.1f} °C
                    </td>
                    <td>20-26°C<br><span style="font-size: 0.85em; color: #6c757d;">ASHRAE 55</span></td>
                </tr>
                <tr>
                    <td>Temperature Range<br><span style="font-size: 0.9em; color: #6c757d;">Min/max extremes</span></td>
                    <td class="metric-value">[{metrics['T_min']:.1f}, {metrics['T_max']:.1f}] °C</td>
                    <td>Within 20-26°C<br><span style="font-size: 0.85em; color: #6c757d;">Optimal range</span></td>
                </tr>
                <tr>
                    <td>Temperature Uniformity σ(T)<br><span style="font-size: 0.9em; color: #6c757d;">Plane std deviation</span></td>
                    <td class="metric-value {'comfort-high' if metrics['T_std'] <= 2 else 'comfort-medium' if metrics['T_std'] <= 3 else 'comfort-low'}">
                        ± {metrics['T_std']:.1f} °C
                    </td>
                    <td>&lt; 2°C<br><span style="font-size: 0.85em; color: #6c757d;">ISO 7730</span></td>
                </tr>
                <tr>
                    <td>Velocity Mean<br><span style="font-size: 0.9em; color: #6c757d;">Plane average</span></td>
                    <td class="metric-value {'comfort-high' if metrics['U_mean'] <= 0.2 else 'comfort-medium' if metrics['U_mean'] <= 0.3 else 'comfort-low'}">
                        {metrics['U_mean']:.2f} m/s
                    </td>
                    <td>&lt; 0.2 m/s<br><span style="font-size: 0.85em; color: #6c757d;">ASHRAE 55</span></td>
                </tr>
                <tr>
                    <td>Velocity Maximum<br><span style="font-size: 0.9em; color: #6c757d;">Peak local value</span></td>
                    <td class="metric-value {'comfort-high' if metrics['U_max'] <= 0.25 else 'comfort-medium' if metrics['U_max'] <= 0.5 else 'comfort-low'}">
                        {metrics['U_max']:.2f} m/s
                    </td>
                    <td>&lt; 0.25 m/s<br><span style="font-size: 0.85em; color: #6c757d;">ISO 7730</span></td>
                </tr>
            </table>
        </div>
"""
    
    # Add Advanced Metrics section (ΔT vertical + DR)
    html_content += """
        <!-- Advanced Metrics Section -->
        <div class="plane-section">
            <div class="plane-header">
                <h2>Advanced Comfort Metrics</h2>
                <div class="height-info">Vertical Gradient & Draft Risk</div>
            </div>
            
            <table class="metrics-table">
                <tr>
                    <th>Metric</th>
                    <th>Value Simulation</th>
                    <th>Target</th>
                </tr>
"""
    
    # Add Vertical Temperature Gradient if available
    if 'vertical_gradient' in results:
        vg = results['vertical_gradient']
        delta_T = vg['delta_T_head_ankle']
        compliant_A = vg['compliant_ISO_7730_A']
        compliant_B = vg['compliant_ISO_7730_B']
        
        html_content += f"""
                <tr>
                    <td>Vertical Temperature Gradient ΔT(head-ankle)<br><span style="font-size: 0.9em; color: #6c757d;">Between z=0.1m (ankle) and z=1.7m (head)</span></td>
                    <td class="metric-value {'comfort-high' if compliant_A else 'comfort-medium' if compliant_B else 'comfort-low'}">
                        {delta_T:.2f} °C
                    </td>
                    <td>&lt; 2°C (ISO 7730 Cat A) {'✓' if compliant_A else '✗'}<br>&lt; 3°C (ISO 7730 Cat B / ASHRAE 55) {'✓' if compliant_B else '✗'}<br><span style="font-size: 0.85em; color: #6c757d;">T(ankle)={vg['T_ankle']:.2f}°C, T(head)={vg['T_head']:.2f}°C</span></td>
                </tr>
"""
    
    # Add Draft Risk metrics if available
    if 'ankle' in results and 'DR_mean' in results['ankle']:
        dr = results['ankle']
        DR_mean = dr['DR_mean']
        DR_max = dr['DR_max']
        DR_high_risk_pct = dr['DR_high_risk_pct']
        Tu_mean = dr.get('Tu_mean', 0)
        Tu_method = dr.get('Tu_method', 'N/A')
        
        html_content += f"""
                <tr>
                    <td>Draft Risk (DR) Mean<br><span style="font-size: 0.9em; color: #6c757d;">At ankle level (z=0.1m) - ISO 7730</span></td>
                    <td class="metric-value {'comfort-high' if DR_mean <= 15 else 'comfort-medium' if DR_mean <= 20 else 'comfort-low'}">
                        {DR_mean:.1f}%
                    </td>
                    <td>&lt; 15% (Acceptable)<br>&lt; 20% (Limit)<br><span style="font-size: 0.85em; color: #6c757d;">DR = (34-T)×(U-0.05)^0.62×(0.37×U×Tu+3.14)</span></td>
                </tr>
                <tr>
                    <td>Draft Risk (DR) Maximum<br><span style="font-size: 0.9em; color: #6c757d;">Worst case at ankle level</span></td>
                    <td class="metric-value {'comfort-high' if DR_max <= 20 else 'comfort-medium' if DR_max <= 30 else 'comfort-low'}">
                        {DR_max:.1f}%
                    </td>
                    <td>&lt; 20% (Target)<br><span style="font-size: 0.85em; color: #6c757d;">Localized peak value</span></td>
                </tr>
                <tr>
                    <td>High Draft Risk Area<br><span style="font-size: 0.9em; color: #6c757d;">% of ankle area with DR &gt; 20%</span></td>
                    <td class="metric-value {'comfort-high' if DR_high_risk_pct <= 10 else 'comfort-medium' if DR_high_risk_pct <= 25 else 'comfort-low'}">
                        {DR_high_risk_pct:.1f}%
                    </td>
                    <td>0% (Ideal)<br>&lt; 10% (Acceptable)<br><span style="font-size: 0.85em; color: #6c757d;">Areas requiring attention</span></td>
                </tr>
                <tr>
                    <td>Turbulence Intensity (Tu)<br><span style="font-size: 0.9em; color: #6c757d;">Used for DR calculation</span></td>
                    <td class="metric-value">
                        {Tu_mean:.1f}%
                    </td>
                    <td>Typical: 30-50% (offices)<br><span style="font-size: 0.85em; color: #6c757d;"><strong>{Tu_method}</strong></span></td>
                </tr>
"""
    
    html_content += """
            </table>
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <div class="info-row">
                <span class="label">Analysis Standard:</span>
                <span class="value">ASHRAE 55 & ISO 7730</span>
            </div>
            <div class="info-row">
                <span class="label">Solver:</span>
                <span class="value">OpenFOAM buoyantBoussinesqPimpleFoam</span>
            </div>
            <div class="info-row">
                <span class="label">Fields:</span>
                <span class="value">Temperature (T) & Velocity (U)</span>
            </div>
        </div>
    </div>
</body>
</html>
"""
    
    # Write HTML file
    html_path = os.path.join(post_path, 'flow_report.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"    * HTML flow report generated: {os.path.basename(html_path)}")
    logger.info(f"    * Open in browser: file://{html_path}")
    
    return html_path
