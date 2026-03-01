"""
Setup Summary module for CFD post-processing.

Visualizes simulation boundary conditions (BCs):
- Floor & Ceiling patches with temperatures
- Wall & Window patches with temperatures  
- Air Entries (vents/inlets) with temperatures and mass flows
- Complete patch table with all setup parameters

Uses 3D renders with PyVista (isometric view) + labels for exact values.
"""

import os
import json
import glob
import logging
import numpy as np
import pandas as pd
import pyvista as pv

from src.components.tools.export_debug import load_foam_results


logger = logging.getLogger(__name__)


def read_mass_flows_from_postprocessing(sim_path):
    """
    Read mass flow rates from postProcessing/massFlow_*/surfaceFieldValue.dat files.
    
    Args:
        sim_path: Path to simulation directory
        
    Returns:
        dict: {patch_name: phi_value [m³/s]}
    """
    postproc_path = os.path.join(sim_path, 'postProcessing')
    massflow_pattern = os.path.join(postproc_path, 'massFlow_*', '0', 'surfaceFieldValue.dat')
    massflow_files = glob.glob(massflow_pattern)
    
    mass_flows = {}
    
    for filepath in massflow_files:
        try:
            # Extract patch name from path
            # e.g., .../postProcessing/massFlow_vent_0F_1/0/surfaceFieldValue.dat
            patch_name = os.path.basename(os.path.dirname(os.path.dirname(filepath)))
            patch_name = patch_name.replace('massFlow_', '')
            
            # Read file (format: # Time phi_sum)
            data = np.loadtxt(filepath, comments='#')
            
            # Get last timestep (steady-state)
            if data.ndim == 1:
                phi_latest = float(data[1]) if len(data) > 1 else 0.0
            else:
                phi_latest = float(data[-1, 1])
            
            mass_flows[patch_name] = phi_latest
            
        except Exception as e:
            logger.warning(f"       Failed to read mass flow for {filepath}: {e}")
            continue
    
    logger.info(f"    * Read mass flows for {len(mass_flows)} patches")
    return mass_flows


def render_setup_patches_3d(patches_dict, patch_df, mass_flows, post_path, group='floor_ceil'):
    """
    Render 3D visualization of setup patches with temperatures.
    
    Uses colormap + labels at centroids for exact T values.
    
    Args:
        patches_dict: dict {patch_id: pyvista_mesh}
        patch_df: DataFrame with patch info (id, type, T)
        mass_flows: dict {patch_id: phi [m³/s]}
        post_path: Path to post-processing directory
        group: 'floor_ceil', 'walls', or 'entries'
    """
    if not patches_dict:
        logger.warning(f"       No patches found for group '{group}'")
        return
    
    logger.info(f"    * Rendering {len(patches_dict)} patches for group '{group}'")
    
    # Create plotter
    plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
    
    # Define group titles and filters
    group_config = {
        'floor_ceil': {
            'title': 'Floor & Ceiling - Temperature Setup',
            'filter_fn': lambda patch_id: 'floor' in patch_id.lower() or 'ceil' in patch_id.lower(),
            'show_mass_flow': False
        },
        'walls': {
            'title': 'Walls & Windows - Temperature Setup',
            'filter_fn': lambda patch_id: 'wall' in patch_id.lower() or 'window' in patch_id.lower() or 'door' in patch_id.lower(),
            'show_mass_flow': False
        },
        'entries': {
            'title': 'Air Entries - Temperature & Mass Flow Setup',
            'filter_fn': lambda patch_id: 'vent' in patch_id.lower() or 'inlet' in patch_id.lower() or 'outlet' in patch_id.lower(),
            'show_mass_flow': True
        }
    }
    
    config = group_config.get(group, group_config['floor_ceil'])
    filter_fn = config['filter_fn']
    show_mass_flow = config['show_mass_flow']
    
    # Filter patches by group
    filtered_patches = {k: v for k, v in patches_dict.items() if filter_fn(k)}
    
    if not filtered_patches:
        logger.warning(f"       No patches match filter for '{group}'")
        plotter.close()
        return
    
    logger.info(f"       Filtered to {len(filtered_patches)} patches: {list(filtered_patches.keys())}")
    
    # Collect all T values for colormap limits
    T_values = []
    for patch_id in filtered_patches.keys():
        if patch_id in patch_df.index:
            T_values.append(patch_df.loc[patch_id, 'T_(°C)'])
    
    if not T_values:
        logger.warning(f"       No temperature data found for group '{group}'")
        plotter.close()
        return
    
    T_min = min(T_values)
    T_max = max(T_values)
    
    # Import colormap
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    
    # Create colormap normalizer
    norm = mcolors.Normalize(vmin=T_min - 2, vmax=T_max + 2)
    cmap = plt.get_cmap('coolwarm')
    
    # Render each patch with mapped color
    for patch_id, patch_mesh in filtered_patches.items():
        if patch_id not in patch_df.index:
            continue
        
        T_value = patch_df.loc[patch_id, 'T_(°C)']
        
        # Map temperature to RGB color using colormap
        rgba = cmap(norm(T_value))
        rgb = rgba[:3]  # Extract RGB, ignore alpha
        
        # Render patch with direct color (no scalar field)
        plotter.add_mesh(
            patch_mesh,
            color=rgb,
            show_edges=False,  # Disabled to get clean surfaces
            opacity=1.0
        )
        
        # Add label at centroid with exact T value
        centroid = patch_mesh.center
        
        # Build label text
        if show_mass_flow and patch_id in mass_flows:
            phi_value = mass_flows[patch_id]
            label_text = f'{patch_id}\n{T_value}°C\nφ={phi_value:.4f} m³/s'
        else:
            label_text = f'{patch_id}\n{T_value}°C'
        
        plotter.add_point_labels(
            [centroid],
            [label_text],
            font_size=20,
            text_color='black',
            font_family='arial',
            fill_shape=True,
            shape='rounded_rect',
            shape_color='white',
            shape_opacity=0.9,
            point_size=0,
            render_points_as_spheres=False,
            always_visible=True
        )
    
    # Add scalar bar (colorbar) using dummy mesh
    # Create invisible dummy mesh with temperature scalars for colorbar
    dummy = pv.Plane()
    dummy.point_data['Temperature'] = np.linspace(T_min - 2, T_max + 2, dummy.n_points)
    plotter.add_mesh(
        dummy,
        scalars='Temperature',
        cmap='coolwarm',
        clim=(T_min - 2, T_max + 2),
        opacity=0,  # Invisible mesh, only for colorbar
        show_scalar_bar=True,
        scalar_bar_args={
            'title': 'Temperature [°C]',
            'title_font_size': 24,
            'label_font_size': 20,
            'shadow': True,
            'n_labels': 5,
            'italic': False,
            'fmt': '%.0f',
            'font_family': 'arial',
            'vertical': True,
            'position_x': 0.85,
            'position_y': 0.3,
            'height': 0.4,
            'width': 0.08
        }
    )
    
    # Set isometric camera view
    plotter.camera_position = 'iso'
    plotter.camera.zoom(1.0)
    
    # Add title
    plotter.add_text(config['title'], position='upper_edge', font_size=18, color='black', font='arial')
    
    # Add axes widget
    plotter.show_axes()
    
    # Save PNG
    images_dir = os.path.join(post_path, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    png_path = os.path.join(images_dir, f'setup_{group}_3d.png')
    plotter.screenshot(png_path, transparent_background=False)
    plotter.close()
    
    logger.info(f"       Saved PNG: {os.path.basename(png_path)}")


def analyze_setup_summary(sim_path, post_path):
    """
    Analyze and visualize simulation setup (boundary conditions).
    
    Generates:
    - 3 PNG images (floor/ceil, walls, entries) with T labels
    - JSON file with setup summary
    - HTML table with all patches
    
    Args:
        sim_path: Path to simulation directory
        post_path: Path to post-processing output directory
        
    Returns:
        dict: Setup summary with all patch data
    """
    logger.info("    * Analyzing simulation setup (boundary conditions)")
    
    # Load patch_info.csv
    geo_df_file = os.path.join(sim_path, "..", "geo", "patch_info.csv")
    if not os.path.isfile(geo_df_file):
        logger.error(f"    * patch_info.csv not found: {geo_df_file}")
        raise FileNotFoundError("patch_info.csv not found")
    
    patch_df = pd.read_csv(geo_df_file, index_col='id')
    logger.info(f"    * Loaded {len(patch_df)} patches from patch_info.csv")
    
    # Read mass flows
    mass_flows = read_mass_flows_from_postprocessing(sim_path)
    
    # Load multiblock with boundary patches
    logger.info("    * Loading boundary patches from VTK")
    internal_mesh, surfaces_mesh, multiblock = load_foam_results(sim_path)
    
    if multiblock is None or 'boundary' not in multiblock.keys():
        logger.error("    * boundary patches not found in VTK multiblock")
        raise ValueError("Boundary patches not available")
    
    boundary_block = multiblock['boundary']
    
    if not isinstance(boundary_block, pv.MultiBlock):
        logger.error("    * boundary is not MultiBlock")
        raise ValueError("boundary is not MultiBlock")
    
    logger.info(f"    * Found {len(boundary_block)} boundary patches")
    
    # Build dict of patches {patch_id: pyvista_mesh}
    patches_dict = {}
    for i, patch_name in enumerate(boundary_block.keys()):
        patch = boundary_block[i]
        if patch is not None and hasattr(patch, 'n_cells') and patch.n_cells > 0:
            patches_dict[str(patch_name)] = patch
    
    logger.info(f"    * Extracted {len(patches_dict)} valid patches")
    
    # Create output directories
    os.makedirs(os.path.join(post_path, 'images'), exist_ok=True)
    
    # Render 3 groups
    render_setup_patches_3d(patches_dict, patch_df, mass_flows, post_path, group='floor_ceil')
    render_setup_patches_3d(patches_dict, patch_df, mass_flows, post_path, group='walls')
    render_setup_patches_3d(patches_dict, patch_df, mass_flows, post_path, group='entries')
    
    # Build summary dict
    summary = {
        'n_patches': len(patch_df),
        'emissivity_default': 0.9,  # From boundaryRadiationProperties
        'patches': []
    }
    
    for patch_id, row in patch_df.iterrows():
        patch_data = {
            'id': patch_id,
            'type': row['type'],
            'T_celsius': float(row['T_(°C)']),
            'mass_flow_m3s': mass_flows.get(patch_id, 0.0),
            'emissivity': 0.9,  # Default from boundaryRadiationProperties
            'absorptivity': 0.9,  # Default (grey body assumption)
            'bc_type': 'fixedValue' if row['type'] == 'wall' else 'pressureInletOutletVelocity'
        }
        summary['patches'].append(patch_data)
    
    # Save summary JSON
    summary_path = os.path.join(post_path, 'setup_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"    * Saved setup summary: {os.path.basename(summary_path)}")
    logger.info("    * Setup summary analysis completed successfully")
    
    return summary


def generate_setup_html_report(summary, post_path, case_name='CFD_Case'):
    """
    Generate HTML report with setup summary.
    
    Args:
        summary: dict with setup data from analyze_setup_summary()
        post_path: Path to post-processing directory
        case_name: Name of the simulation case
    """
    logger.info("    * Generating HTML setup report")
    
    # Get timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Build HTML content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Setup Summary Report - {case_name}</title>
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
            background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
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
        
        .images-section {{
            padding: 40px;
        }}
        
        .images-section h2 {{
            margin-bottom: 25px;
            color: #495057;
        }}
        
        .image-box {{
            background: #f8f9fa;
            border-radius: 6px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
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
        
        .table-section {{
            padding: 40px;
            background: #f8f9fa;
        }}
        
        .table-section h2 {{
            margin-bottom: 25px;
            color: #495057;
        }}
        
        .patches-table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border-radius: 6px;
            overflow: hidden;
        }}
        
        .patches-table th,
        .patches-table td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #e9ecef;
        }}
        
        .patches-table th {{
            background: #343a40;
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 0.5px;
        }}
        
        .patches-table tr:last-child td {{
            border-bottom: none;
        }}
        
        .patches-table tr:hover {{
            background: #f8f9fa;
        }}
        
        .type-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.75em;
            font-weight: 600;
            text-transform: uppercase;
        }}
        
        .type-wall {{
            background: #e9ecef;
            color: #495057;
        }}
        
        .type-floor {{
            background: #d1ecf1;
            color: #0c5460;
        }}
        
        .type-ceil {{
            background: #fff3cd;
            color: #856404;
        }}
        
        .type-vent {{
            background: #d4edda;
            color: #155724;
        }}
        
        .type-window {{
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
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>Setup Summary Report</h1>
            <div class="subtitle">Simulation Boundary Conditions</div>
            <div class="subtitle">{case_name}</div>
            <div class="subtitle">Generated: {timestamp}</div>
        </div>
        
        <!-- Images Section -->
        <div class="images-section">
            <h2>3D Visualization of Boundary Conditions</h2>
            
            <div class="image-box">
                <img src="images/setup_floor_ceil_3d.png" alt="Floor & Ceiling Setup">
                <div class="caption">Floor & Ceiling - Temperature Setup</div>
            </div>
            
            <div class="image-box">
                <img src="images/setup_walls_3d.png" alt="Walls Setup">
                <div class="caption">Walls & Windows - Temperature Setup</div>
            </div>
            
            <div class="image-box">
                <img src="images/setup_entries_3d.png" alt="Air Entries Setup">
                <div class="caption">Air Entries - Temperature & Mass Flow Setup</div>
            </div>
        </div>
        
        <!-- Table Section -->
        <div class="table-section">
            <h2>Complete Patch Configuration Table</h2>
            <table class="patches-table">
                <thead>
                    <tr>
                        <th>Patch ID</th>
                        <th>Type</th>
                        <th>BC Type</th>
                        <th>T [°C]</th>
                        <th>φ [m³/s]</th>
                        <th>ε</th>
                        <th>α</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    # Add rows for each patch
    for patch in summary['patches']:
        patch_id = patch['id']
        patch_type = patch['type']
        T_celsius = patch['T_celsius']
        mass_flow = patch['mass_flow_m3s']
        emissivity = patch['emissivity']
        absorptivity = patch['absorptivity']
        bc_type = patch['bc_type']
        
        # Determine badge class based on patch type
        if 'floor' in patch_id.lower():
            badge_class = 'type-floor'
        elif 'ceil' in patch_id.lower():
            badge_class = 'type-ceil'
        elif 'vent' in patch_id.lower() or 'inlet' in patch_id.lower() or 'outlet' in patch_id.lower():
            badge_class = 'type-vent'
        elif 'window' in patch_id.lower():
            badge_class = 'type-window'
        else:
            badge_class = 'type-wall'
        
        # Format mass flow (show only if non-zero)
        if abs(mass_flow) > 1e-6:
            mass_flow_str = f'{mass_flow:.4f}'
        else:
            mass_flow_str = '—'
        
        html_content += f"""
                    <tr>
                        <td><strong>{patch_id}</strong></td>
                        <td><span class="type-badge {badge_class}">{patch_type}</span></td>
                        <td><code>{bc_type}</code></td>
                        <td>{T_celsius:.1f}°C</td>
                        <td>{mass_flow_str}</td>
                        <td>{emissivity:.2f}</td>
                        <td>{absorptivity:.2f}</td>
                    </tr>
"""
    
    html_content += f"""
                </tbody>
            </table>
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <div class="info-row">
                <span class="label">Total Patches:</span>
                <span class="value">{summary['n_patches']}</span>
            </div>
            <div class="info-row">
                <span class="label">Solver:</span>
                <span class="value">OpenFOAM buoyantBoussinesqPimpleFoam</span>
            </div>
            <div class="info-row">
                <span class="label">Boundary Conditions:</span>
                <span class="value">Temperature (fixedValue) & Mass Flow (flowRateInletVelocity)</span>
            </div>
        </div>
    </div>
</body>
</html>
"""
    
    # Write HTML file
    html_path = os.path.join(post_path, 'setup_report.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"    * HTML setup report generated: {os.path.basename(html_path)}")
    logger.info(f"    * Open in browser: file://{html_path}")
    
    return html_path
