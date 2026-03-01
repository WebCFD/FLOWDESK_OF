"""
Ventilation analysis module for CFD post-processing.

Analyzes ventilation metrics:
- ACH (Air Changes per Hour)
- CO2 concentration and distribution
- Age of Air estimation
- Ventilation Effectiveness (εv)
- Air Change Effectiveness (ACE)
- ADPI (Air Distribution Performance Index)
"""

import os
import json
import logging
import numpy as np
import pyvista as pv

from src.components.tools.export_debug import load_foam_results, load_foam_results_transient


logger = logging.getLogger(__name__)


# CO2 normalization constants (user configurable)
C_INITIAL_PPM = 2000.0  # ppm - typical occupied room initial CO2
C_SUPPLY_PPM = 400.0    # ppm - outdoor fresh air CO2


def calculate_ventilation_metrics(slice_mesh, has_co2=False):
    """
    Calculate ventilation metrics from a slice mesh.
    
    Args:
        slice_mesh: PyVista mesh slice with CO2 and U fields
        has_co2: Whether CO2 field is available
        
    Returns:
        dict: Ventilation metrics (CO2 stats, stagnation zones, etc.)
    """
    # Extract CO2 and U arrays
    CO2 = slice_mesh.point_data.get('CO2', None) if has_co2 else None
    U = slice_mesh.point_data.get('U', None)
    T = slice_mesh.point_data.get('T', None)
    
    if U is None:
        logger.warning("U field not found in slice mesh")
        return {'error': 'U field not found'}
    
    # Calculate U magnitude
    if len(U.shape) > 1 and U.shape[1] == 3:
        U_mag = np.linalg.norm(U, axis=1)
    else:
        U_mag = U
    
    # CO2 statistics
    if CO2 is not None and has_co2:
        CO2_mean = float(CO2.mean())
        CO2_std = float(CO2.std())
        CO2_min = float(CO2.min())
        CO2_max = float(CO2.max())
        
        # CO2 compliance (<1000 ppm)
        CO2_compliant = (CO2 < 1000).sum()
        CO2_compliant_pct = 100.0 * CO2_compliant / len(CO2)
    else:
        CO2_mean = 0.0
        CO2_std = 0.0
        CO2_min = 0.0
        CO2_max = 0.0
        CO2_compliant_pct = 0.0
    
    # Stagnation zones (U < 0.05 m/s)
    stagnation_mask = U_mag < 0.05
    stagnation_pct = 100.0 * stagnation_mask.sum() / len(U_mag)
    
    # ADPI (Air Distribution Performance Index)
    # % of points with EDT < 3°C and U < 0.35 m/s
    if T is not None:
        T_celsius = T.copy()
        if T_celsius.mean() > 100:
            T_celsius = T_celsius - 273.15
        
        # EDT (Effective Draft Temperature) ≈ |T - T_target|
        # Assume T_target = 22°C (typical office)
        T_target = 22.0
        EDT = np.abs(T_celsius - T_target)
        
        # ADPI criteria: EDT < 3°C AND U < 0.35 m/s
        ADPI_mask = (EDT < 3.0) & (U_mag < 0.35)
        ADPI_pct = 100.0 * ADPI_mask.sum() / len(U_mag)
    else:
        ADPI_pct = 0.0
    
    return {
        'CO2_mean': CO2_mean,
        'CO2_std': CO2_std,
        'CO2_min': CO2_min,
        'CO2_max': CO2_max,
        'CO2_compliant_pct': CO2_compliant_pct,
        'stagnation_pct': stagnation_pct,
        'ADPI_pct': ADPI_pct,
        'U_mean': float(U_mag.mean()),
        'U_max': float(U_mag.max()),
        'n_points': int(len(U_mag))
    }


def render_ventilation_png(slice_mesh, name, z_height, post_path, variable='CO2'):
    """
    Render PNG image of ventilation field (CO2 or Age of Air).
    
    Args:
        slice_mesh: PyVista mesh slice
        name: Plane name (seated, standing, head)
        z_height: Height of plane [m]
        post_path: Path to post-processing directory
        variable: Variable to render ('CO2' or 'age')
    """
    # Create plotter
    plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
    
    if variable == 'CO2':
        # CO2 concentration colormap
        if 'CO2' in slice_mesh.point_data:
            plotter.add_mesh(
                slice_mesh,
                scalars='CO2',
                cmap='inferno_r',  # High CO2 = red (bad)
                clim=(400, 1500),  # Outdoor (~400) to high (1500)
                show_edges=False,
                scalar_bar_args={
                    'title': 'CO₂ [ppm]',
                    'title_font_size': 38,
                    'label_font_size': 64,
                    'shadow': True,
                    'n_labels': 5,
                    'italic': False,
                    'fmt': '%.0f',
                    'font_family': 'arial',
                    'vertical': True,
                    'position_x': 0.85,
                    'position_y': 0.25,
                    'height': 0.5,
                    'width': 0.12
                }
            )
            
            # Add CO2 isocontour lines
            try:
                contour_levels = [400, 600, 800, 1000, 1200, 1500]  # CO2 levels in ppm
                contours = slice_mesh.contour(isosurfaces=contour_levels, scalars='CO2')
                
                if contours.n_points > 0:
                    plotter.add_mesh(contours, color='black', line_width=3, render_lines_as_tubes=False, lighting=False)
                    
                    # Add labels
                    for level in contour_levels:
                        level_contour = slice_mesh.contour(isosurfaces=[level], scalars='CO2')
                        if level_contour.n_points > 0:
                            mid_idx = level_contour.n_points // 2
                            point = level_contour.points[mid_idx]
                            plotter.add_point_labels(
                                [point], [f'{level:.0f} ppm'],
                                font_size=48, text_color='black', font_family='arial',
                                fill_shape=True, shape='rounded_rect', shape_color='white',
                                shape_opacity=0.8, point_size=0, render_points_as_spheres=False, always_visible=False
                            )
                    logger.info(f"       Added {len(contour_levels)} CO2 isocontour lines with labels")
            except Exception as e:
                logger.warning(f"       Failed to add CO2 contours: {e}")
    elif variable == 'age':
        # Age of Air colormap (if available)
        if 'age' in slice_mesh.point_data:
            plotter.add_mesh(
                slice_mesh,
                scalars='age',
                cmap='plasma',  # Yellow to purple
                show_edges=False,
                scalar_bar_args={
                    'title': 'Age of Air [min]',
                    'title_font_size': 38,
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
    
    # Set isometric camera view
    plotter.camera_position = 'iso'
    plotter.camera.zoom(1.2)
    
    # Add title
    if variable == 'CO2':
        title = f'CO₂ Concentration [ppm] at {z_height}m ({name.capitalize()})'
    elif variable == 'age':
        title = f'Age of Air [min] at {z_height}m ({name.capitalize()})'
    else:
        title = f'{variable} at {z_height}m ({name.capitalize()})'
    
    plotter.add_text(title, position='upper_edge', font_size=16, color='black')
    
    # Add axes widget
    plotter.show_axes()
    
    # Save PNG
    images_dir = os.path.join(post_path, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    png_path = os.path.join(images_dir, f'ventilation_plane_{name}_{z_height}m_{variable}.png')
    plotter.screenshot(png_path, transparent_background=False)
    plotter.close()
    
    logger.info(f"       Saved PNG: {os.path.basename(png_path)}")


def integrate_co2_on_boundaries(multiblock):
    """
    Integrate CO2 concentration on inlet and outlet boundary patches.
    
    Args:
        multiblock: PyVista MultiBlock with boundary patches
        
    Returns:
        dict: {'C_supply': float, 'C_exhaust': float, 'has_data': bool}
    """
    if multiblock is None or 'boundary' not in multiblock.keys():
        logger.warning("    * No boundary patches found in multiblock")
        return {'C_supply': 0.0, 'C_exhaust': 0.0, 'has_data': False}
    
    boundary_block = multiblock['boundary']
    
    if not isinstance(boundary_block, pv.MultiBlock):
        logger.warning("    * Boundary is not MultiBlock")
        return {'C_supply': 0.0, 'C_exhaust': 0.0, 'has_data': False}
    
    # Collect CO2 values from inlet/outlet patches
    inlet_co2_values = []
    outlet_co2_values = []
    
    for i, patch_name in enumerate(boundary_block.keys()):
        patch = boundary_block[i]
        
        if patch is None or not hasattr(patch, 'point_data'):
            continue
        
        # Check if CO2 field exists
        if 'CO2' not in patch.point_data:
            continue
        
        CO2 = patch.point_data['CO2']
        patch_name_lower = str(patch_name).lower()
        
        # Classify as inlet or outlet based on name
        if 'inlet' in patch_name_lower or 'supply' in patch_name_lower:
            inlet_co2_values.extend(CO2.tolist())
        elif 'outlet' in patch_name_lower or 'exhaust' in patch_name_lower or 'extract' in patch_name_lower:
            outlet_co2_values.extend(CO2.tolist())
    
    # Calculate mean CO2 at supply and exhaust
    if inlet_co2_values:
        C_supply = float(np.mean(inlet_co2_values))
    else:
        C_supply = 400.0  # Assume fresh air (outdoor CO2)
        logger.warning("    * No inlet CO2 data - assuming C_supply = 400 ppm")
    
    if outlet_co2_values:
        C_exhaust = float(np.mean(outlet_co2_values))
    else:
        C_exhaust = 0.0
        logger.warning("    * No outlet CO2 data - cannot calculate εv")
    
    has_data = len(inlet_co2_values) > 0 and len(outlet_co2_values) > 0
    
    logger.info(f"    * C_supply (inlet): {C_supply:.1f} ppm")
    logger.info(f"    * C_exhaust (outlet): {C_exhaust:.1f} ppm")
    
    return {
        'C_supply': C_supply,
        'C_exhaust': C_exhaust,
        'has_data': has_data
    }


def calculate_ev_from_transient_decay(mesh_t0, mesh_tf, Q_inlet, V_room, delta_t):
    """
    Calculate Ventilation Effectiveness (εv) from transient CO2 decay.
    
    Uses exponential decay method:
    - C(t) = C_supply + (C_initial - C_supply) × exp(-k_calculado × t)
    - εv = k_calculado / k_ideal
    
    Args:
        mesh_t0: Mesh at t=0 (initial state with high CO2)
        mesh_tf: Mesh at t=final (after ventilation)
        Q_inlet: Inlet volumetric flow rate [m³/s]
        V_room: Room volume [m³]
        delta_t: Time difference between timesteps [s]
        
    Returns:
        dict: {'ev': float, 'C_0_mean': float, 'C_f_mean': float, 'k_calculado': float, 'k_ideal': float}
    """
    # Extract CO2 scalar fields
    CO2_scalar_0 = mesh_t0.point_data.get('CO2', None)
    CO2_scalar_f = mesh_tf.point_data.get('CO2', None)
    
    if CO2_scalar_0 is None or CO2_scalar_f is None:
        logger.warning("    * CO2 field not found in t=0 or t=final - cannot calculate εv from decay")
        return {
            'ev': 0.0,
            'C_0_mean': 0.0,
            'C_f_mean': 0.0,
            'k_real': 0.0,
            'k_theory': 0.0,
            'method': 'transient_decay',
            'error': 'CO2 field not found'
        }
    
    # Convert scalar [0,1] → ppm using linear interpolation
    # CO2[ppm] = C_supply + scalar × (C_initial - C_supply)
    C_0_ppm = C_SUPPLY_PPM + CO2_scalar_0 * (C_INITIAL_PPM - C_SUPPLY_PPM)
    C_f_ppm = C_SUPPLY_PPM + CO2_scalar_f * (C_INITIAL_PPM - C_SUPPLY_PPM)
    
    # Calculate volumetric means
    C_0_mean = float(C_0_ppm.mean())
    C_f_mean = float(C_f_ppm.mean())
    
    logger.info(f"    * Transient CO2 decay analysis:")
    logger.info(f"       t=0: C_mean = {C_0_mean:.1f} ppm")
    logger.info(f"       t={delta_t:.1f}s: C_mean = {C_f_mean:.1f} ppm")
    logger.info(f"       ΔC = {C_0_mean - C_f_mean:.1f} ppm ({100*(C_0_mean - C_f_mean)/C_0_mean:.1f}% reduction)")
    
    # Calculate decay rate real (measured from simulation)
    # C_f = C_s + (C_0 - C_s) × exp(-k_real × Δt)
    # k_real = -ln[(C_f - C_s)/(C_0 - C_s)] / Δt
    
    denom = C_0_mean - C_SUPPLY_PPM
    if denom < 1.0:
        logger.warning(f"    * ΔC_0 too small ({denom:.1f} ppm) - cannot calculate decay rate")
        return {
            'ev': 0.0,
            'C_0_mean': C_0_mean,
            'C_f_mean': C_f_mean,
            'k_real': 0.0,
            'k_theory': 0.0,
            'method': 'transient_decay',
            'error': 'ΔC_0 too small'
        }
    
    ratio = (C_f_mean - C_SUPPLY_PPM) / denom
    
    if ratio <= 0:
        logger.warning(f"    * Invalid decay ratio ({ratio:.3f}) - C_f may be below C_supply")
        return {
            'ev': 0.0,
            'C_0_mean': C_0_mean,
            'C_f_mean': C_f_mean,
            'k_real': 0.0,
            'k_theory': 0.0,
            'method': 'transient_decay',
            'error': 'Invalid decay ratio'
        }
    
    k_calculado = -np.log(ratio) / delta_t  # [1/s] - calculated from CFD decay
    
    # Calculate decay rate ideal (perfect mixing assumption)
    # k_ideal = Q_inlet / V_room [1/s]
    if V_room > 0 and Q_inlet > 0:
        k_ideal = Q_inlet / V_room  # [1/s]
    else:
        logger.warning(f"    * Invalid V_room ({V_room:.2f}) or Q_inlet ({Q_inlet:.6f})")
        return {
            'ev': 0.0,
            'C_0_mean': C_0_mean,
            'C_f_mean': C_f_mean,
            'k_calculado': k_calculado,
            'k_ideal': 0.0,
            'tau_calculado': 0.0,
            'tau_ideal': 0.0,
            'method': 'transient_decay',
            'error': 'Invalid V_room or Q_inlet'
        }
    
    # Calculate εv and Age of Air
    ev = k_calculado / k_ideal
    tau_calculado = 1.0 / k_calculado / 60.0  # [min] - real residence time
    tau_ideal = 1.0 / k_ideal / 60.0  # [min] - theoretical residence time
    
    logger.info(f"    * Decay rate calculado (CFD): k_calculado = {k_calculado:.6f} [1/s] = {k_calculado*3600:.2f} [1/h]")
    logger.info(f"    * Decay rate ideal (perfect mix): k_ideal = {k_ideal:.6f} [1/s] = {k_ideal*3600:.2f} [1/h]")
    logger.info(f"    * Age of Air calculado (CFD): τ_calculado = {tau_calculado:.2f} min")
    logger.info(f"    * Age of Air ideal (perfect mix): τ_ideal = {tau_ideal:.2f} min")
    logger.info(f"    * ✓ Ventilation Effectiveness (transient): εv = {ev:.3f}")
    
    # Interpretation
    if ev >= 1.0:
        logger.info(f"       ✓ EXCELLENT: εv ≥ 1.0 (displacement or better than perfect mixing)")
    elif ev >= 0.8:
        logger.info(f"       ✓ ACCEPTABLE: 0.8 ≤ εv < 1.0 (good mixing with minor short-circuiting)")
    else:
        logger.warning(f"       ✗ POOR: εv < 0.8 (significant dead zones or short-circuiting)")
    
    return {
        'ev': float(ev),
        'C_0_mean': C_0_mean,
        'C_f_mean': C_f_mean,
        'k_calculado': float(k_calculado),
        'k_ideal': float(k_ideal),
        'tau_calculado': float(tau_calculado),
        'tau_ideal': float(tau_ideal),
        'method': 'transient_decay',
        'delta_t': float(delta_t),
        'C_supply': C_SUPPLY_PPM,
        'C_initial': C_INITIAL_PPM
    }


def calculate_global_ventilation_metrics(internal_mesh, sim_path, multiblock=None, has_co2=False):
    """
    Calculate global (volumetric) ventilation metrics.
    
    Args:
        internal_mesh: Full 3D PyVista mesh
        sim_path: Path to simulation directory (for reading postProcessing data)
        multiblock: PyVista MultiBlock with boundary patches (for CO2 integration)
        has_co2: Whether CO2 field is available
        
    Returns:
        dict: Global metrics (ACH, εv, mean Age of Air)
    """
    import glob
    
    # Calculate room volume
    V_room = float(internal_mesh.volume) if hasattr(internal_mesh, 'volume') else 0.0
    
    # Read mass flow from postProcessing/massFlow_*/0/surfaceFieldValue.dat
    postproc_path = os.path.join(sim_path, 'postProcessing')
    massflow_pattern = os.path.join(postproc_path, 'massFlow_*', '0', 'surfaceFieldValue.dat')
    massflow_files = glob.glob(massflow_pattern)
    
    if not massflow_files:
        logger.warning(f"    * No mass flow files found in {postproc_path}")
        logger.warning(f"    * ACH cannot be calculated - returning 0.0")
        return {
            'ACH': 0.0,
            'Q_inlet': 0.0,
            'Q_outlet': 0.0,
            'mass_imbalance_pct': 0.0,
            'ev': 0.0,
            'mean_age': 0.0,
            'volume_m3': V_room
        }
    
    logger.info(f"    * Found {len(massflow_files)} mass flow files")
    
    # Read all mass flows
    phi_all = []
    for filepath in massflow_files:
        try:
            # Read file (format: # Time phi_sum)
            data = np.loadtxt(filepath, comments='#')
            
            # Get last timestep (steady-state)
            if data.ndim == 1:
                # Single line
                phi_latest = float(data[1]) if len(data) > 1 else 0.0
            else:
                # Multiple lines - get last row
                phi_latest = float(data[-1, 1])
            
            phi_all.append(phi_latest)
            
            # Extract patch name from path
            patch_name = os.path.basename(os.path.dirname(os.path.dirname(filepath)))
            logger.info(f"       {patch_name}: phi = {phi_latest:.6f} m³/s")
            
        except Exception as e:
            logger.warning(f"       Failed to read {filepath}: {e}")
            continue
    
    if not phi_all:
        logger.warning("    * No valid mass flow data found")
        return {
            'ACH': 0.0,
            'Q_inlet': 0.0,
            'Q_outlet': 0.0,
            'mass_imbalance_pct': 0.0,
            'ev': 0.0,
            'mean_age': 0.0,
            'volume_m3': V_room
        }
    
    # Separate inlet (phi > 0) and outlet (phi < 0)
    Q_inlet = sum([phi for phi in phi_all if phi > 0])
    Q_outlet = sum([phi for phi in phi_all if phi < 0])
    
    # Check mass balance (should be ~0 in steady-state)
    if Q_inlet > 1e-10:
        mass_imbalance_pct = abs(Q_inlet + Q_outlet) / Q_inlet * 100.0
    else:
        mass_imbalance_pct = 0.0
    
    logger.info(f"    * Q_inlet  = {Q_inlet:.6f} m³/s")
    logger.info(f"    * Q_outlet = {Q_outlet:.6f} m³/s")
    logger.info(f"    * Mass imbalance: {mass_imbalance_pct:.2f}%")
    
    if mass_imbalance_pct > 5.0:
        logger.warning(f"    * ⚠️  Mass imbalance > 5% - simulation may not be converged!")
    
    # Calculate ACH (Air Changes per Hour)
    # Use inlet flow only (outlet should be equal in magnitude)
    if V_room > 0 and Q_inlet > 0:
        ACH = Q_inlet * 3600.0 / V_room  # [1/h]
        logger.info(f"    * ACH = {ACH:.2f} renovaciones/h (V_room = {V_room:.2f} m³)")
    else:
        ACH = 0.0
        logger.warning(f"    * ACH = 0.0 (V_room={V_room:.2f}, Q_inlet={Q_inlet:.6f})")
    
    # Calculate εv (Ventilation Effectiveness) from CO2
    ev = 0.0
    mean_age = 0.0
    C_supply = 400.0
    C_exhaust = 0.0
    
    if has_co2 and multiblock is not None:
        logger.info("    * Calculating Ventilation Effectiveness (εv) from CO2")
        
        # Integrate CO2 on boundaries
        co2_boundary = integrate_co2_on_boundaries(multiblock)
        C_supply = co2_boundary['C_supply']
        C_exhaust = co2_boundary['C_exhaust']
        
        if co2_boundary['has_data']:
            # Get C_breathing from standing height (1.1m breathing zone)
            # Calculate mean CO2 in occupied zone (volumetric average)
            CO2_field = internal_mesh.point_data.get('CO2', None)
            if CO2_field is not None:
                # Filter valid CO2 values (non-zero)
                valid_mask = CO2_field > 0
                if valid_mask.sum() > 0:
                    C_breathing = float(CO2_field[valid_mask].mean())
                else:
                    C_breathing = C_supply
                    logger.warning("    * No valid CO2 data in volume - using C_supply")
                
                # Calculate εv
                denom = C_breathing - C_supply
                if abs(denom) > 1.0:  # Avoid division by ~0
                    ev = (C_exhaust - C_supply) / denom
                    logger.info(f"    * Ventilation Effectiveness εv = {ev:.2f}")
                    logger.info(f"       C_supply={C_supply:.1f}, C_exhaust={C_exhaust:.1f}, C_breathing={C_breathing:.1f} ppm")
                    
                    # Estimate mean Age of Air from CO2 steady-state
                    # τ = V_room × (C_breathing - C_supply) / (Q_inlet × (C_exhaust - C_supply))
                    if Q_inlet > 1e-10 and abs(C_exhaust - C_supply) > 1.0:
                        tau_seconds = V_room * (C_breathing - C_supply) / (Q_inlet * (C_exhaust - C_supply))
                        mean_age = tau_seconds / 60.0  # Convert to minutes
                        logger.info(f"    * Mean Age of Air τ = {mean_age:.1f} min")
                    else:
                        logger.warning("    * Cannot calculate Age of Air (Q_inlet or ΔC_exhaust too small)")
                else:
                    logger.warning("    * Cannot calculate εv (C_breathing ≈ C_supply)")
            else:
                logger.warning("    * CO2 field not found in internal_mesh")
        else:
            logger.warning("    * Incomplete CO2 boundary data - cannot calculate εv")
    else:
        logger.info("    * Skipping εv calculation (CO2 or multiblock not available)")
    
    return {
        'ACH': float(ACH),
        'Q_inlet': float(Q_inlet),
        'Q_outlet': float(Q_outlet),
        'mass_imbalance_pct': float(mass_imbalance_pct),
        'ev': float(ev),
        'mean_age': float(mean_age),
        'C_supply': float(C_supply),
        'C_exhaust': float(C_exhaust),
        'volume_m3': V_room
    }


def analyze_ventilation_planes(sim_path, post_path):
    """
    Analyze ventilation metrics in 3 horizontal planes.
    
    Generates:
    - VTK files of slices for each plane
    - PNG images for CO2 (and Age of Air if available)
    - JSON file with ventilation metrics
    
    Args:
        sim_path: Path to simulation directory
        post_path: Path to post-processing output directory
        
    Returns:
        dict: Ventilation metrics for all planes + global metrics
    """
    logger.info("    * Analyzing ventilation in horizontal planes")
    
    # Try to load transient data first (for εv calculation)
    logger.info("    * Attempting to load transient data for εv calculation")
    mesh_t0, mesh_tf, multiblock_t0, multiblock_tf, time_values = load_foam_results_transient(sim_path)
    
    # Load 3D mesh (final timestep for plane analysis)
    logger.info("    * Loading CFD results (final timestep)")
    internal_mesh, surfaces_mesh, multiblock = load_foam_results(sim_path)
    
    logger.info(f"    * Loaded mesh with {internal_mesh.n_cells:,} cells")
    
    # Check if CO2 field exists
    has_co2 = 'CO2' in internal_mesh.point_data
    
    if has_co2:
        logger.info("    * CO2 field found - will analyze concentration")
    else:
        logger.warning("    * CO2 field not found - limited ventilation analysis")
    
    # Calculate global metrics (ACH, εv, Age of Air, etc.)
    global_metrics = calculate_global_ventilation_metrics(internal_mesh, sim_path, multiblock=multiblock, has_co2=has_co2)
    
    # Try to calculate εv from transient decay if data available
    if mesh_t0 is not None and mesh_tf is not None and has_co2:
        logger.info("    * Computing εv from transient CO2 decay (improved method)")
        
        Q_inlet = global_metrics.get('Q_inlet', 0.0)
        V_room = global_metrics.get('volume_m3', 0.0)
        
        if time_values is not None and len(time_values) >= 2:
            delta_t = time_values[-1] - time_values[0]
            
            ev_transient = calculate_ev_from_transient_decay(
                mesh_t0, mesh_tf, Q_inlet, V_room, delta_t
            )
            
            # Update global metrics with transient εv (overrides steady-state)
            if 'error' not in ev_transient:
                global_metrics['ev'] = ev_transient['ev']
                global_metrics['ev_method'] = 'transient_decay'
                global_metrics['ev_transient_data'] = ev_transient
                logger.info(f"    * ✓ Updated εv with transient method: {ev_transient['ev']:.3f}")
            else:
                logger.warning(f"    * Transient εv calculation failed: {ev_transient.get('error', 'unknown')}")
                global_metrics['ev_method'] = 'steady_state_fallback'
        else:
            logger.warning("    * Time values not available for transient εv")
            global_metrics['ev_method'] = 'steady_state_fallback'
    else:
        if not has_co2:
            logger.info("    * Skipping transient εv (no CO2 field)")
        else:
            logger.info("    * Skipping transient εv (transient data not available)")
        global_metrics['ev_method'] = 'steady_state' if has_co2 else 'none'
    
    # Define analysis planes
    planes = {
        'seated':  0.6,
        'standing': 1.1,
        'head':    1.7
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
        
        # Calculate metrics
        metrics = calculate_ventilation_metrics(slice_mesh, has_co2=has_co2)
        metrics['height_m'] = z_height
        metrics['plane_name'] = name
        
        if has_co2:
            logger.info(f"       CO2: mean={metrics['CO2_mean']:.0f} ppm, range=[{metrics['CO2_min']:.0f}, {metrics['CO2_max']:.0f}]")
            logger.info(f"       CO2 compliance (<1000 ppm): {metrics['CO2_compliant_pct']:.1f}%")
        
        logger.info(f"       Stagnation zones (U<0.05 m/s): {metrics['stagnation_pct']:.1f}%")
        logger.info(f"       ADPI: {metrics['ADPI_pct']:.1f}%")
        
        results[name] = metrics
        
        # Save VTK slice
        vtk_path = os.path.join(vtk_dir, f'ventilation_plane_{name}_{z_height}m.vtk')
        slice_mesh.save(vtk_path)
        logger.info(f"       Saved VTK: {os.path.basename(vtk_path)}")
        
        # Render CO2 image if available
        if has_co2:
            render_ventilation_png(slice_mesh, name, z_height, post_path, variable='CO2')
    
    # Add global metrics to results
    results['global'] = global_metrics
    
    # Save metrics to JSON
    metrics_path = os.path.join(post_path, 'ventilation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"    * Saved ventilation metrics: {os.path.basename(metrics_path)}")
    logger.info("    * Ventilation analysis completed successfully")
    
    return results


def generate_ventilation_html_report(results, post_path, case_name='CFD_Case'):
    """
    Generate HTML report with ventilation analysis results.
    
    Args:
        results: dict with ventilation metrics from analyze_ventilation_planes()
        post_path: Path to post-processing directory
        case_name: Name of the simulation case
    """
    logger.info("    * Generating HTML ventilation report")
    
    # Get timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Extract global metrics
    global_metrics = results.get('global', {})
    ACH = global_metrics.get('ACH', 0.0)
    ev = global_metrics.get('ev', 0.0)
    mean_age = global_metrics.get('mean_age', 0.0)
    
    # Calculate average CO2 compliance across planes
    plane_results = {k: v for k, v in results.items() if k != 'global'}
    
    if plane_results:
        global_CO2_compliance = np.mean([r.get('CO2_compliant_pct', 0) for r in plane_results.values()])
        global_ADPI = np.mean([r.get('ADPI_pct', 0) for r in plane_results.values()])
    else:
        global_CO2_compliance = 0.0
        global_ADPI = 0.0
    
    # Build HTML (similar structure to comfort/flow reports)
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ventilation Analysis Report - {case_name}</title>
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
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
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
            grid-template-columns: 1fr;
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
            <h1>Ventilation Analysis Report</h1>
            <div class="subtitle">{case_name}</div>
            <div class="subtitle">Generated: {timestamp}</div>
        </div>
        
        <!-- Executive Summary -->
        <div class="executive-summary">
            <h2>Executive Summary</h2>
            <div class="summary-metric">
                <div class="label">ACH (Air Changes per Hour)</div>
                <div class="value {'comfort-high' if ACH >= 6 else 'comfort-medium' if ACH >= 4 else 'comfort-low'}">
                    {ACH:.1f}
                </div>
                <div style="font-size: 0.75em; color: #6c757d; margin-top: 8px; line-height: 1.4;">
                    Global ventilation rate (Q_inlet/V_room)<br>
                    ASHRAE 62.1 target: ≥6 ACH (offices)
                </div>
            </div>
            <div class="summary-metric">
                <div class="label">CO₂ Compliance (&lt;1000 ppm)</div>
                <div class="value {'comfort-high' if global_CO2_compliance >= 90 else 'comfort-medium' if global_CO2_compliance >= 70 else 'comfort-low'}">
                    {global_CO2_compliance:.1f}%
                </div>
                <div style="font-size: 0.75em; color: #6c757d; margin-top: 8px; line-height: 1.4;">
                    Average across 3 analysis planes (0.6m, 1.1m, 1.7m)<br>
                    ASHRAE 62.1 indoor air quality standard
                </div>
            </div>
            <div class="summary-metric">
                <div class="label">ADPI (Air Distribution)</div>
                <div class="value {'comfort-high' if global_ADPI >= 80 else 'comfort-medium' if global_ADPI >= 60 else 'comfort-low'}">
                    {global_ADPI:.1f}%
                </div>
                <div style="font-size: 0.75em; color: #6c757d; margin-top: 8px; line-height: 1.4;">
                    Area with EDT&lt;3°C and U&lt;0.35 m/s<br>
                    ASHRAE 55 target: ≥80%
                </div>
            </div>
        </div>
        
        <!-- Global Metrics Section -->
        <div class="plane-section">
            <div class="plane-header">
                <h2>Global Ventilation Metrics</h2>
                <div class="height-info">System-wide performance indicators</div>
            </div>
            
            <table class="metrics-table">
                <tr>
                    <th>Metric</th>
                    <th>Value Simulation</th>
                    <th>Target / Notes</th>
                </tr>
                <tr>
                    <td>ACH (Air Changes per Hour)<br><span style="font-size: 0.9em; color: #6c757d;">Q_inlet × 3600 / V_room</span></td>
                    <td class="metric-value {'comfort-high' if ACH >= 6 else 'comfort-medium' if ACH >= 4 else 'comfort-low'}">
                        {ACH:.2f} renovaciones/h
                    </td>
                    <td>&ge; 6 ACH<br><span style="font-size: 0.85em; color: #6c757d;">ASHRAE 62.1 offices | V_room = {global_metrics.get('volume_m3', 0):.1f} m³</span></td>
                </tr>
                <tr>
                    <td>Inlet Flow Rate<br><span style="font-size: 0.9em; color: #6c757d;">Volumetric flow rate (φ &gt; 0)</span></td>
                    <td class="metric-value">
                        {global_metrics.get('Q_inlet', 0):.4f} m³/s
                    </td>
                    <td>-<br><span style="font-size: 0.85em; color: #6c757d;">Sum of all inlet patches</span></td>
                </tr>
                <tr>
                    <td>Outlet Flow Rate<br><span style="font-size: 0.9em; color: #6c757d;">Volumetric flow rate (φ &lt; 0)</span></td>
                    <td class="metric-value">
                        {global_metrics.get('Q_outlet', 0):.4f} m³/s
                    </td>
                    <td>-<br><span style="font-size: 0.85em; color: #6c757d;">Sum of all outlet patches</span></td>
                </tr>
                <tr>
                    <td>Ventilation Effectiveness (εᵥ)<br><span style="font-size: 0.9em; color: #6c757d;">(C_exhaust - C_supply) / (C_breathing - C_supply)</span></td>
                    <td class="metric-value {'comfort-high' if ev >= 1.0 else 'comfort-medium' if ev >= 0.8 else 'comfort-low'}">
                        {global_metrics.get('ev', 0):.2f}
                    </td>
                    <td>&gt; 1.0 (ideal), &gt; 0.8 (acceptable)<br><span style="font-size: 0.85em; color: #6c757d;">ISO 16000-40 | C_supply={global_metrics.get('C_supply', 0):.0f}, C_exhaust={global_metrics.get('C_exhaust', 0):.0f} ppm</span></td>
                </tr>
                <tr>
                    <td>Mean Age of Air<br><span style="font-size: 0.9em; color: #6c757d;">Volumetric average residence time</span></td>
                    <td class="metric-value {'comfort-high' if mean_age > 0 and mean_age <= 10 else 'comfort-medium' if mean_age <= 20 else 'comfort-low'}">
                        {global_metrics.get('mean_age', 0):.1f} min
                    </td>
                    <td>Lower is better<br><span style="font-size: 0.85em; color: #6c757d;">Estimated from CO₂ steady-state | τ_nominal={60/ACH if ACH > 0 else 0:.1f} min</span></td>
                </tr>
            </table>
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
                    <img src="images/ventilation_plane_{plane_name}_{z_height}m_CO2.png" alt="CO2 at {z_height}m">
                    <div class="caption">CO₂ Concentration</div>
                </div>
            </div>
            
            <table class="metrics-table">
                <tr>
                    <th>Metric</th>
                    <th>Value Simulation</th>
                    <th>Target</th>
                </tr>
                <tr>
                    <td>CO₂ Mean<br><span style="font-size: 0.9em; color: #6c757d;">Plane average</span></td>
                    <td class="metric-value {'comfort-high' if metrics.get('CO2_mean', 0) <= 800 else 'comfort-medium' if metrics.get('CO2_mean', 0) <= 1000 else 'comfort-low'}">
                        {metrics.get('CO2_mean', 0):.0f} ppm
                    </td>
                    <td>&lt; 1000 ppm<br><span style="font-size: 0.85em; color: #6c757d;">ASHRAE 62.1</span></td>
                </tr>
                <tr>
                    <td>CO₂ Range<br><span style="font-size: 0.9em; color: #6c757d;">Min/max extremes</span></td>
                    <td class="metric-value">[{metrics.get('CO2_min', 0):.0f}, {metrics.get('CO2_max', 0):.0f}] ppm</td>
                    <td>Within &lt;1000 ppm<br><span style="font-size: 0.85em; color: #6c757d;">Acceptable range</span></td>
                </tr>
                <tr>
                    <td>CO₂ Compliance<br><span style="font-size: 0.9em; color: #6c757d;">% area &lt;1000 ppm</span></td>
                    <td class="metric-value {'comfort-high' if metrics.get('CO2_compliant_pct', 0) >= 90 else 'comfort-medium' if metrics.get('CO2_compliant_pct', 0) >= 70 else 'comfort-low'}">
                        {metrics.get('CO2_compliant_pct', 0):.1f}%
                    </td>
                    <td>&ge; 90%<br><span style="font-size: 0.85em; color: #6c757d;">Target coverage</span></td>
                </tr>
                <tr>
                    <td>Stagnation Zones<br><span style="font-size: 0.9em; color: #6c757d;">% area U&lt;0.05 m/s</span></td>
                    <td class="metric-value {'comfort-high' if metrics.get('stagnation_pct', 0) <= 10 else 'comfort-medium' if metrics.get('stagnation_pct', 0) <= 25 else 'comfort-low'}">
                        {metrics.get('stagnation_pct', 0):.1f}%
                    </td>
                    <td>&lt; 10%<br><span style="font-size: 0.85em; color: #6c757d;">Minimize dead zones</span></td>
                </tr>
                <tr>
                    <td>ADPI (Air Distribution)<br><span style="font-size: 0.9em; color: #6c757d;">EDT&lt;3°C & U&lt;0.35 m/s</span></td>
                    <td class="metric-value {'comfort-high' if metrics.get('ADPI_pct', 0) >= 80 else 'comfort-medium' if metrics.get('ADPI_pct', 0) >= 60 else 'comfort-low'}">
                        {metrics.get('ADPI_pct', 0):.1f}%
                    </td>
                    <td>&ge; 80%<br><span style="font-size: 0.85em; color: #6c757d;">ASHRAE 55</span></td>
                </tr>
            </table>
        </div>
"""
    
    html_content += """
        <!-- Footer -->
        <div class="footer">
            <div class="info-row">
                <span class="label">Analysis Standard:</span>
                <span class="value">ASHRAE 62.1 (Ventilation) & ASHRAE 55 (ADPI)</span>
            </div>
            <div class="info-row">
                <span class="label">Solver:</span>
                <span class="value">OpenFOAM buoyantBoussinesqPimpleFoam</span>
            </div>
            <div class="info-row">
                <span class="label">Metrics:</span>
                <span class="value">CO₂, ADPI, Stagnation Zones</span>
            </div>
        </div>
    </div>
</body>
</html>
"""
    
    # Write HTML file
    html_path = os.path.join(post_path, 'ventilation_report.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"    * HTML ventilation report generated: {os.path.basename(html_path)}")
    logger.info(f"    * Open in browser: file://{html_path}")
    
    return html_path
