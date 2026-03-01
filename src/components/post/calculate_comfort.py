#!/usr/bin/env python3.12
"""
Calculate PMV/PPD thermal comfort fields from OpenFOAM T and U fields.
Executes in Inductiva container after reconstructPar.

Uses ISO 7730 standard with hardcoded parameters:
- Met = 1.0 (sedentary office work)
- Clo = 0.7 (light office clothing)
- RH = 50% (standard relative humidity)
- Tmrt = Tair (mean radiant temperature equals air temperature)

Usage:
    python3 calculate_comfort.py <case_path>              # Process only latest timestep
    python3 calculate_comfort.py <case_path> --all-times  # Process all timesteps (excluding t=0)
"""

import os
import sys
import argparse
import numpy as np
from foamlib import FoamCase
import logging
from pythermalcomfort.models import pmv_ppd_iso
from pythermalcomfort.utilities import v_relative

# HARDCODED COMFORT PARAMETERS (ISO 7730 standard office)
MET = 1.0      # Metabolic rate [met] - sedentary office work
CLO = 0.7      # Clothing insulation [clo] - light office clothing (shirt + trousers)
RH = 50.0      # Relative humidity [%]
# Tmrt = Tair (mean radiant temperature equals air temperature - valid for HVAC without radiation)

logger = logging.getLogger(__name__)


def calculate_pmv_fanger(tdb, tr, vel, rh, met, clo):
    """
    Calculate PMV using ISO 7730 standard via pythermalcomfort library.
    
    This is a wrapper function that calls the validated pythermalcomfort.models.pmv_ppd_iso()
    implementation, which is compliant with ISO 7730-2005 standard.
    
    Args:
        tdb: Dry bulb air temperature [°C]
        tr: Mean radiant temperature [°C]
        vel: Air velocity [m/s]
        rh: Relative humidity [%]
        met: Metabolic rate [met]
        clo: Clothing insulation [clo]
    
    Returns:
        PMV value (Predicted Mean Vote)
    """
    # Calculate relative velocity accounting for activity-generated air speed
    vr = v_relative(v=vel, met=met)
    
    # Call validated pythermalcomfort implementation
    # limit_inputs=False allows calculations outside standard ranges (for extreme conditions)
    result = pmv_ppd_iso(
        tdb=tdb, 
        tr=tr, 
        vr=vr, 
        rh=rh, 
        met=met, 
        clo=clo,
        wme=0.0,  # External work = 0 (standard assumption)
        limit_inputs=False  # Accept values outside standard comfort range
    )
    
    return result.pmv


def calculate_ppd(pmv):
    """
    Calculate PPD (Predicted Percentage of Dissatisfied) from PMV (ISO 7730).
    
    Args:
        pmv: Predicted Mean Vote
    
    Returns:
        PPD value [%] (percentage of people dissatisfied with thermal environment)
    """
    ppd = 100.0 - 95.0 * np.exp(-0.03353 * pmv ** 4 - 0.2179 * pmv ** 2)
    return ppd


def process_timestep(case, time_str):
    """
    Process a single timestep to calculate PMV/PPD fields.
    
    Args:
        case: FoamCase object
        time_str: Timestep string (e.g., "500", "1000")
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Processing timestep: {time_str}")
    logger.info(f"{'='*70}")
    
    # Read temperature field [K]
    logger.info("Reading temperature field (T)...")
    with case[time_str]['T'] as T_field:
        T_internal = T_field.internal_field  # [K]
        T_boundary = T_field.boundary_field
    
    # Read velocity field [m/s]
    logger.info("Reading velocity field (U)...")
    with case[time_str]['U'] as U_field:
        U_internal = U_field.internal_field  # [m/s]
        U_boundary = U_field.boundary_field
    
    # Try to read incident radiation field G [W/m²] for accurate T_mrt calculation
    logger.info("Reading incident radiation field (G)...")
    G_internal = None
    try:
        with case[time_str]['G'] as G_field:
            G_internal = G_field.internal_field  # [W/m²]
        logger.info("✓ G field found - will calculate T_mrt from radiation")
    except:
        logger.warning("✗ G field not found - will use T_mrt = T_air (simplified)")
    
    # Handle uniform fields (when solver writes "uniform <value>" instead of cell-by-cell data)
    # This happens when the field didn't evolve or converged to uniform value
    
    # Get number of cells from a non-uniform field or from boundary file
    # Strategy: Try to read from p field (should always be non-uniform after simulation)
    # If that fails, count from mesh files
    try:
        with case[time_str]['p'] as p_field:
            p_internal = p_field.internal_field
            if isinstance(p_internal, (int, float)):
                # p is also uniform, need to read from mesh
                raise ValueError("p is uniform, need mesh method")
            n_cells = len(p_internal)
            logger.info(f"Detected {n_cells} cells from p field")
    except:
        # Fallback: read from polyMesh/owner header
        owner_file = os.path.join(case.path, 'constant', 'polyMesh', 'owner')
        with open(owner_file, 'r') as f:
            in_header = True
            for line in f:
                line_stripped = line.strip()
                # Skip empty lines and comments
                if not line_stripped or line_stripped.startswith('//') or line_stripped.startswith('/*'):
                    continue
                # Skip FoamFile dictionary
                if '{' in line or '}' in line or line_stripped in ['FoamFile', 'version', 'format', 'class', 'object', 'note']:
                    continue
                # Check if line starts with a digit (could be version number or data count)
                if line_stripped[0].isdigit() and ';' not in line:
                    # This is likely the count line (number before opening parenthesis)
                    n_internal_faces = int(line_stripped)
                    # Number of cells is approximately n_internal_faces (close estimate)
                    # But we need to count unique cell indices from owner list
                    break
            
            # Now read owner list to get max cell index
            reading_data = False
            max_owner = -1
            for line in f:
                line_stripped = line.strip()
                if line_stripped == '(':
                    reading_data = True
                    continue
                if line_stripped == ')':
                    break
                if reading_data and line_stripped and not line_stripped.startswith('//'):
                    owner_idx = int(line_stripped)
                    if owner_idx > max_owner:
                        max_owner = owner_idx
            
            n_cells = max_owner + 1
            logger.info(f"Detected {n_cells} cells from polyMesh/owner")
    
    if isinstance(T_internal, (int, float)):
        logger.info(f"  T is uniform field (value={T_internal:.2f} K), expanding to {n_cells} cells...")
        T_internal = np.full(n_cells, T_internal)
    
    if isinstance(U_internal, (int, float)):
        logger.info(f"  U is uniform field (value={U_internal:.2e} m/s), expanding to {n_cells} cells...")
        U_internal = np.full((n_cells, 3), U_internal)
    elif len(U_internal.shape) == 1 and U_internal.shape[0] == 3:
        # U is a single 3D vector, expand to all cells
        logger.info(f"  U is uniform 3D vector, expanding to {n_cells} cells...")
        U_internal = np.tile(U_internal, (n_cells, 1))
    
    n_cells = len(T_internal)
    logger.info(f"Number of cells: {n_cells}")
    
    # Convert temperature to Celsius
    T_celsius = T_internal - 273.15
    
    # Calculate velocity magnitude
    U_mag = np.linalg.norm(U_internal, axis=1)
    
    logger.info("Calculating PMV/PPD fields...")
    
    # Debug: Sample first 5 cells
    logger.info(f"\nDEBUG - Sample of first 5 cells:")
    for i in range(min(5, n_cells)):
        logger.info(f"  Cell {i}: T={T_celsius[i]:.2f}°C, U_mag={np.linalg.norm(U_internal[i]):.4f} m/s")
    
    # Initialize output arrays
    pmv_field = np.zeros(n_cells)
    ppd_field = np.zeros(n_cells)
    
    # Calculate PMV and PPD for each cell
    # Apply valid range limits for PMV/PPD calculation (Fanger model is valid for thermal comfort zones)
    VALID_TEMP_MIN = 10.0  # °C
    VALID_TEMP_MAX = 40.0  # °C  
    VALID_VEL_MAX = 5.0    # m/s
    INVALID_VALUE = -1000.0  # Sentinel value for invalid/out-of-range cells
    
    # Theoretical valid ranges for PMV and PPD results
    # NOTE: ISO 7730 comfort range is -0.5 to +0.5, but PMV model can give values beyond [-3, +3]
    # We accept wider range to capture extreme conditions (very cold/hot)
    PMV_MIN = -10.0  # Accept very cold conditions
    PMV_MAX = 10.0   # Accept very hot conditions
    PPD_MIN = 0.0    # PPD is a percentage from 0% to 100%
    PPD_MAX = 100.0
    
    invalid_count = 0
    for i in range(n_cells):
        tdb = T_celsius[i]
        
        # Calculate mean radiant temperature from incident radiation G
        if G_internal is not None:
            # G [W/m²] = 4σT_mrt⁴ (isotropic radiation equilibrium)
            # σ = 5.67e-8 W/(m²·K⁴) (Stefan-Boltzmann constant)
            # Solve for T_mrt: T_mrt = (G / (4σ))^0.25
            sigma = 5.67e-8  # Stefan-Boltzmann constant [W/(m²·K⁴)]
            
            # Handle G expansion if uniform
            if isinstance(G_internal, (int, float)):
                G_val = G_internal
            else:
                G_val = G_internal[i]
            
            # Ensure G is positive (radiation intensity cannot be negative)
            G_val = max(G_val, 0.1)  # Minimum 0.1 W/m² to avoid singularity
            
            # Calculate T_mrt [K] from incident radiation (CORRECTED with factor 4)
            T_mrt_K = (G_val / (4 * sigma)) ** 0.25
            tr = T_mrt_K - 273.15  # Convert to °C
            
            # Sanity check: T_mrt should be in reasonable range
            tr = np.clip(tr, -10, 80)  # Clip to physical range [-10°C, 80°C]
        else:
            # Fallback: T_mrt = T_air (simplified - no radiation data)
            tr = tdb
        
        vel = U_mag[i]
        
        # Check if values are within valid range for PMV/PPD calculation
        if (VALID_TEMP_MIN <= tdb <= VALID_TEMP_MAX and 
            VALID_TEMP_MIN <= tr <= VALID_TEMP_MAX and 
            vel <= VALID_VEL_MAX):
            # Calculate PMV
            try:
                pmv_val = calculate_pmv_fanger(tdb, tr, vel, RH, MET, CLO)
                ppd_val = calculate_ppd(pmv_val)
                
                # Check if results are finite (not NaN, not inf) AND within theoretical ranges
                if (np.isfinite(pmv_val) and np.isfinite(ppd_val) and
                    PMV_MIN <= pmv_val <= PMV_MAX and
                    PPD_MIN <= ppd_val <= PPD_MAX):
                    # Valid result - use calculated values
                    pmv_field[i] = pmv_val
                    ppd_field[i] = ppd_val
                else:
                    # Result is finite but outside theoretical range OR infinite - use sentinel
                    pmv_field[i] = INVALID_VALUE
                    ppd_field[i] = INVALID_VALUE
                    invalid_count += 1
            except (OverflowError, ValueError):
                # Handle numerical errors - use sentinel value
                pmv_field[i] = INVALID_VALUE
                ppd_field[i] = INVALID_VALUE
                invalid_count += 1
        else:
            # Input conditions out of valid range - use sentinel value
            pmv_field[i] = INVALID_VALUE
            ppd_field[i] = INVALID_VALUE
            invalid_count += 1
    
    # Statistics
    logger.info("")
    logger.info("Results:")
    valid_cells = n_cells - invalid_count
    valid_pct = (valid_cells / n_cells) * 100
    logger.info(f"  Valid cells for PMV/PPD: {valid_cells}/{n_cells} ({valid_pct:.1f}%)")
    logger.info(f"  Invalid/out-of-range cells (marked as {INVALID_VALUE}): {invalid_count} ({(invalid_count/n_cells)*100:.1f}%)")
    
    # Only calculate stats on valid values (exclude INVALID_VALUE sentinel)
    valid_pmv = pmv_field[pmv_field != INVALID_VALUE]
    valid_ppd = ppd_field[ppd_field != INVALID_VALUE]
    
    if len(valid_pmv) > 0:
        logger.info(f"  PMV: min={valid_pmv.min():.2f}, max={valid_pmv.max():.2f}, mean={valid_pmv.mean():.2f}")
        logger.info(f"  PPD: min={valid_ppd.min():.1f}%, max={valid_ppd.max():.1f}%, mean={valid_ppd.mean():.1f}%")
    else:
        logger.info(f"  PMV: No valid values (all marked as {INVALID_VALUE})")
        logger.info(f"  PPD: No valid values (all marked as {INVALID_VALUE})")
    
    # Comfort classification (ISO 7730)
    comfortable = np.sum((pmv_field >= -0.5) & (pmv_field <= 0.5))
    comfortable_pct = 100.0 * comfortable / n_cells
    logger.info(f"  Cells in comfort zone (-0.5 < PMV < 0.5): {comfortable_pct:.1f}%")
    
    # Write PMV field
    logger.info("\nWriting PMV field...")
    with case[time_str]['PMV'] as pmv_out:
        pmv_out.internal_field = pmv_field
        # Keep existing boundary fields (type: 'calculated')
        pmv_out.boundary_field = {}
        for patch_name, patch_data in T_boundary.items():
            pmv_out.boundary_field[patch_name] = {
                'type': 'calculated',
                'value': pmv_field.mean()  # Use mean value for boundaries
            }
    
    # Write PPD field
    logger.info("Writing PPD field...")
    with case[time_str]['PPD'] as ppd_out:
        ppd_out.internal_field = ppd_field
        # Keep existing boundary fields (type: 'calculated')
        ppd_out.boundary_field = {}
        for patch_name, patch_data in T_boundary.items():
            ppd_out.boundary_field[patch_name] = {
                'type': 'calculated',
                'value': ppd_field.mean()  # Use mean value for boundaries
            }
    
    logger.info(f"✓ PMV/PPD calculated for timestep {time_str}")


def main():
    """
    Main function to calculate PMV/PPD fields from OpenFOAM case.
    """
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Calculate PMV/PPD thermal comfort fields from OpenFOAM results')
    parser.add_argument('case_path', nargs='?', default='.', help='Path to OpenFOAM case directory')
    parser.add_argument('--all-times', action='store_true', 
                        help='Calculate PMV/PPD for all timesteps (excluding t=0)')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger.info("=" * 70)
    logger.info("THERMAL COMFORT CALCULATION (PMV/PPD)")
    logger.info("=" * 70)
    logger.info(f"Comfort parameters:")
    logger.info(f"  - Metabolic rate (Met): {MET} met")
    logger.info(f"  - Clothing insulation (Clo): {CLO} clo")
    logger.info(f"  - Relative humidity (RH): {RH} %")
    logger.info(f"  - Mean radiant temp (Tmrt): Equal to air temperature")
    
    # Open OpenFOAM case
    case = FoamCase(args.case_path)
    
    # Find all timesteps
    time_dirs = [d for d in os.listdir(args.case_path) if d.replace('.', '').replace('-', '').isdigit()]
    if len(time_dirs) == 0:
        logger.error("\nNo timesteps found in case!")
        sys.exit(1)
    
    # Sort timesteps
    time_values = [float(t) for t in time_dirs]
    time_values.sort()
    
    # Determine which timesteps to process
    if args.all_times:
        # Process all timesteps except 0
        times_to_process = [t for t in time_values if t > 0]
        logger.info(f"\nMode: Processing ALL timesteps (excluding t=0)")
        logger.info(f"Timesteps to process: {times_to_process}")
    else:
        # Process only latest timestep (backward compatibility)
        times_to_process = [time_values[-1]]
        logger.info(f"\nMode: Processing ONLY latest timestep")
        logger.info(f"Latest timestep: {times_to_process[0]}")
    
    if len(times_to_process) == 0:
        logger.error("\nNo timesteps to process!")
        sys.exit(1)
    
    # Process each timestep
    for time_val in times_to_process:
        # Convert to string: use integer format if it's a whole number
        if time_val.is_integer():
            time_str = str(int(time_val))
        else:
            time_str = str(time_val)
        
        try:
            process_timestep(case, time_str)
        except Exception as e:
            logger.error(f"\n❌ Error processing timestep {time_str}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # Continue with next timestep instead of failing completely
            continue
    
    logger.info("\n" + "=" * 70)
    logger.info(f"✅ PMV/PPD calculation completed for {len(times_to_process)} timestep(s)")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
