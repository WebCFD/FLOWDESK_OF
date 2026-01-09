#!/usr/bin/env python3
"""
Calculate PMV/PPD thermal comfort fields from OpenFOAM T and U fields.
Executes in Inductiva container after reconstructPar.

Uses ISO 7730 standard with hardcoded parameters:
- Met = 1.0 (sedentary office work)
- Clo = 0.7 (light office clothing)
- RH = 50% (standard relative humidity)
- Tmrt = Tair (mean radiant temperature equals air temperature)
"""

import os
import sys
import numpy as np
from foamlib import FoamCase
import logging

# HARDCODED COMFORT PARAMETERS (ISO 7730 standard office)
MET = 1.0      # Metabolic rate [met] - sedentary office work
CLO = 0.7      # Clothing insulation [clo] - light office clothing (shirt + trousers)
RH = 50.0      # Relative humidity [%]
# Tmrt = Tair (mean radiant temperature equals air temperature - valid for HVAC without radiation)

logger = logging.getLogger(__name__)


def calculate_pmv_fanger(tdb, tr, vel, rh, met, clo):
    """
    Calculate PMV (Predicted Mean Vote) using Fanger's equation (ISO 7730).
    
    Args:
        tdb: Dry bulb air temperature [°C]
        tr: Mean radiant temperature [°C]
        vel: Relative air velocity [m/s]
        rh: Relative humidity [%]
        met: Metabolic rate [met]
        clo: Clothing insulation [clo]
    
    Returns:
        PMV value (thermal sensation scale: -3 cold, 0 neutral, +3 hot)
    """
    # Convert metabolic rate to W/m²
    M = met * 58.15  # [W/m²]
    W = 0.0  # External work [W/m²] (assumed zero for sedentary activities)
    
    # Clothing thermal resistance [m²·K/W]
    Icl = clo * 0.155
    
    # Calculate clothing surface area factor
    if Icl <= 0.078:
        fcl = 1.0 + 1.290 * Icl
    else:
        fcl = 1.05 + 0.645 * Icl
    
    # Calculate partial vapor pressure [Pa]
    pa = rh / 100.0 * 10.0 * np.exp(16.6536 - 4030.183 / (tdb + 235.0))
    
    # Calculate clothing surface temperature by iteration
    tcl = tdb + 273.15  # Initial guess [K]
    hc = 0.0  # Initialize convection coefficient
    for _ in range(10):  # Iterative solution
        hc = 2.38 * np.abs(tcl - tdb - 273.15) ** 0.25
        if 12.1 * np.sqrt(vel) > hc:
            hc = 12.1 * np.sqrt(vel)
        
        tcl_new = 35.7 - 0.028 * (M - W) - Icl * fcl * (
            3.96e-8 * fcl * (tcl ** 4 - (tr + 273.15) ** 4) +
            fcl * hc * (tcl - tdb - 273.15)
        )
        tcl = tcl_new + 273.15
    
    # Calculate heat transfer components
    hl1 = 3.05e-3 * (5733.0 - 6.99 * (M - W) - pa)  # Heat loss by skin diffusion
    hl2 = 0.42 * ((M - W) - 58.15) if (M - W) > 58.15 else 0.0  # Heat loss by sweating
    hl3 = 1.7e-5 * M * (5867.0 - pa)  # Latent respiration heat loss
    hl4 = 0.0014 * M * (34.0 - tdb)  # Dry respiration heat loss
    hl5 = 3.96e-8 * fcl * (tcl ** 4 - (tr + 273.15) ** 4)  # Heat loss by radiation
    hl6 = fcl * hc * (tcl - tdb - 273.15)  # Heat loss by convection
    
    # Calculate thermal load
    ts = 0.303 * np.exp(-0.036 * M) + 0.028
    pmv = ts * ((M - W) - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)
    
    return pmv


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


def main(case_path):
    """
    Main function to calculate PMV/PPD fields from OpenFOAM case.
    
    Args:
        case_path: Path to OpenFOAM case directory
    """
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger.info("=" * 70)
    logger.info("THERMAL COMFORT CALCULATION (PMV/PPD)")
    logger.info("=" * 70)
    
    # Open OpenFOAM case
    case = FoamCase(case_path)
    
    # Find latest timestep
    time_dirs = [d for d in os.listdir(case_path) if d.replace('.', '').replace('-', '').isdigit()]
    if len(time_dirs) == 0:
        logger.error("No timesteps found in case!")
        sys.exit(1)
    
    # Sort and get latest
    time_values = [float(t) for t in time_dirs]
    time_values.sort()
    latest_time_float = time_values[-1]
    
    # Convert to string: use integer format if it's a whole number
    if latest_time_float.is_integer():
        latest_time = str(int(latest_time_float))
    else:
        latest_time = str(latest_time_float)
    
    logger.info(f"Latest timestep: {latest_time}")
    logger.info(f"Comfort parameters:")
    logger.info(f"  - Metabolic rate (Met): {MET} met")
    logger.info(f"  - Clothing insulation (Clo): {CLO} clo")
    logger.info(f"  - Relative humidity (RH): {RH} %")
    logger.info(f"  - Mean radiant temp (Tmrt): Equal to air temperature")
    logger.info("")
    
    # Read temperature field [K]
    logger.info("Reading temperature field (T)...")
    with case[latest_time]['T'] as T_field:
        T_internal = T_field.internal_field  # [K]
        T_boundary = T_field.boundary_field
    
    # Read velocity field [m/s]
    logger.info("Reading velocity field (U)...")
    with case[latest_time]['U'] as U_field:
        U_internal = U_field.internal_field  # [m/s]
        U_boundary = U_field.boundary_field
    
    n_cells = len(T_internal)
    logger.info(f"Number of cells: {n_cells}")
    logger.info("")
    
    # Convert temperature to Celsius
    T_celsius = T_internal - 273.15
    
    # Calculate velocity magnitude
    U_mag = np.linalg.norm(U_internal, axis=1)
    
    logger.info("Calculating PMV/PPD fields...")
    
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
    PMV_MIN = -3.0  # Typical comfort range is -0.5 to +0.5, extreme is -3 to +3
    PMV_MAX = 3.0
    PPD_MIN = 0.0   # PPD is a percentage from 0% to 100%
    PPD_MAX = 100.0
    
    invalid_count = 0
    for i in range(n_cells):
        tdb = T_celsius[i]
        tr = tdb  # Tmrt = Tair (simplification)
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
    logger.info("")
    
    # Write PMV field
    logger.info("Writing PMV field...")
    with case[latest_time]['PMV'] as pmv_out:
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
    with case[latest_time]['PPD'] as ppd_out:
        ppd_out.internal_field = ppd_field
        # Keep existing boundary fields (type: 'calculated')
        ppd_out.boundary_field = {}
        for patch_name, patch_data in T_boundary.items():
            ppd_out.boundary_field[patch_name] = {
                'type': 'calculated',
                'value': ppd_field.mean()  # Use mean value for boundaries
            }
    
    logger.info("")
    logger.info("✓ PMV/PPD thermal comfort fields calculated successfully")
    logger.info("=" * 70)


if __name__ == "__main__":
    case_path = sys.argv[1] if len(sys.argv) > 1 else "."
    main(case_path)
