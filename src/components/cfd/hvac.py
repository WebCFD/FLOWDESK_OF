import os
import shutil
import logging
import numpy as np
import pandas as pd

from foamlib import FoamCase, FoamFile


logger = logging.getLogger(__name__)



# CONSTANTS FOR THERMOPHYSICS
CP = 1005.0  # Specific heat at constant pressure [J/(kg·K)]
HF = 0.0  # Formation enthalpy [J/kg]
# Note: Using Boussinesq (incompressiblePerfectGas + hConst): h = Cp×T, ρ = pRef/(R×T)

# DIMENSIONS - MINIMAL FIELDS FOR buoyantSimpleFoam LAMINAR
# Only 5 essential fields required for solver operation
DIMENSIONS_DICT = {
    'h':        FoamFile.DimensionSet(length=2, time=-2),  # J/kg = m²/s² - Primary thermodynamic variable
    'p':        FoamFile.DimensionSet(mass=1, length=-1, time=-2),  # Pressure [Pa]
    'p_rgh':    FoamFile.DimensionSet(mass=1, length=-1, time=-2),  # Buoyancy-corrected pressure [Pa]
    'T':        FoamFile.DimensionSet(temperature=1),  # Temperature [K]
    'U':        FoamFile.DimensionSet(length=1, time=-1),  # Velocity [m/s]
}

INTERNALFIELD_DICT = {
    'h':        294515.75,  # h = Cp×T = 1005×293.15 for Boussinesq (20°C)
    'p':        101325,     # Atmospheric pressure [Pa] (will be modified by setFields for hydrostatic gradient)
    'p_rgh':    101325,     # Modified pressure (constant in hydrostatic equilibrium)
    'T':        293.15,     # Reference temperature for Boussinesq [K] (20°C)
    'U':        np.array([0, 0, 0]),  # Initial velocity (quiescent fluid)
}

# Reference values for pressure calculations
P_ATM = 101325  # Atmospheric pressure [Pa]
RHO_REF = 1.2   # Reference air density at 20°C [kg/m³]
G = 9.81        # Gravitational acceleration [m/s²]
# For hydrostatic equilibrium: p_rgh = constant = p_atm
# This ensures p(z) = p_rgh - rho*g*z has the correct gradient
P_RGH_APERTURE = P_ATM  # p_rgh at atmospheric pressure openings = 101325 Pa


def define_constant_files(template_path, sim_path):
    source_constant_path = os.path.join(template_path, 'constant')
    target_constant_path = os.path.join(sim_path, 'constant')
    for filename in os.listdir(source_constant_path):
        source_file = os.path.join(source_constant_path, filename)
        target_file = os.path.join(target_constant_path, filename)
        shutil.copy(src=source_file, dst=target_file)


def define_system_files(template_path, sim_path):
    source_constant_path = os.path.join(template_path, 'system')
    target_constant_path = os.path.join(sim_path, 'system')
    for filename in os.listdir(source_constant_path):
        source_file = os.path.join(source_constant_path, filename)
        target_file = os.path.join(target_constant_path, filename)
        shutil.copy(src=source_file, dst=target_file)


def update_controldict_iterations(case_path, simulation_type):
    """
    Update controlDict endTime, writeInterval, and purgeWrite based on simulation type.
    
    Maps simulation types to iteration counts:
    - comfortTest: 3 iterations, write every iteration, keep all timesteps
    - comfort30Iter: 500 iterations, write only last iteration, keep only last timestep
    - test_calculation: 3 iterations (default)
    
    Args:
        case_path: Path to case directory
        simulation_type: Type of simulation (comfortTest, comfort30Iter, test_calculation)
    """
    logger.info(f"    * Updating controlDict iterations for simulation type: {simulation_type}")
    
    # Map simulation types to (iterations, writeInterval, purgeWrite)
    config_map = {
        'comfortTest': (3, 1, 0),      # 3 iter, write every iteration, keep all
        'comfort30Iter': (500, 1, 1),  # 500 iter, write every iteration, keep only last
        'test_calculation': (3, 1, 0)   # 3 iter, write every iteration, keep all
    }
    
    iterations, write_interval, purge_write = config_map.get(simulation_type, (3, 1, 0))
    logger.info(f"    * Setting endTime={iterations}, writeInterval={write_interval}, purgeWrite={purge_write}")
    
    sim_path = os.path.join(case_path, "sim")
    case = FoamCase(sim_path)
    with case['system']['controlDict'] as ctrl:
        ctrl['endTime'] = iterations
        ctrl['writeInterval'] = write_interval
        ctrl['purgeWrite'] = purge_write
        
        # CRITICAL: Also update VTK function writeInterval
        if 'functions' in ctrl and 'writeVTK' in ctrl['functions']:
            ctrl['functions']['writeVTK']['writeInterval'] = write_interval
            logger.info(f"    * Updated VTK writeInterval to {write_interval}")
        
        logger.info(f"    * ✅ controlDict updated: endTime={iterations}, writeInterval={write_interval}, purgeWrite={purge_write}")


def define_initial_files(sim_path, patch_df):
    os.makedirs(sim_path, exist_ok=True)

    # Create 0.orig/ directory for initial conditions
    # This will be copied to 0/ after snappyHexMesh by Allrun script
    initial_path = os.path.join(sim_path, "0.orig")
    os.makedirs(initial_path, exist_ok=True)

    # ============ STABILITY TEST MODE ============
    # Set to True to test solver stability with equilibrium solution (all walls, uniform fields)
    # Set to False to use normal boundary conditions from JSON
    STABILITY_TEST_MODE = False
    # =============================================
    
    if STABILITY_TEST_MODE:
        logger.info("=" * 80)
        logger.info("*** STABILITY TEST MODE ACTIVE ***")
        logger.info("*** All patches forced to wall BCs with uniform initial conditions ***")
        logger.info("*** This tests if solver is stable starting from equilibrium ***")
        logger.info("=" * 80)
        
        case = FoamCase(sim_path)
        for variable in DIMENSIONS_DICT.keys():
            with case['0.orig'][variable] as f:
                f.dimensions = DIMENSIONS_DICT[variable]
                f.internal_field = INTERNALFIELD_DICT[variable]
                
                f.boundary_field = dict()
                for _, row in patch_df.iterrows():
                    new_bc_data = dict()
                    
                    # Force all patches to wall-type boundary conditions
                    if(variable == 'h'):
                        # Uniform enthalpy at 20°C (equilibrium)
                        new_bc_data["type"] = 'fixedValue'
                        new_bc_data["value"] = CP * 293.15  # h = Cp×T = 294515.75 J/kg
                    elif(variable == 'p'):
                        new_bc_data["type"] = 'calculated'
                        new_bc_data["value"] = P_ATM
                    elif(variable == 'p_rgh'):
                        new_bc_data["type"] = 'fixedFluxPressure'
                        new_bc_data["value"] = '$internalField'
                    elif(variable == 'T'):
                        # Uniform temperature at 20°C (equilibrium)
                        new_bc_data["type"] = 'fixedValue'
                        new_bc_data["value"] = 293.15  # 20°C
                    elif(variable == 'U'):
                        # No-slip walls (zero velocity everywhere)
                        new_bc_data["type"] = 'noSlip'
                    else:
                        raise BaseException('Unknown variable')
                    
                    f.boundary_field[row['id']] = new_bc_data
        
        logger.info("*** Stability test initial conditions created successfully ***")
        return
    
    # ============ NORMAL MODE (ORIGINAL CODE) ============
    case = FoamCase(sim_path)
    for variable in DIMENSIONS_DICT.keys():
        with case['0.orig'][variable] as f:
            f.dimensions = DIMENSIONS_DICT[variable]
            f.internal_field = INTERNALFIELD_DICT[variable]

            f.boundary_field = dict()
            for _, row in patch_df.iterrows():
                new_bc_data = dict()
                if(row['type'] == 'wall'):
                    if(variable == 'h'):
                        # Enthalpy for perfectGas: h = Cp×T
                        new_bc_data["type"] = 'fixedValue'
                        T_celsius = row['T']
                        T_wall = T_celsius + 273.15
                        h_value = CP * T_wall
                        logger.info(f"    BC {row['id']} ({row['type']}): T={T_celsius}°C → T_K={T_wall}K → h={h_value} J/kg")
                        new_bc_data["value"] = h_value
                    elif(variable == 'p'):
                        # p = p_rgh + p_atm (for walls, p_rgh ≈ 0)
                        new_bc_data["type"] = 'calculated'
                        new_bc_data["value"] = P_ATM
                    elif(variable == 'p_rgh'):
                        new_bc_data["type"] = 'fixedFluxPressure'
                        new_bc_data["value"] = '$internalField'
                    elif(variable == 'T'):
                        new_bc_data["type"] = 'fixedValue'
                        new_bc_data["value"] = row['T'] + 273.15
                    elif(variable == 'U'):
                        new_bc_data["type"] = 'noSlip'
                    else:
                        raise BaseException('Unknown variable')
                elif(row['type'] == 'velocity_inlet'):
                    if(variable == 'h'):
                        # Enthalpy for perfectGas: h = Cp×T
                        new_bc_data["type"] = 'fixedValue'
                        T_celsius = row['T']
                        T_wall = T_celsius + 273.15
                        h_value = CP * T_wall
                        logger.info(f"    BC {row['id']} ({row['type']}): T={T_celsius}°C → T_K={T_wall}K → h={h_value} J/kg")
                        new_bc_data["value"] = h_value
                    elif(variable == 'p'):
                        # p = p_rgh + p_atm (for walls, p_rgh ≈ 0)
                        new_bc_data["type"] = 'calculated'
                        new_bc_data["value"] = P_ATM
                    elif(variable == 'p_rgh'):
                        new_bc_data["type"] = 'fixedFluxPressure'
                        new_bc_data["value"] = '$internalField'
                    elif(variable == 'T'):
                        new_bc_data["type"] = 'fixedValue'
                        new_bc_data["value"] = row['T'] + 273.15
                    elif(variable == 'U'):
                        new_bc_data["type"] = 'fixedValue'
                        if (row['open']):
                            new_bc_data["value"] = row['U'] * np.array([row['nx'], row['ny'], row['nz']])
                        else:
                            new_bc_data["value"] = np.array([0, 0, 0])
                    else:
                        raise BaseException('Unknown variable')
                elif(row['type'] == 'pressure_inlet'):
                    if(variable == 'h'):
                        # Enthalpy for perfectGas: h = Cp×T (use exterior temperature from CSV)
                        new_bc_data["type"] = 'fixedValue'
                        T_celsius = row['T']
                        T_exterior = T_celsius + 273.15  # Convert °C → K
                        h_value = CP * T_exterior
                        logger.info(f"    BC {row['id']} ({row['type']}): T={T_celsius}°C → T_K={T_exterior}K → h={h_value} J/kg")
                        new_bc_data["value"] = h_value
                    elif(variable == 'p'):
                        # Let solver calculate p from p_rgh + ρ·g·h (hydrostatic consistency)
                        new_bc_data["type"] = 'calculated'
                        new_bc_data["value"] = '$internalField'
                    elif(variable == 'p_rgh'):
                        # p_rgh value for atmospheric pressure at typical aperture height
                        # This ensures p ≈ 101325 Pa at the opening, not ~0 Pa
                        new_bc_data["type"] = 'fixedValue'
                        new_bc_data["value"] = P_RGH_APERTURE  # 101325 Pa
                        logger.info(f"    BC {row['id']} ({row['type']}): p_rgh = {P_RGH_APERTURE:.1f} Pa → p ≈ {P_ATM:.0f} Pa")
                    elif(variable == 'T'):
                        # Temperature inlet: fixedValue to impose exterior temperature
                        new_bc_data["type"] = 'fixedValue'
                        new_bc_data["value"] = row['T'] + 273.15  # Convert °C → K
                        logger.info(f"    BC {row['id']} ({row['type']}): T = {row['T']}°C = {row['T'] + 273.15}K")
                    elif(variable == 'U'):
                        if (row['open']):
                            # Use pressureInletOutletVelocity for pressure-driven inflow
                            new_bc_data["type"] = 'pressureInletOutletVelocity'
                            new_bc_data["value"] = '$internalField'
                        else:
                            new_bc_data["type"] = 'fixedValue'
                            new_bc_data["value"] = np.array([0, 0, 0])
                    else:
                        raise BaseException('Unknown variable')
                elif(row['type'] == 'pressure_outlet'):
                    if(variable == 'h'):
                        # Enthalpy for perfectGas: h = Cp×T (use exterior temperature from CSV)
                        new_bc_data["type"] = 'fixedValue'
                        T_celsius = row['T']
                        T_exterior = T_celsius + 273.15  # Convert °C → K
                        h_value = CP * T_exterior
                        logger.info(f"    BC {row['id']} ({row['type']}): T={T_celsius}°C → T_K={T_exterior}K → h={h_value} J/kg")
                        new_bc_data["value"] = h_value
                    elif(variable == 'p'):
                        # Let solver calculate p from p_rgh + ρ·g·h (hydrostatic consistency)
                        new_bc_data["type"] = 'calculated'
                        new_bc_data["value"] = '$internalField'
                    elif(variable == 'p_rgh'):
                        # p_rgh value for atmospheric pressure at typical aperture height
                        # This ensures p ≈ 101325 Pa at the opening, not ~0 Pa
                        new_bc_data["type"] = 'fixedValue'
                        new_bc_data["value"] = P_RGH_APERTURE  # 101325 Pa
                        logger.info(f"    BC {row['id']} ({row['type']}): p_rgh = {P_RGH_APERTURE:.1f} Pa → p ≈ {P_ATM:.0f} Pa")
                    elif(variable == 'T'):
                        # Temperature outlet: fixedValue to impose exterior temperature
                        new_bc_data["type"] = 'fixedValue'
                        new_bc_data["value"] = row['T'] + 273.15  # Convert °C → K
                        logger.info(f"    BC {row['id']} ({row['type']}): T = {row['T']}°C = {row['T'] + 273.15}K")
                    elif(variable == 'U'):
                        if (row['open']):
                            # Use inletOutlet for adjustable mass flow conservation
                            new_bc_data["type"] = 'inletOutlet'
                            new_bc_data["inletValue"] = '$internalField'
                            new_bc_data["value"] = '$internalField'
                            new_bc_data["phi"] = 'phi'  # Required for SIMPLE to recognize as adjustable outflow
                        else:
                            new_bc_data["type"] = 'fixedValue'
                            new_bc_data["value"] = np.array([0, 0, 0])
                    else:
                        raise BaseException('Unknown variable')
                elif(row['type'] == 'mass_flow_inlet'):
                    if(variable == 'h'):
                        # Enthalpy for perfectGas: h = Cp×T
                        new_bc_data["type"] = 'fixedValue'
                        T_inlet = row['T'] + 273.15
                        new_bc_data["value"] = CP * T_inlet
                    elif(variable == 'p'):
                        # p = p_rgh + p_atm (for walls, p_rgh ≈ 0)
                        new_bc_data["type"] = 'calculated'
                        new_bc_data["value"] = P_ATM
                    elif(variable == 'p_rgh'):
                        new_bc_data["type"] = 'fixedFluxPressure'
                        new_bc_data["value"] = '$internalField'
                    elif(variable == 'T'):
                        new_bc_data["type"] = 'fixedValue'
                        new_bc_data["value"] = row['T'] + 273.15
                    elif(variable == 'U'):
                        # Use flowRateInletVelocity for mass flow inlet
                        # Convert m³/h to m³/s: massFlow (m³/h) / 3600
                        new_bc_data["type"] = 'flowRateInletVelocity'
                        new_bc_data["volumetricFlowRate"] = row['massFlow'] / 3600.0
                        new_bc_data["value"] = '$internalField'
                    else:
                        raise BaseException('Unknown variable')
                else:
                    raise BaseException('Boundary Condition Type Unknown')

                f.boundary_field[row['id']] = new_bc_data


def setup(case_path: str, simulation_type: str = 'comfortTest') -> list:
    """
    Set up HVAC CFD simulation case with boundary conditions and solver configuration.
    
    Args:
        case_path: Path to the case directory
        simulation_type: Simulation iteration type (comfortTest=3 iter, comfort30Iter=30 iter)
        
    Returns:
        List of script commands for CFD simulation
    """
    logger.info(f"    * Setting up HVAC CFD simulation case: {case_path}")
    logger.info(f"    * Simulation type: {simulation_type}")
    
    # Load boundary condition information
    geo_df_file = os.path.join(case_path, "geo", "patch_info.csv")
    logger.info(f"    * Loading boundary condition data from: {geo_df_file}")
    if(not os.path.isfile(geo_df_file)):
        logger.error("    * Boundary condition file not found")
        raise BaseException("The case has no information about boundary conditions")
    geo_df = pd.read_csv(geo_df_file)
    logger.info(f"    * Loaded {len(geo_df)} boundary condition patches")

    sim_path = os.path.join(case_path, "sim")
    logger.info(f"    * Setting up simulation directory: {sim_path}")
    
    logger.info("    * Creating initial field files")
    define_initial_files(sim_path, geo_df)

    template_path = os.path.join(os.getcwd(), "data", "settings", "cfd", "hvac")
    logger.info(f"    * Loading CFD configuration templates from: {template_path}")
    
    logger.info("    * Setting up constant files (thermophysical properties, turbulence models)")
    define_constant_files(template_path, sim_path)
    
    logger.info("    * Setting up system files (solver settings, discretization schemes)")
    define_system_files(template_path, sim_path)
    
    logger.info(f"    * Updating controlDict iterations based on simulation type: {simulation_type}")
    update_controldict_iterations(case_path, simulation_type)

    # Copy calculate_comfort.py script to case directory for PMV/PPD calculations
    logger.info("    * Copying calculate_comfort.py script to case directory")
    comfort_script_source = os.path.join(os.getcwd(), "src", "components", "post", "calculate_comfort.py")
    comfort_script_dest = os.path.join(sim_path, "calculate_comfort.py")
    shutil.copy(src=comfort_script_source, dst=comfort_script_dest)
    logger.info(f"    * Comfort script copied to: {comfort_script_dest}")

    script_commands = [
        # Copy initial conditions from 0.orig to 0
        'echo "==================== COPYING INITIAL CONDITIONS FROM 0.orig TO 0 ===================="',
        'rm -rf 0',
        'cp -r 0.orig 0',
        'echo "==================== INITIAL CONDITIONS COPIED ===================="',
        
        # Apply hydrostatic pressure distribution for physical consistency
        'echo "==================== APPLYING HYDROSTATIC PRESSURE GRADIENT ===================="',
        'runApplication setFields',
        'echo "==================== HYDROSTATIC PRESSURE INITIALIZED: p(z) = p_atm - rho*g*z ===================="',
        
        # Generate VTK for time 0 (initial fields with hydrostatic pressure) - BEFORE potentialFoam
        'echo "==================== GENERATING VTK FOR TIME 0 (INITIAL STATE) ===================="',
        'foamToVTK -time 0 -fields "(T U p p_rgh h)" 2>&1 | tee log.foamToVTK_time0',
        'echo "==================== TIME 0 VTK COMPLETED ===================="',
        
        # Decompose for parallel execution
        'rm -rf processor*',
        'runApplication decomposePar',
        
        # DEBUG: Copiar archivos de processor0 para inspección
        'echo "==================== DEBUG: Copying processor0 files ===================="',
        'mkdir -p debug_files',
        'cp processor0/0/h debug_files/processor0_h',
        'cp processor0/0/U debug_files/processor0_U',
        'cp processor0/0/p_rgh debug_files/processor0_p_rgh',
        'cp processor0/constant/thermophysicalProperties debug_files/processor0_thermo',
        'echo "==================== DEBUG FILES COPIED ===================="',
        
        # Initialize velocity and pressure fields with Laplacian solution for better stability
        'echo "==================== INITIALIZING FIELDS WITH potentialFoam ===================="',
        'runParallel -np 16 potentialFoam -initialiseUBCs -parallel',
        'echo "==================== FIELD INITIALIZATION COMPLETED ===================="',
        
        # Run solver in parallel
        'runParallel -np 16 buoyantSimpleFoam -parallel',

        # 3. Reconstruct ALL timesteps (not just latestTime) for complete iteration history
        'echo "==================== RECONSTRUCTING ALL ITERATIONS ===================="',
        'runApplication reconstructPar',  # Without -latestTime = reconstruct all
        'echo "==================== RECONSTRUCTION COMPLETED ===================="',

        # 3.5. Calculate PMV/PPD thermal comfort fields from T and U (latest timestep only)
        'echo "==================== CALCULATING PMV/PPD THERMAL COMFORT ===================="',
        'python3 ./calculate_comfort.py . 2>&1 | tee log.comfort',
        'echo "==================== PMV/PPD CALCULATION COMPLETED ===================="',

        # 4. Generate VTK files for ALL timesteps with VOLUMETRIC data
        'echo "==================== GENERATING VTK FOR ALL ITERATIONS ===================="',
        # Process all timesteps (no -latestTime flag)
        # -excludePatches: Skip internal patches to reduce file size
        # Note: Generates VTK/ directory with subdirs for each timestep
        'runApplication foamToVTK -fields "(T U p p_rgh PMV PPD)" -excludePatches "(.*_master|.*_slave)"',
        'echo "==================== VTK GENERATION COMPLETED ===================="',
        
        # Also generate lightweight surface-only VTK for quick preview
        'echo "==================== GENERATING SURFACE VTK (QUICK PREVIEW) ===================="',
        'foamToVTK -latestTime -surfaceFields -fields "(T U p p_rgh PMV PPD)" 2>&1 | tee log.foamToVTK_surface',
        'echo "==================== SURFACE VTK COMPLETED ===================="',

        # Clean processors
        'rm -rf processor*',
        ]
    
    logger.info("    * HVAC CFD case setup completed successfully")
    return script_commands


if __name__ == "__main__":
    case_folder = "case"
    result = setup(case_folder)