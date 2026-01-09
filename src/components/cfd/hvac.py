import os
import shutil
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from foamlib import FoamCase, FoamFile
from src.components.tools.cpu_cores_partitions import best_cpu_partition
from src.components.tools.populate_template_file import replace_in_file


logger = logging.getLogger(__name__)

# Calculate project root: 4 levels up from this file
# src/components/cfd/hvac.py -> src/components/cfd/ -> src/components/ -> src/ -> FLOWDESK_OF/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent



# CONSTANTS FOR THERMOPHYSICS
CP = 1005.0  # Specific heat at constant pressure [J/(kg·K)]
HF = 0.0  # Formation enthalpy [J/kg]
# Note: Using Boussinesq (incompressiblePerfectGas + hConst): h = Cp×T, ρ = pRef/(R×T)

# DIMENSIONS - FIELDS FOR buoyantSimpleFoam (LAMINAR) AND buoyantPimpleFoam (TURBULENT)
DIMENSIONS_DICT = {
    # Base fields (both solvers)
    'h':        FoamFile.DimensionSet(length=2, time=-2),  # J/kg = m²/s² - Primary thermodynamic variable
    'p':        FoamFile.DimensionSet(mass=1, length=-1, time=-2),  # Pressure [Pa]
    'p_rgh':    FoamFile.DimensionSet(mass=1, length=-1, time=-2),  # Buoyancy-corrected pressure [Pa]
    'T':        FoamFile.DimensionSet(temperature=1),  # Temperature [K]
    'U':        FoamFile.DimensionSet(length=1, time=-1),  # Velocity [m/s]
    
    # Turbulence fields (for buoyantPimpleFoam with kOmegaSST)
    'k':        FoamFile.DimensionSet(length=2, time=-2),  # Turbulent kinetic energy [m²/s²]
    'omega':    FoamFile.DimensionSet(time=-1),  # Specific dissipation rate [1/s]
    'alphat':   FoamFile.DimensionSet(mass=1, length=-1, time=-1),  # Turbulent thermal diffusivity [kg/(m·s)]
    'nut':      FoamFile.DimensionSet(length=2, time=-1),  # Turbulent kinematic viscosity [m²/s]
    
    # Scalar transport (CO2)
    'CO2':      FoamFile.DimensionSet(),  # Dimensionless concentration (or kg/m³ if dimensional)
    
    # Radiation fields (for P1/fvDOM model)
    'qr':       FoamFile.DimensionSet(mass=1, time=-3),  # Radiative heat flux [W/m²]
    'G':        FoamFile.DimensionSet(mass=1, time=-3),  # Incident radiation [W/m²]
}

INTERNALFIELD_DICT = {
    # Base fields
    'h':        294515.75,  # h = Cp×T = 1005×293.15 for Boussinesq (20°C)
    'p':        101325,     # Atmospheric pressure [Pa] (will be modified by setFields for hydrostatic gradient)
    'p_rgh':    101325,     # Modified pressure (constant in hydrostatic equilibrium)
    'T':        293.15,     # Reference temperature for Boussinesq [K] (20°C)
    'U':        np.array([0, 0, 0]),  # Initial velocity (quiescent fluid)
    
    # Turbulence fields (default values from OpenFOAM BC Guide)
    # k ≈ 0.01 m²/s² (typical low turbulence)
    # ω ≈ 0.5 1/s (typical for building ventilation)
    # nut = k/ω = 0.01/0.5 = 0.02 m²/s
    # alphat = ρ·nut/Prt = 1.2 × 0.02 / 0.85 ≈ 0.028 kg/(m·s)
    'k':        0.01,       # Turbulent kinetic energy [m²/s²]
    'omega':    0.5,        # Specific dissipation rate [1/s]
    'alphat':   0.028,      # Turbulent thermal diffusivity [kg/(m·s)] (alphat = ρ·nut/Prt)
    'nut':      0.02,       # Turbulent kinematic viscosity [m²/s] (nut = k/ω)
    
    # Scalar transport
    'CO2':      400e-6,     # 400 ppm CO2 (exterior ambient)
    
    # Radiation
    'qr':       0,          # Initial radiative flux (computed by solver)
    'G':        0,          # Initial incident radiation (computed by solver)
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


def define_boundary_radiation_properties(sim_path, patch_df):
    """
    Create boundaryRadiationProperties file for radiation boundary conditions.
    
    This file is required when using MarshakRadiation BC with emissivityMode="lookup".
    Uses a wildcard pattern ".*" to apply default radiation properties to all patches.
    
    TODO (FUTURE): Emissivity/absorptivity values should come from JSON input file
    for patch-specific material properties.
    
    Args:
        sim_path: Path to simulation directory
        patch_df: DataFrame with patch information (id, type) - used for logging only
    """
    logger.info("    * Creating boundaryRadiationProperties file for radiation model")
    
    # Path to output file
    constant_path = os.path.join(sim_path, 'constant')
    output_file = os.path.join(constant_path, 'boundaryRadiationProperties')
    
    # Default emissivity/absorptivity for typical building materials
    # TODO (FUTURE): Make this configurable per-patch from JSON
    default_emissivity = 0.9  # Typical for painted surfaces, concrete, plaster
    default_absorptivity = 0.9  # For grey body assumption: absorptivity = emissivity
    
    # Build content string with OpenFOAM format
    content_lines = []
    
    # Add FoamFile header
    content_lines.append("/*--------------------------------*- C++ -*----------------------------------*\\")
    content_lines.append("| =========                 |                                                 |")
    content_lines.append("| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |")
    content_lines.append("|  \\\\    /   O peration     | Version:  v2412                                 |")
    content_lines.append("|   \\\\  /    A nd           | Website:  www.openfoam.com                      |")
    content_lines.append("|    \\\\/     M anipulation  |                                                 |")
    content_lines.append("\\*---------------------------------------------------------------------------*/")
    content_lines.append("FoamFile")
    content_lines.append("{")
    content_lines.append("    version     2.0;")
    content_lines.append("    format      ascii;")
    content_lines.append("    class       dictionary;")
    content_lines.append("    object      boundaryRadiationProperties;")
    content_lines.append("}")
    content_lines.append("// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //")
    content_lines.append("")
    content_lines.append("// Default radiation properties for all boundary patches")
    content_lines.append("// Using wildcard pattern \".*\" to apply to all patches")
    content_lines.append("// TODO (FUTURE): Define patch-specific properties from JSON input")
    content_lines.append("")
    content_lines.append('".*"')
    content_lines.append("{")
    content_lines.append("    type            opaqueDiffusive;")
    content_lines.append("    wallAbsorptionEmissionModel")
    content_lines.append("    {")
    content_lines.append("        type            constantAbsorption;")
    content_lines.append(f"        absorptivity    {default_absorptivity};")
    content_lines.append(f"        emissivity      {default_emissivity};")
    content_lines.append("    }")
    content_lines.append("}")
    content_lines.append("")
    content_lines.append("// ************************************************************************* //")
    
    # Write file
    with open(output_file, 'w') as f:
        f.write('\n'.join(content_lines))
    
    logger.info(f"    * boundaryRadiationProperties created with wildcard pattern")
    logger.info(f"    * Default emissivity/absorptivity: {default_emissivity}")
    logger.info(f"    * Applies to {len(patch_df)} patches")
    logger.info(f"    * File written to: {output_file}")


def create_decomposeParDict_local(sim_path):
    """
    Create decomposeParDict for local parallel execution using available CPU cores - 1.
    
    Args:
        sim_path: Path to simulation directory
    """
    # Detect available cores (use all available cores - 1 to leave one for system)
    import os as os_module
    total_cores = os_module.cpu_count()
    if total_cores is None:
        total_cores = 16  # Fallback if detection fails
    
    n_cores = max(total_cores - 1, 1)  # At least 1 core
    logger.info(f"    * Detected {total_cores} CPU cores, using {n_cores} for parallel simulation")
    
    # Calculate best partition for the number of cores
    n_cpu_available, (n_x, n_y, n_z) = best_cpu_partition(n_cores)
    logger.info(f"    * Optimal partition: {n_cpu_available} cores = ({n_x}, {n_y}, {n_z})")
    
    # Use template from inductiva settings (use PROJECT_ROOT for robust path resolution)
    template_path = str(PROJECT_ROOT / "data" / "settings" / "solve" / "inductiva")
    input_path = os.path.join(template_path, "system", "decomposeParDict")
    output_path = os.path.join(sim_path, "system", "decomposeParDict")
    
    # Replace placeholders with actual values
    str_replace_dict = {
        "$NUM_CPUS": str(n_cpu_available),
        "$PARTITION_X": str(n_x),
        "$PARTITION_Y": str(n_y),
        "$PARTITION_Z": str(n_z)
    }
    
    replace_in_file(input_path, output_path, str_replace_dict)
    logger.info(f"    * decomposeParDict created: {n_cpu_available} subdomains")


def define_turbulence_bcs(variable, patch_type, patch_row):
    """
    Define boundary conditions for turbulence fields (k, omega, alphat).
    
    Args:
        variable: 'k', 'omega', or 'alphat'
        patch_type: Type of patch (wall, velocity_inlet, pressure_inlet, etc.)
        patch_row: Row from patch_df with boundary info
        
    Returns:
        dict: Boundary condition data
    """
    bc = {}
    
    if variable == 'k':
        if patch_type == 'wall':
            # Wall function for turbulent kinetic energy
            bc["type"] = "kqRWallFunction"
            bc["value"] = INTERNALFIELD_DICT['k']
        elif patch_type in ['velocity_inlet', 'mass_flow_inlet']:
            # Turbulent intensity inlet
            bc["type"] = "turbulentIntensityKineticEnergyInlet"
            bc["intensity"] = 0.05  # 5% turbulence intensity
            bc["value"] = INTERNALFIELD_DICT['k']
        elif patch_type in ['pressure_inlet', 'pressure_outlet']:
            # Inlet/outlet for k
            bc["type"] = "inletOutlet"
            bc["inletValue"] = INTERNALFIELD_DICT['k']
            bc["value"] = INTERNALFIELD_DICT['k']
        else:
            bc["type"] = "zeroGradient"
            
    elif variable == 'omega':
        if patch_type == 'wall':
            # Wall function for specific dissipation rate
            bc["type"] = "omegaWallFunction"
            bc["value"] = INTERNALFIELD_DICT['omega']
        elif patch_type in ['velocity_inlet', 'mass_flow_inlet']:
            # Mixing length frequency inlet
            bc["type"] = "turbulentMixingLengthFrequencyInlet"
            bc["mixingLength"] = 0.1  # 0.1m mixing length
            bc["value"] = INTERNALFIELD_DICT['omega']
        elif patch_type in ['pressure_inlet', 'pressure_outlet']:
            # Inlet/outlet for omega
            bc["type"] = "inletOutlet"
            bc["inletValue"] = INTERNALFIELD_DICT['omega']
            bc["value"] = INTERNALFIELD_DICT['omega']
        else:
            bc["type"] = "zeroGradient"
    
    elif variable == 'alphat':
        if patch_type == 'wall':
            # Wall function for turbulent thermal diffusivity
            bc["type"] = "compressible::alphatWallFunction"
            bc["value"] = INTERNALFIELD_DICT['alphat']
        else:
            # Calculated for all other patches (inlets, outlets)
            bc["type"] = "calculated"
            bc["value"] = INTERNALFIELD_DICT['alphat']
    
    elif variable == 'nut':
        if patch_type == 'wall':
            # Wall function for turbulent kinematic viscosity (compressible version)
            bc["type"] = "nutkWallFunction"
            bc["value"] = INTERNALFIELD_DICT['nut']
        else:
            # Calculated for all other patches (inlets, outlets)
            bc["type"] = "calculated"
            bc["value"] = INTERNALFIELD_DICT['nut']
    
    return bc


def define_radiation_bcs(variable, patch_type):
    """
    Define boundary conditions for radiation fields (qr, G).
    
    Args:
        variable: 'qr' or 'G'
        patch_type: Type of patch
        
    Returns:
        dict: Boundary condition data
    """
    bc = {}
    
    if variable == 'G':
        if patch_type == 'wall':
            # Marshak radiation boundary condition for walls
            bc["type"] = "MarshakRadiation"
            bc["emissivityMode"] = "lookup"
            bc["emissivity"] = 0.9  # Default wall emissivity
            bc["value"] = 0
        else:
            # Zero gradient for inlets/outlets
            bc["type"] = "zeroGradient"
    
    elif variable == 'qr':
        if patch_type == 'wall':
            # Calculated from G field and wall temperature
            bc["type"] = "calculated"
            bc["value"] = 0
        else:
            # Zero gradient for inlets/outlets
            bc["type"] = "zeroGradient"
    
    return bc


def define_scalar_bcs(variable, patch_type, patch_row):
    """
    Define boundary conditions for scalar transport (CO2).
    
    Args:
        variable: Scalar name (e.g., 'CO2')
        patch_type: Type of patch
        patch_row: Row from patch_df with boundary info
        
    Returns:
        dict: Boundary condition data
    """
    bc = {}
    
    if variable == 'CO2':
        if patch_type in ['velocity_inlet', 'mass_flow_inlet']:
            # Fixed CO2 concentration at inlet (ambient 400 ppm)
            bc["type"] = "fixedValue"
            bc["value"] = 400e-6  # 400 ppm
        elif patch_type == 'pressure_inlet':
            # Bidirectional CO2: inletOutlet for backflow scenarios
            bc["type"] = "inletOutlet"
            bc["inletValue"] = 400e-6  # Exterior air: 400 ppm
            bc["value"] = INTERNALFIELD_DICT['CO2']
        elif patch_type == 'pressure_outlet':
            # Outlet: allow CO2 to leave, prevent backflow contamination
            bc["type"] = "inletOutlet"
            bc["inletValue"] = 400e-6
            bc["value"] = INTERNALFIELD_DICT['CO2']
        elif patch_type == 'wall':
            # Walls: no CO2 flux (impermeable)
            bc["type"] = "zeroGradient"
        else:
            bc["type"] = "zeroGradient"
    
    return bc


def update_controldict_iterations(case_path, simulation_type, transient=False):
    """
    Update controlDict endTime, writeInterval, and purgeWrite based on simulation type.
    
    For STEADY (buoyantSimpleFoam):
    - comfortTest: 3 iterations, write every iteration, keep all
    - comfort30Iter: 500 iterations, write only last iteration, keep only last
    
    For TRANSIENT (buoyantPimpleFoam):
    - Uses writeControl=timeStep (counts timesteps, NOT physical time)
    - comfortTest: 20 timesteps, keep all
    - productionRun: 500 timesteps (~5s @ dt=0.01), keep last
    
    Args:
        case_path: Path to case directory
        simulation_type: Type of simulation (comfortTest, comfort30Iter, productionRun, test_calculation)
        transient: If True, use transient config (timesteps). If False, use steady config (iterations)
    """
    logger.info(f"    * Updating controlDict for simulation type: {simulation_type} (transient={transient})")
    
    if transient:
        # TRANSIENT: writeControl=timeStep, count timesteps
        config_map = {
            'comfortTest': (1000, 20, 0),      # endTime high, 20 timesteps, keep all
            'productionRun': (1000, 500, 1),   # endTime high, 500 timesteps (~5s @ dt=0.01), keep last
            'test_calculation': (1000, 20, 0)  # Same as comfortTest
        }
        write_control = 'timeStep'
    else:
        # STEADY: writeControl=timeStep, count iterations
        config_map = {
            'comfortTest': (3, 1, 0),      # 3 iter, write every iteration, keep all
            'comfort30Iter': (500, 1, 1),  # 500 iter, write every iteration, keep only last
            'test_calculation': (3, 1, 0)   # 3 iter, write every iteration, keep all
        }
        write_control = 'timeStep'
    
    iterations, write_interval, purge_write = config_map.get(simulation_type, (3, 1, 0) if not transient else (1000, 20, 0))
    logger.info(f"    * Setting endTime={iterations}, writeInterval={write_interval}, purgeWrite={purge_write}, writeControl={write_control}")
    
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
                        new_bc_data["value"] = INTERNALFIELD_DICT[variable]
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
                patch_type = row['type']
                new_bc_data = {}  # Initialize dict to avoid UnboundLocalError
                
                # Handle turbulence fields (k, omega, alphat, nut) with dedicated function
                if variable in ['k', 'omega', 'alphat', 'nut']:
                    new_bc_data = define_turbulence_bcs(variable, patch_type, row)
                # Handle radiation fields (qr, G) with dedicated function
                elif variable in ['qr', 'G']:
                    new_bc_data = define_radiation_bcs(variable, patch_type)
                # Handle scalar transport (CO2) with dedicated function
                elif variable == 'CO2':
                    new_bc_data = define_scalar_bcs(variable, patch_type, row)
                # Handle base fields (h, p, p_rgh, T, U) with original logic
                elif patch_type == 'wall':
                    if(variable == 'h'):
                        # Calculated from temperature BC (h = Cp×T)
                        new_bc_data["type"] = 'calculated'
                        new_bc_data["value"] = INTERNALFIELD_DICT['h']
                    elif(variable == 'p'):
                        # p = p_rgh + p_atm (for walls, p_rgh ≈ 0)
                        new_bc_data["type"] = 'calculated'
                        new_bc_data["value"] = P_ATM
                    elif(variable == 'p_rgh'):
                        new_bc_data["type"] = 'fixedFluxPressure'
                        new_bc_data["value"] = INTERNALFIELD_DICT[variable]
                    elif(variable == 'T'):
                        new_bc_data["type"] = 'fixedValue'
                        new_bc_data["value"] = row['T'] + 273.15
                    elif(variable == 'U'):
                        new_bc_data["type"] = 'noSlip'
                    else:
                        raise BaseException('Unknown variable')
                elif(row['type'] == 'velocity_inlet'):
                    if(variable == 'h'):
                        # Calculated from temperature BC (h = Cp×T)
                        new_bc_data["type"] = 'calculated'
                        new_bc_data["value"] = INTERNALFIELD_DICT['h']
                    elif(variable == 'p'):
                        # p = p_rgh + p_atm (for walls, p_rgh ≈ 0)
                        new_bc_data["type"] = 'calculated'
                        new_bc_data["value"] = P_ATM
                    elif(variable == 'p_rgh'):
                        new_bc_data["type"] = 'fixedFluxPressure'
                        new_bc_data["value"] = INTERNALFIELD_DICT[variable]
                    elif(variable == 'T'):
                        new_bc_data["type"] = 'fixedValue'
                        new_bc_data["value"] = row['T'] + 273.15
                    elif(variable == 'U'):
                        if (row['open']):
                            new_bc_data["type"] = 'fixedValue'
                            new_bc_data["value"] = row['U'] * np.array([row['nx'], row['ny'], row['nz']])
                        else:
                            # Closed velocity_inlet behaves as wall with no-slip condition
                            new_bc_data["type"] = 'noSlip'
                    else:
                        raise BaseException('Unknown variable')
                elif(row['type'] == 'pressure_inlet'):
                    if(variable == 'h'):
                        # Calculated from temperature BC (h = Cp×T)
                        new_bc_data["type"] = 'calculated'
                        new_bc_data["value"] = INTERNALFIELD_DICT['h']
                    elif(variable == 'p'):
                        # Initialization identical to p_rgh (calculated from pressure differential)
                        new_bc_data["type"] = 'calculated'
                        new_bc_data["value"] = P_RGH_APERTURE + row['pressure']
                    elif(variable == 'p_rgh'):
                        # p_rgh value with user-defined pressure differential
                        # Apply ΔP from flowIntensity: p_rgh = p_atm + ΔP
                        new_bc_data["type"] = 'fixedValue'
                        p_rgh_value = P_RGH_APERTURE + row['pressure']  # 101325 + ΔP
                        new_bc_data["value"] = p_rgh_value
                        logger.info(f"    BC {row['id']} ({row['type']}): p_rgh = {p_rgh_value:.1f} Pa (ΔP = {row['pressure']:.1f} Pa)")
                    elif(variable == 'T'):
                        # Bidirectional temperature: inletOutlet for backflow scenarios
                        new_bc_data["type"] = 'inletOutlet'
                        T_exterior = row['T'] + 273.15  # Convert °C → K
                        new_bc_data["inletValue"] = T_exterior
                        new_bc_data["value"] = T_exterior
                        logger.info(f"    BC {row['id']} ({row['type']}): T = {row['T']}°C = {T_exterior}K (inletOutlet)")
                    elif(variable == 'U'):
                        if (row['open']):
                            # Use pressureInletOutletVelocity for pressure-driven inflow
                            new_bc_data["type"] = 'pressureInletOutletVelocity'
                            new_bc_data["value"] = INTERNALFIELD_DICT[variable]
                        else:
                            # Closed pressure_inlet behaves as wall with no-slip condition
                            new_bc_data["type"] = 'noSlip'
                    else:
                        raise BaseException('Unknown variable')
                elif(row['type'] == 'pressure_outlet'):
                    if(variable == 'h'):
                        # Calculated from temperature BC (h = Cp×T)
                        new_bc_data["type"] = 'calculated'
                        new_bc_data["value"] = INTERNALFIELD_DICT['h']
                    elif(variable == 'p'):
                        # Initialization identical to p_rgh (calculated from pressure differential)
                        new_bc_data["type"] = 'calculated'
                        new_bc_data["value"] = P_RGH_APERTURE + row['pressure']
                    elif(variable == 'p_rgh'):
                        # p_rgh value with user-defined pressure differential
                        # Apply ΔP from flowIntensity: p_rgh = p_atm + ΔP
                        new_bc_data["type"] = 'fixedValue'
                        p_rgh_value = P_RGH_APERTURE + row['pressure']  # 101325 + ΔP (negative for outlet)
                        new_bc_data["value"] = p_rgh_value
                        logger.info(f"    BC {row['id']} ({row['type']}): p_rgh = {p_rgh_value:.1f} Pa (ΔP = {row['pressure']:.1f} Pa)")
                    elif(variable == 'T'):
                        # Bidirectional temperature: inletOutlet for backflow scenarios
                        new_bc_data["type"] = 'inletOutlet'
                        T_exterior = row['T'] + 273.15  # Convert °C → K
                        new_bc_data["inletValue"] = T_exterior
                        new_bc_data["value"] = T_exterior
                        logger.info(f"    BC {row['id']} ({row['type']}): T = {row['T']}°C = {T_exterior}K (inletOutlet)")
                    elif(variable == 'U'):
                        if (row['open']):
                            # Use pressureInletOutletVelocity for bidirectional pressure-driven flow
                            new_bc_data["type"] = 'pressureInletOutletVelocity'
                            new_bc_data["value"] = INTERNALFIELD_DICT[variable]
                        else:
                            # Closed pressure_outlet behaves as wall with no-slip condition
                            new_bc_data["type"] = 'noSlip'
                    else:
                        raise BaseException('Unknown variable')
                elif(row['type'] == 'mass_flow_inlet'):
                    if(variable == 'h'):
                        # Calculated from temperature BC (h = Cp×T)
                        new_bc_data["type"] = 'calculated'
                        new_bc_data["value"] = INTERNALFIELD_DICT['h']
                    elif(variable == 'p'):
                        # p = p_rgh + p_atm (for walls, p_rgh ≈ 0)
                        new_bc_data["type"] = 'calculated'
                        new_bc_data["value"] = P_ATM
                    elif(variable == 'p_rgh'):
                        new_bc_data["type"] = 'fixedFluxPressure'
                        new_bc_data["value"] = INTERNALFIELD_DICT[variable]
                    elif(variable == 'T'):
                        new_bc_data["type"] = 'fixedValue'
                        new_bc_data["value"] = row['T'] + 273.15
                    elif(variable == 'U'):
                        # Use flowRateInletVelocity for mass flow inlet
                        # Convert m³/h to m³/s: massFlow (m³/h) / 3600
                        new_bc_data["type"] = 'flowRateInletVelocity'
                        new_bc_data["volumetricFlowRate"] = row['massFlow'] / 3600.0
                        new_bc_data["value"] = INTERNALFIELD_DICT[variable]
                    else:
                        raise BaseException('Unknown variable')
                else:
                    raise BaseException('Boundary Condition Type Unknown')

                f.boundary_field[row['id']] = new_bc_data


def setup(case_path: str, simulation_type: str = 'comfortTest', transient: bool = False) -> list:
    """
    Set up HVAC CFD simulation case with boundary conditions and solver configuration.
    
    Args:
        case_path: Path to the case directory
        simulation_type: Simulation iteration type (comfortTest=3 iter, comfort30Iter=30 iter)
        transient: If True, use buoyantPimpleFoam (transient RANS turbulent with radiation/scalars)
                   If False, use buoyantSimpleFoam (steady-state laminar, default)
        
    Returns:
        List of script commands for CFD simulation
    """
    solver_type = "buoyantPimpleFoam (transient)" if transient else "buoyantSimpleFoam (steady)"
    logger.info(f"    * Setting up HVAC CFD simulation case: {case_path}")
    logger.info(f"    * Solver: {solver_type}")
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

    # Select template based on transient flag (use PROJECT_ROOT for robust path resolution)
    if transient:
        template_path = str(PROJECT_ROOT / "data" / "settings" / "cfd" / "hvac_transient")
        logger.info(f"    * Using TRANSIENT templates (buoyantPimpleFoam + kOmegaSST + radiationP1 + CO2)")
    else:
        template_path = str(PROJECT_ROOT / "data" / "settings" / "cfd" / "hvac")
        logger.info(f"    * Using STEADY templates (buoyantSimpleFoam + laminar)")
    
    logger.info(f"    * Loading CFD configuration templates from: {template_path}")
    
    logger.info("    * Setting up constant files (thermophysical properties, turbulence models)")
    define_constant_files(template_path, sim_path)
    
    # Create boundaryRadiationProperties for radiation model (transient only)
    if transient:
        define_boundary_radiation_properties(sim_path, geo_df)
    
    logger.info("    * Setting up system files (solver settings, discretization schemes)")
    define_system_files(template_path, sim_path)
    
    logger.info("    * Creating decomposeParDict for local parallel execution")
    # Detect CPU cores for parallel execution
    import os as os_module
    total_cores = os_module.cpu_count()
    if total_cores is None:
        total_cores = 16
    n_cores = max(total_cores - 1, 1)
    n_cpu_available, _ = best_cpu_partition(n_cores)
    
    create_decomposeParDict_local(sim_path)
    
    logger.info(f"    * Updating controlDict iterations based on simulation type: {simulation_type}")
    update_controldict_iterations(case_path, simulation_type, transient=transient)

    # Copy calculate_comfort.py script to case directory for PMV/PPD calculations (use PROJECT_ROOT)
    logger.info("    * Copying calculate_comfort.py script to case directory")
    comfort_script_source = str(PROJECT_ROOT / "src" / "components" / "post" / "calculate_comfort.py")
    comfort_script_dest = os.path.join(sim_path, "calculate_comfort.py")
    shutil.copy(src=comfort_script_source, dst=comfort_script_dest)
    logger.info(f"    * Comfort script copied to: {comfort_script_dest}")

    # Build script commands dynamically based on solver type
    script_commands = [
        # Copy initial conditions from 0.orig to 0
        'echo "==================== COPYING INITIAL CONDITIONS FROM 0.orig TO 0 ===================="',
        'rm -rf 0',
        'cp -r 0.orig 0',
        'echo "==================== INITIAL CONDITIONS COPIED ===================="',
        
        # Apply hydrostatic pressure distribution for physical consistency
        # APPLIED TO BOTH STEADY AND TRANSIENT: Ensures p(z) = p_atm - rho*g*z from start
        'echo "==================== APPLYING HYDROSTATIC PRESSURE GRADIENT ===================="',
        'runApplication setFields',
        'echo "==================== HYDROSTATIC PRESSURE INITIALIZED: p(z) = p_atm - rho*g*z ===================="',
    ]
    
    # Continue with common steps
    script_commands.extend([
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
        
        # potentialFoam REMOVED: Incompatible with buoyant solvers (variable density)
        # - potentialFoam assumes constant density (incompressible)
        # - buoyantPimpleFoam uses variable density ρ(T) with thermal buoyancy
        # - Starting from U=0 is physically correct and solver develops flow naturally
        # Solution: buoyantPimpleFoam will develop velocity field from rest in 2-3 timesteps
    ])
    
    # Run solver: TRANSIENT uses 3-phase progressive CFL, STEADY uses single execution
    if transient:
        # ========== TRANSIENT: 3-PHASE PROGRESSIVE CFL STRATEGY ==========
        # Phase 1: Conservative start-up (timesteps 0-100, CFL=0.1)
        # Phase 2: Transition ramp-up (timesteps 100-400, CFL=0.5)
        # Phase 3: Normal operation (timesteps 400+, CFL=0.8)
        
        script_commands.extend([
            'echo "=========================================================================="',
            'echo "TRANSIENT SOLVER: 3-PHASE PROGRESSIVE CFL STRATEGY"',
            'echo "Phase 1: Timesteps 0-100 (CFL=0.1) - Conservative start-up"',
            'echo "Phase 2: Timesteps 100-400 (CFL=0.5) - Transition ramp-up"',
            'echo "Phase 3: Timesteps 400+ (CFL=0.8) - Normal operation"',
            'echo "=========================================================================="',
            '',
            # PHASE 1: Conservative start-up
            'echo "==================== PHASE 1: CONSERVATIVE START-UP ===================="',
            'echo "Configuring: maxCo=0.1, maxAlphaCo=0.1, endTime=100"',
            'foamDictionary system/controlDict -entry maxCo -set 0.1',
            'foamDictionary system/controlDict -entry maxAlphaCo -set 0.1',
            'foamDictionary system/controlDict -entry writeControl -set timeStep',
            'foamDictionary system/controlDict -entry writeInterval -set 20',
            'foamDictionary system/controlDict -entry endTime -set 100',
            'echo "Starting buoyantPimpleFoam Phase 1..."',
            f'runParallel -np {n_cpu_available} buoyantPimpleFoam -parallel 2>&1 | tee -a log.buoyantPimpleFoam',
            'echo "==================== PHASE 1 COMPLETED ===================="',
            '',
            # PHASE 2: Transition ramp-up
            'echo "==================== PHASE 2: TRANSITION RAMP-UP ===================="',
            'echo "Configuring: maxCo=0.5, maxAlphaCo=0.5, timesteps 100-400"',
            'foamDictionary system/controlDict -entry maxCo -set 0.5',
            'foamDictionary system/controlDict -entry maxAlphaCo -set 0.5',
            'foamDictionary system/controlDict -entry startTime -set latestTime',
            'foamDictionary system/controlDict -entry endTime -set 400',
            'foamDictionary system/controlDict -entry writeInterval -set 50',
            'echo "Continuing buoyantPimpleFoam Phase 2 from latest time..."',
            f'runParallel -np {n_cpu_available} buoyantPimpleFoam -parallel 2>&1 | tee -a log.buoyantPimpleFoam',
            'echo "==================== PHASE 2 COMPLETED ===================="',
            '',
            # PHASE 3: Normal operation
            'echo "==================== PHASE 3: NORMAL OPERATION ===================="',
            'echo "Configuring: maxCo=0.8, maxAlphaCo=0.8, timesteps 400-1000"',
            'foamDictionary system/controlDict -entry maxCo -set 0.8',
            'foamDictionary system/controlDict -entry maxAlphaCo -set 0.8',
            'foamDictionary system/controlDict -entry startTime -set latestTime',
            'foamDictionary system/controlDict -entry endTime -set 1000',
            'foamDictionary system/controlDict -entry writeInterval -set 100',
            'echo "Continuing buoyantPimpleFoam Phase 3 from latest time..."',
            f'runParallel -np {n_cpu_available} buoyantPimpleFoam -parallel 2>&1 | tee -a log.buoyantPimpleFoam',
            'echo "==================== PHASE 3 COMPLETED ===================="',
            'echo "=========================================================================="',
            'echo "ALL 3 PHASES COMPLETED SUCCESSFULLY"',
            'echo "=========================================================================="',
        ])
    else:
        # ========== STEADY: Single execution ==========
        script_commands.extend([
            'echo "==================== RUNNING STEADY-STATE SOLVER ===================="',
            f'runParallel -np {n_cpu_available} buoyantSimpleFoam -parallel',
            'echo "==================== STEADY SOLVER COMPLETED ===================="',
        ])
    
    # Continue with common post-processing steps
    script_commands.extend([

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
    ])
    
    logger.info("    * HVAC CFD case setup completed successfully")
    return script_commands


if __name__ == "__main__":
    case_folder = "case"
    result = setup(case_folder)
