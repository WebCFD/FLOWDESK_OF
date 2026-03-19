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

# DIMENSIONS - FIELDS FOR buoyantBoussinesqPimpleFoam (INCOMPRESSIBLE)
DIMENSIONS_DICT = {
    # Base fields (incompressible solver)
    # 'h':        FoamFile.DimensionSet(length=2, time=-2),  # J/kg = m²/s² - Primary thermodynamic variable [COMMENTED FOR TEST]
    'p':        FoamFile.DimensionSet(length=2, time=-2),  # Kinematic pressure [m²/s²] - INCOMPRESSIBLE
    'p_rgh':    FoamFile.DimensionSet(length=2, time=-2),  # Kinematic pressure [m²/s²] - INCOMPRESSIBLE
    'T':        FoamFile.DimensionSet(temperature=1),  # Temperature [K]
    'U':        FoamFile.DimensionSet(length=1, time=-1),  # Velocity [m/s]
    
    # Turbulence fields (for buoyantPimpleFoam with kOmegaSST)
    'k':        FoamFile.DimensionSet(length=2, time=-2),  # Turbulent kinetic energy [m²/s²]
    'omega':    FoamFile.DimensionSet(time=-1),  # Specific dissipation rate [1/s]
    'alphat':   FoamFile.DimensionSet(length=2, time=-1),  # Turbulent thermal diffusivity KINEMATIC [m²/s] for incompressible
    'nut':      FoamFile.DimensionSet(length=2, time=-1),  # Turbulent kinematic viscosity [m²/s]
    
    # Scalar transport (CO2)
    'CO2':      FoamFile.DimensionSet(),  # Dimensionless concentration (or kg/m³ if dimensional)
    
    # Radiation fields (for P1/fvDOM model)
    'qr':       FoamFile.DimensionSet(mass=1, time=-3),  # Radiative heat flux [W/m²]
    'G':        FoamFile.DimensionSet(mass=1, time=-3),  # Incident radiation [W/m²]
}

INTERNALFIELD_DICT = {
    # Base fields (INCOMPRESSIBLE - kinematic pressure units)
    # 'h':        294515.75,  # h = Cp×T = 1005×293.15 for Boussinesq (20°C) [COMMENTED FOR TEST]
    'p':        0,          # Gauge pressure [m²/s²] - reference level (p = p_rgh + g·h)
    'p_rgh':    0,          # Dynamic pressure [m²/s²] - hydrostatic component excluded
    'T':        297.15,     # Reference temperature for Boussinesq [K] (24°C)
    'U':        np.array([0, 0, 0]),  # Initial velocity (quiescent fluid - Boussinesq handles stability)
    
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
    'CO2':      1.0,        # 100% CO2 initial concentration (ventilation test scenario)
    
    # Radiation
    'qr':       0,          # Initial radiative flux (computed by solver)
    'G':        450,        # Initial incident radiation [W/m²] - realistic for 25-30°C environment
}

# Reference values for pressure calculations
P_ATM = 101325  # Atmospheric pressure [Pa] - for reference only
RHO_REF = 1.2   # Reference air density at 20°C [kg/m³]
G = 9.81        # Gravitational acceleration [m/s²]
# For Boussinesq approximation: use GAUGE pressure (p_rgh relative to atmosphere)
# p_rgh = 0 → atmospheric pressure (reference)
# p_rgh > 0 → above atmospheric (pressure inlet)
# p_rgh < 0 → below atmospheric (pressure outlet)
P_RGH_APERTURE = 0  # Gauge pressure reference: 0 = atmospheric pressure


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
    Create boundaryRadiationProperties with per-patch emissivity from patch_df.

    Generates:
    - Wildcard ".*" entry (default 0.9) as fallback for open boundaries
    - Explicit per-wall-patch entry with emissivity from JSON simulationProperties

    Args:
        sim_path: Path to simulation directory
        patch_df: DataFrame with columns: id, type, emissivity (from JSON)
    """
    logger.info("    * Creating per-patch boundaryRadiationProperties")

    constant_path = os.path.join(sim_path, 'constant')
    output_file = os.path.join(constant_path, 'boundaryRadiationProperties')

    content_lines = []

    # FoamFile header
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
    content_lines.append("// Wildcard fallback – covers open boundaries (zeroGradient G) harmlessly")
    content_lines.append('".*"')
    content_lines.append("{")
    content_lines.append("    type            opaqueDiffusive;")
    content_lines.append("    wallAbsorptionEmissionModel")
    content_lines.append("    {")
    content_lines.append("        type            constantAbsorption;")
    content_lines.append("        absorptivity    0.9;")
    content_lines.append("        emissivity      0.9;")
    content_lines.append("    }")
    content_lines.append("}")
    content_lines.append("")

    # Per-patch entries for wall patches (override wildcard with JSON-defined emissivity)
    wall_patches = patch_df[patch_df['type'] == 'wall'] if 'type' in patch_df.columns else patch_df
    patches_written = 0
    for _, row in wall_patches.iterrows():
        patch_id  = row['id']
        emissivity = float(row['emissivity']) if 'emissivity' in row and row['emissivity'] == row['emissivity'] else 0.9
        content_lines.append(f'"{patch_id}"')
        content_lines.append("{")
        content_lines.append("    type            opaqueDiffusive;")
        content_lines.append("    wallAbsorptionEmissionModel")
        content_lines.append("    {")
        content_lines.append("        type            constantAbsorption;")
        content_lines.append(f"        absorptivity    {emissivity};")
        content_lines.append(f"        emissivity      {emissivity};")
        content_lines.append("    }")
        content_lines.append("}")
        content_lines.append("")
        patches_written += 1

    content_lines.append("// ************************************************************************* //")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(content_lines))

    logger.info(f"    * boundaryRadiationProperties written: {patches_written} wall patches with per-patch emissivity")
    logger.info(f"    * File: {output_file}")


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


def generate_mass_flow_functions(patch_df):
    """
    Generate function objects to calculate mass flow rate at relevant boundaries.
    
    Creates surfaceFieldValue function objects for patches with flow:
    - velocity_inlet, mass_flow_inlet: Forced inflow
    - pressure_inlet, pressure_outlet: Natural ventilation openings
    
    Output files: postProcessing/massFlow_<patchName>/0/surfaceFieldValue.dat
    Format: iteration phi_sum(m³/s)
    
    Args:
        patch_df: DataFrame with patch information (id, type)
        
    Returns:
        str: functions{} block for controlDict, or empty string if no flow patches
    """
    # Filter patches with flow (exclude walls)
    flow_types = ['velocity_inlet', 'mass_flow_inlet', 'pressure_inlet', 'pressure_outlet']
    flow_patches = patch_df[patch_df['type'].isin(flow_types)]
    
    if len(flow_patches) == 0:
        logger.info("    * No flow patches found - skipping mass flow functions")
        return ""
    
    logger.info(f"    * Generating mass flow functions for {len(flow_patches)} patches")
    
    functions_block = "functions\n{\n"
    
    for _, row in flow_patches.iterrows():
        patch_id = row['id']
        patch_type = row['type']
        
        functions_block += f"""    massFlow_{patch_id}
    {{
        type            surfaceFieldValue;
        libs            ("libfieldFunctionObjects.so");
        
        // Write every 10 iterations to reduce file size
        writeControl    timeStep;
        writeInterval   10;
        
        // Calculate volumetric flow rate phi = U*n integrated over patch
        fields          (phi);
        operation       sum;
        
        regionType      patch;
        name            {patch_id};
        
        writeFields     false;
        
        // Output: postProcessing/massFlow_{patch_id}/0/surfaceFieldValue.dat
        // Convert to mass flow: phi_sum * rho_ref (1.2 kg/m3)
    }}

"""
    
    functions_block += "}\n"
    
    logger.info(f"    * Mass flow tracking enabled for patches: {', '.join(flow_patches['id'].tolist())}")
    return functions_block


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
        # For LAMINAR simulation: alphat is always "calculated" (no wall functions)
        # For TURBULENT: use compressible::alphatWallFunction
        # Since we're in laminar mode, use simple "calculated" BC everywhere
        bc["type"] = "calculated"
        bc["value"] = 0  # alphat=0 in laminar (no turbulent thermal diffusivity)
    
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


def define_radiation_bcs(variable, patch_type, patch_row=None):
    """
    Define boundary conditions for radiation fields (qr, G).
    
    Args:
        variable: 'qr' or 'G'
        patch_type: Type of patch
        patch_row: Row from patch_df with boundary info (optional, for G initialization)
        
    Returns:
        dict: Boundary condition data
    """
    bc = {}
    
    if variable == 'G':
        if patch_type == 'wall':
            # Marshak radiation boundary condition for walls
            # emissivityMode="lookup" reads from boundaryRadiationProperties (per-patch values)
            # bc["emissivity"] here is a fallback – the lookup file takes precedence
            bc["type"] = "MarshakRadiation"
            bc["emissivityMode"] = "lookup"
            # Use per-patch emissivity from patch_df (via patch_row); fall back to 0.9
            if patch_row is not None and 'emissivity' in patch_row:
                emissivity_val = patch_row['emissivity']
                bc["emissivity"] = float(emissivity_val) if emissivity_val == emissivity_val else 0.9
            else:
                bc["emissivity"] = 0.9
            # Let MarshakRadiation calculate value automatically from T field
            bc["value"] = INTERNALFIELD_DICT['G']
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
            # Fixed CO2 concentration at inlet (fresh air without CO2)
            bc["type"] = "fixedValue"
            bc["value"] = 0.0  # Fresh air (0% CO2)
        elif patch_type in ['pressure_inlet', 'pressure_outlet']:
            # Ventana/puerta abierta: entra aire fresco sin CO2 (ventilation scenario)
            bc["type"] = "fixedValue"
            bc["value"] = 0.0  # Fresh air (0% CO2)
        elif patch_type == 'wall':
            # Walls: no CO2 flux (impermeable)
            bc["type"] = "zeroGradient"
        else:
            bc["type"] = "zeroGradient"
    
    return bc


# [DEPRECATED 2026-01-17] Function no longer used - endTime configured in template files
# def update_controldict_iterations(case_path, simulation_type, transient=False):
#     """
#     [DEPRECATED] This function is no longer used.
#     endTime, writeInterval, and purgeWrite are now configured in template controlDict files:
#     - data/settings/cfd/hvac_transient/system/controlDict (transient)
#     - data/settings/cfd/hvac/system/controlDict (steady)
#     
#     The hardcoded values in this function were overriding template configurations,
#     causing unexpected behavior (e.g., endTime=1000 when template had endTime=10).
#     
#     Solution: Configure these parameters directly in the template files.
#     """
#     pass


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
    
    # For buoyantBoussinesqPimpleFoam with kOmegaSST turbulence model
    # Generate all turbulence fields: k, omega, nut
    # alphat is REQUIRED for thermal diffusion calculation
    
    for variable in DIMENSIONS_DICT.keys():
        # NO skip turbulence fields - using RAS turbulence model
        # if variable in ['k', 'omega', 'nut', 'epsilon']:
        #     logger.info(f"    * Skipping turbulence field {variable} (laminar simulation)")
        #     continue
            
        with case['0.orig'][variable] as f:
            f.dimensions = DIMENSIONS_DICT[variable]
            f.internal_field = INTERNALFIELD_DICT[variable]

            f.boundary_field = dict()
            for _, row in patch_df.iterrows():
                patch_type = row['type']
                new_bc_data = {}  # Initialize dict to avoid UnboundLocalError
                
                # Handle turbulence fields (k, omega, alphat, nut) with dedicated function
                # (This block is now unreachable due to skip above, but kept for clarity)
                if variable in ['k', 'omega', 'alphat', 'nut']:
                    new_bc_data = define_turbulence_bcs(variable, patch_type, row)
                # Handle radiation fields (qr, G) with dedicated function
                elif variable in ['qr', 'G']:
                    new_bc_data = define_radiation_bcs(variable, patch_type, row)
                # Handle scalar transport (CO2) with dedicated function
                elif variable == 'CO2':
                    new_bc_data = define_scalar_bcs(variable, patch_type, row)
                # Handle base fields (h, p, p_rgh, T, U) with original logic
                elif patch_type == 'wall':
                    # if(variable == 'h'):  # [COMMENTED FOR TEST - NO h FILE]
                    #     # Calculated from temperature BC (h = Cp×T)
                    #     new_bc_data["type"] = 'calculated'
                    #     new_bc_data["value"] = INTERNALFIELD_DICT['h']
                    if(variable == 'p'):
                        # Incompressible: kinematic pressure [m²/s²] - consistent with internal field
                        new_bc_data["type"] = 'calculated'
                        new_bc_data["value"] = INTERNALFIELD_DICT['p']  # Use consistent value
                    elif(variable == 'p_rgh'):
                        new_bc_data["type"] = 'fixedFluxPressure'
                        new_bc_data["value"] = 0
                    elif(variable == 'T'):
                        new_bc_data["type"] = 'fixedValue'
                        new_bc_data["value"] = row['T_(°C)'] + 273.15
                    elif(variable == 'U'):
                        new_bc_data["type"] = 'noSlip'
                    else:
                        raise BaseException('Unknown variable')
                elif(row['type'] == 'velocity_inlet'):
                    # if(variable == 'h'):  # [COMMENTED FOR TEST - NO h FILE]
                    #     # Calculated from temperature BC (h = Cp×T)
                    #     new_bc_data["type"] = 'calculated'
                    #     new_bc_data["value"] = INTERNALFIELD_DICT['h']
                    if(variable == 'p'):
                        # Kinematic pressure [m²/s²] - consistent with internal field
                        new_bc_data["type"] = 'calculated'
                        new_bc_data["value"] = INTERNALFIELD_DICT['p']
                    elif(variable == 'p_rgh'):
                        new_bc_data["type"] = 'fixedFluxPressure'
                        new_bc_data["value"] = INTERNALFIELD_DICT[variable]
                    elif(variable == 'T'):
                        new_bc_data["type"] = 'fixedValue'
                        new_bc_data["value"] = row['T_(°C)'] + 273.15
                    elif(variable == 'U'):
                        if (row['open']):
                            new_bc_data["type"] = 'fixedValue'
                            new_bc_data["value"] = row['U_(m/s)'] * np.array([row['fluid_nx'], row['fluid_ny'], row['fluid_nz']])
                        else:
                            # Closed velocity_inlet behaves as wall with no-slip condition
                            new_bc_data["type"] = 'noSlip'
                    else:
                        raise BaseException('Unknown variable')
                elif(row['type'] == 'pressure_inlet'):
                    # if(variable == 'h'):  # [COMMENTED FOR TEST - NO h FILE]
                    #     # Calculated from temperature BC (h = Cp×T)
                    #     new_bc_data["type"] = 'calculated'
                    #     new_bc_data["value"] = INTERNALFIELD_DICT['h']
                    if(variable == 'p'):
                        # Gauge kinematic pressure: convert Pa → m²/s²
                        new_bc_data["type"] = 'calculated'
                        new_bc_data["value"] = row['pressure_(Pa)'] / RHO_REF
                    elif(variable == 'p_rgh'):
                        # Convert pressure from Pa to kinematic (m²/s²) for Boussinesq solver
                        # p_rgh_kinematic = ΔP_Pa / rho_ref
                        new_bc_data["type"] = 'fixedValue'
                        p_rgh_kinematic = row['pressure_(Pa)'] / RHO_REF  # Pa / (kg/m³) = m²/s²
                        new_bc_data["value"] = p_rgh_kinematic
                        logger.info(f"    BC {row['id']} ({row['type']}): p_rgh = {p_rgh_kinematic:.2f} m²/s² (ΔP = {row['pressure_(Pa)']:.1f} Pa)")
                    elif(variable == 'T'):
                        # Bidirectional temperature: inletOutlet for backflow scenarios
                        new_bc_data["type"] = 'inletOutlet'
                        T_exterior = row['T_(°C)'] + 273.15  # Convert °C → K
                        new_bc_data["inletValue"] = T_exterior
                        new_bc_data["value"] = T_exterior
                        logger.info(f"    BC {row['id']} ({row['type']}): T = {row['T_(°C)']}°C = {T_exterior}K (inletOutlet)")
                    elif(variable == 'U'):
                        if (row['open']):
                            # pressureDirectedInletOutletVelocity: pressure-driven flow with constrained inlet direction
                            # - inflow (phi<0): direction = fluid_nx/ny/nz (from airOrientation); magnitude from solver via p_rgh
                            # - outflow (phi>0): zeroGradient (unconstrained, follows interior solution)
                            # Works for both normal (airOrientation=0) and oblique (airOrientation≠0) cases
                            new_bc_data["type"] = 'pressureDirectedInletOutletVelocity'
                            new_bc_data["inletDirection"] = np.array([row['fluid_nx'], row['fluid_ny'], row['fluid_nz']])
                            new_bc_data["value"] = INTERNALFIELD_DICT[variable]
                        else:
                            # Closed pressure_inlet behaves as wall with no-slip condition
                            new_bc_data["type"] = 'noSlip'
                    else:
                        raise BaseException('Unknown variable')
                elif(row['type'] == 'pressure_outlet'):
                    # if(variable == 'h'):  # [COMMENTED FOR TEST - NO h FILE]
                    #     # Calculated from temperature BC (h = Cp×T)
                    #     new_bc_data["type"] = 'calculated'
                    #     new_bc_data["value"] = INTERNALFIELD_DICT['h']
                    if(variable == 'p'):
                        # Gauge kinematic pressure: convert Pa → m²/s²
                        new_bc_data["type"] = 'calculated'
                        new_bc_data["value"] = row['pressure_(Pa)'] / RHO_REF
                    elif(variable == 'p_rgh'):
                        # Convert pressure from Pa to kinematic (m²/s²) for Boussinesq solver
                        # p_rgh_kinematic = ΔP_Pa / rho_ref (negative for outlet)
                        new_bc_data["type"] = 'fixedValue'
                        p_rgh_kinematic = row['pressure_(Pa)'] / RHO_REF  # Pa / (kg/m³) = m²/s²
                        new_bc_data["value"] = p_rgh_kinematic
                        logger.info(f"    BC {row['id']} ({row['type']}): p_rgh = {p_rgh_kinematic:.2f} m²/s² (ΔP = {row['pressure_(Pa)']:.1f} Pa)")
                    elif(variable == 'T'):
                        # Bidirectional temperature: inletOutlet for backflow scenarios
                        new_bc_data["type"] = 'inletOutlet'
                        T_exterior = row['T_(°C)'] + 273.15  # Convert °C → K
                        new_bc_data["inletValue"] = T_exterior
                        new_bc_data["value"] = T_exterior
                        logger.info(f"    BC {row['id']} ({row['type']}): T = {row['T_(°C)']}°C = {T_exterior}K (inletOutlet)")
                    elif(variable == 'U'):
                        if (row['open']):
                            # pressureDirectedInletOutletVelocity: pressure-driven flow with constrained inlet direction
                            # - inflow (phi<0): direction = fluid_nx/ny/nz (from airOrientation); magnitude from solver via p_rgh
                            # - outflow (phi>0): zeroGradient (unconstrained, follows interior solution)
                            # Works for both normal (airOrientation=0) and oblique (airOrientation≠0) cases
                            new_bc_data["type"] = 'pressureDirectedInletOutletVelocity'
                            new_bc_data["inletDirection"] = np.array([row['fluid_nx'], row['fluid_ny'], row['fluid_nz']])
                            new_bc_data["value"] = INTERNALFIELD_DICT[variable]
                        else:
                            # Closed pressure_outlet behaves as wall with no-slip condition
                            new_bc_data["type"] = 'noSlip'
                    else:
                        raise BaseException('Unknown variable')
                elif(row['type'] == 'mass_flow_inlet'):
                    # if(variable == 'h'):  # [COMMENTED FOR TEST - NO h FILE]
                    #     # Calculated from temperature BC (h = Cp×T)
                    #     new_bc_data["type"] = 'calculated'
                    #     new_bc_data["value"] = INTERNALFIELD_DICT['h']
                    if(variable == 'p'):
                        # Kinematic pressure [m²/s²] - consistent with internal field
                        new_bc_data["type"] = 'calculated'
                        new_bc_data["value"] = INTERNALFIELD_DICT['p']
                    elif(variable == 'p_rgh'):
                        new_bc_data["type"] = 'fixedFluxPressure'
                        new_bc_data["value"] = INTERNALFIELD_DICT[variable]
                    elif(variable == 'T'):
                        new_bc_data["type"] = 'fixedValue'
                        new_bc_data["value"] = row['T_(°C)'] + 273.15
                    elif(variable == 'U'):
                        # Convert massFlow [m³/h] → velocity vector [m/s] with prescribed direction
                        # Q [m³/s] = massFlow [m³/h] / 3600
                        # U_mag [m/s] = Q [m³/s] / area [m²]  (area from JSON dimensions)
                        # U_vec [m/s] = U_mag * (nx, ny, nz)  (flow direction from airOrientation)
                        Q_m3s = row['massFlow_(m³/h)'] / 3600.0           # m³/h → m³/s
                        area_m2 = float(row['area_(m²)']) if pd.notna(row.get('area', np.nan)) else 1.0
                        U_mag = Q_m3s / area_m2                    # m/s
                        new_bc_data["type"] = 'fixedValue'
                        new_bc_data["value"] = U_mag * np.array([row['fluid_nx'], row['fluid_ny'], row['fluid_nz']])
                        logger.info(f"    BC {row['id']} (mass_flow_inlet): Q={Q_m3s*3600:.1f} m³/h, A={area_m2:.4f} m², U_mag={U_mag:.2f} m/s")
                    else:
                        raise BaseException('Unknown variable')
                elif(row['type'] == 'rack_inlet'):
                    # Rack cold-air intake: air leaves the room domain.
                    # U: flowRateInletVelocity with NEGATIVE volumetricFlowRate (= outflow from domain)
                    # T: zeroGradient placeholder (will be replaced by post-processing)
                    if(variable == 'p'):
                        new_bc_data["type"] = 'calculated'
                        new_bc_data["value"] = INTERNALFIELD_DICT['p']
                    elif(variable == 'p_rgh'):
                        new_bc_data["type"] = 'fixedFluxPressure'
                        new_bc_data["value"] = 0
                    elif(variable == 'T'):
                        # Placeholder — post-processing replaces this with zeroGradient
                        new_bc_data["type"] = 'fixedValue'
                        new_bc_data["value"] = INTERNALFIELD_DICT['T']
                    elif(variable == 'U'):
                        Q_m3s = row['massFlow_(m³/h)'] / 3600.0  # m³/s
                        # Negative = outflow from domain (rack sucks cold air in)
                        new_bc_data["type"] = 'flowRateInletVelocity'
                        new_bc_data["volumetricFlowRate"] = -Q_m3s
                        new_bc_data["value"] = np.array([0.0, 0.0, 0.0])
                        logger.info(f"    BC {row['id']} (rack_inlet): Q={Q_m3s*3600:.1f} m³/h → volumetricFlowRate={-Q_m3s:.4f} m³/s (outflow)")
                    else:
                        raise BaseException('Unknown variable')
                elif(row['type'] == 'rack_outlet'):
                    # Rack hot-air exhaust: hot air enters the room domain.
                    # U: flowRateInletVelocity with POSITIVE volumetricFlowRate (= inflow into domain)
                    # T: codedFixedValue placeholder (will be replaced by post-processing)
                    #    T_outlet = average(T_inlet_patch) + ΔT
                    #    ΔT = thermalPower_(W) / (massFlow_(m³/h)/3600 * RHO_REF * CP)
                    if(variable == 'p'):
                        new_bc_data["type"] = 'calculated'
                        new_bc_data["value"] = INTERNALFIELD_DICT['p']
                    elif(variable == 'p_rgh'):
                        new_bc_data["type"] = 'fixedFluxPressure'
                        new_bc_data["value"] = 0
                    elif(variable == 'T'):
                        # Placeholder — post-processing replaces this with codedFixedValue
                        T_outlet_approx = row['T_(°C)'] + 273.15
                        new_bc_data["type"] = 'fixedValue'
                        new_bc_data["value"] = T_outlet_approx
                    elif(variable == 'U'):
                        Q_m3s = row['massFlow_(m³/h)'] / 3600.0  # m³/s
                        # Positive = inflow into domain (rack blows hot air out)
                        new_bc_data["type"] = 'flowRateInletVelocity'
                        new_bc_data["volumetricFlowRate"] = Q_m3s
                        new_bc_data["value"] = np.array([0.0, 0.0, 0.0])
                        logger.info(f"    BC {row['id']} (rack_outlet): Q={Q_m3s*3600:.1f} m³/h → volumetricFlowRate={Q_m3s:.4f} m³/s (inflow)")
                    else:
                        raise BaseException('Unknown variable')
                else:
                    raise BaseException('Boundary Condition Type Unknown')

                f.boundary_field[row['id']] = new_bc_data

    # ── Post-process 0.orig/U: foamlib writes 'inletDirection (x y z)' ──────
    # OpenFOAM (solver + foamToVTK) requires 'inletDirection uniform (x y z)'.
    # foamlib adds 'uniform' only to the special 'value' key; all other vector
    # entries in BC dicts are written without the keyword, causing a FATAL IO ERROR.
    import re as _re
    u_file = os.path.join(initial_path, 'U')
    with open(u_file, 'r', encoding='utf-8') as _f:
        _content = _f.read()
    _content = _re.sub(r'(inletDirection\s+)(\()', r'\1uniform \2', _content)
    with open(u_file, 'w', encoding='utf-8') as _f:
        _f.write(_content)
    logger.info("    * Fixed inletDirection keyword in 0.orig/U (added 'uniform')")

    # ── Post-process 0.orig/T: replace rack placeholder BCs ──────────────────
    # rack_inlet T: fixedValue placeholder → zeroGradient (takes T from interior)
    # rack_outlet T: fixedValue placeholder → codedFixedValue (T_inlet_avg + ΔT)
    rack_inlets  = patch_df[patch_df['type'] == 'rack_inlet']
    rack_outlets = patch_df[patch_df['type'] == 'rack_outlet']
    if len(rack_inlets) > 0 or len(rack_outlets) > 0:
        t_file = os.path.join(initial_path, 'T')
        with open(t_file, 'r', encoding='utf-8') as _f:
            _t_content = _f.read()

        # --- rack_inlet: replace with zeroGradient ---
        for _, row in rack_inlets.iterrows():
            pid = row['id']
            # Matches the entire patch block (no nested braces in fixedValue)
            _pat = r'([ \t]*)' + _re.escape(pid) + r'[ \t]*\n[ \t]*\{[^}]*?\}'
            _new = (
                f'    {pid}\n'
                f'    {{\n'
                f'        type            zeroGradient;\n'
                f'    }}'
            )
            _t_content = _re.sub(_pat, _new, _t_content, flags=_re.DOTALL)
            logger.info(f"    * rack_inlet T: '{pid}' → zeroGradient")

        # --- rack_outlet: replace with codedFixedValue ---
        for _, row in rack_outlets.iterrows():
            pid        = row['id']
            inlet_pid  = row['inlet_id']
            T_out_K    = row['T_(°C)'] + 273.15   # approximate initial value [K]
            Q_m3s      = row['massFlow_(m³/h)'] / 3600.0
            Q_W        = float(row['thermalPower'])  # already in W (converted in create_volumes.py)
            m_dot      = Q_m3s * RHO_REF            # kg/s
            delta_T    = Q_W / (m_dot * CP)          # K  ← constant design parameter

            coded_name = f"rackOutlet_{pid}"
            _new = (
                f'    {pid}\n'
                f'    {{\n'
                f'        type            codedFixedValue;\n'
                f'        value           uniform {T_out_K:.4f};\n'
                f'        name            {coded_name};\n'
                f'        code\n'
                f'        #{{\n'
                f'            const fvMesh& mesh = this->patch().boundaryMesh().mesh();\n'
                f'            label inletID = mesh.boundaryMesh().findPatchID("{inlet_pid}");\n'
                f'            const scalarField& Tin =\n'
                f'                mesh.boundary()[inletID].lookupPatchField<volScalarField, scalar>("T");\n'
                f'            scalar Tmean = gAverage(Tin);\n'
                f'            operator==(Tmean + {delta_T:.4f});  // deltaT = Q_W/(mdot*Cp)\n'
                f'        #}};\n'
                f'    }}'
            )
            _pat = r'([ \t]*)' + _re.escape(pid) + r'[ \t]*\n[ \t]*\{[^}]*?\}'
            _t_content = _re.sub(_pat, _new, _t_content, flags=_re.DOTALL)
            logger.info(
                f"    * rack_outlet T: '{pid}' → codedFixedValue "
                f"(ΔT={delta_T:.2f} K, Q={Q_W:.0f} W, inlet='{inlet_pid}')"
            )

        with open(t_file, 'w', encoding='utf-8') as _f:
            _f.write(_t_content)
        logger.info("    * Rack T boundary conditions post-processed successfully")


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
    solver_type = "buoyantBoussinesqPimpleFoam (transient)" if transient else "buoyantBoussinesqSimpleFoam (steady)"
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
    
    # Create boundaryRadiationProperties for radiation model (both steady and transient)
    logger.info("    * Creating boundaryRadiationProperties for radiation model")
    define_boundary_radiation_properties(sim_path, geo_df)
    
    logger.info("    * Setting up system files (solver settings, discretization schemes)")
    define_system_files(template_path, sim_path)
    
    # Generate mass flow monitoring functions for flow boundaries
    logger.info("    * Generating mass flow monitoring function objects")
    mass_flow_functions = generate_mass_flow_functions(geo_df)
    
    if mass_flow_functions:
        # Add functions{} block to controlDict
        controldict_path = os.path.join(sim_path, "system", "controlDict")
        
        with open(controldict_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Insert massFlow functions INSIDE existing functions{} block
        # Find the closing brace of functions{} and insert before it
        import re
        
        # Pattern: find "functions\n{\n ... \n}" and insert before final }
        pattern = r'(functions\s*\{.*?)(^\})'
        
        def insert_mass_flow(match):
            functions_body = match.group(1)
            closing_brace = match.group(2)
            
            # Add massFlow functions before closing brace
            # Remove "functions\n{\n" prefix and final "}\n" from mass_flow_functions
            mass_flow_clean = mass_flow_functions.replace('functions\n{\n', '', 1)
            # Remove final "}\n" carefully (only the wrapper, not function braces)
            if mass_flow_clean.endswith('}\n'):
                mass_flow_clean = mass_flow_clean[:-2]  # Remove last "}\n"
            
            return functions_body + '\n' + mass_flow_clean + '\n' + closing_brace
        
        content = re.sub(pattern, insert_mass_flow, content, flags=re.MULTILINE | re.DOTALL)

        with open(controldict_path, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info("    * Mass flow functions added to controlDict")
    
    logger.info("    * Creating decomposeParDict for local parallel execution")
    # Detect CPU cores for parallel execution
    import os as os_module
    total_cores = os_module.cpu_count()
    if total_cores is None:
        total_cores = 16
    n_cores = max(total_cores - 1, 1)
    n_cpu_available, _ = best_cpu_partition(n_cores)
    
    create_decomposeParDict_local(sim_path)
    
    # [DEPRECATED 2026-01-17] endTime is now configured in template controlDict files
    # No need to override template values with hardcoded values
    # logger.info(f"    * Updating controlDict iterations based on simulation type: {simulation_type}")
    # update_controldict_iterations(case_path, simulation_type, transient=transient)

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
        
        # SETFIELDS DISABLED: Let solver handle hydrostatic pressure naturally
        # Uncertainty about correct p vs p_rgh initialization with kinematic units
        # Safer to start from p=0, p_rgh=0 and let solver stabilize in first iterations
        # 'echo "==================== APPLYING HYDROSTATIC PRESSURE GRADIENT ===================="',
        # 'runApplication setFields',
        # 'echo "==================== HYDROSTATIC PRESSURE INITIALIZED: p(z) = p_atm - rho*g*z ===================="',
    ]
    
    # Continue with common steps
    script_commands.extend([
        # Generate VTK for time 0 (initial fields with hydrostatic pressure) - BEFORE potentialFoam
        'echo "==================== GENERATING VTK FOR TIME 0 (INITIAL STATE) ===================="',
        'foamToVTK -time 0 -fields "(T U p p_rgh G qr)" 2>&1 | tee log.foamToVTK_time0',  # Added G, qr for radiation
        'echo "==================== TIME 0 VTK COMPLETED ===================="',
        
        # Decompose for parallel execution
        'rm -rf processor*',
        'runApplication decomposePar',
        
        # DEBUG: Copiar archivos de processor0 para inspección
        'echo "==================== DEBUG: Copying processor0 files ===================="',
        'mkdir -p debug_files',
        # 'cp processor0/0/h debug_files/processor0_h',  # [COMMENTED - NO h FILE FOR TEST]
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
    
    # Run solver: TRANSIENT or STEADY
    if transient:
        # ========== TRANSIENT: Single execution with ultra-conservative settings ==========
        # Uses controlDict values directly (no foamDictionary modifications)
        # deltaT=1e-6, maxDeltaT=1e-4, maxCo=0.1 configured in controlDict template
        
        script_commands.extend([
            'echo "=========================================================================="',
            'echo "TRANSIENT SOLVER: ULTRA-CONSERVATIVE CONFIGURATION (Boussinesq)"',
            'echo "Using controlDict settings: deltaT=1e-6, maxDeltaT=1e-4, maxCo=0.1"',
            'echo "Solver: buoyantBoussinesqPimpleFoam (Boussinesq approximation)"',
            'echo "=========================================================================="',
            '',
            'echo "==================== RUNNING BUOYANTBOUSSINESQPIMPLEFOAM ===================="',
            f'mpirun -np {n_cpu_available} buoyantBoussinesqPimpleFoam -parallel > log.buoyantBoussinesqPimpleFoam 2>&1',
            'RETVAL=$?',
            'if [ $RETVAL -ne 0 ] || grep -q "FOAM FATAL\\|floating point exception\\|Killed" log.buoyantBoussinesqPimpleFoam; then',
            '    echo "❌ ERROR: Solver FAILED (exit code: $RETVAL)"',
            '    echo "Attempting to reconstruct partial results..."',
            '    reconstructPar -latestTime > log.reconstructPar 2>&1 || echo "⚠️  reconstructPar failed"',
            '    echo "Check log.buoyantBoussinesqPimpleFoam for crash details."',
            '    exit 1',
            'fi',
            'echo "✅ Solver completed successfully"',
            'echo "=========================================================================="',
        ])
    else:
        # ========== STEADY: Single execution ==========
        script_commands.extend([
            'echo "==================== RUNNING STEADY-STATE SOLVER (Boussinesq) ===================="',
            f'runParallel -np {n_cpu_available} buoyantBoussinesqSimpleFoam -parallel',
            'echo "==================== STEADY SOLVER COMPLETED ===================="',
        ])
    
    # Continue with common post-processing steps
    script_commands.extend([

        # 3. Reconstruct ALL timesteps (not just latestTime) for complete iteration history
        'echo "==================== RECONSTRUCTING ALL ITERATIONS ===================="',
        'reconstructPar > log.reconstructPar 2>&1',  # Default behavior = reconstruct ALL timesteps
        'echo "==================== RECONSTRUCTION COMPLETED ===================="',

        # 3.5. Calculate PMV/PPD thermal comfort fields from T and U (latest timestep only)
        'echo "==================== CALCULATING PMV/PPD THERMAL COMFORT ===================="',
        'python3 ./calculate_comfort.py . --all-times 2>&1 | tee log.comfort',
        'echo "==================== PMV/PPD CALCULATION COMPLETED ===================="',

        # 4. Generate VTK files for ALL timesteps with VOLUMETRIC data
        'echo "==================== GENERATING VTK FOR ALL ITERATIONS ===================="',
        # Process all timesteps (no -latestTime flag)
        # -excludePatches: Skip internal patches to reduce file size
        # Note: Generates VTK/ directory with subdirs for each timestep
        'runApplication foamToVTK -fields "(T U p p_rgh PMV PPD G qr)" -excludePatches "(.*_master|.*_slave)"',
        'echo "==================== VTK GENERATION COMPLETED ===================="',
        
        # Also generate lightweight surface-only VTK for quick preview
        'echo "==================== GENERATING SURFACE VTK (QUICK PREVIEW) ===================="',
        'foamToVTK -latestTime -surfaceFields -fields "(T U p p_rgh PMV PPD G qr)" 2>&1 | tee log.foamToVTK_surface',
        'echo "==================== SURFACE VTK COMPLETED ===================="',

        # Clean processors
        'rm -rf processor*',
    ])
    
    logger.info("    * HVAC CFD case setup completed successfully")
    return script_commands


if __name__ == "__main__":
    case_folder = "case"
    result = setup(case_folder)
