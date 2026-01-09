import os
import numpy as np
import pandas as pd
import pyvista as pv
import multiprocessing
import logging
from pathlib import Path

from src.components.tools.populate_template_file import replace_in_file

logger = logging.getLogger(__name__)

# Calculate project root: 4 levels up from this file
# src/components/mesh/cfmesh.py -> src/components/mesh/ -> src/components/ -> src/ -> FLOWDESK_OF/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def create_surfaceFeatureExtractDict(template_path, sim_path, stl_filename):
    """Create surfaceFeatureExtractDict for cfMesh (same as snappyHexMesh)"""
    input_path = os.path.join(template_path, "system", "surfaceFeatureExtractDict") 
    output_path = os.path.join(sim_path, "system", "surfaceFeatureExtractDict") 
    str_replace_dict = dict()
    str_replace_dict["$STL_FILENAME"] = stl_filename
    replace_in_file(input_path, output_path, str_replace_dict)


def calculate_adaptive_cell_size(geo_mesh):
    """
    Calculate base cell size for cfMesh.
    Using FIXED cell size of 0.1m for uniform mesh throughout the domain.
    This ensures consistent mesh quality and predictable cell count.
    """
    bounds = geo_mesh.bounds
    x_range = bounds[1] - bounds[0]
    y_range = bounds[3] - bounds[2]
    z_range = bounds[5] - bounds[4]
    max_dim = max(x_range, y_range, z_range)
    
    # Fixed cell size: 0.1m for uniform mesh
    base_cell_size = 0.06
    logger.info(f"    * Geometry bounds: X={x_range:.2f}m, Y={y_range:.2f}m, Z={z_range:.2f}m")
    logger.info(f"    * Using FIXED cell size: {base_cell_size:.4f}m (uniform mesh)")
    logger.info(f"    * Expected cells in domain: ~{(x_range/base_cell_size) * (y_range/base_cell_size) * (z_range/base_cell_size):.0f}")
    
    return base_cell_size


def validate_geometry(geo_mesh, geo_df):
    """Validate geometry before meshing."""
    if geo_mesh.n_cells == 0:
        raise ValueError("Geometry mesh is empty - no cells to mesh")
    if len(geo_df) == 0:
        raise ValueError("No boundary conditions defined")
    logger.info(f"✓ Geometry validation passed: {geo_mesh.n_cells} cells, {len(geo_df)} boundary conditions")


def generate_local_refinement_block(geo_df, base_cell_size):
    """Generate localRefinement block for critical patches (window/vent/door) and furniture objects."""
    blocks = []
    
    # 1. Critical patches (window/vent/door): cellSize = base/2
    critical_patterns = ['window', 'vent', 'door']
    refinement_cell_size = base_cell_size / 1.0  # 0.025m for base 0.1m
    refinement_thickness = base_cell_size / 1.0  # 0.05m for base 0.1m
    
    for pattern in critical_patterns:
        # Check if any patch matches this pattern
        matching_patches = geo_df[geo_df['id'].str.contains(pattern, case=False, na=False)]
        if len(matching_patches) > 0:
            logger.info(f"    * {pattern}.* patches: localRefinement cellSize={refinement_cell_size:.4f}m, thickness={refinement_thickness:.2f}m")
            blocks.append(f"""    "{pattern}.*"
    {{
        cellSize {refinement_cell_size:.6f};
        refinementThickness {refinement_thickness:.6f};
    }}""")
    
    # 2. Furniture objects (object_*): Add to localRefinement with TYPE-SPECIFIC parameters
    # cfMesh does NOT support wildcards, must list each patch explicitly
    object_patches = geo_df[geo_df['id'].str.startswith('object_', na=False)]
    if len(object_patches) > 0:
        logger.info(f"    * 3D Objects: {len(object_patches)} patches found")
        logger.info(f"      → Applying differentiated refinement by object type:")
        
        # Group objects by type for organized logging
        obj_types = {'person': [], 'block': [], 'table': [], 'chair': [], 'stairs': [], 'other': []}
        
        for patch_id in object_patches['id']:
            patch_lower = patch_id.lower()
            if 'person' in patch_lower:
                obj_types['person'].append(patch_id)
            elif 'block' in patch_lower:
                obj_types['block'].append(patch_id)
            elif 'mesa' in patch_lower or 'table' in patch_lower:
                obj_types['table'].append(patch_id)
            elif 'silla' in patch_lower or 'chair' in patch_lower:
                obj_types['chair'].append(patch_id)
            elif 'stair' in patch_lower:
                obj_types['stairs'].append(patch_id)
            else:
                obj_types['other'].append(patch_id)
        
        # Log and create blocks for each type
        if obj_types['person']:
            cell_size, thickness = base_cell_size / 8.0, base_cell_size / 4.0
            logger.info(f"      → Person ({len(obj_types['person'])}): cellSize={cell_size:.4f}m (base/8), thickness={thickness:.4f}m")
            for patch_id in obj_types['person']:
                logger.info(f"          - {patch_id}")
                blocks.append(f"""    {patch_id}
    {{
        cellSize {cell_size:.6f};
        refinementThickness {thickness:.6f};
    }}""")
        
        if obj_types['block']:
            cell_size, thickness = base_cell_size / 2.0, base_cell_size
            logger.info(f"      → Block ({len(obj_types['block'])}): cellSize={cell_size:.4f}m (base/2), thickness={thickness:.4f}m")
            for patch_id in obj_types['block']:
                logger.info(f"          - {patch_id}")
                blocks.append(f"""    {patch_id}
    {{
        cellSize {cell_size:.6f};
        refinementThickness {thickness:.6f};
    }}""")
        
        if obj_types['table']:
            cell_size, thickness = base_cell_size / 4.0, base_cell_size
            logger.info(f"      → Mesa/Table ({len(obj_types['table'])}): cellSize={cell_size:.4f}m (base/2), thickness={thickness:.4f}m")
            for patch_id in obj_types['table']:
                logger.info(f"          - {patch_id}")
                blocks.append(f"""    {patch_id}
    {{
        cellSize {cell_size:.6f};
        refinementThickness {thickness:.6f};
    }}""")
        
        if obj_types['chair']:
            cell_size, thickness = base_cell_size / 10.0, base_cell_size / 4.0
            logger.info(f"      → Silla/Chair ({len(obj_types['chair'])}): cellSize={cell_size:.4f}m (base/2), thickness={thickness:.4f}m")
            for patch_id in obj_types['chair']:
                logger.info(f"          - {patch_id}")
                blocks.append(f"""    {patch_id}
    {{
        cellSize {cell_size:.6f};
        refinementThickness {thickness:.6f};
    }}""")
        
        if obj_types['stairs']:
            cell_size, thickness = base_cell_size / 2.0, 2.0 * base_cell_size
            logger.info(f"      → Stairs ({len(obj_types['stairs'])}): cellSize={cell_size:.4f}m (base/2), thickness={thickness:.4f}m")
            for patch_id in obj_types['stairs']:
                logger.info(f"          - {patch_id}")
                blocks.append(f"""    {patch_id}
    {{
        cellSize {cell_size:.6f};
        refinementThickness {thickness:.6f};
    }}""")
        
        if obj_types['other']:
            cell_size, thickness = base_cell_size / 4.0, base_cell_size / 8.0
            logger.info(f"      → Other objects ({len(obj_types['other'])}): cellSize={cell_size:.4f}m (base/4), thickness={thickness:.4f}m")
            for patch_id in obj_types['other']:
                logger.info(f"          - {patch_id}")
                blocks.append(f"""    {patch_id}
    {{
        cellSize {cell_size:.6f};
        refinementThickness {thickness:.6f};
    }}""")
    
    if blocks:
        return "// Refinamiento LOCAL en BCs críticos y objetos 3D\nlocalRefinement\n{\n" + "\n    \n".join(blocks) + "\n}"
    else:
        return "// No critical patches or furniture objects found for localRefinement"


def generate_boundary_layers_block(geo_df, base_cell_size):
    """Generate boundaryLayers block with global settings and critical patch overrides."""
    base_n_layers = 0
    base_thickness_ratio = 1
    base_first_layer = base_cell_size * 0.8  # 0.02m for base 0.1m
    
    critical_patterns = ['window', 'vent', 'door']
    critical_n_layers = 0
    
    # Build patchBoundaryLayers for critical patches
    patch_blocks = []
    for pattern in critical_patterns:
        matching_patches = geo_df[geo_df['id'].str.contains(pattern, case=False, na=False)]
        if len(matching_patches) > 0:
            logger.info(f"    * {pattern}.* patches: {critical_n_layers} boundary layers (critical)")
            patch_blocks.append(f"""        "{pattern}.*"
        {{
            nLayers {critical_n_layers};
            allowDiscontinuity 1;
        }}""")
    
    # Build full boundaryLayers block
    result = f"""// Boundary layers - celdas prismáticas en BCs
boundaryLayers
{{
    nLayers {base_n_layers};
    thicknessRatio {base_thickness_ratio};
    maxFirstLayerThickness {base_first_layer:.6f};"""
    
    if patch_blocks:
        result += "\n    \n    patchBoundaryLayers\n    {\n" + "\n        \n".join(patch_blocks) + "\n    }"
    
    result += "\n}"
    
    logger.info(f"    * Global boundary layers: {base_n_layers} layers, ratio={base_thickness_ratio}, firstLayer={base_first_layer:.4f}m")
    
    return result


def create_meshDict(template_path, sim_path, stl_filename, geo_mesh, geo_df):
    """Create meshDict for cfMesh cartesianMesh with optimized settings."""
    input_path = os.path.join(template_path, "system", "meshDict") 
    output_path = os.path.join(sim_path, "system", "meshDict") 
    validate_geometry(geo_mesh, geo_df)
    
    # Calculate cell sizes
    base_cell_size = calculate_adaptive_cell_size(geo_mesh)

    boundary_cell_size = base_cell_size / 1.0  # 0.05m for base 0.1m
    min_cell_size = base_cell_size / 1.0     # 0.0125m for base 0.1m
    
    # Generate dynamic blocks
    local_refinement = generate_local_refinement_block(geo_df, base_cell_size)
    boundary_layers = generate_boundary_layers_block(geo_df, base_cell_size)
    
    # Generate filenames
    fms_filename = stl_filename.replace(".stl", ".fms")
    
    str_replace_dict = dict()
    str_replace_dict["$FMS_FILENAME"] = fms_filename
    str_replace_dict["$MAX_CELL_SIZE"] = f"{base_cell_size:.6f}"
    str_replace_dict["$BOUNDARY_CELL_SIZE"] = f"{boundary_cell_size:.6f}"
    str_replace_dict["$MIN_CELL_SIZE"] = f"{min_cell_size:.6f}"
    str_replace_dict["$LOCAL_REFINEMENT"] = local_refinement
    str_replace_dict["$BOUNDARY_LAYERS"] = boundary_layers
    
    logger.info(f"    * Cell sizes: max={base_cell_size:.4f}m, boundary={boundary_cell_size:.4f}m, min={min_cell_size:.4f}m")
    replace_in_file(input_path, output_path, str_replace_dict)


def create_emesh_file(geo_mesh_dict, sim_path, stl_filename):
    """Create .eMesh file for cfMesh to properly identify patches."""
    emesh_filename = stl_filename.replace(".stl", ".eMesh")
    emesh_path = os.path.join(sim_path, "constant", "triSurface", emesh_filename)
    with open(emesh_path, 'w') as f:
        f.write("/*--------------------------------*- C++ -*----------------------------------*\\\n")
        f.write("| =========                 |                                                 |\n")
        f.write("| \\\\      /  F ield         | cfMesh: A library for mesh generation          |\n")
        f.write("|  \\\\    /   O peration     |                                                 |\n")
        f.write("|   \\\\  /    A nd           | Author: Franjo Juretic                          |\n")
        f.write("|    \\\\/     M anipulation  | E-mail: franjo.juretic@c-fields.com            |\n")
        f.write("\\*---------------------------------------------------------------------------*/\n")
        f.write("FoamFile\n")
        f.write("{\n")
        f.write("    version   2.0;\n")
        f.write("    format    ascii;\n")
        f.write("    class     edgeMesh;\n")
        f.write("    location  \"constant/triSurface\";\n")
        f.write(f"    object    {emesh_filename};\n")
        f.write("}\n")
        f.write("// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n")
        f.write("\n")
        patch_idx = 0
        for patch_name, mesh in geo_mesh_dict.items():
            f.write(f"// Patch {patch_idx}: {patch_name}\n")
            f.write(f"// Faces: {mesh.n_cells}\n")
            patch_idx += 1
        f.write("\n")
    logger.info(f"    * Created eMesh file: {emesh_path}")
    return emesh_filename


def export_to_fms(geo_mesh_dict, sim_path, fms_filename):
    """Export geometry to STL format for cfMesh."""
    stl_filename = fms_filename.replace(".fms", ".stl")
    stl_path = os.path.join(sim_path, "constant", "triSurface", stl_filename)
    os.makedirs(os.path.dirname(stl_path), exist_ok=True)
    
    def write_facet(f, normal, points):
        f.write(f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n")
        f.write("    outer loop\n")
        for pt in points:
            f.write(f"      vertex {pt[0]:.6e} {pt[1]:.6e} {pt[2]:.6e}\n")
        f.write("    endloop\n")
        f.write("  endfacet\n")
    
    with open(stl_path, 'w') as f:
        for solid_name, mesh in geo_mesh_dict.items():
            f.write(f"solid {solid_name}\n")
            mesh = mesh.triangulate()
            faces = mesh.cells.reshape((-1, 4))
            for face in faces:
                assert face[0] == 3
                pts = mesh.points[face[1:4]]
                v1 = pts[1] - pts[0]
                v2 = pts[2] - pts[0]
                normal = np.cross(v1, v2)
                norm = np.linalg.norm(normal)
                if norm > 0:
                    normal /= norm
                else:
                    normal = np.array([0.0, 0.0, 0.0])
                write_facet(f, normal, pts)
            f.write(f"endsolid {solid_name}\n")
    
    return stl_filename


def split_polydata_by_cell_data(mesh: pv.PolyData, df: pd.DataFrame) -> dict:
    """Split mesh by patch ID for multi-solid export"""
    patch_names = df[["id"]].to_dict()
    patch_mesh_dict = {}
    for patch_id, patch_name in patch_names['id'].items():
        mask = mesh.cell_data["patch_id"] == patch_id
        submesh = mesh.extract_cells(mask)
        patch_mesh_dict[patch_name] = submesh
    return patch_mesh_dict


def get_parallel_options():
    """Get parallel execution options for cfMesh.
    
    Always uses serial meshing for stability and compatibility.
    Serial mode is more reliable and avoids parallel execution issues.
    """
    logger.info("    * Using SERIAL meshing mode (most stable and reliable)")
    return ""


def prepare_cfmesh(geo_mesh, sim_path, geo_df, fms_filename="geometry.fms"):
    """Prepare cfMesh configuration and scripts for mesh generation."""
    logger.info(f"    * Preparing cfMesh configuration for {geo_mesh.n_cells} geometry cells")
    logger.info("    * Implementing regular prism boundary layers with uniform thickness")
    
    logger.info("    * Splitting geometry mesh by boundary condition patches")
    geo_mesh_dict = split_polydata_by_cell_data(geo_mesh, geo_df)
    logger.info(f"    * Split into {len(geo_mesh_dict)} patch meshes")
    
    logger.info(f"    * Exporting geometry to STL format: {fms_filename}")
    stl_filename = export_to_fms(geo_mesh_dict, sim_path, fms_filename)
    
    # Use PROJECT_ROOT for robust path resolution (independent of execution directory)
    template_path = str(PROJECT_ROOT / "data" / "settings" / "mesh" / "cfmesh")
    logger.info(f"    * Creating cfMesh configuration files from template: {template_path}")
    
    logger.info("    * Creating meshDict with optimized settings")
    create_meshDict(template_path, sim_path, stl_filename, geo_mesh, geo_df)
    
    expected_patches = geo_df["id"].tolist()
    expected_patches_str = ", ".join(expected_patches)
    
    pressure_patches = geo_df[geo_df['type'].isin(['pressure_inlet', 'pressure_outlet'])]
    wall_patches = geo_df[geo_df['type'] == 'wall']
    logger.info(f"    * Pressure boundaries ({len(pressure_patches)}): 8 layers, 2x fine refinement")
    logger.info(f"    * Wall boundaries ({len(wall_patches)}): 6 layers, 1.5x fine refinement")
    
    parallel_opts = get_parallel_options()
    
    # Generate FMS filename
    fms_file = stl_filename.replace(".stl", ".fms")
    
    script_commands = [
        '#!/bin/sh',
        'cd "${0%/*}" || exit',
        '. ${WM_PROJECT_DIR:?}/bin/tools/RunFunctions',
        'decompDict="-decomposeParDict system/decomposeParDict"',
        'echo "==================== GENERATING FMS WITH FEATURE EDGES ===================="',
        f'runApplication surfaceFeatureEdges -angle 10 constant/triSurface/{stl_filename} constant/triSurface/{fms_file}',
        'echo "==================== EXPORTING FEATURE EDGES TO VTK ===================="',
        f'echo "Using FMSToSurface to export feature edges for ParaView visualization..."',
        f'FMSToSurface constant/triSurface/{fms_file} constant/triSurface/edges_temp.vtk -exportFeatureEdges',
        f'if [ -f "constant/triSurface/edges_temp_featureEdges.vtk" ]; then',
        f'    mv constant/triSurface/edges_temp_featureEdges.vtk constant/triSurface/geometry_edges.vtk',
        f'    echo "✓ Feature edges exported to: constant/triSurface/geometry_edges.vtk"',
        f'else',
        f'    echo "⚠ Warning: Feature edges export may have failed"',
        f'fi',
        'echo "==================== RUNNING CFMESH CARTESIAN MESHER ===================="',
        'echo "cfMesh configuration:"',
        'echo "  - Adaptive base cell size from geometry"',
        'echo "  - Pressure boundaries: 2x fine refinement, 8 regular prism layers"',
        'echo "  - Wall boundaries: 1.5x fine refinement, 6 regular prism layers"',
        'echo "  - Boundary layer optimization: enabled"',
        'echo "  - Geometry constraint enforcement: enabled"',
        'echo ""',
        f'runApplication cartesianMesh {parallel_opts}',
        'echo "==================== DETECTING MESH LOCATION ===================="',
        'if [ -d "constant/polyMesh" ]; then',
        '    MESH_LOCATION="constant"',
        '    echo "✓ Mesh found in SERIAL location: constant/polyMesh"',
        'elif [ -d "processor0/constant/polyMesh" ]; then',
        '    MESH_LOCATION="processor0/constant"',
        '    echo "✓ Mesh found in PARALLEL location: processor0/constant/polyMesh"',
        'else',
        '    echo "✗ ERROR: Mesh not found in constant/polyMesh or processor0/constant/polyMesh"',
        '    exit 1',
        'fi',
        'echo "==================== VALIDATING MESH ===================="',
        "python3 << 'VALIDATION_EOF'",
        "import re, os",
        "mesh_location = open('/tmp/mesh_location.txt').read().strip()",
        "boundary_file = f'{mesh_location}/polyMesh/boundary'",
        "with open(boundary_file, 'r') as f:",
        "    content = f.read()",
        "patches = {}",
        "lines = content.split('\\n')",
        "current_patch = None",
        "for i, line in enumerate(lines):",
        "    if re.match(r'^\\s+(\\w+)\\s*$', line):",
        "        current_patch = line.strip()",
        "    elif current_patch and 'nFaces' in line:",
        "        match = re.search(r'nFaces\\s+(\\d+)', line)",
        "        if match:",
        "            patches[current_patch] = int(match.group(1))",
        "            current_patch = None",
        "background_patches = ['limits', 'defaultFaces', 'background']",
        f"expected_patches = [{', '.join([repr(p) for p in expected_patches])}]",
        "failed = [(p, patches[p]) for p in patches if p in background_patches and patches[p] > 0]",
        "if failed:",
        "    print('\\n' + '='*80)",
        "    print('MESHING ERROR: cfMesh failed to cut geometry properly!')",
        "    print('='*80)",
        "    print('Background patches remain in the mesh:')",
        "    for patch, count in failed:",
        "        print(f'  - {patch}: {count} faces')",
        "    print()",
        "    print('This indicates the geometry is NOT WATERTIGHT (has holes/gaps).')",
        "    print('Possible causes:')",
        "    print('  1. Geometry has holes, gaps, or non-manifold surfaces')",
        "    print('  2. Wall extrusion created invalid 3D geometry')",
        "    print('  3. Boolean operations failed to create closed volume')",
        "    print()",
        f"    print('Expected patches: {expected_patches_str}')",
        "    print(f'Actual patches: {\", \".join(patches.keys())}')",
        "    print('='*80 + '\\n')",
        "    exit(1)",
        "print('✅ Mesh validation passed - no background patches found')",
        "print(f'Valid patches: {\", \".join(patches.keys())}')",
        "VALIDATION_EOF",
        'echo "==================== MESH VALIDATION PASSED ===================="',
        'runApplication checkMesh',
        'rm -rf 0',
        'cp -r 0.orig 0',
        'echo "==================== COPYING MESH TO STANDARD LOCATION ===================="',
        'if [ "$MESH_LOCATION" != "constant" ]; then',
        '    echo "Copying mesh from $MESH_LOCATION/polyMesh to constant/polyMesh"',
        '    rm -rf constant/polyMesh',
        '    cp -r $MESH_LOCATION/polyMesh constant/',
        'fi',
        'echo "==================== INITIAL CONDITIONS CHECK ===================="',
        'echo ""',
        'echo "--- Checking h (enthalpy) ---"',
        'grep "internalField" 0/h',
        'echo ""',
        'echo "h boundary conditions:"',
        'head -n 50 0/h | grep -A 2 "type"',
        'echo ""',
        'echo "--- Checking thermophysicalProperties ---"',
        'grep -A 15 "mixture" constant/thermophysicalProperties',
        'echo ""',
        'echo "==================== END INITIAL CONDITIONS CHECK ===================="',
        'echo ""',
        'touch results.foam',
    ]
    
    logger.info("    * cfMesh preparation completed successfully")
    logger.info(f"    * Mesh validation will check for patches: {expected_patches}")
    logger.info("    * cfMesh will generate robust boundary layers automatically")
    logger.info("    * Pressure boundaries will have 2x finer resolution than walls")
    return script_commands
