import os
import numpy as np
import pandas as pd
import pyvista as pv

from src.components.tools.populate_template_file import replace_in_file, generate_regions_block


def calculate_patch_normal(geo_mesh, patch_id):
    """
    Calculate the average normal vector of a patch from geo_mesh.
    
    Args:
        geo_mesh: PyVista PolyData with cell_data["patch_id"]
        patch_id: ID of the patch to analyze
    
    Returns:
        normal: np.array([x, y, z]) - normalized average normal
    """
    import numpy as np
    
    # Extract cells for this patch
    mask = geo_mesh.cell_data["patch_id"] == patch_id
    patch_cells = geo_mesh.extract_cells(mask)
    
    # Convert to PolyData and extract surface
    patch_surface = patch_cells.extract_surface()
    
    # Compute normals for the surface
    patch_surface = patch_surface.compute_normals(cell_normals=True, point_normals=False)
    
    # Average normals (area-weighted would be better, but this is sufficient)
    normals = patch_surface.cell_data["Normals"]
    avg_normal = np.mean(normals, axis=0)
    
    # Normalize
    norm = np.linalg.norm(avg_normal)
    if norm > 0:
        avg_normal = avg_normal / norm
    
    return avg_normal


def classify_bc_alignment(normal):
    """
    Classify BC according to alignment with principal axes.
    
    Args:
        normal: np.array([x, y, z]) - surface normal
    
    Returns:
        'aligned' | 'quasi_aligned' | 'non_aligned'
    """
    import numpy as np
    
    # Principal axes
    axes = [
        np.array([1, 0, 0]),  # X
        np.array([0, 1, 0]),  # Y  
        np.array([0, 0, 1])   # Z
    ]
    
    # Calculate minimum angle with any axis
    min_angle = 90.0
    for axis in axes:
        dot_product = np.abs(np.dot(normal, axis))
        angle = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
        min_angle = min(min_angle, angle)
    
    # Classify according to thresholds
    if min_angle < 5.0:
        return 'aligned'
    elif min_angle < 15.0:  # Strict threshold (was 25.0)
        return 'quasi_aligned'
    else:
        return 'non_aligned'


def create_surfaceFeatureExtractDict(template_path, sim_path, stl_filename):
    input_path = os.path.join(template_path, "system", "surfaceFeatureExtractDict") 
    output_path = os.path.join(sim_path, "system", "surfaceFeatureExtractDict") 
    str_replace_dict = dict()
    str_replace_dict["$STL_FILENAME"] = stl_filename
    replace_in_file(input_path, output_path, str_replace_dict)


def create_blockMeshDict(template_path, sim_path, geo_mesh):
    input_path = os.path.join(template_path, "system", "blockMeshDict") 
    output_path = os.path.join(sim_path, "system", "blockMeshDict") 

    BLOCKMESH_MARGIN = 0.1
    bounds = geo_mesh.bounds
    str_replace_dict = dict()
    str_replace_dict["$XMIN"] = str(bounds[0] - BLOCKMESH_MARGIN)
    str_replace_dict["$XMAX"] = str(bounds[1] + BLOCKMESH_MARGIN)
    str_replace_dict["$YMIN"] = str(bounds[2] - BLOCKMESH_MARGIN)
    str_replace_dict["$YMAX"] = str(bounds[3] + BLOCKMESH_MARGIN)
    str_replace_dict["$ZMIN"] = str(bounds[4] - BLOCKMESH_MARGIN)
    str_replace_dict["$ZMAX"] = str(bounds[5] + BLOCKMESH_MARGIN)
    replace_in_file(input_path, output_path, str_replace_dict)


def generate_location_inside_mesh(mesh):
    """
    Generate locationInMesh point for snappyHexMesh with robust fallback strategy.
    
    Primary strategy:
    1. Generate candidate points from cell centers
    2. Filter out points too close to Z boundaries (floor/ceiling)
    3. Validate points are enclosed using select_enclosed_points
    4. Select point farthest from boundary for robustness
    
    Fallback strategy (if primary fails):
    5. Use geometric center of bounding box
    """
    import logging
    import numpy as np
    
    logger = logging.getLogger(__name__)
    
    try:
        # PRIMARY METHOD - Advanced validation
        bounds = mesh.bounds
        z_min, z_max = bounds[4], bounds[5]
        z_range = z_max - z_min
        
        # Safety margin from boundaries (10% of Z range, minimum 0.05m)
        z_margin = max(0.1 * z_range, 0.05)
        
        logger.info(f"    * Geometry Z bounds: [{z_min:.3f}, {z_max:.3f}], range: {z_range:.3f} m")
        logger.info(f"    * Using Z margin: {z_margin:.3f} m to avoid floor/ceiling boundaries")
        
        # Generate candidate points from cell centers
        cell_centers = mesh.cell_centers()
        volumes = mesh.compute_cell_sizes()["Volume"]
        
        # Get top 20 largest cells as candidates
        n_candidates = min(20, len(volumes))
        top_indices = np.argsort(volumes)[-n_candidates:]
        candidate_points = cell_centers.points[top_indices]
        
        # Filter out points too close to floor/ceiling
        z_coords = candidate_points[:, 2]
        away_from_boundaries = (z_coords > z_min + z_margin) & (z_coords < z_max - z_margin)
        
        if not np.any(away_from_boundaries):
            logger.warning(f"    * All {n_candidates} candidates near boundaries, relaxing Z margin to {z_margin/2:.3f} m")
            z_margin_relaxed = z_margin / 2
            away_from_boundaries = (z_coords > z_min + z_margin_relaxed) & (z_coords < z_max - z_margin_relaxed)
        
        filtered_points = candidate_points[away_from_boundaries]
        n_filtered = len(filtered_points)
        
        logger.info(f"    * Testing {n_filtered} candidate interior points (filtered from {n_candidates})")
        
        if n_filtered == 0:
            logger.warning(f"    * No candidates away from boundaries, using all {n_candidates} candidates")
            filtered_points = candidate_points
            n_filtered = n_candidates
        
        # Create surface mesh and validate enclosure
        surface_mesh = mesh.extract_surface()
        candidate_cloud = pv.PolyData(filtered_points)
        enclosed_mask = candidate_cloud.select_enclosed_points(surface_mesh, tolerance=1e-6, check_surface=True)
        is_inside = enclosed_mask['SelectedPoints'].astype(bool)
        
        logger.info(f"    * Found {np.sum(is_inside)} valid interior points out of {n_filtered} candidates")
        
        if not np.any(is_inside):
            raise Exception("No valid interior points found")
        
        # Select point farthest from boundary AND close to geometric center
        valid_points = filtered_points[is_inside]
        
        # Calculate geometric center as reference
        center_x = (bounds[0] + bounds[1]) / 2.0
        center_y = (bounds[2] + bounds[3]) / 2.0
        center_z = (bounds[4] + bounds[5]) / 2.0
        geometric_center = np.array([center_x, center_y, center_z])
        
        # Score each point: distance from boundary (good) - distance from center (bad)
        scores = []
        for pt in valid_points:
            closest_point = surface_mesh.find_closest_point(pt)
            dist_to_boundary = np.linalg.norm(pt - surface_mesh.points[closest_point])
            dist_to_center = np.linalg.norm(pt - geometric_center)
            
            # Score: prefer points far from boundary but close to center
            # Weight boundary distance more heavily (factor of 3)
            score = 3.0 * dist_to_boundary - dist_to_center
            scores.append(score)
        
        best_idx = np.argmax(scores)
        point = valid_points[best_idx]
        
        # Validate point is reasonably centered
        dist_from_center = np.linalg.norm(point - geometric_center)
        # Use 30% threshold instead of 50% to be more strict
        max_expected_offset = 0.3 * max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
        
        if dist_from_center > max_expected_offset:
            logger.warning(f"    * Selected point far from center ({dist_from_center:.2f}m), using geometric center instead")
            point = geometric_center
        
        logger.info(f"    * Selected interior point: ({point[0]:.6f}, {point[1]:.6f}, {point[2]:.6f})")
        logger.info(f"    * Geometric center: ({center_x:.6f}, {center_y:.6f}, {center_z:.6f})")
        logger.info(f"    * Offset from center: {np.linalg.norm(point - geometric_center):.6f} m")
        
        return f"({point[0]:.6f} {point[1]:.6f} {point[2]:.6f})"
    
    except Exception as e:
        # FALLBACK: Geometric center of bounding box
        logger.warning(f"    ⚠️  Primary method failed: {str(e)}")
        logger.warning(f"    * Using FALLBACK: geometric center of bounding box")
        
        bounds = mesh.bounds
        center_x = (bounds[0] + bounds[1]) / 2.0
        center_y = (bounds[2] + bounds[3]) / 2.0
        center_z = (bounds[4] + bounds[5]) / 2.0
        
        logger.info(f"    * Fallback point: ({center_x:.6f}, {center_y:.6f}, {center_z:.6f})")
        logger.warning(f"    * ⚠️  This may fail if geometry is non-convex or has holes")
        
        return f"({center_x:.6f} {center_y:.6f} {center_z:.6f})"


def generate_refinement_block_with_alignment(geo_mesh, geo_df):
    """
    Generate refinement surfaces with levels based on type, alignment, and area.
    
    Strategy:
    - Aligned BC: level (2 3)
    - Non-aligned BC: level (3 4) - more aggressive
    - Small area (<0.5m²): +1 additional refinement level
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("    * Generating refinement surfaces with alignment-based levels")
    
    blocks = []
    
    for idx, row in geo_df.iterrows():
        patch_id = idx
        patch_name = row['id']
        bc_type = row['type']
        
        # Calculate normal and classify alignment
        normal = calculate_patch_normal(geo_mesh, patch_id)
        alignment = classify_bc_alignment(normal)
        
        # Calculate area if dimensions are available in DataFrame
        if 'width' in geo_df.columns and 'height' in geo_df.columns:
            area = row.get('width', 1.0) * row.get('height', 1.0)
        else:
            # Estimate area from patch surface
            mask = geo_mesh.cell_data["patch_id"] == patch_id
            patch_cells = geo_mesh.extract_cells(mask)
            patch_surface = patch_cells.extract_surface()
            area = patch_surface.area
        
        is_small = area < 0.5  # m²
        
        # Determine refinement levels based on type and alignment
        if bc_type in ['pressure_inlet', 'pressure_outlet']:
            if alignment == 'aligned':
                base_level = 2
                max_level = 3
            else:  # quasi or non-aligned
                base_level = 3
                max_level = 4
            
            # Additional refinement for small areas
            if is_small:
                base_level += 1
                max_level += 1
                logger.info(f"      - {patch_name}: SMALL AREA ({area:.2f}m²) → +1 refinement")
            
            level = f"({base_level} {max_level})"
        else:  # walls
            level = "(0 0)"
        
        logger.info(f"      - {patch_name}: type={bc_type}, alignment={alignment}, area={area:.2f}m², level={level}")
        
        block = f"""            {patch_name}
            {{
                level {level};
                patchInfo {{ type wall; }}
            }}"""
        blocks.append(block)
    
    # Format as OpenFOAM dict
    blocks_str = "\n".join(blocks)
    return f"regions\n{blocks_str}"


def generate_volumetric_refinement_regions(geo_df, geo_mesh):
    """
    Generate volumetric refinement regions with distance-based levels according to alignment.
    
    Strategy:
    - Aligned BC: ((0.08 3) (0.25 2) (0.50 1)) - moderate refinement
    - Non-aligned BC: ((0.05 4) (0.10 3) (0.20 2) (0.40 1)) - aggressive refinement
    """
    import logging
    logger = logging.getLogger(__name__)
    
    pressure_patches = geo_df[geo_df['type'].isin(['pressure_inlet', 'pressure_outlet'])]
    
    if len(pressure_patches) == 0:
        logger.info("    * No pressure boundaries found, skipping volumetric refinement")
        return ""
    
    logger.info(f"    * Creating volumetric refinement regions for {len(pressure_patches)} pressure boundaries")
    
    blocks = []
    for idx, row in pressure_patches.iterrows():
        patch_id = idx
        patch_name = row['id']
        
        # Calculate alignment
        normal = calculate_patch_normal(geo_mesh, patch_id)
        alignment = classify_bc_alignment(normal)
        
        # Refinement according to alignment
        if alignment == 'aligned':
            levels = "((0.08 3) (0.25 2) (0.50 1))"
            logger.info(f"      - {patch_name}: ALIGNED - moderate refinement")
        else:  # non-aligned - MORE AGGRESSIVE
            levels = "((0.05 4) (0.10 3) (0.20 2) (0.40 1))"
            logger.info(f"      - {patch_name}: NON-ALIGNED - aggressive refinement")
        
        block = f"""        {patch_name}_volume
        {{
            mode    distance;
            levels  {levels};
        }}"""
        blocks.append(block)
    
    return "\n".join(blocks)


def generate_boundary_layers_config(geo_df, geo_mesh):
    """
    Generate boundary layer configuration with alignment-based parameters.
    
    Strategy:
    - Pressure boundaries:
      * Non-aligned: 8 layers, expansion 1.8 (fast transition to volume)
      * Aligned: 5 layers, expansion 1.2 (conservative growth)
      * firstLayerThickness: 0.001m
    - Walls: 3 layers, expansion 1.2, firstLayerThickness: 0.002m
    
    Philosophy: High expansion ratio for non-aligned BCs = layers grow fast = 
    rapid transition from orthogonal layers to permissive volume mesh.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("    * Generating boundary layer configuration with alignment-based parameters")
    
    blocks = []
    layer_counts = {}
    
    for idx, row in geo_df.iterrows():
        patch_id = idx
        patch_name = row['id']
        bc_type = row['type']
        
        # Calculate alignment
        normal = calculate_patch_normal(geo_mesh, patch_id)
        alignment = classify_bc_alignment(normal)
        
        # Parameters according to type and alignment
        if bc_type in ['pressure_inlet', 'pressure_outlet']:
            first_layer = 0.001  # 0.001m for pressure BC
            
            if alignment == 'non_aligned':
                n_layers = 8  # More layers for smooth transition
                expansion = 1.8  # HIGH ratio - fast transition to volume
                logger.info(f"      - {patch_name}: pressure BC, NON-ALIGNED → {n_layers} layers, ratio {expansion}")
            else:
                n_layers = 5
                expansion = 1.2  # Conservative ratio for aligned BC
                logger.info(f"      - {patch_name}: pressure BC, ALIGNED → {n_layers} layers, ratio {expansion}")
            
            layer_counts[bc_type] = layer_counts.get(bc_type, 0) + 1
            
        elif bc_type == 'wall':
            first_layer = 0.002  # 0.002m for walls
            n_layers = 3
            expansion = 1.2
            layer_counts[bc_type] = layer_counts.get(bc_type, 0) + 1
        else:
            continue
        
        block = f"""        "{patch_name}"
        {{
            nSurfaceLayers {n_layers};
            firstLayerThickness {first_layer};
            expansionRatio {expansion};
        }}"""
        blocks.append(block)
    
    logger.info(f"    * Boundary layers summary:")
    for bc_type, count in layer_counts.items():
        logger.info(f"      - {bc_type}: {count} patches configured")
    
    return "\n".join(blocks) if blocks else ""


def create_snappyHexMeshDict(template_path, sim_path, stl_filename, geo_mesh, geo_df):
    import logging
    logger = logging.getLogger(__name__)
    
    input_path = os.path.join(template_path, "system", "snappyHexMeshDict") 
    output_path = os.path.join(sim_path, "system", "snappyHexMeshDict") 

    patch_names = geo_df["id"].tolist()
    geometry_regions = generate_regions_block(patch_names)
    emesh_filename = stl_filename.replace(".stl", ".eMesh")
    
    # Use new alignment-based refinement
    refinement_surfaces = generate_refinement_block_with_alignment(geo_mesh, geo_df)
    location_inside_mesh = generate_location_inside_mesh(geo_mesh)
    
    # Generate volumetric refinement regions with alignment
    volumetric_refinement = generate_volumetric_refinement_regions(geo_df, geo_mesh)
    
    # Generate boundary layer configuration with alignment
    boundary_layers = generate_boundary_layers_config(geo_df, geo_mesh)
    
    # Check if we should enable boundary layers
    enable_layers = "true" if boundary_layers else "false"

    str_replace_dict = dict()
    str_replace_dict["$STL_FILENAME"] = stl_filename
    str_replace_dict["$GEOMETRY_REGIONS"] = geometry_regions
    str_replace_dict["$EMESH_FILENAME"] = emesh_filename
    str_replace_dict["$REFINEMENT_SURFACES"] = refinement_surfaces
    str_replace_dict["$LOCATION_INSIDE_MESH"] = location_inside_mesh
    str_replace_dict["$VOLUMETRIC_REFINEMENT"] = volumetric_refinement if volumetric_refinement else "// No volumetric refinement"
    str_replace_dict["$BOUNDARY_LAYERS"] = boundary_layers if boundary_layers else "// No boundary layers configured"
    str_replace_dict["$ENABLE_LAYERS"] = enable_layers
    
    logger.info(f"    * Boundary layers enabled: {enable_layers}")
    if volumetric_refinement:
        logger.info("    * Alignment-based volumetric refinement enabled for pressure boundaries")

    replace_in_file(input_path, output_path, str_replace_dict)


def export_to_stl(geo_mesh_dict, sim_path, stl_filename):
    """
    Write multiple PyVista PolyData solids to one multi-solid ASCII STL file.

    Parameters:
        geo_mesh_dict (dict): Keys are solid names (str), values are pv.PolyData objects.
        stl_path (str): Output STL file path.
    """

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

            # Ensure mesh has triangles
            mesh = mesh.triangulate()

            # Iterate over faces
            faces = mesh.cells.reshape((-1, 4))  # VTK faces: [3, i0, i1, i2] for triangles
            for face in faces:
                assert face[0] == 3  # triangle

                pts = mesh.points[face[1:4]]
                # Calculate normal (can use PyVista or numpy)
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


def validate_mesh_patches(sim_path: str, expected_patches: list[str]) -> None:
    """
    Validate that the mesh only contains user-defined patches.
    Raises exception if background patches like 'limits' remain, indicating meshing failure.
    
    Args:
        sim_path: Path to simulation directory
        expected_patches: List of patch names that should exist (from user config)
        
    Raises:
        Exception: If background patches remain or if meshing failed
    """
    import logging
    from foamlib import FoamCase
    
    logger = logging.getLogger(__name__)
    
    boundary_file = os.path.join(sim_path, "constant", "polyMesh", "boundary")
    if not os.path.exists(boundary_file):
        raise Exception(f"Mesh boundary file not found: {boundary_file}. Meshing may have failed.")
    
    # Read boundary file using FoamCase
    case = FoamCase(sim_path)
    try:
        with case['constant']['polyMesh']['boundary'] as bnd:
            # Get all patches from the mesh
            mesh_patches = {}
            for patch_name in bnd.keys():
                patch_info = bnd[patch_name]
                n_faces = patch_info.get('nFaces', 0)
                mesh_patches[patch_name] = n_faces
            
            logger.info(f"    * Mesh validation: Found {len(mesh_patches)} patches in mesh")
            for patch_name, n_faces in mesh_patches.items():
                logger.info(f"      - {patch_name}: {n_faces} faces")
            
            # Check for background patches that indicate meshing failure
            background_patches = ['limits', 'defaultFaces']
            failed_patches = []
            
            for bg_patch in background_patches:
                if bg_patch in mesh_patches and mesh_patches[bg_patch] > 0:
                    failed_patches.append(f"{bg_patch} ({mesh_patches[bg_patch]} faces)")
            
            if failed_patches:
                error_msg = (
                    f"\n{'='*80}\n"
                    f"MESHING ERROR: SnappyHexMesh failed to cut geometry properly!\n"
                    f"{'='*80}\n"
                    f"Background patches remain in the mesh:\n"
                    f"  {', '.join(failed_patches)}\n\n"
                    f"This indicates that snappyHexMesh could not classify cells correctly.\n"
                    f"Possible causes:\n"
                    f"  1. locationInMesh point is outside or on the boundary of the geometry\n"
                    f"  2. Geometry has holes, gaps, or non-manifold surfaces\n"
                    f"  3. STL file has errors or invalid normals\n\n"
                    f"Expected patches: {', '.join(expected_patches)}\n"
                    f"Actual patches: {', '.join(mesh_patches.keys())}\n"
                    f"{'='*80}\n"
                )
                logger.error(error_msg)
                raise Exception(error_msg)
            
            # Check for unexpected patches (not in user config and not background)
            unexpected = set(mesh_patches.keys()) - set(expected_patches) - set(background_patches)
            if unexpected:
                logger.warning(f"    * Warning: Unexpected patches found: {', '.join(unexpected)}")
            
            logger.info("    * Mesh validation PASSED: No background patches remain")
            
    except Exception as e:
        if "MESHING ERROR" in str(e):
            raise  # Re-raise our custom error
        else:
            raise Exception(f"Failed to read mesh boundary file: {e}")


def split_polydata_by_cell_data(mesh: pv.PolyData, df: pd.DataFrame) -> dict[int, pv.PolyData]:
    patch_names = df[["id"]].to_dict()

    patch_mesh_dict = {}
    for patch_id, patch_name in patch_names['id'].items():
        mask = mesh.cell_data["patch_id"] == patch_id
        submesh = mesh.extract_cells(mask)
        patch_mesh_dict[patch_name] = submesh
    return patch_mesh_dict


def prepare_snappy(geo_mesh, sim_path, geo_df, stl_filename = "geometry.stl"):
    """
    Prepare SnappyHexMesh configuration and scripts for mesh generation.
    
    Args:
        geo_mesh: Geometry mesh to be meshed
        sim_path: Path to simulation directory
        geo_df: DataFrame containing boundary condition information
        stl_filename: Name of the STL file to export
        
    Returns:
        List of script commands for mesh generation
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"    * Preparing SnappyHexMesh configuration for {geo_mesh.n_cells} geometry cells")
    
    logger.info("    * Splitting geometry mesh by boundary condition patches")
    geo_mesh_dict = split_polydata_by_cell_data(geo_mesh, geo_df)
    logger.info(f"    * Split into {len(geo_mesh_dict)} patch meshes")
    
    logger.info(f"    * Exporting geometry to STL format: {stl_filename}")
    export_to_stl(geo_mesh_dict, sim_path, stl_filename)
    
    template_path = os.path.join(os.getcwd(), "data", "settings", "mesh", "snappy")
    logger.info(f"    * Creating SnappyHexMesh configuration files from template: {template_path}")
    
    logger.info("    * Creating surfaceFeatureExtractDict")
    create_surfaceFeatureExtractDict(template_path, sim_path, stl_filename)
    
    logger.info("    * Creating blockMeshDict")
    create_blockMeshDict(template_path, sim_path, geo_mesh)
    
    logger.info("    * Creating snappyHexMeshDict")
    create_snappyHexMeshDict(template_path, sim_path, stl_filename, geo_mesh, geo_df)





    # Get expected patches for validation
    expected_patches = geo_df["id"].tolist()
    expected_patches_str = ", ".join(expected_patches)
    
    script_commands = [
        '#!/bin/sh', 
        'cd "${0%/*}" || exit',
        '. ${WM_PROJECT_DIR:?}/bin/tools/RunFunctions',

        'decompDict="-decomposeParDict system/decomposeParDict"',

        # 1. Extract features for snapping and refinement in snappyHexMesh
        'runApplication surfaceFeatureExtract',

        # 2. This creates a simple background mesh in constant/polyMesh
        'runApplication blockMesh',

        # 3. Run snappyHexMesh in a single core
        'runApplication snappyHexMesh -overwrite',

        # 4. VALIDATE MESH - Fail fast if background patches remain
        'echo "==================== VALIDATING MESH ===================="',
        "python3 << 'VALIDATION_EOF'",
        "import re",
        "",
        "# Read boundary file",
        "with open('constant/polyMesh/boundary', 'r') as f:",
        "    content = f.read()",
        "",
        "# Extract patch names and face counts",
        "patches = {}",
        "lines = content.split('\\n')",
        "current_patch = None",
        "for i, line in enumerate(lines):",
        "    # Match patch name (word followed by newline and opening brace)",
        "    if re.match(r'^\\s+(\\w+)\\s*$', line):",
        "        current_patch = line.strip()",
        "    # Match nFaces",
        "    elif current_patch and 'nFaces' in line:",
        "        match = re.search(r'nFaces\\s+(\\d+)', line)",
        "        if match:",
        "            patches[current_patch] = int(match.group(1))",
        "            current_patch = None",
        "",
        "# Check for background patches that indicate meshing failure",
        "background_patches = ['limits', 'defaultFaces', 'background']",
        f"expected_patches = [{', '.join([repr(p) for p in expected_patches])}]",
        "failed = [(p, patches[p]) for p in patches if p in background_patches and patches[p] > 0]",
        "",
        "if failed:",
        "    print('\\n' + '='*80)",
        "    print('MESHING ERROR: SnappyHexMesh failed to cut geometry properly!')",
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
        "",
        "print('✅ Mesh validation passed - no background patches found')",
        "print(f'Valid patches: {\", \".join(patches.keys())}')",
        "VALIDATION_EOF",
        'echo "==================== MESH VALIDATION PASSED ===================="',

        # 5. Check mesh quality (skewness, aspect ratio, non-orthogonality)
        'runApplication checkMesh',

        # 6. Prepare initial fields
        'rm -rf 0',
        'cp -r 0.orig 0',

        # 7. Check initial conditions before decomposition
        'echo "==================== INITIAL CONDITIONS CHECK ===================="',
        'echo ""',
        'echo "--- Checking h (enthalpy) ---"',
        'grep "internalField" 0/h',
        'echo ""',
        'echo "h boundary conditions:"',
        'grep -A 2 "window_0F_1" 0/h',
        'grep -A 2 "door_0F_1" 0/h',
        'grep -A 2 "window_0F_2" 0/h',
        'echo ""',
        'echo "--- Checking p_rgh (pressure) ---"',
        'grep "internalField" 0/p_rgh',
        'echo ""',
        'echo "p_rgh boundary conditions:"',
        'grep -A 3 "window_0F_1" 0/p_rgh',
        'grep -A 3 "door_0F_1" 0/p_rgh',
        'grep -A 3 "window_0F_2" 0/p_rgh',
        'echo ""',
        'echo "--- Checking U (velocity) ---"',
        'grep "internalField" 0/U',
        'echo ""',
        'echo "U boundary conditions:"',
        'grep -A 3 "window_0F_1" 0/U',
        'grep -A 3 "door_0F_1" 0/U',
        'grep -A 3 "window_0F_2" 0/U',
        'echo ""',
        'echo "--- Checking thermophysicalProperties ---"',
        'grep -A 15 "mixture" constant/thermophysicalProperties',
        'echo ""',
        'echo "==================== END INITIAL CONDITIONS CHECK ===================="',
        'echo ""',

        # 8. Create foam marker file for GUI usage
        'touch results.foam',
        ]
    
    logger.info("    * SnappyHexMesh preparation completed successfully")
    logger.info(f"    * Mesh validation will check for patches: {expected_patches}")
    logger.info("    * Validation will run in Inductiva after meshing completes")
    return script_commands