"""
HVAC Professional Meshing Module
=================================

Generates high-quality CFD meshes optimized for HVAC applications.
Design based on physical principles and industry best practices.

Key Features:
- Physics-based cell sizing (jets, boundary layers, stratification)
- Multi-zone volumetric refinement (jet core → near-field → far-field)
- Optimized boundary layers (y+ < 30 for wall functions)
- Strict quality controls for CFD stability
"""

import os
import logging
import numpy as np
import pandas as pd
import pyvista as pv

from src.components.tools.populate_template_file import replace_in_file, generate_regions_block

logger = logging.getLogger(__name__)


# ============================================================================
# MESH QUALITY LEVELS (PARAMETRIC)
# ============================================================================

class MeshQualityLevel:
    """
    Parametric mesh quality configurations for different use cases.
    
    Quality Level 1 (COARSE): ~50k cells
    - Quick validation, fast meshing (~2 min)
    - Basic flow patterns only
    - Not suitable for final results
    
    Quality Level 2 (MEDIUM): ~500k cells ⭐ RECOMMENDED
    - Production quality for most HVAC cases
    - Good balance of accuracy and speed (~5-10 min meshing)
    - Suitable for design iterations
    
    Quality Level 3 (FINE): ~5M cells
    - Research-grade mesh for publication
    - Maximum accuracy, slow meshing (~20-30 min)
    - Use for final validation only
    """
    
    CONFIGS = {
        1: {  # COARSE - ~300-400k cells (conservative isotropic strategy with extended BC zones)
            'name': 'Conservative isotropic (no layers)',
            'description': 'No boundary layers, isotropic refinement, strict quality (maxNonOrtho<65)',
            'base_cell_size': 0.10,  # 10cm base cells (coarse domain)
            'global_level': 0,       # No global refinement (10cm base cells)
            'maxGlobalCells': 1000000,  # Increased from 400k to allow extended volumetric zones (Nov 4, 2025)
            'maxLocalCells': 500000,   # Increased from 200k for extended volumetric zones (Nov 4, 2025)
            'maxLoadUnbalance': 0.25,  # Relaxed from 0.10 to allow extended refinement (Nov 4, 2025)
            'levels': {
                'pressure_inlet': 2,      # 2.5cm surface (reduced refinement)
                'pressure_outlet': 2,     # 2.5cm surface (reduced refinement)
                'wall': 1,                # 5cm (improves rectangular capture, reduces corner degeneracy)
                'floor_ceiling': 0,       # Same as base (no jump)
                'default': 0              # Base level
            },
            'volumetric_zones': [
                # Three-zone progressive isotropic refinement (smooth 2→1→0)
                # Extended distances for better BC capture
                {'distance': 0.35, 'level': 2, 'name': 'bc_core_0_35cm'},     # 0-35cm: 2.5cm cells
                {'distance': 0.45, 'level': 1, 'name': 'bc_near_35_45cm'},    # 35-45cm: 5cm cells
                {'distance': 0.60, 'level': 0, 'name': 'bc_far_45_60cm'},     # 45-60cm: 10cm cells (base)
            ],
            'boundary_layers': None,  # NO LAYERS - isotropic refinement only
            'feature_edge_refinement': {
                'enabled': True,         # Enabled to capture 90° edges (windows/doors)
                'min_level': 1,          # 5cm distance
                'max_level': 2,          # 1cm distance  
                'feature_angle': 30
            },
            'mesh_quality': {
                # PROFESSIONAL CFD quality controls
                'maxNonOrtho': 50,              # ✅ Improved from 65
                'maxBoundarySkewness': 4,       # ✅ Improved from 6
                'maxInternalSkewness': 2,       # ✅ Improved from 4
                'maxConcave': 75,               # Conservative
                'nCellsBetweenLevels': 3,       # SMOOTH transitions
                'minRefinementCells': 5,        # Allow small zones
                # Relaxed fallback (still reasonable)
                'relaxed': {
                    'maxNonOrtho': 60,          # ✅ Improved from 65
                    'maxBoundarySkewness': 6,   # ✅ Improved from 8
                    'maxInternalSkewness': 3    # ✅ Improved from 5
                }
            }
        },
        2: {  # MEDIUM - ~500k cells ⭐ DEFAULT
            'name': 'Medium (production)',
            'description': 'Professional HVAC mesh for design iterations',
            'base_cell_size': 0.25,
            'levels': {
                'pressure_inlet': 6,      # 0.0039m (~4mm)
                'pressure_outlet': 5,     # 0.0078m (~8mm)
                'wall': 3,                # 0.0312m (~3cm)
                'floor_ceiling': 2,       # 0.0625m (~6cm)
                'default': 2
            },
            'volumetric_zones': [
                {'distance': 0.3, 'level': 5, 'name': 'jet_core'},
                {'distance': 1.0, 'level': 4, 'name': 'near_field'},
                {'distance': 2.0, 'level': 3, 'name': 'mid_field'},
            ],
            'boundary_layers': {
                'pressure_inlet': {'nLayers': 7, 'firstLayerThickness': 0.001, 'expansionRatio': 1.15},
                'pressure_outlet': {'nLayers': 7, 'firstLayerThickness': 0.001, 'expansionRatio': 1.15},
                'wall': {'nLayers': 5, 'firstLayerThickness': 0.002, 'expansionRatio': 1.2},
            }
        },
        3: {  # FINE - ~5M cells
            'name': 'Fine (research grade)',
            'description': 'Maximum resolution for publication-quality results',
            'base_cell_size': 0.20,  # Smaller base mesh
            'levels': {
                'pressure_inlet': 8,      # 0.00078m (~0.8mm)
                'pressure_outlet': 7,     # 0.00156m (~1.5mm)
                'wall': 5,                # 0.00625m (~6mm)
                'floor_ceiling': 3,       # 0.025m (~2.5cm)
                'default': 3
            },
            'volumetric_zones': [
                {'distance': 0.2, 'level': 7, 'name': 'jet_core'},
                {'distance': 0.5, 'level': 6, 'name': 'jet_expansion'},
                {'distance': 1.0, 'level': 5, 'name': 'near_field'},
                {'distance': 2.0, 'level': 4, 'name': 'mid_field'},
            ],
            'boundary_layers': {
                'pressure_inlet': {'nLayers': 10, 'firstLayerThickness': 0.0005, 'expansionRatio': 1.1},
                'pressure_outlet': {'nLayers': 10, 'firstLayerThickness': 0.0005, 'expansionRatio': 1.1},
                'wall': {'nLayers': 7, 'firstLayerThickness': 0.001, 'expansionRatio': 1.15},
            }
        }
    }
    
    @staticmethod
    def get_config(quality_level: int):
        """Get configuration for specified quality level"""
        if quality_level not in MeshQualityLevel.CONFIGS:
            raise ValueError(f"Invalid quality level: {quality_level}. Must be 1 (coarse), 2 (medium), or 3 (fine)")
        return MeshQualityLevel.CONFIGS[quality_level]


# ============================================================================
# LEGACY CLASS (for backwards compatibility)
# ============================================================================

class HVACMeshParams:
    """Legacy parameters - now use MeshQualityLevel.get_config(2) instead"""
    
    BASE_CELL_SIZE = 0.25
    LEVELS = MeshQualityLevel.CONFIGS[2]['levels']
    VOLUMETRIC_ZONES = MeshQualityLevel.CONFIGS[2]['volumetric_zones']
    BOUNDARY_LAYERS = MeshQualityLevel.CONFIGS[2]['boundary_layers']


# ============================================================================
# REFINEMENT BLOCK GENERATION
# ============================================================================

def generate_hvac_refinement_surfaces(geo_df: pd.DataFrame, quality_level: int = 2) -> str:
    """
    Generate surface refinement configuration based on boundary condition types.
    
    Args:
        geo_df: DataFrame with boundary condition information
        quality_level: 1 (coarse ~50k), 2 (medium ~500k), 3 (fine ~5M cells)
    
    Uses physics-based refinement levels based on quality setting.
    """
    config = MeshQualityLevel.get_config(quality_level)
    levels = config['levels']
    base_size = config['base_cell_size']
    
    logger.info("=" * 80)
    logger.info(f"HVAC SURFACE REFINEMENT - Quality Level {quality_level}: {config['name']}")
    logger.info("=" * 80)
    logger.info(f"  Base cell size: {base_size}m")
    logger.info(f"  Target cells: {config['description']}")
    logger.info("")
    
    blocks = []
    refinement_summary = {}
    
    for idx, row in geo_df.iterrows():
        patch_name = row['id']
        bc_type = row['type']
        
        # Determine refinement level based on BC type
        if bc_type == 'pressure_inlet':
            level = levels['pressure_inlet']
            cell_size = base_size / (2 ** level)
            patch_type = 'patch'
        elif bc_type == 'pressure_outlet':
            level = levels['pressure_outlet']
            cell_size = base_size / (2 ** level)
            patch_type = 'patch'
        elif bc_type == 'wall':
            level = levels['wall']
            cell_size = base_size / (2 ** level)
            patch_type = 'wall'
        else:
            level = levels['default']
            cell_size = base_size / (2 ** level)
            patch_type = 'patch'
        
        # Generate refinement block with correct structure
        # For level 0, use (0 0) to avoid negative min level
        min_level = max(0, level - 1)
        block = f"""{patch_name}
                {{
                    level ({min_level} {level});
                    patchInfo {{ type {patch_type}; }}
                }}"""
        blocks.append(block)
        
        # Track refinement summary
        key = f"{bc_type} (level {level})"
        refinement_summary[key] = refinement_summary.get(key, 0) + 1
        
        logger.info(f"  {patch_name:20s} → level {level} ({cell_size*1000:.1f}mm cells)")
    
    # Log summary
    logger.info("-" * 80)
    logger.info("Refinement Summary:")
    for config, count in sorted(refinement_summary.items()):
        logger.info(f"  {config}: {count} patches")
    logger.info("=" * 80)
    
    # Format as OpenFOAM regions block
    if blocks:
        indent = " " * 12
        formatted_blocks = [indent + block for block in blocks]
        return "regions\n" + indent + "{\n" + "\n".join(formatted_blocks) + "\n" + indent + "}"
    else:
        return ""


# ============================================================================
# VOLUMETRIC REFINEMENT
# ============================================================================

def generate_hvac_volumetric_refinement(geo_df: pd.DataFrame, quality_level: int = 2) -> tuple:
    """
    Generate multi-zone volumetric refinement around pressure boundaries.
    
    Args:
        geo_df: DataFrame with boundary condition information
        quality_level: 1 (coarse), 2 (medium), 3 (fine)
    
    Returns:
        tuple: (geometry_surfaces, refinement_regions)
            - geometry_surfaces: searchableSurfaces for geometry{} section
            - refinement_regions: refinementRegions configuration
    
    Creates graduated refinement zones to capture jet development.
    """
    config = MeshQualityLevel.get_config(quality_level)
    volumetric_zones = config['volumetric_zones']
    base_size = config['base_cell_size']
    
    pressure_patches = geo_df[geo_df['type'].isin(['pressure_inlet', 'pressure_outlet'])]
    
    if len(pressure_patches) == 0:
        logger.info("  * No pressure boundaries → volumetric refinement disabled")
        return ("", "    // No pressure boundaries for volumetric refinement")
    
    logger.info("=" * 80)
    logger.info(f"HVAC VOLUMETRIC REFINEMENT - Quality Level {quality_level}")
    logger.info("=" * 80)
    logger.info(f"  Creating multi-zone refinement for {len(pressure_patches)} pressure boundaries")
    logger.info("")
    
    # Log zone configuration
    for zone in volumetric_zones:
        cell_size = base_size / (2 ** zone['level'])
        logger.info(f"  Zone '{zone['name']}': 0-{zone['distance']}m → level {zone['level']} ({cell_size*1000:.1f}mm cells)")
    
    logger.info("")
    
    # Generate searchableSurfaces for geometry{} section
    geometry_blocks = []
    refinement_blocks = []
    
    for idx, row in pressure_patches.iterrows():
        patch_name = row['id']
        bc_type = row['type']
        
        # Create searchableSurface that references the STL region
        geometry_block = f"""    {patch_name}
    {{
        type triSurfaceMesh;
        file geometry.stl;
        regions
        {{
            {patch_name} {{ name {patch_name}; }}
        }}
    }}"""
        geometry_blocks.append(geometry_block)
        
        # Build distance-based refinement levels
        level_spec = " ".join([f"({z['distance']} {z['level']})" for z in volumetric_zones])
        
        # Build refinementRegions entry (referencing the searchableSurface)
        # Indentation: 8 spaces (directly inside refinementRegions{}, no geometry{} wrapper)
        refinement_block = f"""        {patch_name}
        {{
            mode    distance;
            levels  ({level_spec});
        }}"""
        refinement_blocks.append(refinement_block)
        
        logger.info(f"  {patch_name} ({bc_type}): multi-zone refinement enabled")
    
    logger.info("=" * 80)
    
    # Format output
    geometry_content = "\n".join(geometry_blocks) if geometry_blocks else ""
    refinement_content = "\n".join(refinement_blocks) if refinement_blocks else "        // No volumetric refinement"
    
    return (geometry_content, refinement_content)


# ============================================================================
# BOUNDARY LAYERS
# ============================================================================

def generate_hvac_boundary_layers(geo_df: pd.DataFrame, quality_level: int = 2) -> str:
    """
    Generate optimized boundary layer configuration.
    
    Args:
        geo_df: DataFrame with boundary condition information
        quality_level: 1 (coarse), 2 (medium), 3 (fine)
    
    Layer sizing based on y+ requirements, scaled by quality level.
    Returns empty string if boundary_layers is None (no layers strategy).
    """
    config = MeshQualityLevel.get_config(quality_level)
    boundary_layers = config['boundary_layers']
    
    # Handle None case (no layers strategy)
    if boundary_layers is None:
        logger.info("=" * 80)
        logger.info(f"HVAC BOUNDARY LAYERS - Quality Level {quality_level}")
        logger.info("=" * 80)
        logger.info("  ⚠️  Boundary layers DISABLED (isotropic refinement strategy)")
        logger.info("  Strategy: No layers → isotropic cells → natural orthogonality")
        logger.info("=" * 80)
        return "        // No boundary layers (isotropic refinement strategy)"
    
    logger.info("=" * 80)
    logger.info(f"HVAC BOUNDARY LAYERS - Quality Level {quality_level}")
    logger.info("=" * 80)
    
    blocks = []
    layer_summary = {}
    
    for idx, row in geo_df.iterrows():
        patch_name = row['id']
        bc_type = row['type']
        
        # Get layer configuration for this BC type
        if bc_type in boundary_layers:
            layer_config = boundary_layers[bc_type]
        elif bc_type == 'wall':
            layer_config = boundary_layers['wall']
        else:
            continue  # No layers for this BC type
        
        # Generate layer block with nSurfaceLayers + firstLayerThickness + expansionRatio
        # CRITICAL: OpenFOAM v2406 requires exactly TWO thickness parameters
        # We use: firstLayerThickness + expansionRatio (most common combination)
        # Check if using relative or absolute sizing
        use_relative = layer_config.get('relativeSizes', True)
        
        if use_relative:
            # Relative sizing: 0.25 means 25% of local cell size
            first_layer = 0.25
        else:
            # Absolute sizing: use actual thickness in meters
            first_layer = layer_config['firstLayerThickness']
        
        block = f"""        "{patch_name}"
        {{
            nSurfaceLayers {layer_config['nLayers']};
            firstLayerThickness {first_layer};
            expansionRatio {layer_config['expansionRatio']};
        }}"""
        blocks.append(block)
        
        # Track summary
        key = f"{bc_type} ({layer_config['nLayers']} layers, {layer_config['firstLayerThickness']*1000:.1f}mm first)"
        layer_summary[key] = layer_summary.get(key, 0) + 1
        
        logger.info(f"  {patch_name:20s} → {layer_config['nLayers']} layers (first: {layer_config['firstLayerThickness']*1000:.1f}mm)")
    
    # Log summary
    logger.info("-" * 80)
    logger.info("Boundary Layer Summary:")
    for config, count in sorted(layer_summary.items()):
        logger.info(f"  {config}: {count} patches")
    logger.info("  Target coverage: >95% on pressure boundaries, >90% on walls")
    logger.info("=" * 80)
    
    return "\n".join(blocks) if blocks else "        // No boundary layers configured"


# ============================================================================
# MAIN CONFIGURATION FUNCTION
# ============================================================================

def create_hvac_pro_snappyHexMeshDict(template_path, sim_path, stl_filename, geo_mesh, geo_df, quality_level: int = 2):
    """
    Create professional HVAC snappyHexMeshDict from scratch.
    
    Args:
        template_path: Path to template directory
        sim_path: Path to simulation directory
        stl_filename: STL geometry filename
        geo_mesh: PyVista mesh for locationInMesh
        geo_df: DataFrame with boundary condition information
        quality_level: 1 (uniform 10mm), 2 (medium ~500k), 3 (fine ~5M cells)
    """
    from src.components.mesh.snappy import generate_location_inside_mesh
    
    config = MeshQualityLevel.get_config(quality_level)
    
    input_path = os.path.join(template_path, "system", "snappyHexMeshDict")
    output_path = os.path.join(sim_path, "system", "snappyHexMeshDict")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"GENERATING HVAC MESH - Quality Level {quality_level}: {config['name']}")
    logger.info("=" * 80)
    logger.info(f"  {config['description']}")
    logger.info("")
    
    # Generate configuration blocks
    geometry_regions = generate_regions_block(geo_df["id"].tolist())
    emesh_filename = stl_filename.replace(".stl", ".eMesh")
    refinement_surfaces = generate_hvac_refinement_surfaces(geo_df, quality_level)
    volumetric_geometry, volumetric_refinement = generate_hvac_volumetric_refinement(geo_df, quality_level)
    boundary_layers = generate_hvac_boundary_layers(geo_df, quality_level)
    location_inside_mesh = generate_location_inside_mesh(geo_mesh)
    
    # Feature edge refinement configuration
    if quality_level == 1 and 'feature_edge_refinement' in config:
        # Level 1: Check if feature refinement is enabled
        feature_config = config['feature_edge_refinement']
        
        if feature_config.get('enabled', False):
            # Feature edge refinement enabled with HVAC-realistic distances
            feature_refinement_levels = f"            levels ((0.01 {feature_config['max_level']}) (0.05 {feature_config['min_level']}));"
            feature_angle = str(feature_config['feature_angle'])
            
            logger.info(f"  Feature edge refinement: 0-1cm → level {feature_config['max_level']}, 1-5cm → level {feature_config['min_level']}")
        else:
            # Feature edge refinement disabled
            feature_refinement_levels = "            levels ((0.001 0));"
            feature_angle = "30"
            logger.info(f"  Feature edge refinement: DISABLED (isotropic strategy)")
    else:
        # Level 2/3: Standard feature refinement
        feature_refinement_levels = "            levels ((0.001 6) (0.005 4) (0.020 2));  // Fine feature resolution"
        feature_angle = "25"
    
    # Enable layers if any boundary layers configured (handle None case)
    boundary_layers_config = config.get('boundary_layers')
    add_layers = "true" if (boundary_layers_config is not None and len(boundary_layers_config) > 0) else "false"
    
    # Global refinement level
    global_level = str(config.get('global_level', 1))
    
    # Cell count limits
    max_global_cells = str(config.get('maxGlobalCells', 5000000))
    max_local_cells = str(config.get('maxLocalCells', 200000))
    max_load_unbalance = str(config.get('maxLoadUnbalance', 0.10))
    
    # Mesh quality parameters (from config or defaults)
    mesh_quality = config.get('mesh_quality', {})
    n_cells_between = str(mesh_quality.get('nCellsBetweenLevels', 2))
    min_refinement_cells = str(mesh_quality.get('minRefinementCells', 0))
    max_non_ortho = str(mesh_quality.get('maxNonOrtho', 55))
    max_boundary_skew = str(mesh_quality.get('maxBoundarySkewness', 12))
    max_internal_skew = str(mesh_quality.get('maxInternalSkewness', 2.5))
    max_concave = str(mesh_quality.get('maxConcave', 70))
    
    # Relaxed quality controls (fallback - MORE permissive than main)
    relaxed_quality = mesh_quality.get('relaxed', {})
    max_non_ortho_relaxed = str(relaxed_quality.get('maxNonOrtho', int(max_non_ortho) + 10))
    max_boundary_skew_relaxed = str(relaxed_quality.get('maxBoundarySkewness', int(max_boundary_skew) + 10))
    max_internal_skew_relaxed = str(relaxed_quality.get('maxInternalSkewness', float(max_internal_skew) + 1.5))
    
    # Boundary layer parameters (detect if using relative or absolute sizing)
    # Check first boundary layer config to determine sizing mode
    # Handle None case for boundary_layers
    if boundary_layers_config is not None and len(boundary_layers_config) > 0:
        first_bc_config = next(iter(boundary_layers_config.values()))
        use_relative = first_bc_config.get('relativeSizes', True)
    else:
        use_relative = True
    relative_sizes = "true" if use_relative else "false"
    
    # Layer controls (from config or defaults)
    # Use empty dict if layer_controls not present (for no-layers strategy)
    layer_controls = config.get('layer_controls', {})
    layer_feature_angle = str(layer_controls.get('featureAngle', 75))
    layer_n_relax_iter = str(layer_controls.get('nRelaxIter', 12))
    layer_n_smooth_normals = str(layer_controls.get('nSmoothNormals', 8))
    layer_max_face_thickness = str(layer_controls.get('maxFaceThicknessRatio', 0.6))
    layer_max_thickness_medial = str(layer_controls.get('maxThicknessToMedialRatio', 0.6))
    layer_min_thickness = str(layer_controls.get('minThickness', 0.1 if use_relative else 0.0005))
    layer_min_medial_angle = str(layer_controls.get('minMedialAxisAngle', 80))
    layer_n_layer_iter = str(layer_controls.get('nLayerIter', 150))
    layer_n_relaxed_iter = str(layer_controls.get('nRelaxedIter', 75))
    
    # Global fallback parameters for addLayersControls (required by OpenFOAM v2406)
    # Use conservative defaults from wall configuration
    if boundary_layers_config is not None and 'wall' in boundary_layers_config:
        wall_config = boundary_layers_config['wall']
        if use_relative:
            global_first_layer = "0.25"
        else:
            global_first_layer = str(wall_config['firstLayerThickness'])
        global_expansion = str(wall_config['expansionRatio'])
    else:
        # Ultimate fallback if no boundary layers configured
        global_first_layer = "0.002" if not use_relative else "0.25"
        global_expansion = "1.2"
    
    # Replace placeholders
    str_replace_dict = {
        "$STL_FILENAME": stl_filename,
        "$GEOMETRY_REGIONS": geometry_regions,
        "$VOLUMETRIC_GEOMETRY": volumetric_geometry,
        "$EMESH_FILENAME": emesh_filename,
        "$FEATURE_REFINEMENT_LEVELS": feature_refinement_levels,
        "$FEATURE_ANGLE": feature_angle,
        "$REFINEMENT_SURFACES": refinement_surfaces,
        "$VOLUMETRIC_REFINEMENT": volumetric_refinement,
        "$BOUNDARY_LAYERS": boundary_layers,
        "$LOCATION_INSIDE_MESH": location_inside_mesh,
        "$ADD_LAYERS": add_layers,
        "$GLOBAL_LEVEL": global_level,
        "$MAX_GLOBAL_CELLS": max_global_cells,
        "$MAX_LOCAL_CELLS": max_local_cells,
        "$MAX_LOAD_UNBALANCE": max_load_unbalance,
        "$NCELLS_BETWEEN_LEVELS": n_cells_between,
        "$MIN_REFINEMENT_CELLS": min_refinement_cells,
        "$MAX_NON_ORTHO": max_non_ortho,
        "$MAX_BOUNDARY_SKEWNESS": max_boundary_skew,
        "$MAX_INTERNAL_SKEWNESS": max_internal_skew,
        "$MAX_CONCAVE": max_concave,
        "$MAX_NON_ORTHO_RELAXED": max_non_ortho_relaxed,
        "$MAX_BOUNDARY_SKEWNESS_RELAXED": max_boundary_skew_relaxed,
        "$MAX_INTERNAL_SKEWNESS_RELAXED": max_internal_skew_relaxed,
        "$RELATIVE_SIZES": relative_sizes,
        "$LAYER_FEATURE_ANGLE": layer_feature_angle,
        "$LAYER_N_RELAX_ITER": layer_n_relax_iter,
        "$LAYER_N_SMOOTH_NORMALS": layer_n_smooth_normals,
        "$LAYER_MAX_FACE_THICKNESS_RATIO": layer_max_face_thickness,
        "$LAYER_MAX_THICKNESS_TO_MEDIAL_RATIO": layer_max_thickness_medial,
        "$LAYER_MIN_THICKNESS": layer_min_thickness,
        "$LAYER_MIN_MEDIAL_AXIS_ANGLE": layer_min_medial_angle,
        "$LAYER_N_LAYER_ITER": layer_n_layer_iter,
        "$LAYER_N_RELAXED_ITER": layer_n_relaxed_iter,
        "$GLOBAL_EXPANSION_RATIO": global_expansion,
        "$GLOBAL_FIRST_LAYER_THICKNESS": global_first_layer,
    }
    
    replace_in_file(input_path, output_path, str_replace_dict)
    
    logger.info("")
    logger.info("✓ HVAC mesh configuration generated successfully")
    logger.info(f"  Quality level: {quality_level} ({config['name']})")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Boundary layers: {add_layers}")
    logger.info("=" * 80)
    logger.info("")
