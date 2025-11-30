"""
Centralized mesher configuration for HVAC CFD pipeline.

Change DEFAULT_MESHER to switch between meshing strategies globally.
"""

# =============================================================================
# MESHER CONFIGURATION
# =============================================================================
DEFAULT_MESHER = "cfmesh"  # Options: "cfmesh" (recommended), "hvac_pro", "snappy"

# MESH QUALITY LEVEL (only for hvac_pro)
# Level 1: Coarse (~50k cells)   - Fast validation, ~2 min meshing ⭐ DEBUG MODE
# Level 2: Medium (~500k cells)  - Production quality, ~5-10 min meshing
# Level 3: Fine (~5M cells)      - Research grade, ~20-30 min meshing
DEFAULT_QUALITY_LEVEL = 1

# =============================================================================
# NOTE: 
#   - "cfmesh" is now the recommended fast automatic mesher (available on Inductiva: inductiva/kutu:openfoam-cfmesh_v2412_dev)
#   - "hvac_pro" is professional HVAC configuration using snappyHexMesh
#   - "snappy" is the basic snappyHexMesh configuration

# Mesher descriptions for documentation
MESHER_INFO = {
    "hvac_pro": {
        "name": "HVAC Professional (snappyHexMesh optimized)",
        "description": "⭐ RECOMMENDED - Professional HVAC mesh from scratch with physics-based sizing",
        "advantages": [
            "Level 6 (4mm cells) on pressure inlets - finest resolution for jet capture",
            "Level 5 (8mm cells) on pressure outlets - fine for return flow",
            "7 boundary layers on pressure boundaries (y+ ≈ 20-30)",
            "5 boundary layers on walls (thermal boundary layer)",
            "Multi-zone volumetric refinement (jet core → near-field → far-field)",
            "Strict quality controls (maxNonOrtho 55, maxSkew 12/2.5)"
        ],
        "best_for": "All HVAC applications - designed from scratch for thermal comfort"
    },
    "cfmesh": {
        "name": "cfMesh (OpenFOAM ESI v2412)",
        "description": "⭐ RECOMMENDED - Fast automatic mesher (available on Inductiva: kutu:openfoam-cfmesh_v2412_dev)",
        "advantages": [
            "Automatic robust boundary layers (>90% coverage)",
            "Differentiated refinement (pressure boundaries 2x finer)",
            "2-5x faster meshing than snappyHexMesh",
            "Single-command workflow (cartesianMesh)",
            "Simpler configuration and setup",
            "Better suited for HVAC with pressure boundaries (windows/doors/vents)"
        ],
        "best_for": "All HVAC applications - fast, robust, and production-quality"
    },
    "snappy": {
        "name": "snappyHexMesh (basic)",
        "description": "Standard OpenFOAM mesher with basic configuration",
        "advantages": [
            "Flexible multi-region meshing",
            "Good for rotating machinery",
            "Mature and well-documented"
        ],
        "best_for": "Multi-region cases or when hvac_pro is too aggressive"
    }
}


def get_default_mesher():
    """Get the configured default mesher."""
    return DEFAULT_MESHER


def get_default_quality_level():
    """Get the configured default mesh quality level."""
    return DEFAULT_QUALITY_LEVEL


def validate_mesher(mesher_type):
    """Validate that the mesher type is supported."""
    if mesher_type not in ["hvac_pro", "cfmesh", "snappy"]:
        raise ValueError(f"Unknown mesher type: {mesher_type}. Must be 'hvac_pro', 'cfmesh', or 'snappy'")
    return mesher_type


def validate_quality_level(quality_level):
    """Validate that the quality level is supported."""
    if quality_level not in [1, 2, 3]:
        raise ValueError(f"Unknown quality level: {quality_level}. Must be 1 (coarse), 2 (medium), or 3 (fine)")
    return quality_level
