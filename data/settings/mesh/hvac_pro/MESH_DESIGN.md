# HVAC Professional Mesh Design Strategy
## Design Philosophy

This mesh configuration is designed from scratch for HVAC CFD simulations based on physical principles and industry best practices.

---

## Physical Requirements

### 1. **Jet Resolution from Inlets**
- **Typical HVAC velocity**: 1-5 m/s at inlets
- **Jet spreading angle**: ~15-20° for ventilation
- **Minimum cells across jet diameter**: 10-15 cells
- **Target cell size at inlet**: **0.01m (1cm)** for typical 0.15m vents

### 2. **Boundary Layer Resolution**
- **Thermal boundary layer thickness**: δ_T ≈ 0.05-0.10m for natural convection
- **First cell height**: y+ < 30 for wall functions (RANS)
  - For Re=10^4, u_τ ≈ 0.05 m/s → **y ≈ 0.001m (1mm)**
- **Boundary layer cells**: 5-8 cells minimum
- **Growth ratio**: 1.1-1.2 for smooth transition

### 3. **Thermal Stratification**
- **Vertical gradient**: ΔT/Δz ≈ 1-3°C/m typical
- **Minimum vertical cells**: 15-20 cells/floor height (2.5m)
- **Target vertical resolution**: **0.10-0.15m**

### 4. **Bulk Flow Region**
- **Away from boundaries**: 0.20-0.30m cells acceptable
- **Transition ratio**: Max 1.5 between adjacent cells

---

## Mesh Sizing Strategy

### **Base Mesh (Background)**
```
Domain size: 5m × 5m × 2.5m (typical room)
Base cell size: 0.25m
Background cells: 20 × 20 × 10 = 4,000 cells
```

### **Surface Refinement Levels**

| Boundary Type | Level | Cell Size | Rationale |
|---|---|---|---|
| **Pressure Inlets** (rejillas, windows) | **6** | **0.0039m** (~4mm) | Resolve jet core and velocity gradients |
| **Pressure Outlets** | **5** | **0.0078m** (~8mm) | Capture return flow patterns |
| **Walls** (thermal) | **3** | **0.0312m** (~3cm) | Thermal boundary layer base |
| **Floor/Ceiling** | **2** | **0.0625m** (~6cm) | Vertical stratification |

**Cell size formula**: size = base_size / 2^level

### **Volumetric Refinement Zones**

#### Zone 1: Inlet Jet Core (0-0.3m from inlet)
```
Purpose: Resolve jet expansion and mixing
Refinement level: 5
Cell size: 0.0078m (8mm)
Extent: Cone 30° angle, 0.5m depth
```

#### Zone 2: Near-Field (0.3-1.0m from inlet)
```
Purpose: Capture jet deceleration and entrainment
Refinement level: 4
Cell size: 0.0156m (1.5cm)
```

#### Zone 3: Mid-Field (1.0-2.0m from inlet)
```
Purpose: Track flow development
Refinement level: 3
Cell size: 0.0312m (3cm)
```

### **Boundary Layers**

#### Pressure Boundaries (Inlets/Outlets)
```
Number of layers: 7
First layer thickness: 0.001m (1mm) → y+ ≈ 20-30
Expansion ratio: 1.15 (conservative)
Total thickness: ~0.012m
Coverage target: >95%
```

#### Walls (Thermal)
```
Number of layers: 5
First layer thickness: 0.002m (2mm)
Expansion ratio: 1.2
Total thickness: ~0.020m
Coverage target: >90%
```

---

## Quality Targets

### **Orthogonality**
```
maxNonOrtho: 55 (strict for HVAC)
maxBoundarySkewness: 12
maxInternalSkewness: 2.5
```

### **Aspect Ratio**
```
Near boundaries: < 5:1 (with layers)
Bulk region: < 3:1
Transition zones: < 4:1
```

### **Cell Count Estimate**
```
Small room (20m²): 200,000-400,000 cells
Medium room (50m²): 500,000-800,000 cells
Large space (100m²): 1,000,000-1,500,000 cells
```

---

## Implementation Notes

1. **Adaptive approach**: Start with level 5 on inlets, increase to 6 if convergence issues
2. **Layer coverage**: If <90%, reduce nSurfaceLayers by 1
3. **Parallel efficiency**: Target 10,000-20,000 cells/core for optimal speedup
4. **Memory**: 8GB RAM handles ~500k cells comfortably

---

## Validation Criteria

✅ **Good mesh indicators**:
- Boundary layer coverage >90% on all patches
- No "limits" patch in final mesh
- checkMesh passes with warnings <5%
- Velocity profiles smooth in jet region
- Temperature stratification captured (15+ vertical cells)

❌ **Bad mesh indicators**:
- Sudden jumps in cell size (>2x)
- Boundary layer collapse (<70% coverage)
- Jet requires >20 iterations to converge
- Non-physical recirculation zones

---

## References

- ASHRAE Fundamentals Handbook: Ventilation and air flow patterns
- OpenFOAM Best Practices Guide: snappyHexMesh quality
- Wilcox: Turbulence Modeling for CFD (y+ requirements)
