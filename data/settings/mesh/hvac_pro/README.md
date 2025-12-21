# HVAC Professional Mesh Configuration

## 📊 Parametric Quality Levels

This meshing system provides **3 quality levels** for different use cases, from quick validation to research-grade meshes.

### Quality Level Comparison

| Parameter | Level 1 (Coarse) | Level 2 (Medium) ⭐ | Level 3 (Fine) |
|---|---|---|---|
| **Target Cells** | ~50,000 | ~500,000 | ~5,000,000 |
| **Use Case** | Quick validation | Production design | Research/Publication |
| **Mesh Time** | ~2 min | ~5-10 min | ~20-30 min |
| **CFD Time** | ~5 min | ~15-30 min | ~2-4 hours |
| **Accuracy** | Basic flow patterns | Production quality | Maximum precision |
| **Best For** | Rapid iterations | Most HVAC projects | Final validation |

---

## 🔬 Detailed Configuration

### Level 1 - COARSE (~50k cells)

**Purpose**: Fast validation of flow patterns during early design

**Cell Sizes**:
```
Base mesh:         0.30m
Pressure inlets:   ~2cm (level 4)
Pressure outlets:  ~4cm (level 3)
Walls:            ~7cm (level 2)
Floor/ceiling:     15cm (level 1)
```

**Boundary Layers**:
```
Pressure boundaries: 3 layers, first 5mm
Walls:              2 layers, first 10mm
```

**Volumetric Refinement**:
```
0-0.5m from inlets:    Level 3
0.5-1.5m from inlets:  Level 2
```

**When to Use**:
- Early design exploration
- Rough flow pattern visualization
- Testing geometry changes quickly
- NOT suitable for thermal comfort analysis

---

### Level 2 - MEDIUM (~500k cells) ⭐ RECOMMENDED

**Purpose**: Production-quality mesh for design iterations and client presentations

**Cell Sizes**:
```
Base mesh:         0.25m
Pressure inlets:   ~4mm (level 6)  ← Jet resolution
Pressure outlets:  ~8mm (level 5)  ← Return flow
Walls:            ~3cm (level 3)  ← Thermal BL
Floor/ceiling:     ~6cm (level 2)  ← Stratification
```

**Boundary Layers**:
```
Pressure boundaries: 7 layers, first 1mm (y+ ≈ 20-30)
Walls:              5 layers, first 2mm (thermal BL)
```

**Volumetric Refinement**:
```
0-0.3m:   Level 5 (jet core expansion)
0.3-1.0m: Level 4 (near-field deceleration)
1.0-2.0m: Level 3 (mid-field development)
```

**When to Use**:
- Design iterations and optimization
- Client presentations and reports
- Building permit submissions
- Most professional HVAC projects

---

### Level 3 - FINE (~5M cells)

**Purpose**: Research-grade mesh for publication and maximum accuracy

**Cell Sizes**:
```
Base mesh:         0.20m
Pressure inlets:   ~0.8mm (level 8)  ← Ultra-fine jets
Pressure outlets:  ~1.5mm (level 7)
Walls:            ~6mm (level 5)
Floor/ceiling:     ~2.5cm (level 3)
```

**Boundary Layers**:
```
Pressure boundaries: 10 layers, first 0.5mm (y+ ≈ 10-15)
Walls:               7 layers, first 1mm
```

**Volumetric Refinement**:
```
0-0.2m:   Level 7 (ultra-fine jet core)
0.2-0.5m: Level 6 (jet expansion)
0.5-1.0m: Level 5 (near-field)
1.0-2.0m: Level 4 (mid-field)
```

**When to Use**:
- Research publications
- Validation against experimental data
- Critical installations (hospitals, cleanrooms)
- Final verification before construction

---

## 🎛️ How to Select Quality Level

### Option 1: Configuration File (Recommended)

Edit `mesher_config.py`:
```python
DEFAULT_QUALITY_LEVEL = 2  # Change to 1, 2, or 3
```

### Option 2: Runtime Parameter

Pass `quality_level` to the meshing function:
```python
from step02_geo2mesh import run

run(case_name, geo_mesh, geo_df, 
    type="hvac_pro", 
    quality_level=2)  # 1, 2, or 3
```

---

## 📈 Performance Guidelines

### Small Room (20m²)
| Level | Cells | Mesh Time | CFD Time (30 iter) | Total |
|---|---|---|---|---|
| 1 | 40k | 2 min | 5 min | **7 min** |
| 2 | 300k | 7 min | 20 min | **27 min** |
| 3 | 3M | 25 min | 2 hours | **~2.5 hours** |

### Medium Room (50m²)
| Level | Cells | Mesh Time | CFD Time (30 iter) | Total |
|---|---|---|---|---|
| 1 | 60k | 3 min | 8 min | **11 min** |
| 2 | 500k | 10 min | 30 min | **40 min** |
| 3 | 5M | 30 min | 4 hours | **~4.5 hours** |

### Large Space (100m²)
| Level | Cells | Mesh Time | CFD Time (30 iter) | Total |
|---|---|---|---|---|
| 1 | 80k | 4 min | 12 min | **16 min** |
| 2 | 800k | 15 min | 45 min | **1 hour** |
| 3 | 8M | 40 min | 8 hours | **~9 hours** |

*Times are estimates for Inductiva cloud with 4 cores*

---

## ✅ Quality Validation

### Good Mesh Indicators (All Levels)
- ✓ Boundary layer coverage >85%
- ✓ No "limits" patch in final mesh
- ✓ checkMesh passes with <5% warnings
- ✓ Smooth cell transitions (<2x size ratio)
- ✓ Velocity profiles converge smoothly

### Level-Specific Validation

**Level 1 (Coarse)**:
- Acceptable if: Flow directions are correct
- Warning if: Temperature gradients poorly resolved
- Reject if: Boundary layer coverage <70%

**Level 2 (Medium)**:
- Acceptable if: Thermal comfort metrics stable
- Warning if: Small recirculation zones missed
- Reject if: Jet spreading angle differs >10%

**Level 3 (Fine)**:
- Acceptable if: Matches experimental data
- Warning if: Any mesh quality metric fails
- Reject if: Boundary layer coverage <95%

---

## 🚀 Quick Start

1. **Choose your quality level** (edit `mesher_config.py`):
   ```python
   DEFAULT_QUALITY_LEVEL = 2  # Start with level 2
   ```

2. **Run simulation** - The system automatically uses the configured level

3. **Check results** - Verify mesh quality in logs

4. **Iterate**:
   - If mesh is too slow: Try level 1
   - If results are questionable: Try level 3
   - If results are good: Stick with level 2

---

## 📝 Notes

- **Level 2 is recommended** for 95% of HVAC projects
- **Level 1** for rapid design exploration only
- **Level 3** for critical projects or publications
- All levels use **physics-based sizing** (not arbitrary)
- Cell counts scale with room volume

---

## 🔗 References

- ASHRAE Fundamentals Handbook (Ventilation chapter)
- Wilcox: Turbulence Modeling for CFD (y+ requirements)
- OpenFOAM Best Practices Guide (mesh quality)
- ISO 7730 (Thermal comfort simulation requirements)
