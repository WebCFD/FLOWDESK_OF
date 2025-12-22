# Correcciones en hvac.py para Transient

## INCOHERENCIA 1: Valores de turbulencia corregidos ✅

### Problema original:
```python
'k':        0.375,      # INCORRECTO: 4x mayor que el calculado
'omega':    1.78,       # INCORRECTO: 6x menor que el calculado
```

**Error matemático:**
- k = 1.5×(0.05×5)² = 0.375 ❌ (comentario dice U=5m/s pero debería ser 1m/s)
- ω = √0.375/(0.09^0.25×0.1) = 11.18 ❌ (no coincide con 1.78)

### Corrección aplicada:
```python
# For I=5%, U_ref=1m/s: k = 1.5*(0.05*1)² = 0.00375 m²/s²
# For L=0.1m, C_μ=0.09: ω = √k/(C_μ^0.25*L) = √0.00375/(0.09^0.25*0.1) ≈ 1.12 [1/s]
'k':        0.00375,    # Turbulent kinetic energy [m²/s²]
'omega':    1.12,       # Specific dissipation rate [1/s]
```

**Verificación matemática:**
- k = 1.5×(0.05×1)² = 1.5×0.0025 = 0.00375 ✓
- ω = √0.00375/(0.548×0.1) = 0.0612/0.0548 = 1.12 ✓

---

## INCOHERENCIA 2: setFields condicional para transient ✅

### Problema original:
```python
# setFields se ejecutaba SIEMPRE, tanto en steady como transient
'runApplication setFields',
```

**Error conceptual:**
- setFields aplica gradiente hidrostático a p (no p_rgh)
- buoyantPimpleFoam resuelve p_rgh, no p
- Gradiente hidrostático en p causa inicialización incorrecta para transient

### Correcciones aplicadas:

**1. setFieldsDict específico para transient creado:**

`data/settings/cfd/hvac_transient/system/setFieldsDict`:
```
defaultFieldValues
(
    // No field modifications - keep values from 0.orig/ directory
    // This is intentionally empty for transient cases
);

regions
(
    // No regions defined - keep uniform initial fields
);
```

**Razón:** Transient necesita campos uniformes iniciales (p_rgh=0) para evitar ondas de presión espurias.

**2. Script_commands modificado con condicional:**

```python
# Hydrostatic pressure initialization (ONLY for steady-state)
if not transient:
    script_commands.extend([
        'echo "... APPLYING HYDROSTATIC PRESSURE GRADIENT (STEADY) ..."',
        'runApplication setFields',
        'echo "... HYDROSTATIC PRESSURE INITIALIZED: p(z) = p_atm - rho*g*z ..."',
    ])
else:
    script_commands.extend([
        'echo "... TRANSIENT MODE: Keeping uniform initial fields (no setFields) ..."',
    ])
```

---

## Resumen de cambios

### Archivos modificados:

1. **`src/components/cfd/hvac.py`**
   - Líneas 47-48: Corregidos k=0.00375, omega=1.12
   - Líneas 593-603: Script_commands con condicional transient

2. **`data/settings/cfd/hvac_transient/system/setFieldsDict`** (CREADO)
   - setFieldsDict vacío para transient
   - Documentación del porqué

### Impacto:

**Steady (buoyantSimpleFoam):**
- ✅ Sin cambios - mantiene setFields con gradiente hidrostático
- ✅ Valores k/omega corregidos pero solo relevantes si se añade turbulencia a steady

**Transient (buoyantPimpleFoam):**
- ✅ NO ejecuta setFields → p_rgh permanece uniforme
- ✅ Valores k/omega correctos (I=5%, U_ref=1m/s)
- ✅ Inicialización suave para PIMPLE transient

### Validación matemática:

| Variable | Antes | Después | Fórmula | Verificación |
|----------|-------|---------|---------|--------------|
| k | 0.375 | 0.00375 | 1.5×(I×U)² = 1.5×(0.05×1)² | ✓ Correcto |
| omega | 1.78 | 1.12 | √k/(Cμ^0.25×L) = √0.00375/0.0548 | ✓ Correcto |

---

## Testing requerido

### Caso steady (sin cambios esperados):
```bash
cd /home/plprm/FLOWDESK_OF
python PYTHON_STEPS/step03_mesh2cfd.py
# Verificar que setFields se ejecuta
grep "APPLYING HYDROSTATIC" cases/*/sim/Allrun
```

### Caso transient (nuevos comportamientos):
```python
from src.components.cfd import hvac
hvac.setup(case_path, transient=True)
# Verificar:
# 1. NO ejecuta setFields
# 2. Campos k/omega creados con valores correctos
# 3. buoyantPimpleFoam en Allrun
```

---

## Fecha corrección
21 de diciembre de 2025
