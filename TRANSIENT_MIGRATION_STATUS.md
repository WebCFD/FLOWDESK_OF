# Migración a buoyantPimpleFoam Transient - Estado

## FASE 1: Templates hvac_transient/ ✅ COMPLETA

### Archivos creados en `data/settings/cfd/hvac_transient/`:

**constant/**
- ✅ `turbulenceProperties` - kOmegaSST RANS
- ✅ `radiationProperties` - P1 modelo con absorptivity/emissivity
- ✅ `thermophysicalProperties` - Copiado de hvac/
- ✅ `g` - Copiado de hvac/

**system/**
- ✅ `controlDict` - buoyantPimpleFoam, backward, CFL adaptativo (maxCo=0.5), function objects
- ✅ `fvSchemes` - linearUpwind para momento, limitedLinear para energía/turbulencia
- ✅ `fvSolution` - PIMPLE con 3 outer correctors, relajación 0.3/0.7
- ✅ `fvOptions` - Fuentes volumétricas (calor 100W + CO2 0.005kg/s)
- ✅ `topoSetDict` - cellZones para occupants/equipment
- ✅ `setFieldsDict` - Copiado de hvac/

---

## FASE 2: Modificar hvac.py - ✅ COMPLETA

### ✅ 2.1 DIMENSIONS_DICT actualizado (10 campos)
### ✅ 2.2 INTERNALFIELD_DICT actualizado
### ✅ 2.3 Función `define_turbulence_bcs()` creada
### ✅ 2.4 Función `define_radiation_bcs()` creada
### ✅ 2.5 Función `define_scalar_bcs()` creada
### ✅ 2.6 `define_initial_files()` modificada
- Ahora procesa 10 campos: h/p/p_rgh/T/U + k/omega + CO2 + qr/G
- Switch que detecta tipo de campo y llama función apropiada

### ✅ 2.7 `setup()` modificado - COMPLETO
- Parámetro `transient=False` añadido (backward compatible)
- Switch automático de templates: hvac/ vs hvac_transient/
- Solver dinámico en script_commands: buoyantSimpleFoam vs buoyantPimpleFoam

---

## FASE 3-5: Testing - PENDIENTE

**READY TO TEST:** Código funcional, falta validación práctica

**FASE 3:** Test mínimo (flujo + térmica + turbulencia)  
**FASE 4:** Añadir radiación P1  
**FASE 5:** Añadir CO2 + fvOptions

---

## ESTADO CRÍTICO ACTUAL

⚠️ **Código actual NO funcional para transient**

**Razón:**
- Templates existen ✅
- Diccionarios expandidos ✅
- PERO `define_initial_files()` aún NO crea campos k/omega/CO2/qr/G
- SOLO crea: h, p, p_rgh, T, U

**Siguiente paso obligatorio:**
1. Crear funciones BCs (2.3, 2.4, 2.5)
2. Modificar bucle en `define_initial_files()` para usar estas funciones
3. Modificar `setup()` para switch steady/transient

**Tiempo estimado:** 30-45 min

---

## Decisión requerida

¿Continuar con FASE 2.3-2.7 o pausar aquí?

**Opción A:** Continuar ahora → Completar hvac.py funcional  
**Opción B:** Pausar → Revisar templates, testear sintaxis OF primero  
**Opción C:** Simplificar → Solo turbulencia primero (sin radiación/CO2)
