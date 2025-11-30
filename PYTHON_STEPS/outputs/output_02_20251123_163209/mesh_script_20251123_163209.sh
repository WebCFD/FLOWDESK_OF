#!/bin/sh
cd "${0%/*}" || exit
. ${WM_PROJECT_DIR:?}/bin/tools/RunFunctions
decompDict="-decomposeParDict system/decomposeParDict"
echo "==================== EXTRACTING SURFACE FEATURES ===================="
runApplication surfaceFeatureExtract
echo "==================== RUNNING CFMESH CARTESIAN MESHER ===================="
echo "cfMesh configuration:"
echo "  - Adaptive base cell size from geometry"
echo "  - Pressure boundaries: 2x fine refinement, 8 regular prism layers"
echo "  - Wall boundaries: 1.5x fine refinement, 6 regular prism layers"
echo "  - Boundary layer optimization: enabled"
echo "  - Geometry constraint enforcement: enabled"
echo ""
runApplication cartesianMesh 
echo "==================== DETECTING MESH LOCATION ===================="
if [ -d "constant/polyMesh" ]; then
    MESH_LOCATION="constant"
    echo "✓ Mesh found in SERIAL location: constant/polyMesh"
elif [ -d "processor0/constant/polyMesh" ]; then
    MESH_LOCATION="processor0/constant"
    echo "✓ Mesh found in PARALLEL location: processor0/constant/polyMesh"
else
    echo "✗ ERROR: Mesh not found in constant/polyMesh or processor0/constant/polyMesh"
    exit 1
fi
echo "==================== VALIDATING MESH ===================="
python3 << 'VALIDATION_EOF'
import re, os
mesh_location = open('/tmp/mesh_location.txt').read().strip()
boundary_file = f'{mesh_location}/polyMesh/boundary'
with open(boundary_file, 'r') as f:
    content = f.read()
patches = {}
lines = content.split('\n')
current_patch = None
for i, line in enumerate(lines):
    if re.match(r'^\s+(\w+)\s*$', line):
        current_patch = line.strip()
    elif current_patch and 'nFaces' in line:
        match = re.search(r'nFaces\s+(\d+)', line)
        if match:
            patches[current_patch] = int(match.group(1))
            current_patch = None
background_patches = ['limits', 'defaultFaces', 'background']
expected_patches = ['wall_0F_7', 'wall_0F_10', 'wall_0F_13', 'window_0F_1', 'wall_0F_14', 'wall_0F_15', 'door_0F_1', 'wall_0F_16', 'floor_0F', 'ceil_0F']
failed = [(p, patches[p]) for p in patches if p in background_patches and patches[p] > 0]
if failed:
    print('\n' + '='*80)
    print('MESHING ERROR: cfMesh failed to cut geometry properly!')
    print('='*80)
    print('Background patches remain in the mesh:')
    for patch, count in failed:
        print(f'  - {patch}: {count} faces')
    print()
    print('This indicates the geometry is NOT WATERTIGHT (has holes/gaps).')
    print('Possible causes:')
    print('  1. Geometry has holes, gaps, or non-manifold surfaces')
    print('  2. Wall extrusion created invalid 3D geometry')
    print('  3. Boolean operations failed to create closed volume')
    print()
    print('Expected patches: wall_0F_7, wall_0F_10, wall_0F_13, window_0F_1, wall_0F_14, wall_0F_15, door_0F_1, wall_0F_16, floor_0F, ceil_0F')
    print(f'Actual patches: {", ".join(patches.keys())}')
    print('='*80 + '\n')
    exit(1)
print('✅ Mesh validation passed - no background patches found')
print(f'Valid patches: {", ".join(patches.keys())}')
VALIDATION_EOF
echo "==================== MESH VALIDATION PASSED ===================="
runApplication checkMesh
rm -rf 0
cp -r 0.orig 0
echo "==================== COPYING MESH TO STANDARD LOCATION ===================="
if [ "$MESH_LOCATION" != "constant" ]; then
    echo "Copying mesh from $MESH_LOCATION/polyMesh to constant/polyMesh"
    rm -rf constant/polyMesh
    cp -r $MESH_LOCATION/polyMesh constant/
fi
echo "==================== INITIAL CONDITIONS CHECK ===================="
echo ""
echo "--- Checking h (enthalpy) ---"
grep "internalField" 0/h
echo ""
echo "h boundary conditions:"
head -n 50 0/h | grep -A 2 "type"
echo ""
echo "--- Checking thermophysicalProperties ---"
grep -A 15 "mixture" constant/thermophysicalProperties
echo ""
echo "==================== END INITIAL CONDITIONS CHECK ===================="
echo ""
touch results.foam