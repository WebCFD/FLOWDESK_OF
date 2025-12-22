#!/bin/bash

# Script para actualizar casos existentes de OpenFOAM v2012 a v2412
# Actualiza todos los scripts Allrun en cases/*/sim/

echo "========================================================================"
echo "ACTUALIZANDO CASOS EXISTENTES: OpenFOAM v2012 → v2412"
echo "========================================================================"
echo ""

OLD_PATH="/home/plprm/OpenFOAM/OpenFOAM-v2012/etc/bashrc"
NEW_PATH="/usr/lib/openfoam/openfoam2412/etc/bashrc"

# Buscar todos los archivos Allrun en cases/*/sim/
ALLRUN_FILES=$(find cases/*/sim/Allrun 2>/dev/null)

if [ -z "$ALLRUN_FILES" ]; then
    echo "⚠️  No se encontraron archivos Allrun en cases/*/sim/"
    echo "   No hay casos existentes para actualizar."
    exit 0
fi

COUNT=0
UPDATED=0

for FILE in $ALLRUN_FILES; do
    COUNT=$((COUNT + 1))
    
    # Verificar si el archivo contiene la ruta antigua
    if grep -q "$OLD_PATH" "$FILE"; then
        echo "🔄 Actualizando: $FILE"
        
        # Crear backup
        cp "$FILE" "${FILE}.backup_v2012"
        echo "   ✓ Backup creado: ${FILE}.backup_v2012"
        
        # Reemplazar ruta antigua por nueva
        sed -i "s|$OLD_PATH|$NEW_PATH|g" "$FILE"
        echo "   ✓ Ruta actualizada: v2012 → v2412"
        
        UPDATED=$((UPDATED + 1))
    else
        echo "✓ Ya actualizado: $FILE"
    fi
done

echo ""
echo "========================================================================"
echo "RESUMEN"
echo "========================================================================"
echo "   Total de archivos Allrun: $COUNT"
echo "   Actualizados: $UPDATED"
echo "   Ya actualizados: $((COUNT - UPDATED))"
echo ""
echo "✅ Actualización completada"
echo ""
echo "NOTA: Los backups se guardaron con extensión .backup_v2012"
echo "      Para revertir: mv FILE.backup_v2012 FILE"
echo "========================================================================"
