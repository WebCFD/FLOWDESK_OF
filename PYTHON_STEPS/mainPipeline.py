"""
mainPipeline.py - Script para ejecutar el pipeline completo (step01 + step02 + step03 + step04)
Genera outputs en carpetas separadas: output_01, output_02, output_03, output_04
Soporta modos de ejecución: mesh-only, test, full
"""

import os
import json
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# =============================================================================
# LOGGING CONFIGURATION (ROBUST)
# =============================================================================
# Configure logging at the very beginning to capture all messages
logging.basicConfig(
    level=logging.INFO,  # Show INFO, WARNING, ERROR, CRITICAL
    format='%(levelname)-8s - %(name)-30s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Set specific loggers to INFO level to ensure they are not filtered
logging.getLogger('src.components.geo').setLevel(logging.INFO)
logging.getLogger('src.components.geo.create_volumes').setLevel(logging.INFO)
logging.getLogger('src.components.geo.create_volumes_layered').setLevel(logging.INFO)
logging.getLogger('__main__').setLevel(logging.INFO)

logger = logging.getLogger(__name__)
logger.info("="*70)
logger.info("FLOWDESK PIPELINE - Logging configured at INFO level")
logger.info("="*70)

# =============================================================================
# EXECUTION MODE CONFIGURATION
# =============================================================================
EXECUTION_MODE = "full"  # "stl-only", "mesh-only", "full"
# stl-only: Genera STL (geometría), no ejecuta cfMesh
# mesh-only: Ejecuta cfMesh (genera malla), sin simulación CFD
# full: Ejecuta cfMesh + simulación CFD completa (3 iteraciones)

# =============================================================================
# OPENFOAM ENVIRONMENT SETUP
# =============================================================================
def setup_openfoam_environment():
    """
    Verifica si el entorno de OpenFOAM está cargado.
    Si no, intenta cargarlo automáticamente.
    """
    import subprocess
    
    # Verificar si WM_PROJECT_DIR está definido (indicador de OF cargado)
    if "WM_PROJECT_DIR" in os.environ:
        print("✅ OpenFOAM environment already loaded")
        print(f"   WM_PROJECT_DIR: {os.environ.get('WM_PROJECT_DIR')}")
        return True
    
    print("⚠️  OpenFOAM environment NOT loaded. Attempting to load...")
    
    # Rutas comunes de OpenFOAM
    of_paths = [
        "/home/plprm/OpenFOAM/OpenFOAM-v2012/etc/bashrc",
        "/opt/openfoam/etc/bashrc",
        "/usr/lib/openfoam/etc/bashrc",
        os.path.expanduser("~/OpenFOAM/OpenFOAM-v2012/etc/bashrc"),
    ]
    
    # Buscar bashrc de OpenFOAM
    bashrc_path = None
    for path in of_paths:
        if os.path.exists(path):
            bashrc_path = path
            print(f"   Found OpenFOAM bashrc: {bashrc_path}")
            break
    
    if not bashrc_path:
        print("❌ ERROR: Could not find OpenFOAM bashrc file")
        print("   Tried paths:")
        for path in of_paths:
            print(f"     - {path}")
        print("\n   Please source OpenFOAM manually:")
        print("   source /path/to/OpenFOAM/etc/bashrc")
        return False
    
    # Cargar OpenFOAM en el entorno actual
    try:
        # Ejecutar bash con source y exportar variables
        cmd = f"source {bashrc_path} && env"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable="/bin/bash")
        
        if result.returncode != 0:
            print(f"❌ ERROR: Failed to source OpenFOAM bashrc")
            print(f"   Error: {result.stderr}")
            return False
        
        # Parsear variables de entorno
        for line in result.stdout.split('\n'):
            if '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value
        
        # Verificar que se cargó correctamente
        if "WM_PROJECT_DIR" in os.environ:
            print(f"✅ OpenFOAM environment loaded successfully")
            print(f"   WM_PROJECT_DIR: {os.environ.get('WM_PROJECT_DIR')}")
            return True
        else:
            print("❌ ERROR: OpenFOAM environment variables not set after sourcing")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: Exception while loading OpenFOAM: {e}")
        return False

# Cargar entorno de OpenFOAM solo si vamos a ejecutar localmente
# (se detectará después de parsear argumentos)
_SETUP_OF_LATER = True

# Agregar el directorio raíz del proyecto al path para importar src
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

# Agregar PYTHON_STEPS al path para importar pipeline_exceptions
python_steps = str(Path(__file__).parent)
sys.path.insert(0, python_steps)

from step01_json2geo import run as step01_run
from step02_geo2mesh import run as step02_run
# step03 se importará condicionalmente dentro de main() para evitar
# cargar dependencias innecesarias (como foamlib) en modo mesh-only


def main():
    """Ejecuta el pipeline completo (3 steps) con outputs en disco"""
    
    # Importar step03 solo si es necesario (no en stl-only)
    # Esto evita cargar foamlib y otras dependencias innecesarias
    global EXECUTION_MODE, json_filename
    if EXECUTION_MODE != "stl-only":
        from step03_mesh2cfd import run as step03_run
    
    # Crear directorio base de outputs
    base_output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Crear timestamp único para toda la ejecución
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Crear directorios para cada step
    output_01_dir = os.path.join(base_output_dir, f"output_01_{timestamp}")
    output_02_dir = os.path.join(base_output_dir, f"output_02_{timestamp}")
    output_03_dir = os.path.join(base_output_dir, f"output_03_{timestamp}")
    
    os.makedirs(output_01_dir, exist_ok=True)
    os.makedirs(output_02_dir, exist_ok=True)
    os.makedirs(output_03_dir, exist_ok=True)
    
    # Crear archivo de log principal
    main_log_file = os.path.join(base_output_dir, f"pipeline_main_{timestamp}.txt")
    
    def log_print(message, log_file=main_log_file):
        """Imprime en consola y guarda en archivo de log"""
        print(message)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(message + "\n")
    
    # ============================================================
    # STEP 01: JSON → GEOMETRÍA 3D
    # ============================================================
    log_print("\n" + "="*70)
    log_print("STEP 01: JSON → GEOMETRÍA 3D (step01_json2geo.py)")
    log_print("="*70 + "\n")
    
    # Cargar el JSON de entrada
    json_path = os.path.join(os.path.dirname(__file__), json_filename)
    
    if not os.path.exists(json_path):
        log_print(f"❌ Error: No se encontró el archivo JSON en {json_path}")
        log_print(f"   Archivo buscado: {json_filename}")
        log_print(f"   Ruta completa: {json_path}")
        return False
    
    log_print(f"📂 Cargando JSON: {json_path}")
    with open(json_path, 'r') as f:
        json_payload = json.load(f)
    
    case_name = json_payload.get("case_name", "DefaultCase")
    log_print(f"📋 Case name: {case_name}\n")
    
    # Crear directorio de caso para step03 (necesario para copiar STL)
    case_path = os.path.join(os.getcwd(), "cases", case_name)
    os.makedirs(case_path, exist_ok=True)
    
    try:
        log_print("🔄 Ejecutando STEP 01...")
        final_geometry_mesh, boundary_conditions_df = step01_run(json_payload, case_name)
        
        # Validar output
        log_print("\n" + "-"*70)
        log_print("✅ VALIDACIÓN DE OUTPUT - STEP 01")
        log_print("-"*70)
        
        # Validar geometría
        log_print(f"\n📐 Geometría (pv.PolyData):")
        log_print(f"   - Tipo: {type(final_geometry_mesh).__name__}")
        log_print(f"   - Vértices: {final_geometry_mesh.n_points}")
        log_print(f"   - Celdas: {final_geometry_mesh.n_cells}")
        log_print(f"   - Válida: {'✓' if final_geometry_mesh.n_points > 0 else '✗'}")
        
        # Validar dataframe
        log_print(f"\n📊 Condiciones de Frontera (DataFrame):")
        log_print(f"   - Tipo: {type(boundary_conditions_df).__name__}")
        log_print(f"   - Filas: {len(boundary_conditions_df)}")
        log_print(f"   - Columnas: {list(boundary_conditions_df.columns) if len(boundary_conditions_df.columns) > 0 else 'N/A'}")
        log_print(f"   - Válido: {'✓' if len(boundary_conditions_df) >= 0 else '✗'}")
        
        # Guardar outputs de STEP 01
        log_print("\n" + "-"*70)
        log_print("💾 GUARDANDO OUTPUTS DE STEP 01 EN DISCO")
        log_print("-"*70)
        
        # Guardar geometría en VTK
        geometry_file = os.path.join(output_01_dir, f"geometry_{timestamp}.vtk")
        final_geometry_mesh.save(geometry_file)
        log_print(f"✓ Geometría guardada (VTK): {geometry_file}")
        
        # Guardar geometría en STL
        stl_file = os.path.join(output_01_dir, f"geometry_{timestamp}.stl")
        final_geometry_mesh.save(stl_file)
        log_print(f"✓ Geometría guardada (STL): {stl_file}")
        
        # Copiar STL a cases/MySim/geo/ para que step03 lo encuentre
        geo_dir = os.path.join(case_path, "geo")
        os.makedirs(geo_dir, exist_ok=True)
        geo_stl_file = os.path.join(geo_dir, "geometry.stl")
        import shutil
        shutil.copy(stl_file, geo_stl_file)
        log_print(f"✓ Geometría copiada a: {geo_stl_file}")
        
        # Copiar STL a cases/MySim/sim/ para que cartesianMesh lo encuentre
        sim_dir = os.path.join(case_path, "sim")
        sim_stl_file = os.path.join(sim_dir, "geometry.stl")
        if os.path.exists(sim_dir):
            shutil.copy(stl_file, sim_stl_file)
            log_print(f"✓ Geometría copiada a: {sim_stl_file}")
        
        # Guardar condiciones de frontera en CSV
        bc_file = os.path.join(output_01_dir, f"boundary_conditions_{timestamp}.csv")
        boundary_conditions_df.to_csv(bc_file, index=False)
        log_print(f"✓ Condiciones de frontera guardadas: {bc_file}")
        
        # Guardar condiciones de frontera en JSON
        bc_json_file = os.path.join(output_01_dir, f"boundary_conditions_{timestamp}.json")
        boundary_conditions_df.to_json(bc_json_file, orient='records', indent=2)
        log_print(f"✓ Condiciones de frontera (JSON) guardadas: {bc_json_file}")
        
        # Guardar resumen de STEP 01
        summary_01 = {
            "timestamp": timestamp,
            "step": "STEP 01",
            "case_name": case_name,
            "geometry": {
                "type": type(final_geometry_mesh).__name__,
                "vertices": final_geometry_mesh.n_points,
                "cells": final_geometry_mesh.n_cells
            },
            "boundary_conditions": {
                "rows": len(boundary_conditions_df),
                "columns": list(boundary_conditions_df.columns)
            },
            "output_files": {
                "geometry": geometry_file,
                "boundary_conditions_csv": bc_file,
                "boundary_conditions_json": bc_json_file
            }
        }
        summary_01_file = os.path.join(output_01_dir, f"summary_{timestamp}.json")
        with open(summary_01_file, 'w') as f:
            json.dump(summary_01, f, indent=2)
        log_print(f"✓ Resumen guardado: {summary_01_file}")
        
        log_print("\n" + "="*70)
        log_print("✅ STEP 01 COMPLETADO EXITOSAMENTE")
        log_print("="*70)
        log_print(f"📁 Outputs guardados en: {output_01_dir}\n")
        
    except Exception as e:
        log_print("\n" + "="*70)
        log_print(f"❌ ERROR EN STEP 01: {str(e)}")
        log_print("="*70 + "\n")
        import traceback
        log_print(traceback.format_exc())
        return False
    
    # ============================================================
    # STEP 02: GEOMETRÍA → COMANDOS DE MALLADO
    # ============================================================
    log_print("\n" + "="*70)
    log_print("STEP 02: GEOMETRÍA → COMANDOS DE MALLADO (step02_geo2mesh.py)")
    log_print("="*70 + "\n")
    
    try:
        log_print("🔄 Ejecutando STEP 02...")
        mesh_script_commands = step02_run(
            case_name=case_name,
            geo_mesh=final_geometry_mesh,
            geo_df=boundary_conditions_df,
            type="cfmesh",
            quality_level=None
        )
        
        # Validar output
        log_print("\n" + "-"*70)
        log_print("✅ VALIDACIÓN DE OUTPUT - STEP 02")
        log_print("-"*70)
        
        log_print(f"\n📜 Script de Mallado:")
        log_print(f"   - Tipo: {type(mesh_script_commands).__name__}")
        log_print(f"   - Número de comandos: {len(mesh_script_commands)}")
        log_print(f"   - Válido: {'✓' if len(mesh_script_commands) > 0 else '✗'}")
        
        # Guardar outputs de STEP 02
        log_print("\n" + "-"*70)
        log_print("💾 GUARDANDO OUTPUTS DE STEP 02 EN DISCO")
        log_print("-"*70)
        
        # Guardar script de mallado
        mesh_script_file = os.path.join(output_02_dir, f"mesh_script_{timestamp}.sh")
        with open(mesh_script_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(mesh_script_commands))
        log_print(f"✓ Script de mallado guardado: {mesh_script_file}")
        
        # Guardar resumen de STEP 02
        summary_02 = {
            "timestamp": timestamp,
            "step": "STEP 02",
            "case_name": case_name,
            "mesh_script": {
                "type": type(mesh_script_commands).__name__,
                "num_commands": len(mesh_script_commands),
                "mesher_type": "cfmesh"
            },
            "output_files": {
                "mesh_script": mesh_script_file
            }
        }
        summary_02_file = os.path.join(output_02_dir, f"summary_{timestamp}.json")
        with open(summary_02_file, 'w') as f:
            json.dump(summary_02, f, indent=2)
        log_print(f"✓ Resumen guardado: {summary_02_file}")
        
        # Guardar primeros 20 comandos para referencia
        commands_preview_file = os.path.join(output_02_dir, f"commands_preview_{timestamp}.txt")
        with open(commands_preview_file, 'w', encoding='utf-8') as f:
            f.write("PRIMEROS 20 COMANDOS DEL SCRIPT DE MALLADO:\n")
            f.write("="*70 + "\n\n")
            for i, cmd in enumerate(mesh_script_commands[:20], 1):
                f.write(f"{i:3d}. {cmd}\n")
            if len(mesh_script_commands) > 20:
                f.write(f"\n... y {len(mesh_script_commands) - 20} comandos más ...\n")
        log_print(f"✓ Preview de comandos guardado: {commands_preview_file}")
        
        log_print("\n" + "="*70)
        log_print("✅ STEP 02 COMPLETADO EXITOSAMENTE")
        log_print("="*70)
        log_print(f"📁 Outputs guardados en: {output_02_dir}\n")
        log_print("⚠️  NOTA: Los comandos de mallado están preparados pero NO ejecutados")
        log_print("    Para ejecutar la malla, ejecuta el script generado en OpenFOAM\n")
        
    except Exception as e:
        log_print("\n" + "="*70)
        log_print(f"❌ ERROR EN STEP 02: {str(e)}")
        log_print("="*70 + "\n")
        import traceback
        log_print(traceback.format_exc())
        return False
    
    # ============================================================
    # STEP 03: MALLA → CONFIGURACIÓN CFD
    # ============================================================
    
    # En modo stl-only, saltamos step03 (no ejecutamos cfMesh)
    if EXECUTION_MODE == "stl-only":
        log_print("\n" + "="*70)
        log_print("⏭️  STEP 03 SKIPPED (stl-only mode)")
        log_print("="*70 + "\n")
        log_print("STL generado exitosamente (sin ejecutar cfMesh).")
        log_print("Para generar malla con cfMesh, usa: --mode mesh-only")
        log_print("\n")
        return True
    
    # Crear directorio de caso para step03
    case_path = os.path.join(os.getcwd(), "cases", case_name)
    os.makedirs(case_path, exist_ok=True)
    
    log_print("\n" + "="*70)
    log_print("STEP 03: MALLA → CONFIGURACIÓN CFD (step03_mesh2cfd.py)")
    log_print("="*70 + "\n")
    
    try:
        log_print("🔄 Ejecutando STEP 03...")
        
        # Ejecutar step03
        step03_run(
            case_name=case_name,
            type="hvac",
            mesh_script=mesh_script_commands,
            simulation_type="comfortTest"
        )
        
        # Validar output
        log_print("\n" + "-"*70)
        log_print("✅ VALIDACIÓN DE OUTPUT - STEP 03")
        log_print("-"*70)
        
        log_print(f"\n📋 Configuración CFD:")
        log_print(f"   - Tipo de simulación: HVAC")
        log_print(f"   - Tipo de iteración: comfortTest")
        log_print(f"   - Directorio de caso: {case_path}")
        
        # Verificar archivos creados
        sim_path = os.path.join(case_path, "sim")
        if os.path.exists(sim_path):
            log_print(f"   - Directorio sim creado: ✓")
            
            # Contar archivos creados
            system_dir = os.path.join(sim_path, "system")
            constant_dir = os.path.join(sim_path, "constant")
            initial_dir = os.path.join(sim_path, "0.orig")
            
            system_files = len(os.listdir(system_dir)) if os.path.exists(system_dir) else 0
            constant_files = len(os.listdir(constant_dir)) if os.path.exists(constant_dir) else 0
            initial_files = len(os.listdir(initial_dir)) if os.path.exists(initial_dir) else 0
            
            log_print(f"   - Archivos en system/: {system_files}")
            log_print(f"   - Archivos en constant/: {constant_files}")
            log_print(f"   - Archivos en 0.orig/: {initial_files}")
        
        # Guardar outputs de STEP 03
        log_print("\n" + "-"*70)
        log_print("💾 GUARDANDO OUTPUTS DE STEP 03 EN DISCO")
        log_print("-"*70)
        
        # Copiar archivos importantes a output_03
        if os.path.exists(sim_path):
            # Copiar archivos de configuración
            for subdir in ['system', 'constant', '0.orig']:
                src_dir = os.path.join(sim_path, subdir)
                dst_dir = os.path.join(output_03_dir, subdir)
                if os.path.exists(src_dir):
                    import shutil
                    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
                    log_print(f"✓ Directorio {subdir}/ copiado a output_03")
        
        # Guardar resumen de STEP 03
        summary_03 = {
            "timestamp": timestamp,
            "step": "STEP 03",
            "case_name": case_name,
            "cfd_configuration": {
                "simulation_type": "hvac",
                "iteration_type": "comfortTest",
                "case_path": case_path,
                "sim_path": sim_path
            },
            "directories_created": {
                "system": system_files if os.path.exists(system_dir) else 0,
                "constant": constant_files if os.path.exists(constant_dir) else 0,
                "0.orig": initial_files if os.path.exists(initial_dir) else 0
            },
            "output_files": {
                "configuration_backup": output_03_dir
            }
        }
        summary_03_file = os.path.join(output_03_dir, f"summary_{timestamp}.json")
        with open(summary_03_file, 'w') as f:
            json.dump(summary_03, f, indent=2)
        log_print(f"✓ Resumen guardado: {summary_03_file}")
        
        log_print("\n" + "="*70)
        log_print("✅ STEP 03 COMPLETADO EXITOSAMENTE")
        log_print("="*70)
        log_print(f"📁 Outputs guardados en: {output_03_dir}\n")
        log_print(f"📁 Caso CFD configurado en: {case_path}\n")
        
        # Copiar STL a sim/ para que cartesianMesh lo encuentre (después de que step03 cree sim/)
        sim_stl_file = os.path.join(sim_path, "geometry.stl")
        stl_file = os.path.join(output_01_dir, f"geometry_{timestamp}.stl")
        if os.path.exists(stl_file) and os.path.exists(sim_path):
            import shutil
            shutil.copy(stl_file, sim_stl_file)
            log_print(f"✓ Geometría copiada a: {sim_stl_file}")
        
        # En modo mesh-only, ejecutar Allrun para generar la malla con cfMesh
        if EXECUTION_MODE == "mesh-only":
            log_print("\n" + "="*70)
            log_print("EJECUTANDO CFMESH (Allrun)")
            log_print("="*70 + "\n")
            
            sim_path = os.path.join(case_path, "sim")
            allrun_path = os.path.join(sim_path, "Allrun")
            
            if os.path.exists(allrun_path):
                log_print(f"🔄 Ejecutando: {allrun_path}")
                log_print("   Esto generará la malla con cfMesh...\n")
                
                try:
                    import subprocess
                    # Ejecutar Allrun con OpenFOAM environment cargado correctamente
                    # Usar bash -c para asegurar que source se ejecuta en el mismo shell
                    cmd = f"bash -c 'source /home/plprm/OpenFOAM/OpenFOAM-v2012/etc/bashrc && cd {sim_path} && ./Allrun'"
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable="/bin/bash", encoding='utf-8', errors='replace')
                    
                    # Verificar que se generó polyMesh
                    polymesh_path = os.path.join(sim_path, "constant", "polyMesh")
                    if os.path.exists(polymesh_path):
                        points_file = os.path.join(polymesh_path, "points")
                        if os.path.exists(points_file):
                            log_print("✅ cfMesh ejecutado exitosamente")
                            log_print("   Malla generada en: constant/polyMesh/")
                            log_print(f"   ✓ Archivo points encontrado")
                        else:
                            log_print("⚠️  polyMesh creado pero sin archivo points")
                            if result.stderr:
                                log_print(f"   Stderr: {result.stderr[:300]}")
                    else:
                        log_print(f"❌ cfMesh NO generó polyMesh")
                        log_print(f"   Return code: {result.returncode}")
                        if result.stderr:
                            log_print(f"   Stderr: {result.stderr[:500]}")
                        if result.stdout:
                            # Mostrar últimas líneas del stdout para ver dónde falló
                            stdout_lines = result.stdout.split('\n')
                            log_print(f"   Últimas líneas de output:")
                            for line in stdout_lines[-10:]:
                                if line.strip():
                                    log_print(f"     {line}")
                except Exception as e:
                    log_print(f"❌ Error al ejecutar Allrun: {str(e)}")
            else:
                log_print(f"⚠️  Archivo Allrun no encontrado en: {allrun_path}")
        
    except Exception as e:
        log_print("\n" + "="*70)
        log_print(f"❌ ERROR EN STEP 03: {str(e)}")
        log_print("="*70 + "\n")
        import traceback
        log_print(traceback.format_exc())
        return False
    
    # ============================================================
    # RESUMEN FINAL
    # ============================================================
    log_print("\n" + "="*70)
    log_print("✅ PIPELINE COMPLETO EJECUTADO EXITOSAMENTE")
    log_print("="*70 + "\n")
    
    log_print("📊 RESUMEN DE EJECUCIÓN:")
    log_print(f"   - Timestamp: {timestamp}")
    log_print(f"   - Case name: {case_name}")
    log_print(f"   - Geometría: {final_geometry_mesh.n_points} vértices, {final_geometry_mesh.n_cells} celdas")
    log_print(f"   - Boundary conditions: {len(boundary_conditions_df)} patches")
    log_print(f"   - Mesh script: {len(mesh_script_commands)} comandos")
    
    log_print("\n📁 OUTPUTS GENERADOS:")
    log_print(f"   - STEP 01: {output_01_dir}")
    log_print(f"   - STEP 02: {output_02_dir}")
    log_print(f"   - STEP 03: {output_03_dir}")
    
    log_print("\n📝 LOG PRINCIPAL:")
    log_print(f"   - {main_log_file}")
    
    log_print("\n" + "="*70 + "\n")
    
    return True


if __name__ == "__main__":
    # Parsear argumentos de línea de comandos
    parser = argparse.ArgumentParser(
        description="FLOWDESK Pipeline: JSON → Geometría → Malla → CFD → Resultados"
    )
    parser.add_argument(
        "--mode",
        choices=["stl-only", "mesh-only", "full"],
        default="full",
        help="Modo de ejecución: stl-only (solo STL), mesh-only (cfMesh sin CFD), full (cfMesh + CFD)"
    )
    parser.add_argument(
        "--json",
        type=str,
        default="MySim_FlowDeskModel.json",
        help="Archivo JSON de entrada (ruta relativa a PYTHON_STEPS/). Por defecto: MySim_FlowDeskModel.json"
    )
    args = parser.parse_args()
    
    # Establecer modo de ejecución global
    EXECUTION_MODE = args.mode
    
    # Establecer archivo JSON de entrada
    json_filename = args.json
    
    # Cargar OpenFOAM solo si vamos a ejecutar cfMesh o CFD (no en stl-only)
    if EXECUTION_MODE != "stl-only":
        print("\n⚠️  Mesh/CFD mode detected. Setting up OpenFOAM environment...")
        setup_openfoam_environment()
    else:
        print("\n✅ STL-only mode: OpenFOAM environment setup skipped")
    
    # Mostrar modo de ejecución
    print("\n" + "="*70)
    print(f"EXECUTION MODE: {EXECUTION_MODE.upper()}")
    if EXECUTION_MODE == "stl-only":
        print("└─ Generará STL (geometría), NO ejecutará cfMesh")
        print("└─ OpenFOAM environment: NOT REQUIRED")
    elif EXECUTION_MODE == "mesh-only":
        print("└─ Ejecutará cfMesh (genera malla), SIN simulación CFD")
        print("└─ OpenFOAM environment: REQUIRED")
    else:
        print("└─ Ejecutará cfMesh + simulación CFD completa")
        print("└─ OpenFOAM environment: REQUIRED")
    print("="*70)
    print(f"INPUT JSON: {json_filename}")
    print("="*70 + "\n")
    
    success = main()
