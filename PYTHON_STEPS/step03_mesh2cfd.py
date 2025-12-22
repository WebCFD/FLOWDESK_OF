import os
import logging

from src.components.cfd.hvac import setup as hvac_setup
from src.components.tools.clear_case import clean_case_keep_mesh
from src.components.tools.create_scripts import save_scripts
from src.components.tools.performance import PerformanceMonitor
from pipeline_exceptions import CFDSetupError

logger = logging.getLogger(__name__)


def run(case_name: str, type: str, mesh_script: list = [], simulation_type: str = 'comfortTest', transient: bool = False) -> None:
    """
    Convert mesh to CFD simulation setup with parallel processing and memory management.
    
    This function orchestrates the complete CFD setup pipeline:
    1. Cleans case directory while preserving mesh files
    2. Sets up CFD simulation configuration based on simulation type
    3. Configures boundary conditions and solver parameters
    4. Generates and saves execution scripts for mesh and solve operations
    
    Args:
        case_name: Name of the simulation case
        type: Type of CFD simulation ("hvac" or other supported types)
        mesh_script: List of mesh generation script commands
        simulation_type: Simulation iteration type (comfortTest=3 iter, comfort30Iter=30 iter)
        transient: If True, use buoyantPimpleFoam (transient RANS). If False, use buoyantSimpleFoam (steady laminar)
        
    Returns:
        None
    """
    performance_monitor = PerformanceMonitor()
    performance_monitor.start()
    
    logger.info("\n=========== RUNNING MESH TO CFD CONVERSION ===========")

    # Step 1: Clean case directory while preserving mesh
    case_path = os.path.join(os.getcwd(), "cases", case_name)
    logger.info(f"1 - Cleaning case directory while preserving mesh: {case_path}")
    clean_case_keep_mesh(case_path)
    performance_monitor.update_memory()

    # Step 2: Set up CFD simulation configuration
    logger.info(f"2 - Setting up {type} CFD simulation configuration")
    performance_monitor.update_memory()
    
    try:
        if(type == "hvac"):
            solve_script = hvac_setup(case_path, simulation_type=simulation_type, transient=transient)
        else:
            raise CFDSetupError(
                f"Unknown simulation type: {type}",
                {
                    'case_name': case_name,
                    'simulation_type': type,
                    'suggestion': 'Use "hvac" simulation type'
                }
            )
    except CFDSetupError:
        # Re-raise CFDSetupError without wrapping
        raise
    except Exception as e:
        raise CFDSetupError(
            f"CFD setup failed: {str(e)}",
            {
                'case_name': case_name,
                'simulation_type': type,
                'suggestion': 'Check if mesh is valid and boundary conditions are properly configured'
            }
        )
    
    performance_monitor.update_memory()

    # Step 3: Generate and save execution scripts
    logger.info("3 - Generating and saving execution scripts")
    save_scripts(case_path, mesh_script, solve_script)
    performance_monitor.update_memory()
    
    # Log performance summary
    performance_summary = performance_monitor.get_summary()
    logger.info(f"Total processing time: {performance_summary['total_time']:.2f}s")
    logger.info(f"Peak memory usage: {performance_summary['peak_memory_mb']:.1f}MB")
    logger.info(f"✅ CFD case prepared successfully at {case_path}")


if __name__ == "__main__":
    result = run(case_name="FDM_iter2", type="hvac")
