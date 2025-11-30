import os
import logging
import sys
from pathlib import Path

from src.components.solve.local import solve_local
from src.components.solve.inductiva import solve_inductiva
from src.components.tools.performance import PerformanceMonitor

logger = logging.getLogger(__name__)

# Importar EXECUTION_MODE desde mainPipeline
python_steps = str(Path(__file__).parent.parent / "PYTHON_STEPS")
sys.path.insert(0, python_steps)
try:
    from mainPipeline import EXECUTION_MODE
except ImportError:
    EXECUTION_MODE = "full"  # Default si no se puede importar


def run(case_name: str = "cases/cfd_case", type: str = "local") -> None:
    """
    Execute CFD simulation with parallel processing and memory management.
    
    This function orchestrates the complete CFD solving pipeline:
    1. Sets up simulation environment and paths
    2. Executes CFD simulation on selected platform (local or cloud)
    3. Monitors simulation progress and performance
    4. Handles simulation completion and result validation
    
    Args:
        case_name: Name of the simulation case
        type: Type of execution platform ("local" or "inductiva")
        
    Returns:
        None
    """
    performance_monitor = PerformanceMonitor()
    performance_monitor.start()
    
    logger.info("\n=========== RUNNING CFD SIMULATION ===========")
    logger.info(f"Execution mode: {EXECUTION_MODE}")
    
    # ========== MESH-ONLY MODE: Skip simulation ==========
    if EXECUTION_MODE == "mesh-only":
        logger.info("\n⚠️  MESH-ONLY MODE: Skipping CFD simulation")
        logger.info("Mesh generation will be executed, but simulation will NOT run")
        logger.info("To visualize the mesh in ParaView:")
        logger.info(f"  1. Open: cases/{case_name}/sim/results.foam")
        logger.info(f"  2. Or open: cases/{case_name}/sim/constant/polyMesh/")
        logger.info("✅ Mesh-only mode completed\n")
        return

    # Step 1: Set up simulation environment
    case_path = os.path.join(os.getcwd(), "cases", case_name)
    sim_path = os.path.join(case_path, "sim")
    logger.info(f"1 - Setting up simulation environment: {sim_path}")
    performance_monitor.update_memory()

    # Step 2: Execute CFD simulation
    logger.info(f"2 - Executing CFD simulation on {type} platform")
    if EXECUTION_MODE == "test":
        logger.info("   (TEST MODE: 2 iterations only)")
    logger.info(f"   Platform: {type}")
    performance_monitor.update_memory()
    
    if(type == "inductiva"):
        logger.info(f"Running CFD simulation in INDUCTIVA cloud platform")
        solve_inductiva(sim_path, machine_type="c2d-highcpu-16")
    elif(type == "local"):
        logger.info(f"Running CFD simulation locally in {sim_path}") 
        solve_local(sim_path)
    else:
        logger.error(f"Unknown solve location: {type}")
        raise ValueError(f"Unknown solve location: {type}")
    
    performance_monitor.update_memory()
    
    # Log performance summary
    performance_summary = performance_monitor.get_summary()
    logger.info(f"Total processing time: {performance_summary['total_time']:.2f}s")
    logger.info(f"Peak memory usage: {performance_summary['peak_memory_mb']:.1f}MB")
    logger.info(f"✅ CFD simulation completed successfully")

if __name__ == "__main__":
    result = run(case_name="FDM_iter2", type="inductiva")
