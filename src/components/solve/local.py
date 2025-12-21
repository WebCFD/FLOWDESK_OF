import os
import subprocess
import logging

logger = logging.getLogger(__name__)


def solve_local(sim_path):
    """
    Solve the CFD case locally using OpenFOAM.
    
    Args:
        sim_path: Path to the simulation directory
    """
    logger.info(f"    * Starting local CFD simulation in: {sim_path}")
    
    # Check if Allrun script exists
    allrun_path = os.path.join(sim_path, "Allrun")
    
    if not os.path.exists(allrun_path):
        logger.error(f"    * ERROR: Allrun script not found at {allrun_path}")
        raise FileNotFoundError(f"Allrun script not found: {allrun_path}")
    
    # Make script executable
    os.chmod(allrun_path, 0o755)
    
    # Execute Allrun script
    logger.info(f"    * Executing Allrun script")
    
    try:
        result = subprocess.run(
            ["./Allrun"],
            cwd=sim_path,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            logger.info(f"    * Local CFD simulation completed successfully")
            logger.info(f"    * Output: {result.stdout[-500:]}")  # Last 500 chars
        else:
            logger.error(f"    * ERROR: CFD simulation failed with return code {result.returncode}")
            logger.error(f"    * Error output: {result.stderr}")
            raise RuntimeError(f"CFD simulation failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        logger.error(f"    * ERROR: CFD simulation timed out after 1 hour")
        raise RuntimeError("CFD simulation timed out")
    except Exception as e:
        logger.error(f"    * ERROR: Failed to execute CFD simulation: {e}")
        raise
