import os
import logging

from src.components.post.objects import post_objects
from src.components.tools.performance import PerformanceMonitor

# NOTE: PDF and residuals analysis removed to reduce memory usage
# from src.components.post.pdf import post_pdf
# from src.components.post.residuals import analyse_residuals

logger = logging.getLogger(__name__)


def run(case_name: str = "cases/cfd_case") -> None:
    """
    Process CFD simulation results - OPTIMIZED FOR MEMORY EFFICIENCY.
    
    This function generates only VTK files for interactive web visualization:
    1. Validates simulation results and sets up post-processing environment
    2. Generates VTK slice files and complete mesh for web viewer
    3. Converts VTK to vtkjs format for browser compatibility
    
    NOTE: Image rendering, PDF reports, and residual plots have been removed
    to prevent OOM (Out of Memory) issues in production.
    
    Args:
        case_name: Name of the simulation case
        
    Returns:
        None
    """
    performance_monitor = PerformanceMonitor()
    performance_monitor.start()
    
    logger.info("\n=========== RUNNING CFD POST-PROCESSING ===========")

    # Step 1: Set up post-processing environment and validate results
    sim_path = os.path.join(os.getcwd(), "cases", case_name, "sim")
    post_path = os.path.join(os.getcwd(), "cases", case_name, "post")
    logger.info(f"1 - Setting up post-processing environment: {post_path}")
    performance_monitor.update_memory()
    
    # NOTE: Residual analysis and PDF generation removed to reduce memory usage
    # The web viewer provides interactive 3D visualization of results
    
    # Generate VTK files for web viewer (lightweight, no rendering)
    logger.info("2 - Generating VTK files for web viewer")
    internal_mesh = post_objects(sim_path, post_path)
    performance_monitor.update_memory()
    
    # Log performance summary
    performance_summary = performance_monitor.get_summary()
    logger.info(f"Total processing time: {performance_summary['total_time']:.2f}s")
    logger.info(f"Peak memory usage: {performance_summary['peak_memory_mb']:.1f}MB")
    logger.info(f"✅ Post-processing completed successfully - results saved at {post_path}")


if __name__ == "__main__":
    import sys
    
    # Configure logging to print to stderr for subprocess visibility
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stderr)]
    )
    
    # Accept case_name as command-line argument
    if len(sys.argv) > 1:
        case_name = sys.argv[1]
    else:
        # Default for testing
        case_name = "FDM_iter2"
    
    logger.info(f"Starting post-processing for case: {case_name}")
    
    try:
        result = run(case_name=case_name)
        logger.info("✅ Post-processing completed successfully")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Post-processing failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)