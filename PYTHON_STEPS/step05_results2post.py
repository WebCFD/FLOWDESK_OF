import os
import sys
import logging
from pathlib import Path

# Add project root to path for src imports
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from src.components.post.objects import analyze_comfort_planes
from src.components.tools.performance import PerformanceMonitor


logger = logging.getLogger(__name__)


def run(case_name: str = "cases/cfd_case") -> None:
    """
    Post-process CFD simulation results: Thermal comfort analysis.
    
    Analyzes PMV/PPD fields in 3 horizontal planes (0.6m, 1.1m, 1.7m) and generates:
    - VTK slice files for each plane
    - PNG images (isometric view) for PMV and PPD
    - JSON file with comfort metrics
    
    Args:
        case_name: Name of the simulation case
        
    Returns:
        None
    """
    performance_monitor = PerformanceMonitor()
    performance_monitor.start()
    
    logger.info("\n=========== RUNNING THERMAL COMFORT POST-PROCESSING ===========")

    # Setup paths
    sim_path = os.path.join(os.getcwd(), "PYTHON_STEPS", "cases", case_name, "sim")
    post_path = os.path.join(os.getcwd(), "PYTHON_STEPS", "cases", case_name, "post")
    
    logger.info(f"Simulation path: {sim_path}")
    logger.info(f"Post-processing output: {post_path}")
    
    # Analyze comfort in horizontal planes
    logger.info("\n1 - Analyzing PMV/PPD thermal comfort in horizontal planes")
    performance_monitor.update_memory()
    
    results = analyze_comfort_planes(sim_path, post_path)
    performance_monitor.update_memory()
    
    # Summary
    logger.info("\n=========== COMFORT ANALYSIS SUMMARY ===========")
    for plane_name, metrics in results.items():
        logger.info(f"\n{plane_name.upper()} (z={metrics['height_m']}m):")
        logger.info(f"  Comfort area: {metrics['comfort_area_pct']:.1f}%")
        logger.info(f"  PMV: {metrics['pmv_mean']:.2f} ± {metrics['pmv_std']:.2f} (range: [{metrics['pmv_min']:.2f}, {metrics['pmv_max']:.2f}])")
        logger.info(f"  PPD: {metrics['ppd_mean']:.1f}% (max: {metrics['ppd_max']:.1f}%)")
    
    # Log performance summary
    performance_summary = performance_monitor.get_summary()
    logger.info(f"\nTotal processing time: {performance_summary['total_time']:.2f}s")
    logger.info(f"Peak memory usage: {performance_summary['peak_memory_mb']:.1f}MB")
    logger.info(f"\n✅ Post-processing completed successfully")
    logger.info(f"Results saved at: {post_path}")


if __name__ == "__main__":
    import sys
    
    # Configure logging
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
