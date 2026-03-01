import os
import sys
import logging
import subprocess
from pathlib import Path

# Add project root to path for src imports
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from src.components.post.objects import analyze_comfort_planes, generate_html_report, analyze_flow_planes, generate_flow_html_report
from src.components.post.ventilation import analyze_ventilation_planes, generate_ventilation_html_report
from src.components.post.setup_summary import analyze_setup_summary, generate_setup_html_report
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

    # Setup paths (handle both PYTHON_STEPS/cases and cases directories)
    # Try PYTHON_STEPS/cases first (mainPipeline.py location)
    sim_path_python_steps = os.path.join(os.getcwd(), "PYTHON_STEPS", "cases", case_name, "sim")
    sim_path_root = os.path.join(os.getcwd(), "cases", case_name, "sim")
    
    if os.path.isdir(sim_path_python_steps):
        sim_path = sim_path_python_steps
        post_path = os.path.join(os.getcwd(), "PYTHON_STEPS", "cases", case_name, "post")
    elif os.path.isdir(sim_path_root):
        sim_path = sim_path_root
        post_path = os.path.join(os.getcwd(), "cases", case_name, "post")
    else:
        raise FileNotFoundError(f"Case not found in PYTHON_STEPS/cases/{case_name} or cases/{case_name}")
    
    logger.info(f"Simulation path: {sim_path}")
    logger.info(f"Post-processing output: {post_path}")
    
    # Calculate PMV/PPD fields first (required for comfort analysis)
    logger.info("\n1 - Calculating PMV/PPD comfort fields")
    comfort_script = os.path.join(project_root, 'src', 'components', 'post', 'calculate_comfort.py')
    logger.info(f"   Running: python3.12 {comfort_script} {sim_path}")
    
    result = subprocess.run(
        ['python3.12', comfort_script, sim_path],
        capture_output=True,
        text=True,
        encoding='utf-8'  # Fix: Explicit UTF-8 encoding for subprocess output
    )
    
    if result.returncode != 0:
        logger.error(f"   calculate_comfort.py failed with exit code {result.returncode}")
        logger.error(f"   STDERR: {result.stderr}")
        raise RuntimeError(f"PMV/PPD calculation failed: {result.stderr}")
    
    # Log output from calculate_comfort.py
    if result.stdout:
        for line in result.stdout.strip().split('\n'):
            logger.info(f"   {line}")
    
    logger.info("   ✓ PMV/PPD fields calculated successfully")
    
    # Analyze comfort in horizontal planes
    logger.info("\n2 - Analyzing PMV/PPD thermal comfort in horizontal planes")
    performance_monitor.update_memory()
    
    results = analyze_comfort_planes(sim_path, post_path)
    performance_monitor.update_memory()
    
    # Generate HTML comfort report
    logger.info("\n3 - Generating HTML comfort report")
    html_path = generate_html_report(results, post_path, case_name=case_name)
    logger.info(f"HTML report: {html_path}")
    
    # Analyze flow fields (T, U) in horizontal planes
    logger.info("\n4 - Analyzing T/U flow fields in horizontal planes")
    performance_monitor.update_memory()
    
    flow_results = analyze_flow_planes(sim_path, post_path)
    performance_monitor.update_memory()
    
    # Generate HTML flow report
    logger.info("\n5 - Generating HTML flow report")
    flow_html_path = generate_flow_html_report(flow_results, post_path, case_name=case_name)
    logger.info(f"Flow report: {flow_html_path}")
    
    # Analyze ventilation (CO2, ADPI, stagnation zones)
    logger.info("\n6 - Analyzing ventilation metrics in horizontal planes")
    performance_monitor.update_memory()
    
    ventilation_results = analyze_ventilation_planes(sim_path, post_path)
    performance_monitor.update_memory()
    
    # Generate HTML ventilation report
    logger.info("\n7 - Generating HTML ventilation report")
    ventilation_html_path = generate_ventilation_html_report(ventilation_results, post_path, case_name=case_name)
    logger.info(f"Ventilation report: {ventilation_html_path}")
    
    # Analyze setup summary (boundary conditions)
    logger.info("\n8 - Analyzing simulation setup (boundary conditions)")
    performance_monitor.update_memory()
    
    setup_summary = analyze_setup_summary(sim_path, post_path)
    performance_monitor.update_memory()
    
    # Generate HTML setup report
    logger.info("\n9 - Generating HTML setup report")
    setup_html_path = generate_setup_html_report(setup_summary, post_path, case_name=case_name)
    logger.info(f"Setup report: {setup_html_path}")
    
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
