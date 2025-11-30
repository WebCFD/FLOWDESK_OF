import os
import re
import logging
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict

logger = logging.getLogger(__name__)


def analyse_residuals(logfile_path, post_path):
    """
    Analyze convergence residuals from CFD simulation log file.
    
    Args:
        logfile_path: Path to the simulation log file
        post_path: Path to post-processing output directory
    """
    logger.info(f"    * Analyzing convergence residuals from: {logfile_path}")
    
    # Parse all available fields first to detect simulation type
    all_fields = set()
    residual_pattern_temp = re.compile(
        r"Solving for (\w+), Initial residual = ([\deE\.\-]+)"
    )
    with open(logfile_path, "r") as f:
        for line in f:
            match = residual_pattern_temp.search(line)
            if match:
                all_fields.add(match.group(1))
    
    # Auto-detect target fields based on simulation type
    # Turbulent: h, p_rgh, k, omega
    # Laminar: h, p_rgh, Ux, Uy, Uz
    if "omega" in all_fields or "k" in all_fields:
        target_fields = ["h", "p_rgh", "k", "omega"]
        logger.info(f"    * Detected turbulent simulation (kOmegaSST)")
    else:
        target_fields = ["h", "p_rgh", "Ux", "Uy", "Uz"]
        logger.info(f"    * Detected laminar simulation")
    
    logger.info(f"    * Target residual fields: {target_fields}")

    # === REGEX TO MATCH RESIDUAL LINES ===
    residual_pattern = re.compile(
        r"Solving for (\w+), Initial residual = ([\deE\.\-]+), Final residual = ([\deE\.\-]+)"
    )

    # === DATA CONTAINERS ===
    residuals = defaultdict(list)
    iterations = []
    iteration = 0
    current_fields = set()

    # === PARSE LOG FILE ===
    logger.info("    * Parsing simulation log file for residual data")
    with open(logfile_path, "r") as f:
        for line in f:
            match = residual_pattern.search(line)
            if match:
                field, initial, _ = match.groups()
                if field in target_fields:
                    residuals[field].append(float(initial))
                    current_fields.add(field)

                    # Register new iteration once all fields seen
                    if len(current_fields) == len(target_fields):
                        iterations.append(iteration)
                        iteration += 1
                        current_fields.clear()

    logger.info(f"    * Parsed {len(iterations)} iterations of residual data")
    logger.info(f"    * Fields found: {list(residuals.keys())}")

    # === SAVE TO CSV ===
    csv_dir = os.path.join(post_path, "csv")
    os.makedirs(csv_dir, exist_ok=True)
    output_csv = os.path.join(csv_dir, "residuals.csv")
    logger.info(f"    * Saving residual data to CSV: {output_csv}")

    min_len = min(len(residuals[f]) for f in target_fields)
    data = {field: residuals[field][:min_len] for field in target_fields}
    data["iteration"] = iterations[:min_len]
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    logger.info(f"    * Residuals saved to {output_csv}")

    # === PLOT ALL RESIDUALS ===
    img_dir = os.path.join(post_path, "images")
    os.makedirs(img_dir, exist_ok=True)
    output_png = os.path.join(img_dir, "residuals_plot.png")
    logger.info(f"    * Generating residual convergence plot: {output_png}")

    plt.figure(figsize=(10, 6))
    for column in df.columns:
        if column != "iteration":
            plt.semilogy(df["iteration"], df[column], label=column)
    plt.xlabel("Iteration")
    plt.ylabel("Initial Residual")
    plt.title("Residuals vs Iteration (from CSV)")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png)
    logger.info(f"    * Residual analysis completed successfully")