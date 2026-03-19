"""
Data Center Cooling Post-Processing Module.

Analyzes CFD simulation results for Data Center thermal management:
- Per-rack inlet/outlet temperature (ASHRAE A2 compliance)
- RTI  — Return Temperature Index
- RCI  — Rack Cooling Index (HI and LO) with percentiles P95/P99
- Cooling Efficiency η_cooling  = Q_IT / Q_CRAC
- Horizontal temperature slices at rack heights (0.5m, 1.0m, 1.5m)
- Hot-spot detection and airflow uniformity

Standards referenced:
  ASHRAE TC9.9  — Thermal Guidelines for Data Processing Environments
  ASHRAE A2 class supply: 10–35°C  |  return: up to 45°C
"""

import os
import json
import glob
import logging
import numpy as np
import pyvista as pv
from datetime import datetime
from pathlib import Path

from src.components.tools.export_debug import load_foam_results

logger = logging.getLogger(__name__)

# ── ASHRAE A2 limits ─────────────────────────────────────────────────────────
ASHRAE_A2_T_SUPPLY_MIN = 10.0   # °C  — minimum cold aisle supply
ASHRAE_A2_T_SUPPLY_MAX = 35.0   # °C  — maximum cold aisle supply (RCI_HI limit)
ASHRAE_A2_T_SUPPLY_LOW = 15.0   # °C  — lower recommended (RCI_LO limit)
ASHRAE_A2_T_RETURN_MAX = 45.0   # °C  — maximum hot aisle return

# Air specific heat capacity [J/(kg·K)] and density [kg/m³] at ~25°C
CP_AIR = 1006.0
RHO_AIR = 1.184

# CFM → m³/s conversion factor
CFM_TO_M3S = 4.719e-4


# ─────────────────────────────────────────────────────────────────────────────
# JSON PARSING
# ─────────────────────────────────────────────────────────────────────────────

def extract_rack_info_from_json(json_payload: dict) -> dict:
    """
    Parse the FlowDesk JSON payload and extract rack metadata.

    Returns:
        dict keyed by rack_id:
          {
            'id':              str,
            'inlet_face_id':   str,   # e.g. "object_0F_rack_1_front"
            'outlet_face_id':  str,
            'T_inlet_set':     float, # °C from JSON
            'T_outlet_set':    float, # °C from JSON
            'power_kW':        float, # thermalPower_kW
            'airflow_cfm':     float, # airFlow in CFM
            'airflow_m3s':     float,
          }
    """
    racks = {}

    levels = json_payload.get("levels", {})
    for level_id, level_data in levels.items():
        for obj in level_data.get("furniture", []):
            obj_id = obj.get("id", "")
            faces = obj.get("faces", {})
            if not faces:
                continue

            # Identify rack objects by the presence of a "front" face with thermalPower_kW
            front = faces.get("front", {})
            back = faces.get("back", {})
            if front.get("role") not in ("inlet",) or "thermalPower_kW" not in front:
                continue

            rack = {
                "id": obj_id,
                "inlet_face_id": f"{obj_id}_front",
                "outlet_face_id": f"{obj_id}_back",
                "T_inlet_set": float(front.get("temperature", 22.0)),
                "T_outlet_set": float(back.get("temperature", 45.0)),
                "power_kW": float(front.get("thermalPower_kW", 0.0)),
                "airflow_cfm": float(front.get("airFlow", 0.0)),
                "airflow_m3s": float(front.get("airFlow", 0.0)) * CFM_TO_M3S,
            }
            racks[obj_id] = rack

    logger.info(f"    * Extracted {len(racks)} racks from JSON")
    for rid, r in racks.items():
        logger.info(f"       {rid}: P={r['power_kW']:.1f} kW, Q={r['airflow_cfm']:.0f} CFM, "
                    f"T_in={r['T_inlet_set']:.0f}°C, T_out={r['T_outlet_set']:.0f}°C")
    return racks


# ─────────────────────────────────────────────────────────────────────────────
# RACK PATCH TEMPERATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def analyze_dc_rack_patches(internal_mesh, multiblock, rack_info: dict) -> dict:
    """
    Extract per-rack inlet temperature from the CFD field and compute outlet
    temperature from thermal physics.

    Physics model:
      T_inlet  = T of the CFD field sampled at the rack inlet face geometry
                 (the ambient air arriving at the rack front — from the solver)
      ΔT_rack  = Q_rack [W] / (ṁ_rack [kg/s] × Cp [J/(kg·K)])
      T_outlet = T_inlet_mean + ΔT_rack

    The outlet patch BC in OpenFOAM sets a heat-flux, not a fixed temperature.
    Reading T from the outlet patch would give BC values, not the real exit air
    temperature. We compute it instead from energy balance.

    Raises:
        ValueError: if a rack inlet patch is not found in the multiblock.

    Returns:
        dict keyed by rack_id with measured thermal metrics per rack.
    """
    rack_metrics = {}

    # Build a dict of lowercase patch_name → patch mesh for fast lookup
    patch_dict = {}
    if multiblock is not None:
        # Try direct "boundary" sub-block first
        bnames = list(multiblock.keys()) if hasattr(multiblock, "keys") else []
        boundary_block = None
        if "boundary" in bnames:
            boundary_block = multiblock["boundary"]
        else:
            boundary_block = multiblock  # flat multiblock

        if boundary_block is not None and hasattr(boundary_block, "keys"):
            for i, pname in enumerate(boundary_block.keys()):
                patch = boundary_block[i]
                if patch is not None and patch.n_points > 0:
                    patch_dict[str(pname).lower()] = patch

    logger.info(f"    * Found {len(patch_dict)} boundary patches in multiblock")

    for rack_id, rack in rack_info.items():
        inlet_id  = rack["inlet_face_id"].lower()

        # ── Locate inlet patch ───────────────────────────────────────────────
        inlet_patch = None
        for pname, pmesh in patch_dict.items():
            if pname == inlet_id or pname.endswith(f"_{inlet_id}"):
                inlet_patch = pmesh
                logger.info(f"       Found inlet patch: {pname}  ({pmesh.n_points} pts)")
                break

        if inlet_patch is None:
            available = list(patch_dict.keys())[:10]
            raise ValueError(
                f"Rack inlet patch '{inlet_id}' not found in multiblock.\n"
                f"  Available patches (first 10): {available}\n"
                f"  Check that the case was simulated with the DC JSON geometry."
            )

        # ── Sample CFD T field at inlet patch points ─────────────────────────
        # internal_mesh.sample(source) → evaluates internal_mesh fields at
        # the points of `source` using point interpolation.
        sampled = internal_mesh.sample(inlet_patch)
        T_raw = sampled.point_data.get("T", None)

        if T_raw is None or len(T_raw) == 0:
            raise ValueError(
                f"Field 'T' not found in internal mesh for rack '{rack_id}'. "
                f"Check that the CFD result contains a temperature field."
            )

        # Convert K → °C if stored in Kelvin (buoyantSimpleFoam stores K)
        T_inlet_values = T_raw - 273.15 if float(T_raw.mean()) > 100.0 else T_raw.copy()

        # ── Compute T_outlet from physics ─────────────────────────────────────
        # ΔT = Q [W] / (ṁ [kg/s] × Cp [J/(kg·K)])
        # ṁ  = Q̇_vol [m³/s] × ρ_air [kg/m³]
        airflow_kgs = rack["airflow_m3s"] * RHO_AIR
        if airflow_kgs > 1e-9:
            delta_T_physics = rack["power_kW"] * 1000.0 / (airflow_kgs * CP_AIR)
        else:
            logger.warning(f"       Rack {rack_id}: zero airflow — ΔT set to 0")
            delta_T_physics = 0.0

        T_in_mean = float(np.mean(T_inlet_values))
        T_out_mean = T_in_mean + delta_T_physics   # physics-derived outlet T

        # ── Statistics ───────────────────────────────────────────────────────
        T_in_p50 = float(np.percentile(T_inlet_values, 50))
        T_in_p95 = float(np.percentile(T_inlet_values, 95))
        T_in_p99 = float(np.percentile(T_inlet_values, 99))
        T_in_max = float(np.max(T_inlet_values))

        # ASHRAE A2 compliance — based on CFD-sampled inlet temperature
        ashrae_ok    = (ASHRAE_A2_T_SUPPLY_MIN <= T_in_mean <= ASHRAE_A2_T_SUPPLY_MAX)
        ashrae_class = "A2 ✓" if ashrae_ok else "EXCEEDS ✗"

        rack_metrics[rack_id] = {
            "T_inlet_mean":  T_in_mean,
            "T_inlet_p50":   T_in_p50,
            "T_inlet_p95":   T_in_p95,
            "T_inlet_p99":   T_in_p99,
            "T_inlet_max":   T_in_max,
            "T_outlet_mean": T_out_mean,
            "delta_T":       delta_T_physics,
            "power_kW":      rack["power_kW"],
            "airflow_m3s":   rack["airflow_m3s"],
            "ashrae_ok":     ashrae_ok,
            "ashrae_class":  ashrae_class,
        }

        logger.info(
            f"    * Rack {rack_id}: "
            f"T_in={T_in_mean:.1f}°C (P95={T_in_p95:.1f}, max={T_in_max:.1f})  "
            f"ΔT={delta_T_physics:.1f}°C (from Q={rack['power_kW']:.1f}kW, "
            f"ṁ={airflow_kgs:.4f}kg/s)  "
            f"T_out={T_out_mean:.1f}°C  ASHRAE={ashrae_class}"
        )

    return rack_metrics


# ─────────────────────────────────────────────────────────────────────────────
# RCI AND RTI INDICES
# ─────────────────────────────────────────────────────────────────────────────

def calculate_rci(rack_metrics: dict) -> dict:
    """
    Calculate Rack Cooling Index (RCI_HI and RCI_LO).

    RCI_HI: measures overheating risk (T_inlet > ASHRAE_A2_MAX = 35°C)
    RCI_LO: measures undercooling / wasted cold air (T_inlet < ASHRAE_A2_LOW = 15°C)

    Target: RCI_HI ≥ 96%, RCI_LO ≥ 96%

    Returns:
        dict with RCI_HI, RCI_LO, and per-rack exceedance.
    """
    n = len(rack_metrics)
    if n == 0:
        return {"RCI_HI": 0.0, "RCI_LO": 0.0, "n_racks": 0}

    delta_T_max = ASHRAE_A2_T_SUPPLY_MAX  # = 35°C
    delta_T_low = ASHRAE_A2_T_SUPPLY_LOW  # = 15°C

    sum_exceed_hi = 0.0
    sum_exceed_lo = 0.0
    T_inlet_all = []

    for rid, m in rack_metrics.items():
        T_in = m["T_inlet_mean"]
        T_inlet_all.append(T_in)
        # High-end exceedance (overheating)
        sum_exceed_hi += max(0.0, T_in - delta_T_max)
        # Low-end exceedance (undercooling)
        sum_exceed_lo += max(0.0, delta_T_low - T_in)

    # RCI formula: 100% × (1 - Σexceed / (limit × n_racks))
    RCI_HI = max(0.0, (1.0 - sum_exceed_hi / (delta_T_max * n)) * 100.0)
    RCI_LO = max(0.0, (1.0 - sum_exceed_lo / (delta_T_low * n)) * 100.0)

    T_arr = np.array(T_inlet_all)

    logger.info(f"    * RCI_HI = {RCI_HI:.1f}%  (target ≥ 96%)")
    logger.info(f"    * RCI_LO = {RCI_LO:.1f}%  (target ≥ 96%)")

    return {
        "RCI_HI": float(RCI_HI),
        "RCI_LO": float(RCI_LO),
        "n_racks": n,
        "T_inlet_p50": float(np.percentile(T_arr, 50)),
        "T_inlet_p95": float(np.percentile(T_arr, 95)),
        "T_inlet_p99": float(np.percentile(T_arr, 99)),
        "T_inlet_max": float(np.max(T_arr)),
        "n_ashrae_ok": sum(1 for m in rack_metrics.values() if m["ashrae_ok"]),
    }


def calculate_rti(rack_metrics: dict, T_supply_mean: float) -> dict:
    """
    Calculate Return Temperature Index (RTI).

    RTI = (T_return_mean_all - T_supply) / (T_IT_outlet_mean_all - T_supply)

    Interpretation:
      RTI = 1.0  — perfect: all cooling goes to IT
      RTI > 1.0  — hot air recirculation (hot aisle leaking to cold)
      RTI < 1.0  — bypass cooling (cold air reaching outlet without cooling IT)

    Args:
        rack_metrics: per-rack temperature metrics
        T_supply_mean: mean CRAC supply temperature [°C]
    """
    if not rack_metrics:
        return {"RTI": 0.0, "T_return_mean": 0.0, "T_supply_mean": T_supply_mean}

    T_return_values = [m["T_outlet_mean"] for m in rack_metrics.values()]
    T_outlet_values = [m["T_outlet_mean"] for m in rack_metrics.values()]

    T_return_mean  = float(np.mean(T_return_values))
    T_outlet_mean  = float(np.mean(T_outlet_values))

    denom = T_outlet_mean - T_supply_mean
    if abs(denom) < 0.1:
        RTI = 0.0
        logger.warning(f"    * RTI denominator too small (T_IT_outlet ≈ T_supply)")
    else:
        RTI = (T_return_mean - T_supply_mean) / denom

    if RTI > 1.05:
        interpretation = "Hot air recirculation detected"
    elif RTI < 0.95:
        interpretation = "Bypass cooling (cold air short-circuit)"
    else:
        interpretation = "Balanced (efficient cooling)"

    logger.info(f"    * RTI = {RTI:.3f} — {interpretation}")
    logger.info(f"       T_supply={T_supply_mean:.1f}°C, T_return={T_return_mean:.1f}°C, T_IT_out={T_outlet_mean:.1f}°C")

    return {
        "RTI": float(RTI),
        "T_supply_mean": float(T_supply_mean),
        "T_return_mean": float(T_return_mean),
        "T_IT_outlet_mean": float(T_outlet_mean),
        "interpretation": interpretation,
    }


# ─────────────────────────────────────────────────────────────────────────────
# COOLING EFFICIENCY
# ─────────────────────────────────────────────────────────────────────────────

def calculate_cooling_efficiency(rack_info: dict, rack_metrics: dict,
                                  mass_flow_crac: float, T_supply: float) -> dict:
    """
    Calculate partial thermal PUE / cooling efficiency.

    η_cooling = Q_IT_total / Q_CRAC_supply

    Q_IT_total  = Σ thermalPower_kW of all racks (from JSON design values)
    Q_CRAC      = ṁ_CRAC × Cp × (T_return_mean - T_supply)

    Interpretation:
      1.0        — ideal: all CRAC cooling absorbed by IT
      0.85-0.95  — well-designed DC
      < 0.80     — significant bypass or mixing losses

    Args:
        rack_info:       JSON rack metadata (with power_kW)
        rack_metrics:    CFD-derived rack temperatures
        mass_flow_crac:  CRAC mass flow rate [kg/s] (from postProcessing/massFlow)
        T_supply:        CRAC supply temperature [°C]
    """
    Q_IT_kW = sum(r["power_kW"] for r in rack_info.values())

    # Mean return temperature across all rack outlets
    if rack_metrics:
        T_return = float(np.mean([m["T_outlet_mean"] for m in rack_metrics.values()]))
    else:
        T_return = T_supply + 10.0  # fallback estimate

    delta_T_crac = T_return - T_supply
    Q_CRAC_kW = mass_flow_crac * CP_AIR * delta_T_crac / 1000.0  # W → kW

    if Q_CRAC_kW > 0.01:
        eta = Q_IT_kW / Q_CRAC_kW
    else:
        eta = 0.0
        logger.warning(f"    * Q_CRAC too small ({Q_CRAC_kW:.3f} kW) — η_cooling = 0")

    if eta >= 0.95:
        quality = "Excellent"
    elif eta >= 0.85:
        quality = "Good"
    elif eta >= 0.80:
        quality = "Acceptable"
    else:
        quality = "Poor — significant bypass/mixing losses"

    logger.info(f"    * Q_IT_total = {Q_IT_kW:.2f} kW")
    logger.info(f"    * Q_CRAC = {Q_CRAC_kW:.2f} kW  (ṁ={mass_flow_crac:.4f} kg/s, ΔT={delta_T_crac:.1f}°C)")
    logger.info(f"    * η_cooling = {eta:.3f}  [{quality}]")

    return {
        "eta_cooling": float(eta),
        "Q_IT_kW": float(Q_IT_kW),
        "Q_CRAC_kW": float(Q_CRAC_kW),
        "T_supply": float(T_supply),
        "T_return_mean": float(T_return),
        "delta_T_crac": float(delta_T_crac),
        "mass_flow_crac_kgs": float(mass_flow_crac),
        "quality": quality,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MASS FLOW FROM postProcessing
# ─────────────────────────────────────────────────────────────────────────────

def read_crac_mass_flows(sim_path: str, T_field: np.ndarray) -> tuple:
    """
    Read CRAC mass flow and supply temperature from postProcessing files.

    Returns:
        (mass_flow_crac_kgs, T_supply_mean)
    """
    import glob as _glob

    postproc_path = os.path.join(sim_path, "postProcessing")
    pattern = os.path.join(postproc_path, "massFlow_*", "0", "surfaceFieldValue.dat")
    files = _glob.glob(pattern)

    inlet_flows = []
    outlet_flows = []

    for filepath in files:
        try:
            patch_name = os.path.basename(
                os.path.dirname(os.path.dirname(filepath))
            ).replace("massFlow_", "")
            data = np.loadtxt(filepath, comments="#")
            phi = float(data[-1, 1]) if data.ndim > 1 else float(data[1])

            if phi > 0:
                inlet_flows.append((patch_name, phi))
            else:
                outlet_flows.append((patch_name, abs(phi)))

            logger.info(f"       massFlow {patch_name}: φ = {phi:.6f} m³/s")
        except Exception as e:
            logger.warning(f"       Failed to read {filepath}: {e}")

    total_inlet_m3s = sum(phi for _, phi in inlet_flows)
    mass_flow_kgs = total_inlet_m3s * RHO_AIR  # volumetric → mass flow [kg/s]

    # CRAC supply temperature: use minimum T in the domain (cold supply air)
    if T_field is not None and len(T_field) > 0:
        T_celsius = T_field - 273.15 if T_field.mean() > 100 else T_field
        T_supply = float(np.percentile(T_celsius, 5))  # P5 ≈ supply temperature
    else:
        T_supply = 22.0  # fallback

    logger.info(f"    * Total CRAC inlet flow: {total_inlet_m3s:.6f} m³/s = {mass_flow_kgs:.4f} kg/s")
    logger.info(f"    * Estimated T_supply (P5 of domain): {T_supply:.1f}°C")

    return mass_flow_kgs, T_supply


# ─────────────────────────────────────────────────────────────────────────────
# HORIZONTAL SLICE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def render_dc_slice_png(slice_mesh, z_height: float, post_path: str,
                         variable: str = "T") -> str:
    """
    Render a horizontal slice PNG for Data Center analysis.

    Args:
        slice_mesh: PyVista slice mesh
        z_height: Height of the slice [m]
        post_path: Post-processing output directory
        variable: Field to render ('T' or 'U_mag')

    Returns:
        Path to saved PNG.
    """
    plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])

    if variable == "T":
        T = slice_mesh.point_data.get("T", None)
        if T is not None:
            T_c = T - 273.15 if T.mean() > 100 else T.copy()
            slice_mesh.point_data["T_celsius"] = T_c
            scalar_field = "T_celsius"
        else:
            logger.warning(f"       T field missing in slice at z={z_height}m")
            plotter.close()
            return ""

        plotter.add_mesh(
            slice_mesh, scalars=scalar_field,
            cmap="coolwarm", clim=(15, 45), show_edges=False,
            scalar_bar_args={
                "title": "Temperature [°C]", "title_font_size": 38,
                "label_font_size": 64, "n_labels": 5, "fmt": "%.0f",
                "vertical": True, "position_x": 0.85, "position_y": 0.25,
                "height": 0.5, "width": 0.12, "shadow": True,
            },
        )

        # Add ASHRAE A2 limit contour (35°C)
        try:
            c35 = slice_mesh.contour(isosurfaces=[35.0], scalars=scalar_field)
            if c35.n_points > 0:
                plotter.add_mesh(c35, color="red", line_width=4,
                                 render_lines_as_tubes=False, lighting=False)
                if c35.n_points > 0:
                    mid = c35.points[c35.n_points // 2]
                    plotter.add_point_labels(
                        [mid], ["35°C\nASHRAE A2"],
                        font_size=40, text_color="red", font_family="arial",
                        fill_shape=True, shape_color="white",
                        shape_opacity=0.9, point_size=0, always_visible=True,
                    )
        except Exception:
            pass

        title = f"Temperature [°C] at z={z_height}m — DC Thermal Map"

    elif variable == "U_mag":
        U = slice_mesh.point_data.get("U", None)
        if U is not None and len(U.shape) > 1:
            U_mag = np.linalg.norm(U, axis=1)
            slice_mesh.point_data["U_mag"] = U_mag
        elif "U_mag" in slice_mesh.point_data:
            U_mag = slice_mesh.point_data["U_mag"]
        else:
            logger.warning(f"       U field missing in slice at z={z_height}m")
            plotter.close()
            return ""

        plotter.add_mesh(
            slice_mesh, scalars="U_mag",
            cmap="viridis", clim=(0, 2.0), show_edges=False,
            scalar_bar_args={
                "title": "Velocity [m/s]", "title_font_size": 38,
                "label_font_size": 64, "n_labels": 5, "fmt": "%.1f",
                "vertical": True, "position_x": 0.85, "position_y": 0.25,
                "height": 0.5, "width": 0.12, "shadow": True,
            },
        )
        title = f"Velocity [m/s] at z={z_height}m — Airflow Distribution"
    else:
        plotter.close()
        return ""

    plotter.camera_position = "iso"
    plotter.camera.zoom(1.2)
    plotter.add_text(title, position="upper_edge", font_size=16, color="black")
    plotter.show_axes()

    images_dir = os.path.join(post_path, "images")
    os.makedirs(images_dir, exist_ok=True)
    png_path = os.path.join(images_dir, f"dc_{variable}_z{z_height}m.png")
    plotter.screenshot(png_path, transparent_background=False)
    plotter.close()

    logger.info(f"       Saved PNG: {os.path.basename(png_path)}")
    return png_path


def analyze_dc_slices(internal_mesh, post_path: str) -> dict:
    """
    Analyze temperature and velocity in horizontal slices at rack heights.

    Heights: 0.5m (bottom rack), 1.0m (mid rack), 1.5m (top rack)

    Returns:
        dict with slice metrics (T stats, hot_spot_pct, stagnation_pct).
    """
    heights = {"bottom_rack": 0.5, "mid_rack": 1.0, "top_rack": 1.5}
    slice_results = {}

    vtk_dir = os.path.join(post_path, "vtk")
    os.makedirs(vtk_dir, exist_ok=True)
    os.makedirs(os.path.join(post_path, "images"), exist_ok=True)

    for name, z in heights.items():
        logger.info(f"    * DC slice: {name} (z={z}m)")
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sl = internal_mesh.slice(normal="z", origin=(0, 0, z))
        except Exception:
            sl = internal_mesh.slice(normal="z", origin=(0, 0, z))

        T = sl.point_data.get("T", None)
        U = sl.point_data.get("U", None)

        metrics = {"height_m": z, "plane_name": name}

        if T is not None:
            T_c = T - 273.15 if T.mean() > 100 else T.copy()
            metrics["T_mean"] = float(T_c.mean())
            metrics["T_p95"]  = float(np.percentile(T_c, 95))
            metrics["T_p99"]  = float(np.percentile(T_c, 99))
            metrics["T_max"]  = float(T_c.max())
            # Hot spot: T > ASHRAE A2 max (35°C)
            metrics["hot_spot_pct"] = float(100.0 * (T_c > ASHRAE_A2_T_SUPPLY_MAX).sum() / len(T_c))

        if U is not None and len(U.shape) > 1:
            U_mag = np.linalg.norm(U, axis=1)
            metrics["U_mean"] = float(U_mag.mean())
            metrics["U_max"]  = float(U_mag.max())
            metrics["stagnation_pct"] = float(100.0 * (U_mag < 0.05).sum() / len(U_mag))

        slice_results[name] = metrics
        logger.info(f"       T_mean={metrics.get('T_mean', 0):.1f}°C, "
                    f"hot_spot={metrics.get('hot_spot_pct', 0):.1f}%, "
                    f"stagnation={metrics.get('stagnation_pct', 0):.1f}%")

        # Save VTK
        vtk_path = os.path.join(vtk_dir, f"dc_slice_{name}_{z}m.vtk")
        sl.save(vtk_path)

        # Render PNGs
        render_dc_slice_png(sl, z, post_path, variable="T")
        render_dc_slice_png(sl, z, post_path, variable="U_mag")

    return slice_results


# ─────────────────────────────────────────────────────────────────────────────
# HTML REPORT GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def _color(value, good_thresh, warn_thresh, higher_is_better=True):
    """Return CSS color class string based on thresholds."""
    if higher_is_better:
        if value >= good_thresh:
            return "comfort-high"
        elif value >= warn_thresh:
            return "comfort-medium"
        else:
            return "comfort-low"
    else:
        if value <= good_thresh:
            return "comfort-high"
        elif value <= warn_thresh:
            return "comfort-medium"
        else:
            return "comfort-low"


_CSS_BASE = """
        * { margin:0; padding:0; box-sizing:border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                         'Helvetica Neue', Arial, sans-serif;
            background:#f5f5f5; color:#333; line-height:1.6; padding:20px;
        }
        .container { max-width:1400px; margin:0 auto; background:white;
            box-shadow:0 2px 8px rgba(0,0,0,0.1); border-radius:8px; overflow:hidden; }
        .header { padding:40px; text-align:center; color:white; }
        .header h1 { font-size:2.5em; margin-bottom:10px; font-weight:300; }
        .header .subtitle { font-size:1.1em; opacity:0.9; }
        .executive-summary { background:#f8f9fa; padding:30px; border-bottom:3px solid #e9ecef; }
        .executive-summary h2 { margin-bottom:15px; color:#495057; }
        .summary-metric { display:inline-block; padding:15px 30px; margin:10px;
            background:white; border-radius:6px; box-shadow:0 1px 3px rgba(0,0,0,0.1); }
        .summary-metric .label { font-size:0.9em; color:#6c757d;
            text-transform:uppercase; letter-spacing:1px; }
        .summary-metric .value { font-size:2em; font-weight:bold; margin-top:5px; }
        .comfort-high { color:#28a745; } .comfort-medium { color:#ffc107; }
        .comfort-low  { color:#dc3545; }
        .section { padding:40px; border-bottom:1px solid #e9ecef; }
        .section:last-child { border-bottom:none; }
        .section h2 { color:#495057; font-size:1.8em; margin-bottom:20px; }
        .images-container { display:grid; grid-template-columns:1fr 1fr;
            gap:20px; margin-bottom:30px; }
        .image-box { background:#f8f9fa; border-radius:6px; overflow:hidden;
            box-shadow:0 2px 4px rgba(0,0,0,0.1); }
        .image-box img { width:100%; display:block; }
        .image-box .caption { padding:15px; text-align:center; font-weight:600;
            background:white; border-top:2px solid #e9ecef; }
        .metrics-table { width:100%; border-collapse:collapse; margin-top:20px;
            background:white; box-shadow:0 1px 3px rgba(0,0,0,0.1);
            border-radius:6px; overflow:hidden; }
        .metrics-table th, .metrics-table td { padding:12px 15px; text-align:left;
            border-bottom:1px solid #e9ecef; }
        .metrics-table th { background:#f8f9fa; font-weight:600; color:#495057;
            text-transform:uppercase; font-size:0.85em; letter-spacing:0.5px; }
        .metrics-table tr:last-child td { border-bottom:none; }
        .metrics-table .metric-value { font-weight:600; font-size:1.1em; }
        .status-badge { display:inline-block; padding:4px 12px; border-radius:12px;
            font-size:0.8em; font-weight:600; text-transform:uppercase; }
        .badge-ok  { background:#d4edda; color:#155724; }
        .badge-warn{ background:#fff3cd; color:#856404; }
        .badge-fail{ background:#f8d7da; color:#721c24; }
        .footer { background:#343a40; color:#adb5bd; padding:30px; text-align:center; }
        .footer .info-row { margin:5px 0; }
        .footer .label { color:#6c757d; }
        .footer .value { color:#f8f9fa; font-weight:600; }
        @media (max-width:768px) {
            .images-container { grid-template-columns:1fr; }
            .header h1 { font-size:1.8em; }
        }
"""


def generate_dc_rack_report(rack_metrics: dict, rci: dict, rti: dict,
                              cooling_eff: dict, post_path: str,
                              case_name: str = "DC_Case") -> str:
    """
    Generate HTML report: per-rack table with T_inlet percentiles, ΔT, RTI, RCI, η_cooling.
    """
    logger.info("    * Generating DC rack report")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    RCI_HI = rci.get("RCI_HI", 0.0)
    RCI_LO = rci.get("RCI_LO", 0.0)
    RTI    = rti.get("RTI", 0.0)
    eta    = cooling_eff.get("eta_cooling", 0.0)
    n_ok   = rci.get("n_ashrae_ok", 0)
    n_tot  = rci.get("n_racks", 1)

    html = f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>DC Rack Report — {case_name}</title>
<style>{_CSS_BASE}</style></head><body><div class="container">

<div class="header" style="background:linear-gradient(135deg,#1a1a2e 0%,#16213e 60%,#0f3460 100%);">
    <h1>Rack Thermal Performance Report</h1>
    <div class="subtitle">{case_name}</div>
    <div class="subtitle">Generated: {timestamp}</div>
</div>

<div class="executive-summary">
    <h2>Executive Summary</h2>
    <div class="summary-metric">
        <div class="label">RCI High</div>
        <div class="value {_color(RCI_HI, 96, 90)}">{RCI_HI:.1f}%</div>
        <div style="font-size:0.75em;color:#6c757d;margin-top:8px">
            Rack Cooling Index — overheating risk<br>Target ≥ 96%
        </div>
    </div>
    <div class="summary-metric">
        <div class="label">RCI Low</div>
        <div class="value {_color(RCI_LO, 96, 90)}">{RCI_LO:.1f}%</div>
        <div style="font-size:0.75em;color:#6c757d;margin-top:8px">
            Rack Cooling Index — undercooling waste<br>Target ≥ 96%
        </div>
    </div>
    <div class="summary-metric">
        <div class="label">RTI</div>
        <div class="value {_color(abs(RTI - 1.0), 0.05, 0.15, higher_is_better=False)}">{RTI:.3f}</div>
        <div style="font-size:0.75em;color:#6c757d;margin-top:8px">
            Return Temp Index<br>Ideal = 1.0
        </div>
    </div>
    <div class="summary-metric">
        <div class="label">η Cooling</div>
        <div class="value {_color(eta, 0.85, 0.80)}">{eta:.2f}</div>
        <div style="font-size:0.75em;color:#6c757d;margin-top:8px">
            Q_IT / Q_CRAC<br>Target ≥ 0.85
        </div>
    </div>
    <div class="summary-metric">
        <div class="label">ASHRAE A2</div>
        <div class="value {_color(n_ok / max(n_tot, 1) * 100, 90, 70)}">{n_ok}/{n_tot}</div>
        <div style="font-size:0.75em;color:#6c757d;margin-top:8px">
            Racks within 10–35°C<br>Inlet temperature
        </div>
    </div>
</div>

<!-- Per-rack table -->
<div class="section">
    <h2>Per-Rack Thermal Metrics</h2>
    <table class="metrics-table">
        <thead><tr>
            <th>Rack ID</th>
            <th>T_inlet mean</th>
            <th>T_inlet P95</th>
            <th>T_inlet P99</th>
            <th>T_inlet max</th>
            <th>T_outlet</th>
            <th>ΔT</th>
            <th>Power</th>
            <th>ASHRAE A2</th>
        </tr></thead><tbody>
"""
    for rid, m in rack_metrics.items():
        badge = "badge-ok" if m["ashrae_ok"] else "badge-fail"
        t_in_color = _color(m["T_inlet_mean"], ASHRAE_A2_T_SUPPLY_MAX,
                            ASHRAE_A2_T_SUPPLY_MAX + 5, higher_is_better=False)
        html += f"""
        <tr>
            <td><strong>{rid}</strong></td>
            <td class="metric-value {t_in_color}">{m['T_inlet_mean']:.1f}°C</td>
            <td class="metric-value">{m['T_inlet_p95']:.1f}°C</td>
            <td class="metric-value">{m['T_inlet_p99']:.1f}°C</td>
            <td class="metric-value">{m['T_inlet_max']:.1f}°C</td>
            <td class="metric-value">{m['T_outlet_mean']:.1f}°C</td>
            <td class="metric-value">{m['delta_T']:.1f}°C</td>
            <td class="metric-value">{m['power_kW']:.1f} kW</td>
            <td><span class="status-badge {badge}">{m['ashrae_class']}</span></td>
        </tr>"""

    html += f"""
        </tbody>
    </table>
</div>

<!-- Global indices table -->
<div class="section">
    <h2>Global Cooling Indices</h2>
    <table class="metrics-table">
        <tr><th>Metric</th><th>Value</th><th>Target / Notes</th></tr>
        <tr>
            <td>T_inlet P50 (all racks)<br>
                <span style="font-size:0.9em;color:#6c757d">Median supply air temperature</span></td>
            <td class="metric-value {_color(rci.get('T_inlet_p50',25), ASHRAE_A2_T_SUPPLY_MAX, ASHRAE_A2_T_SUPPLY_MAX+5, False)}">{rci.get('T_inlet_p50',0):.1f}°C</td>
            <td>10–35°C (ASHRAE A2)</td>
        </tr>
        <tr>
            <td>T_inlet P95 (all racks)<br>
                <span style="font-size:0.9em;color:#6c757d">95th percentile — systemic vs isolated issues</span></td>
            <td class="metric-value {_color(rci.get('T_inlet_p95',25), ASHRAE_A2_T_SUPPLY_MAX, ASHRAE_A2_T_SUPPLY_MAX+5, False)}">{rci.get('T_inlet_p95',0):.1f}°C</td>
            <td>&lt; 35°C — if P95 &gt; 35°C: generalised problem</td>
        </tr>
        <tr>
            <td>T_inlet P99 (all racks)</td>
            <td class="metric-value">{rci.get('T_inlet_p99',0):.1f}°C</td>
            <td>&lt; 35°C — if P99 &gt; 35°C: structural hot spots</td>
        </tr>
        <tr>
            <td>RTI<br><span style="font-size:0.9em;color:#6c757d">(T_return − T_supply) / (T_IT_out − T_supply)</span></td>
            <td class="metric-value">{RTI:.3f}</td>
            <td>1.0 = perfect | &gt;1.0 recirculation | &lt;1.0 bypass<br>
                <span style="font-size:0.85em;color:#6c757d">{rti.get('interpretation','')}</span></td>
        </tr>
        <tr>
            <td>RCI_HI<br><span style="font-size:0.9em;color:#6c757d">Overheating risk index</span></td>
            <td class="metric-value {_color(RCI_HI, 96, 90)}">{RCI_HI:.1f}%</td>
            <td>≥ 96% target (ASHRAE TC9.9)</td>
        </tr>
        <tr>
            <td>RCI_LO<br><span style="font-size:0.9em;color:#6c757d">Cold air waste index</span></td>
            <td class="metric-value {_color(RCI_LO, 96, 90)}">{RCI_LO:.1f}%</td>
            <td>≥ 96% target</td>
        </tr>
        <tr>
            <td>η_cooling<br><span style="font-size:0.9em;color:#6c757d">Q_IT / Q_CRAC — partial thermal PUE</span></td>
            <td class="metric-value {_color(eta, 0.85, 0.80)}">{eta:.3f}</td>
            <td>0.85–0.95 well-designed | &lt;0.80 significant losses<br>
                <span style="font-size:0.85em;color:#6c757d">
                Q_IT = {cooling_eff.get('Q_IT_kW',0):.1f} kW |
                Q_CRAC = {cooling_eff.get('Q_CRAC_kW',0):.1f} kW |
                ΔT_CRAC = {cooling_eff.get('delta_T_crac',0):.1f}°C
                </span>
            </td>
        </tr>
    </table>
</div>

<div class="footer">
    <div class="info-row"><span class="label">Standards:</span>
        <span class="value">ASHRAE TC9.9 — Thermal Guidelines for Data Processing Environments</span></div>
    <div class="info-row"><span class="label">Solver:</span>
        <span class="value">OpenFOAM buoyantBoussinesqSimpleFoam</span></div>
</div>
</div></body></html>"""

    html_path = os.path.join(post_path, "dc_rack_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"    * DC rack report: {html_path}")
    return html_path


def generate_dc_thermal_report(slice_results: dict, post_path: str,
                                 case_name: str = "DC_Case") -> str:
    """
    Generate HTML thermal map report with horizontal slice images at rack heights.
    """
    logger.info("    * Generating DC thermal report")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    planes = [("bottom_rack", 0.5), ("mid_rack", 1.0), ("top_rack", 1.5)]
    plane_titles = {
        "bottom_rack": "Bottom Rack Level (z = 0.5 m)",
        "mid_rack":    "Mid Rack Level (z = 1.0 m)",
        "top_rack":    "Top Rack Level (z = 1.5 m)",
    }

    global_hot_spot = np.mean([
        slice_results[k]["hot_spot_pct"]
        for k in ("bottom_rack", "mid_rack", "top_rack")
        if k in slice_results and "hot_spot_pct" in slice_results[k]
    ]) if slice_results else 0.0

    html = f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>DC Thermal Map — {case_name}</title>
<style>{_CSS_BASE}</style></head><body><div class="container">

<div class="header" style="background:linear-gradient(135deg,#b91c1c 0%,#991b1b 60%,#7f1d1d 100%);">
    <h1>Data Center Thermal Map Report</h1>
    <div class="subtitle">{case_name}</div>
    <div class="subtitle">Generated: {timestamp}</div>
</div>

<div class="executive-summary">
    <h2>Executive Summary</h2>
    <div class="summary-metric">
        <div class="label">Hot Spot Area (avg)</div>
        <div class="value {_color(global_hot_spot, 5, 15, higher_is_better=False)}">{global_hot_spot:.1f}%</div>
        <div style="font-size:0.75em;color:#6c757d;margin-top:8px">
            % area with T &gt; 35°C (ASHRAE A2 max)<br>
            Target &lt; 5%
        </div>
    </div>
</div>
"""
    for name, z in planes:
        if name not in slice_results:
            continue
        m = slice_results[name]
        t_png = f"images/dc_T_z{z}m.png"
        u_png = f"images/dc_U_mag_z{z}m.png"

        html += f"""
<div class="section">
    <h2>{plane_titles[name]}</h2>
    <div class="images-container">
        <div class="image-box">
            <img src="{t_png}" alt="Temperature at {z}m">
            <div class="caption">Temperature Field — {z} m</div>
        </div>
        <div class="image-box">
            <img src="{u_png}" alt="Velocity at {z}m">
            <div class="caption">Velocity Field — {z} m</div>
        </div>
    </div>
    <table class="metrics-table">
        <tr><th>Metric</th><th>Value</th><th>Target</th></tr>
        <tr>
            <td>T mean</td>
            <td class="metric-value">{m.get('T_mean',0):.1f}°C</td>
            <td>10–35°C (ASHRAE A2)</td>
        </tr>
        <tr>
            <td>T P95</td>
            <td class="metric-value {_color(m.get('T_p95',0), ASHRAE_A2_T_SUPPLY_MAX, ASHRAE_A2_T_SUPPLY_MAX+5, False)}">{m.get('T_p95',0):.1f}°C</td>
            <td>&lt; 35°C</td>
        </tr>
        <tr>
            <td>T max</td>
            <td class="metric-value {_color(m.get('T_max',0), ASHRAE_A2_T_SUPPLY_MAX, ASHRAE_A2_T_SUPPLY_MAX+5, False)}">{m.get('T_max',0):.1f}°C</td>
            <td>&lt; 35°C</td>
        </tr>
        <tr>
            <td>Hot spot area (T &gt; 35°C)</td>
            <td class="metric-value {_color(m.get('hot_spot_pct',0), 5, 15, False)}">{m.get('hot_spot_pct',0):.1f}%</td>
            <td>&lt; 5%</td>
        </tr>
        <tr>
            <td>Stagnation area (U &lt; 0.05 m/s)</td>
            <td class="metric-value {_color(m.get('stagnation_pct',0), 10, 25, False)}">{m.get('stagnation_pct',0):.1f}%</td>
            <td>&lt; 10%</td>
        </tr>
    </table>
</div>
"""

    html += """
<div class="footer">
    <div class="info-row"><span class="label">Red contour:</span>
        <span class="value">ASHRAE A2 limit — 35°C</span></div>
    <div class="info-row"><span class="label">Colormap range:</span>
        <span class="value">15°C (blue) → 45°C (red)</span></div>
</div>
</div></body></html>"""

    html_path = os.path.join(post_path, "dc_thermal_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"    * DC thermal report: {html_path}")
    return html_path


def generate_dc_airflow_report(slice_results: dict, post_path: str,
                                 case_name: str = "DC_Case") -> str:
    """
    Generate HTML airflow report with velocity maps and stagnation analysis.
    """
    logger.info("    * Generating DC airflow report")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    planes = [("bottom_rack", 0.5), ("mid_rack", 1.0), ("top_rack", 1.5)]
    plane_titles = {
        "bottom_rack": "Bottom Rack Level (z = 0.5 m)",
        "mid_rack":    "Mid Rack Level (z = 1.0 m)",
        "top_rack":    "Top Rack Level (z = 1.5 m)",
    }

    global_stagnation = np.mean([
        slice_results[k]["stagnation_pct"]
        for k in ("bottom_rack", "mid_rack", "top_rack")
        if k in slice_results and "stagnation_pct" in slice_results[k]
    ]) if slice_results else 0.0

    html = f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>DC Airflow Report — {case_name}</title>
<style>{_CSS_BASE}</style></head><body><div class="container">

<div class="header" style="background:linear-gradient(135deg,#1e3a5f 0%,#1e40af 60%,#1d4ed8 100%);">
    <h1>Data Center Airflow Report</h1>
    <div class="subtitle">{case_name}</div>
    <div class="subtitle">Generated: {timestamp}</div>
</div>

<div class="executive-summary">
    <h2>Executive Summary</h2>
    <div class="summary-metric">
        <div class="label">Stagnation Area (avg)</div>
        <div class="value {_color(global_stagnation, 10, 25, higher_is_better=False)}">{global_stagnation:.1f}%</div>
        <div style="font-size:0.75em;color:#6c757d;margin-top:8px">
            % area with U &lt; 0.05 m/s (dead zones)<br>
            Target &lt; 10%
        </div>
    </div>
</div>
"""
    for name, z in planes:
        if name not in slice_results:
            continue
        m = slice_results[name]
        u_png = f"images/dc_U_mag_z{z}m.png"

        html += f"""
<div class="section">
    <h2>{plane_titles[name]}</h2>
    <div class="images-container" style="grid-template-columns:1fr;">
        <div class="image-box">
            <img src="{u_png}" alt="Velocity at {z}m">
            <div class="caption">Velocity Magnitude [m/s] — {z} m</div>
        </div>
    </div>
    <table class="metrics-table">
        <tr><th>Metric</th><th>Value</th><th>Target</th></tr>
        <tr>
            <td>Velocity Mean</td>
            <td class="metric-value">{m.get('U_mean',0):.2f} m/s</td>
            <td>0.5–2.0 m/s (typical rack face)</td>
        </tr>
        <tr>
            <td>Velocity Max</td>
            <td class="metric-value">{m.get('U_max',0):.2f} m/s</td>
            <td>—</td>
        </tr>
        <tr>
            <td>Stagnation Zones (U &lt; 0.05 m/s)</td>
            <td class="metric-value {_color(m.get('stagnation_pct',0), 10, 25, False)}">{m.get('stagnation_pct',0):.1f}%</td>
            <td>&lt; 10% (low recirculation risk)</td>
        </tr>
    </table>
</div>
"""

    html += """
<div class="footer">
    <div class="info-row"><span class="label">Standards:</span>
        <span class="value">ASHRAE TC9.9 | ANSI/TIA-942</span></div>
</div>
</div></body></html>"""

    html_path = os.path.join(post_path, "dc_airflow_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"    * DC airflow report: {html_path}")
    return html_path


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def run_data_centers(case_name: str, sim_path: str, post_path: str) -> None:
    """
    Run Data Center post-processing pipeline.

    Reads rack metadata from geo/building_config.json, loads CFD results,
    computes all thermal KPIs and generates 3 HTML reports + JSON metrics.

    Args:
        case_name: Name of the simulation case
        sim_path:  Path to simulation directory (with VTK/)
        post_path: Path to post-processing output directory
    """
    logger.info("\n=========== RUNNING DATA CENTER POST-PROCESSING ===========")
    logger.info(f"Simulation path: {sim_path}")
    logger.info(f"Post-processing output: {post_path}")

    os.makedirs(post_path, exist_ok=True)

    # 1 ── Load JSON rack metadata ────────────────────────────────────────────
    json_path = os.path.join(sim_path, "..", "geo", "building_config.json")
    if not os.path.isfile(json_path):
        logger.warning(f"    * building_config.json not found at {json_path}")
        logger.warning("    * Proceeding with empty rack_info (no JSON rack data)")
        rack_info = {}
        json_payload = {}
    else:
        with open(json_path, "r") as f:
            json_payload = json.load(f)
        logger.info(f"\n1 - Extracting rack info from JSON: {json_path}")
        rack_info = extract_rack_info_from_json(json_payload)

    # 2 ── Load CFD results ───────────────────────────────────────────────────
    logger.info("\n2 - Loading CFD results")
    internal_mesh, surfaces_mesh, multiblock = load_foam_results(sim_path)
    logger.info(f"    * Loaded mesh with {internal_mesh.n_cells:,} cells")

    # 3 ── Mass flow + T_supply from postProcessing ───────────────────────────
    logger.info("\n3 - Reading CRAC mass flows")
    T_field = internal_mesh.point_data.get("T", None)
    mass_flow_crac, T_supply = read_crac_mass_flows(sim_path, T_field)

    # 4 ── Per-rack patch temperature extraction ──────────────────────────────
    logger.info("\n4 - Analyzing rack face temperatures from CFD patches")
    rack_metrics = analyze_dc_rack_patches(internal_mesh, multiblock, rack_info)

    # 5 ── Thermal indices ────────────────────────────────────────────────────
    logger.info("\n5 - Computing RCI, RTI, cooling efficiency")
    rci = calculate_rci(rack_metrics)
    rti = calculate_rti(rack_metrics, T_supply)
    cooling_eff = calculate_cooling_efficiency(rack_info, rack_metrics,
                                                mass_flow_crac, T_supply)

    # 6 ── Horizontal slices at rack heights ──────────────────────────────────
    logger.info("\n6 - Analyzing horizontal slices at rack heights")
    slice_results = analyze_dc_slices(internal_mesh, post_path)

    # 7 ── Generate HTML reports ──────────────────────────────────────────────
    logger.info("\n7 - Generating HTML reports")
    generate_dc_rack_report(rack_metrics, rci, rti, cooling_eff, post_path, case_name)
    generate_dc_thermal_report(slice_results, post_path, case_name)
    generate_dc_airflow_report(slice_results, post_path, case_name)

    # 8 ── Save JSON metrics ──────────────────────────────────────────────────
    def _np_safe(obj):
        if isinstance(obj, dict):
            return {k: _np_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_np_safe(i) for i in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    metrics_all = {
        "rack_metrics":  rack_metrics,
        "rci":           rci,
        "rti":           rti,
        "cooling_efficiency": cooling_eff,
        "slice_results": slice_results,
    }
    metrics_path = os.path.join(post_path, "dc_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(_np_safe(metrics_all), f, indent=2)
    logger.info(f"\n    * Saved DC metrics: {os.path.basename(metrics_path)}")

    logger.info("\n=========== DATA CENTER POST-PROCESSING SUMMARY ===========")
    logger.info(f"  RCI_HI = {rci.get('RCI_HI', 0):.1f}%  (target ≥ 96%)")
    logger.info(f"  RCI_LO = {rci.get('RCI_LO', 0):.1f}%  (target ≥ 96%)")
    logger.info(f"  RTI    = {rti.get('RTI', 0):.3f}  (ideal = 1.0)")
    logger.info(f"  η_cool = {cooling_eff.get('eta_cooling', 0):.3f}  (target ≥ 0.85)")
    logger.info(f"  ASHRAE = {rci.get('n_ashrae_ok', 0)}/{rci.get('n_racks', 0)} racks compliant")
    logger.info(f"\n✅ Data Center post-processing completed")
    logger.info(f"Results saved at: {post_path}")
