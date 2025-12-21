"""
Convert VTK text format files to vtkjs JSON format for web visualization.
"""
import json
import numpy as np
import pyvista as pv
import logging

logger = logging.getLogger(__name__)


def vtk_to_vtkjs(vtk_path: str, output_path: str) -> None:
    """
    Convert a VTK file to vtkjs JSON format.
    
    Args:
        vtk_path: Path to input .vtk file
        output_path: Path to output .vtkjs file
    """
    try:
        # Load VTK file
        mesh = pv.read(vtk_path)
        
        # Convert cell data to point data BEFORE surface extraction
        # OpenFOAM stores fields (T, U, p, PMV, PPD) in cell_data
        if mesh.cell_data:
            logger.info(f"    * Converting cell data to point data: {list(mesh.cell_data.keys())}")
            
            # Try global conversion first (fastest)
            try:
                mesh = mesh.cell_data_to_point_data()
                logger.info(f"    * Successfully converted all cell data")
            except Exception as e:
                logger.warning(f"    * Global conversion failed: {e}")
                logger.info(f"    * Attempting per-field conversion...")
                
                # Fall back to per-field conversion for robustness
                for field_name in list(mesh.cell_data.keys()):
                    try:
                        # Create temporary mesh with only this field
                        temp_mesh = mesh.copy()
                        # Remove all other cell data
                        for other_field in list(temp_mesh.cell_data.keys()):
                            if other_field != field_name:
                                del temp_mesh.cell_data[other_field]
                        # Convert this field
                        temp_mesh = temp_mesh.cell_data_to_point_data()
                        # Copy to original mesh
                        if field_name in temp_mesh.point_data:
                            mesh.point_data[field_name] = temp_mesh.point_data[field_name]
                            logger.info(f"    * ✓ Converted field: {field_name}")
                    except Exception as field_error:
                        logger.warning(f"    * ✗ Failed to convert field {field_name}: {field_error}")
        
        # Verify critical fields are present and valid (ALWAYS validate, even if no cell_data)
        critical_fields = ['T', 'U', 'p', 'p_rgh']
        available_critical = []
        for field in critical_fields:
            if field in mesh.point_data:
                # Check if field contains finite values
                field_data = mesh.point_data[field]
                if np.all(np.isfinite(field_data)):
                    available_critical.append(field)
                else:
                    logger.warning(f"    * Field {field} contains non-finite values")
        
        if available_critical:
            logger.info(f"    * Critical fields available: {available_critical}")
        else:
            raise ValueError(
                f"No critical OpenFOAM fields found in point_data after conversion. "
                f"Expected at least one of: {critical_fields}. "
                f"Available fields: {list(mesh.point_data.keys())}"
            )
        
        # If it's an UnstructuredGrid (3D volume), extract surface to get all faces
        # This is what ParaView does - shows the surface representation of the volume
        # Keep all internal data for slicing in web viewer
        if hasattr(mesh, 'extract_surface') and mesh.n_cells > 1000:
            logger.info(f"    * Extracting surface from 3D volume ({mesh.n_cells} cells)...")
            # Extract surface - point data is already transferred
            surface_mesh = mesh.extract_surface()
            mesh = surface_mesh
            logger.info(f"    * Surface extracted: {mesh.n_points} points, {mesh.n_cells} cells")
            logger.info(f"    * Final scalar fields: {list(mesh.point_data.keys())}")
        
        # Extract geometry
        points = mesh.points.flatten().tolist()
        
        # Extract polygons/cells (now guaranteed to be surface faces)
        if mesh.faces.size > 0:
            # PolyData with faces
            polys = mesh.faces.tolist()
        else:
            polys = []
        
        # Build vtkjs structure
        vtkjs_data = {
            "vtkClass": "vtkPolyData",
            "points": {
                "vtkClass": "vtkPoints",
                "name": "_points",
                "numberOfComponents": 3,
                "dataType": "Float32Array",
                "size": len(mesh.points) * 3,
                "values": points
            },
            "polys": {
                "vtkClass": "vtkCellArray",
                "name": "_polys",
                "numberOfComponents": 1,
                "dataType": "Uint32Array",
                "size": len(polys),
                "values": polys
            },
            "pointData": {
                "vtkClass": "vtkDataSetAttributes",
                "activeScalars": 0,
                "arrays": []
            }
        }
        
        # Add point data arrays (fields like T, U, p)
        for array_name in mesh.point_data.keys():
            array = mesh.point_data[array_name]
            
            # Handle vector and scalar data
            if array.ndim == 1:
                # Scalar data
                n_components = 1
                values = array.tolist()
            else:
                # Vector data (e.g., velocity)
                n_components = array.shape[1]
                values = array.flatten().tolist()
            
            vtkjs_data["pointData"]["arrays"].append({
                "data": {
                    "vtkClass": "vtkDataArray",
                    "name": array_name,
                    "numberOfComponents": n_components,
                    "dataType": "Float32Array",
                    "size": len(values),
                    "values": values
                }
            })
        
        # Set first array as active scalars if available
        if len(vtkjs_data["pointData"]["arrays"]) > 0:
            vtkjs_data["pointData"]["activeScalars"] = 0
        
        # Write JSON file
        with open(output_path, 'w') as f:
            json.dump(vtkjs_data, f)
        
        logger.info(f"    * Converted VTK to vtkjs: {output_path}")
        logger.info(f"      - Points: {len(mesh.points)}")
        logger.info(f"      - Arrays: {len(vtkjs_data['pointData']['arrays'])}")
        
    except Exception as e:
        logger.error(f"    * Failed to convert VTK to vtkjs: {e}")
        raise


if __name__ == "__main__":
    # Test conversion
    import sys
    if len(sys.argv) != 3:
        print("Usage: python vtk_to_vtkjs.py <input.vtk> <output.vtkjs>")
        sys.exit(1)
    
    vtk_to_vtkjs(sys.argv[1], sys.argv[2])
