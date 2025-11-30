import os
import pandas as pd
import pyvista as pv


def load_geo_files(case_name: str):
    geo_path = os.path.join(os.getcwd(), "cases", case_name, "geo")

    geo_file = os.path.join(geo_path, "geometry.vtk")
    if (not os.path.isfile(geo_file)):
        raise BaseException('Geometry file is missing!')
    geo_mesh = pv.read(geo_file)

    df_file = os.path.join(geo_path, "patch_info.csv")
    geo_df = pd.read_csv(df_file)

    return geo_mesh, geo_df
