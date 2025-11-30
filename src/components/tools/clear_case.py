import os
import shutil


def remove_files_inside_folder(dir_path, keep=[]):
    os.makedirs(dir_path, exist_ok=True)
    for filename in os.listdir(dir_path):
        file = os.path.join(dir_path, filename)
        if(os.path.isfile(file)):
            if(not filename in keep):
                os.remove(file)


def clean_case_keep_mesh(case_path):
    sim_path = os.path.join(case_path, "sim")
    initial_path = os.path.join(sim_path, "0.orig")
    remove_files_inside_folder(initial_path)
    system_path = os.path.join(sim_path, 'system')
    remove_files_inside_folder(system_path, keep=['meshDict', 'snappyHexMeshDict', 'blockMeshDict', 'surfaceFeatureExtractDict', 'decomposeParDict'])
    constant_path = os.path.join(sim_path, 'constant')
    remove_files_inside_folder(constant_path)


def clear_case_all(case_name):
    case_path = os.path.join(os.getcwd(), 'cases', case_name)
    os.makedirs(case_path, exist_ok=True)
    if os.path.isdir(case_path):
        for item in os.listdir(case_path):
            item_path = os.path.join(case_path, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)