import os


def save_scripts(case_path, mesh_script, solve_script):
    script_path = os.path.join(case_path, "sim", "Allrun")
    os.makedirs(os.path.dirname(script_path), exist_ok=True)

    all_script = mesh_script + solve_script
    with open(script_path, 'w', encoding='utf-8', newline='\n') as file:
        for line in all_script:
            file.write(line.rstrip() + '\n')

    os.chmod(script_path, 0o755)
    return script_path
