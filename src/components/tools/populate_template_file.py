import os


def indent_lines(lines, num_spaces):
    indent = " " * num_spaces
    return [indent + line for line in lines]


def generate_regions_block(name_map):
    blocks = ["{"]
    for patch in name_map:
        block = f"""    {patch} {{   name {patch}; }}"""
        blocks.append(block)

    blocks.append("}")
    blocks = indent_lines(blocks, 8)

    text_str = "\n".join(blocks)
    text_str = "regions\n" + text_str
    return text_str


def generate_refinement_block(type_map):
    """
    Generate refinement configuration for snappyHexMesh with differentiated levels.
    
    Refinement strategy for HVAC applications:
    - Pressure boundaries (windows/doors/vents): ULTRA-FINE (0 4) - 4 refinement levels for accurate flow
    - Walls: STANDARD (0 2) - 2 refinement levels sufficient for thermal boundaries
    
    This ensures high-quality mesh where it matters most: at flow boundaries.
    """
    blocks = ["{"]
    for patch in type_map:
        # Differentiated refinement based on boundary condition type
        if patch in ["pressure_inlet", "pressure_outlet"]:
            # ULTRA-FINE mesh for pressure boundaries (4 refinement levels)
            # This resolves velocity/pressure gradients accurately
            block = f"""    {patch} {{   level (0 4);     patchInfo {{ type patch; }}}}"""
        elif patch == "wall":
            # STANDARD mesh for walls (2 refinement levels)
            block = f"""    {patch} {{   level (0 2);     patchInfo {{ type wall; }}}}"""
        else:
            # DEFAULT mesh for other boundaries (2 refinement levels)
            block = f"""    {patch} {{   level (0 2);     patchInfo {{ type patch; }}}}"""
        blocks.append(block)
    blocks.append("}")
    blocks = indent_lines(blocks, 12)

    text_str = "regions" + "\n" + "\n".join(blocks)
    return text_str


def replace_in_file(input_path, output_path, str_replace_dict):
    with open(input_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Sort replacements by key length (longest first) to avoid partial replacements
    # Example: $MAX_NON_ORTHO_RELAXED must be replaced before $MAX_NON_ORTHO
    sorted_items = sorted(str_replace_dict.items(), key=lambda x: len(x[0]), reverse=True)
    
    for old_str, new_str in sorted_items:
        content = content.replace(old_str, new_str)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(content)