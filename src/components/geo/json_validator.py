"""
JSON Validator for Building Configuration Data.

This module provides comprehensive validation for building JSON files before
geometry creation. It detects common errors and provides clear error messages.

Architecture:
    - validate_building_json(): Master validation function
    - validate_structure(): Basic JSON structure validation
    - validate_levels(): Level structure validation
    - validate_floor_sequence(): Floor sequence and placeholder validation
    - validate_walls(): Wall geometry validation
    - validate_stairs(): Stair geometry validation
    - validate_air_entries(): Air entry validation

Usage:
    from src.components.geo.json_validator import validate_building_json
    
    results = validate_building_json(json_data)
    if not results['valid']:
        for error in results['errors']:
            print(f"ERROR: {error['message']}")
"""

import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


class JSONValidationError(Exception):
    """Custom exception for JSON validation errors."""
    def __init__(self, message: str, errors: List[Dict] = None):
        super().__init__(message)
        self.errors = errors or []


# ============================================================================
# VALIDATION RESULT HELPERS
# ============================================================================

def add_error(results: Dict, error_type: str, location: str, message: str):
    """Add an error to validation results."""
    results['valid'] = False
    results['errors'].append({
        'type': error_type,
        'location': location,
        'message': message
    })


def add_warning(results: Dict, warning_type: str, location: str, message: str):
    """Add a warning to validation results."""
    results['warnings'].append({
        'type': warning_type,
        'location': location,
        'message': message
    })


# ============================================================================
# PHASE 1: STRUCTURE VALIDATION
# ============================================================================

def validate_structure(json_data: Dict[str, Any], results: Dict):
    """
    Validate basic JSON structure.
    
    Checks:
    - JSON is not empty
    - Has 'levels' field
    - 'levels' is a dictionary
    - 'levels' is not empty
    - Has 'case_name' (optional warning)
    - Has 'version' (optional warning)
    """
    logger.info("  [1/6] Validating basic structure...")
    
    # Check if JSON is empty
    if not json_data:
        add_error(results, 'empty_json', 'root', 'JSON is empty')
        return
    
    # Check if 'levels' field exists
    if 'levels' not in json_data:
        add_error(results, 'missing_field', 'root', "Missing required field 'levels'")
        return
    
    # Check if 'levels' is a dictionary
    if not isinstance(json_data['levels'], dict):
        add_error(results, 'invalid_type', 'levels', "'levels' must be a dictionary")
        return
    
    # Check if 'levels' is not empty
    if not json_data['levels']:
        add_error(results, 'empty_field', 'levels', "'levels' dictionary is empty")
        return
    
    # Optional fields (warnings only)
    if 'case_name' not in json_data:
        add_warning(results, 'missing_optional', 'root', "Optional field 'case_name' not found")
    
    if 'version' not in json_data:
        add_warning(results, 'missing_optional', 'root', "Optional field 'version' not found")
    
    logger.info(f"    ✓ Basic structure valid ({len(json_data['levels'])} levels found)")


# ============================================================================
# PHASE 2: LEVELS VALIDATION
# ============================================================================

def validate_levels(json_data: Dict[str, Any], results: Dict):
    """
    Validate level structure for each floor.
    
    Checks per level:
    - Has 'height' field (number > 0)
    - Has 'deck' field (number >= 0)
    - Has 'walls' field (list)
    - Has 'ceiling' field (dict with 'temp' and 'airEntries')
    - Has 'floor' field (dict with 'temp' and 'airEntries')
    - Has 'stairs' field (list)
    """
    logger.info("  [2/6] Validating level structures...")
    
    levels = json_data.get('levels', {})
    valid_levels = 0
    
    for level_name, level_data in levels.items():
        location = f"levels.{level_name}"
        
        # Check if level_data is a dictionary
        if not isinstance(level_data, dict):
            add_error(results, 'invalid_type', location, f"Level '{level_name}' must be a dictionary")
            continue
        
        # Check 'height' field
        if 'height' not in level_data:
            add_error(results, 'missing_field', f"{location}.height", "Missing required field 'height'")
        elif not isinstance(level_data['height'], (int, float)):
            add_error(results, 'invalid_type', f"{location}.height", "'height' must be a number")
        elif level_data['height'] <= 0:
            add_error(results, 'invalid_value', f"{location}.height", f"'height' must be > 0 (got {level_data['height']})")
        
        # Check 'deck' field
        if 'deck' not in level_data:
            add_error(results, 'missing_field', f"{location}.deck", "Missing required field 'deck'")
        elif not isinstance(level_data['deck'], (int, float)):
            add_error(results, 'invalid_type', f"{location}.deck", "'deck' must be a number")
        elif level_data['deck'] < 0:
            add_error(results, 'invalid_value', f"{location}.deck", f"'deck' must be >= 0 (got {level_data['deck']})")
        
        # Check 'walls' field
        if 'walls' not in level_data:
            add_error(results, 'missing_field', f"{location}.walls", "Missing required field 'walls'")
        elif not isinstance(level_data['walls'], list):
            add_error(results, 'invalid_type', f"{location}.walls", "'walls' must be a list")
        
        # Check 'ceiling' field
        if 'ceiling' not in level_data:
            add_error(results, 'missing_field', f"{location}.ceiling", "Missing required field 'ceiling'")
        elif not isinstance(level_data['ceiling'], dict):
            add_error(results, 'invalid_type', f"{location}.ceiling", "'ceiling' must be a dictionary")
        else:
            ceiling = level_data['ceiling']
            if 'temp' not in ceiling:
                add_error(results, 'missing_field', f"{location}.ceiling.temp", "Missing required field 'temp'")
            elif not isinstance(ceiling['temp'], (int, float)):
                add_error(results, 'invalid_type', f"{location}.ceiling.temp", "'temp' must be a number")
            
            if 'airEntries' not in ceiling:
                add_error(results, 'missing_field', f"{location}.ceiling.airEntries", "Missing required field 'airEntries'")
            elif not isinstance(ceiling['airEntries'], list):
                add_error(results, 'invalid_type', f"{location}.ceiling.airEntries", "'airEntries' must be a list")
        
        # Check 'floor' field
        if 'floor' not in level_data:
            add_error(results, 'missing_field', f"{location}.floor", "Missing required field 'floor'")
        elif not isinstance(level_data['floor'], dict):
            add_error(results, 'invalid_type', f"{location}.floor", "'floor' must be a dictionary")
        else:
            floor = level_data['floor']
            if 'temp' not in floor:
                add_error(results, 'missing_field', f"{location}.floor.temp", "Missing required field 'temp'")
            elif not isinstance(floor['temp'], (int, float)):
                add_error(results, 'invalid_type', f"{location}.floor.temp", "'temp' must be a number")
            
            if 'airEntries' not in floor:
                add_error(results, 'missing_field', f"{location}.floor.airEntries", "Missing required field 'airEntries'")
            elif not isinstance(floor['airEntries'], list):
                add_error(results, 'invalid_type', f"{location}.floor.airEntries", "'airEntries' must be a list")
        
        # Check 'stairs' field
        if 'stairs' not in level_data:
            add_error(results, 'missing_field', f"{location}.stairs", "Missing required field 'stairs'")
        elif not isinstance(level_data['stairs'], list):
            add_error(results, 'invalid_type', f"{location}.stairs", "'stairs' must be a list")
        
        valid_levels += 1
    
    logger.info(f"    ✓ Level structures validated ({valid_levels}/{len(levels)} levels)")


# ============================================================================
# PHASE 3: FLOOR SEQUENCE VALIDATION
# ============================================================================

def validate_floor_sequence(json_data: Dict[str, Any], results: Dict):
    """
    Validate floor sequence and identify valid floors to process.
    
    Rules:
    1. Floor 0 must exist and have walls
    2. Floors must be consecutive (no gaps)
    3. Placeholders (empty walls) only allowed at end of sequence
    4. Once placeholder found, all subsequent floors ignored
    
    Updates results['valid_floors'] with list of floors to process.
    """
    logger.info("  [3/6] Validating floor sequence...")
    
    levels = json_data.get('levels', {})
    
    # Sort floor numbers
    try:
        floor_numbers = sorted([int(k) for k in levels.keys()])
    except ValueError as e:
        add_error(results, 'invalid_floor_name', 'levels', f"Floor names must be integers: {e}")
        return
    
    # Validate floor 0 exists
    if 0 not in floor_numbers:
        add_error(results, 'missing_floor_0', 'levels', "Floor 0 (ground floor) is required but not found")
        return
    
    # Check if floor 0 has walls
    floor_0_walls = levels["0"].get("walls", [])
    if not floor_0_walls or len(floor_0_walls) == 0:
        add_error(results, 'floor_0_empty', 'levels.0.walls', "Floor 0 (ground floor) must have walls defined")
        return
    
    valid_floors = ["0"]
    logger.info(f"    ✓ Floor 0 validated: {len(floor_0_walls)} walls")
    
    # Process floors 1, 2, 3, ... in sequence
    for i in range(1, len(floor_numbers)):
        floor_num = floor_numbers[i]
        floor_name = str(floor_num)
        
        # Check for gaps in sequence
        expected_floor = i
        if floor_num != expected_floor:
            add_warning(results, 'floor_gap', 'levels', 
                       f"Floor sequence gap: expected floor {expected_floor}, found floor {floor_num}. "
                       f"Stopping at floor {floor_numbers[i-1]} (last valid floor)")
            break
        
        # Check if floor has walls
        floor_walls = levels[floor_name].get("walls", [])
        
        if not floor_walls or len(floor_walls) == 0:
            # Floor is a placeholder (empty)
            add_warning(results, 'placeholder_floor', f'levels.{floor_name}', 
                       f"Floor {floor_name} is a placeholder (no walls) - stopping here")
            break
        
        # Floor is valid
        valid_floors.append(floor_name)
        logger.info(f"    ✓ Floor {floor_name} validated: {len(floor_walls)} walls")
    
    results['valid_floors'] = valid_floors
    logger.info(f"    → Total valid floors: {len(valid_floors)}")


# ============================================================================
# PHASE 4: WALLS VALIDATION
# ============================================================================

def check_walls_connectivity(walls: List[Dict]) -> Tuple[bool, List[str]]:
    """
    Check if walls form a closed polygon.
    
    Returns:
        Tuple of (is_closed, gaps_description)
    """
    if len(walls) < 3:
        return False, ["Need at least 3 walls to form a polygon"]
    
    # Build adjacency map: point → [next_points]
    adjacency = {}
    for wall in walls:
        start = (round(wall['start']['x'], 5), round(wall['start']['y'], 5))
        end = (round(wall['end']['x'], 5), round(wall['end']['y'], 5))
        
        if start not in adjacency:
            adjacency[start] = []
        adjacency[start].append(end)
    
    # Try to find a cycle starting from first wall
    start_point = (round(walls[0]['start']['x'], 5), round(walls[0]['start']['y'], 5))
    current = start_point
    visited_points = set([current])
    path_length = 0
    
    while path_length < len(walls):
        next_points = adjacency.get(current, [])
        
        if not next_points:
            return False, [f"Dead end at point {current}"]
        
        # Find next unvisited point (or start point to close)
        found = False
        for next_point in next_points:
            if next_point == start_point and path_length == len(walls) - 1:
                # Closes the loop
                return True, []
            elif next_point not in visited_points:
                visited_points.add(next_point)
                current = next_point
                path_length += 1
                found = True
                break
        
        if not found:
            return False, [f"Cannot continue from point {current}"]
    
    # Check if last point connects back to start
    if current in adjacency and start_point in adjacency[current]:
        return True, []
    
    return False, [f"Last point {current} does not connect back to start {start_point}"]


def check_walls_order(walls: List[Dict]) -> bool:
    """
    Check if walls are in consecutive order (end[i] == start[i+1]).
    
    Returns:
        True if walls are in order, False otherwise
    """
    if len(walls) < 2:
        return True
    
    for i in range(len(walls)):
        current_wall = walls[i]
        next_wall = walls[(i + 1) % len(walls)]
        
        current_end = (round(current_wall['end']['x'], 5), round(current_wall['end']['y'], 5))
        next_start = (round(next_wall['start']['x'], 5), round(next_wall['start']['y'], 5))
        
        if current_end != next_start:
            return False
    
    return True


def validate_walls(json_data: Dict[str, Any], results: Dict):
    """
    Validate wall geometry for each floor.
    
    Checks per wall:
    - Has 'id' field (string)
    - Has 'start' field (dict with x, y)
    - Has 'end' field (dict with x, y)
    - Has 'temp' field (number)
    - Has 'airEntries' field (list)
    - start != end (wall length > 0)
    - Coordinates are valid numbers
    - Walls form a closed polygon (connectivity)
    - Walls are in consecutive order (optional warning)
    """
    logger.info("  [4/6] Validating walls...")
    
    levels = json_data.get('levels', {})
    total_walls = 0
    valid_walls = 0
    
    for level_name, level_data in levels.items():
        walls = level_data.get('walls', [])
        
        # Validate individual walls first
        for wall_idx, wall in enumerate(walls):
            location = f"levels.{level_name}.walls[{wall_idx}]"
            total_walls += 1
            
            # Check if wall is a dictionary
            if not isinstance(wall, dict):
                add_error(results, 'invalid_type', location, "Wall must be a dictionary")
                continue
            
            # Check 'id' field
            if 'id' not in wall:
                add_error(results, 'missing_field', f"{location}.id", "Missing required field 'id'")
            elif not isinstance(wall['id'], str) or not wall['id']:
                add_error(results, 'invalid_value', f"{location}.id", "'id' must be a non-empty string")
            
            # Check 'start' field
            if 'start' not in wall:
                add_error(results, 'missing_field', f"{location}.start", "Missing required field 'start'")
            elif not isinstance(wall['start'], dict):
                add_error(results, 'invalid_type', f"{location}.start", "'start' must be a dictionary")
            else:
                start = wall['start']
                if 'x' not in start or 'y' not in start:
                    add_error(results, 'missing_field', f"{location}.start", "'start' must have 'x' and 'y' fields")
                elif not isinstance(start['x'], (int, float)) or not isinstance(start['y'], (int, float)):
                    add_error(results, 'invalid_type', f"{location}.start", "'x' and 'y' must be numbers")
            
            # Check 'end' field
            if 'end' not in wall:
                add_error(results, 'missing_field', f"{location}.end", "Missing required field 'end'")
            elif not isinstance(wall['end'], dict):
                add_error(results, 'invalid_type', f"{location}.end", "'end' must be a dictionary")
            else:
                end = wall['end']
                if 'x' not in end or 'y' not in end:
                    add_error(results, 'missing_field', f"{location}.end", "'end' must have 'x' and 'y' fields")
                elif not isinstance(end['x'], (int, float)) or not isinstance(end['y'], (int, float)):
                    add_error(results, 'invalid_type', f"{location}.end", "'x' and 'y' must be numbers")
            
            # Check if start == end (zero-length wall)
            if 'start' in wall and 'end' in wall:
                start = wall['start']
                end = wall['end']
                if (isinstance(start, dict) and isinstance(end, dict) and
                    'x' in start and 'y' in start and 'x' in end and 'y' in end):
                    if start['x'] == end['x'] and start['y'] == end['y']:
                        add_error(results, 'zero_length_wall', location, 
                                 f"Wall has zero length (start == end): {wall.get('id', 'unknown')}")
            
            # Check 'temp' field
            if 'temp' not in wall:
                add_error(results, 'missing_field', f"{location}.temp", "Missing required field 'temp'")
            elif not isinstance(wall['temp'], (int, float)):
                add_error(results, 'invalid_type', f"{location}.temp", "'temp' must be a number")
            
            # Check 'airEntries' field
            if 'airEntries' not in wall:
                add_error(results, 'missing_field', f"{location}.airEntries", "Missing required field 'airEntries'")
            elif not isinstance(wall['airEntries'], list):
                add_error(results, 'invalid_type', f"{location}.airEntries", "'airEntries' must be a list")
            
            valid_walls += 1
        
        # NEW: Check wall connectivity and order for this level
        if len(walls) >= 3:
            # Check connectivity (walls form closed polygon)
            is_closed, gaps = check_walls_connectivity(walls)
            
            if not is_closed:
                gap_msg = "; ".join(gaps)
                add_error(results, 'walls_not_closed', f'levels.{level_name}.walls',
                         f"Walls do not form a closed polygon: {gap_msg}")
            
            # Check order (walls in consecutive sequence)
            is_ordered = check_walls_order(walls)
            
            if not is_ordered and is_closed:
                add_warning(results, 'walls_unordered', f'levels.{level_name}.walls',
                           f"Walls are not in consecutive order. "
                           f"Geometry creation will attempt to reorder them automatically.")
    
    logger.info(f"    ✓ Walls validated ({valid_walls}/{total_walls} walls)")
    results['stats']['total_walls'] = total_walls


# ============================================================================
# PHASE 5: STAIRS VALIDATION
# ============================================================================

def validate_stairs(json_data: Dict[str, Any], results: Dict):
    """
    Validate stair geometry for each floor.
    
    Checks per stair:
    - Has 'id' field (string)
    - Has 'lines' field (list, not empty)
    - Has 'temp' field (number)
    - Each line has 'start' and 'end'
    - At least 3 lines (minimum polygon)
    """
    logger.info("  [5/6] Validating stairs...")
    
    levels = json_data.get('levels', {})
    total_stairs = 0
    valid_stairs = 0
    
    for level_name, level_data in levels.items():
        stairs = level_data.get('stairs', [])
        
        for stair_idx, stair in enumerate(stairs):
            location = f"levels.{level_name}.stairs[{stair_idx}]"
            total_stairs += 1
            
            # Check if stair is a dictionary
            if not isinstance(stair, dict):
                add_error(results, 'invalid_type', location, "Stair must be a dictionary")
                continue
            
            # Check 'id' field
            if 'id' not in stair:
                add_error(results, 'missing_field', f"{location}.id", "Missing required field 'id'")
            elif not isinstance(stair['id'], str) or not stair['id']:
                add_error(results, 'invalid_value', f"{location}.id", "'id' must be a non-empty string")
            
            # Check 'lines' field
            if 'lines' not in stair:
                add_error(results, 'missing_field', f"{location}.lines", "Missing required field 'lines'")
                continue
            elif not isinstance(stair['lines'], list):
                add_error(results, 'invalid_type', f"{location}.lines", "'lines' must be a list")
                continue
            elif len(stair['lines']) == 0:
                add_error(results, 'empty_field', f"{location}.lines", "'lines' list is empty")
                continue
            elif len(stair['lines']) < 3:
                add_error(results, 'insufficient_lines', f"{location}.lines", 
                         f"Stair must have at least 3 lines (got {len(stair['lines'])})")
            
            # Check each line
            for line_idx, line in enumerate(stair['lines']):
                line_location = f"{location}.lines[{line_idx}]"
                
                if not isinstance(line, dict):
                    add_error(results, 'invalid_type', line_location, "Line must be a dictionary")
                    continue
                
                # Check 'start' and 'end' fields
                if 'start' not in line:
                    add_error(results, 'missing_field', f"{line_location}.start", "Missing required field 'start'")
                if 'end' not in line:
                    add_error(results, 'missing_field', f"{line_location}.end", "Missing required field 'end'")
            
            # Check 'temp' field
            if 'temp' not in stair:
                add_error(results, 'missing_field', f"{location}.temp", "Missing required field 'temp'")
            elif not isinstance(stair['temp'], (int, float)):
                add_error(results, 'invalid_type', f"{location}.temp", "'temp' must be a number")
            
            valid_stairs += 1
    
    logger.info(f"    ✓ Stairs validated ({valid_stairs}/{total_stairs} stairs)")
    results['stats']['total_stairs'] = total_stairs


# ============================================================================
# PHASE 6: AIR ENTRIES VALIDATION
# ============================================================================

def validate_air_entries(json_data: Dict[str, Any], results: Dict):
    """
    Validate air entries in walls, floors, and ceilings.
    
    Checks per air entry:
    - Has 'id' field (string)
    - Has 'position' field (dict)
    - Has 'dimensions' field (dict with width, height)
    - Has 'simulation' field (dict)
    - dimensions are numbers > 0
    """
    logger.info("  [6/6] Validating air entries...")
    
    levels = json_data.get('levels', {})
    total_entries = 0
    valid_entries = 0
    
    for level_name, level_data in levels.items():
        # Check wall air entries
        for wall_idx, wall in enumerate(level_data.get('walls', [])):
            for entry_idx, entry in enumerate(wall.get('airEntries', [])):
                location = f"levels.{level_name}.walls[{wall_idx}].airEntries[{entry_idx}]"
                total_entries += 1
                
                if validate_single_air_entry(entry, location, results):
                    valid_entries += 1
        
        # Check floor air entries
        for entry_idx, entry in enumerate(level_data.get('floor', {}).get('airEntries', [])):
            location = f"levels.{level_name}.floor.airEntries[{entry_idx}]"
            total_entries += 1
            
            if validate_single_air_entry(entry, location, results):
                valid_entries += 1
        
        # Check ceiling air entries
        for entry_idx, entry in enumerate(level_data.get('ceiling', {}).get('airEntries', [])):
            location = f"levels.{level_name}.ceiling.airEntries[{entry_idx}]"
            total_entries += 1
            
            if validate_single_air_entry(entry, location, results):
                valid_entries += 1
    
    logger.info(f"    ✓ Air entries validated ({valid_entries}/{total_entries} entries)")
    results['stats']['total_air_entries'] = total_entries


def validate_single_air_entry(entry: Dict, location: str, results: Dict) -> bool:
    """Validate a single air entry. Returns True if valid."""
    is_valid = True
    
    # Check if entry is a dictionary
    if not isinstance(entry, dict):
        add_error(results, 'invalid_type', location, "Air entry must be a dictionary")
        return False
    
    # Check 'id' field
    if 'id' not in entry:
        add_error(results, 'missing_field', f"{location}.id", "Missing required field 'id'")
        is_valid = False
    elif not isinstance(entry['id'], str) or not entry['id']:
        add_error(results, 'invalid_value', f"{location}.id", "'id' must be a non-empty string")
        is_valid = False
    
    # Check 'position' field
    if 'position' not in entry:
        add_error(results, 'missing_field', f"{location}.position", "Missing required field 'position'")
        is_valid = False
    elif not isinstance(entry['position'], dict):
        add_error(results, 'invalid_type', f"{location}.position", "'position' must be a dictionary")
        is_valid = False
    
    # Check 'dimensions' field
    if 'dimensions' not in entry:
        add_error(results, 'missing_field', f"{location}.dimensions", "Missing required field 'dimensions'")
        is_valid = False
    elif not isinstance(entry['dimensions'], dict):
        add_error(results, 'invalid_type', f"{location}.dimensions", "'dimensions' must be a dictionary")
        is_valid = False
    else:
        dims = entry['dimensions']
        if 'width' not in dims or 'height' not in dims:
            add_error(results, 'missing_field', f"{location}.dimensions", "'dimensions' must have 'width' and 'height'")
            is_valid = False
        else:
            if not isinstance(dims['width'], (int, float)) or dims['width'] <= 0:
                add_error(results, 'invalid_value', f"{location}.dimensions.width", 
                         f"'width' must be > 0 (got {dims['width']})")
                is_valid = False
            if not isinstance(dims['height'], (int, float)) or dims['height'] <= 0:
                add_error(results, 'invalid_value', f"{location}.dimensions.height", 
                         f"'height' must be > 0 (got {dims['height']})")
                is_valid = False
    
    # Check 'simulation' field
    if 'simulation' not in entry:
        add_error(results, 'missing_field', f"{location}.simulation", "Missing required field 'simulation'")
        is_valid = False
    elif not isinstance(entry['simulation'], dict):
        add_error(results, 'invalid_type', f"{location}.simulation", "'simulation' must be a dictionary")
        is_valid = False
    
    return is_valid


# ============================================================================
# MASTER VALIDATION FUNCTION
# ============================================================================

def validate_building_json(json_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Master validation function that runs all checks.
    
    Args:
        json_data: Building configuration JSON data
        
    Returns:
        Dictionary with validation results:
        {
            'valid': bool,
            'errors': List[Dict],
            'warnings': List[Dict],
            'valid_floors': List[str],
            'stats': Dict
        }
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'valid_floors': [],
        'stats': {
            'total_floors': 0,
            'valid_floors': 0,
            'total_walls': 0,
            'total_stairs': 0,
            'total_air_entries': 0
        }
    }
    
    logger.info("=" * 70)
    logger.info("VALIDATING JSON STRUCTURE")
    logger.info("=" * 70)
    
    # Run all validation phases
    validate_structure(json_data, results)
    
    # Only continue if structure is valid
    if results['valid']:
        validate_levels(json_data, results)
        validate_floor_sequence(json_data, results)
        validate_walls(json_data, results)
        validate_stairs(json_data, results)
        validate_air_entries(json_data, results)
    
    # Update stats
    if 'levels' in json_data:
        results['stats']['total_floors'] = len(json_data['levels'])
        results['stats']['valid_floors'] = len(results['valid_floors'])
    
    # Log summary
    logger.info("=" * 70)
    if results['valid']:
        logger.info("✅ JSON VALIDATION PASSED")
        logger.info(f"   - Valid floors: {results['stats']['valid_floors']}/{results['stats']['total_floors']}")
        logger.info(f"   - Total walls: {results['stats']['total_walls']}")
        logger.info(f"   - Total stairs: {results['stats']['total_stairs']}")
        logger.info(f"   - Total air entries: {results['stats']['total_air_entries']}")
        
        if results['warnings']:
            logger.info(f"   - Warnings: {len(results['warnings'])}")
            for warning in results['warnings']:
                logger.warning(f"     ⚠️  {warning['location']}: {warning['message']}")
    else:
        logger.error("❌ JSON VALIDATION FAILED")
        logger.error(f"   - Errors: {len(results['errors'])}")
        for error in results['errors']:
            logger.error(f"     ❌ {error['location']}: {error['message']}")
    
    logger.info("=" * 70)
    
    return results
