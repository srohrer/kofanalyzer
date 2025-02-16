import os
import json
from kof_dataclasses import Round
from videoutil import make_debug_screenshot_filepath

def calculate_round_id(round: Round) -> str:
    return f"{round.video_id}_{int(round.start_time)}"

def save_round(round: Round) -> None:
    """
    Save a round to character-specific JSON files, avoiding duplicates.
    Only saves rounds that are at least 20 seconds long.
    
    Args:
        round: Round object to save
    """
    # Skip rounds shorter than 20 seconds
    if round.end_time - round.start_time < 20:
        return
    
    # Create unique ID for the round
    round_id = calculate_round_id(round)
    
    # Create rounds directory if it doesn't exist
    os.makedirs('rounds', exist_ok=True)
    
    # Process both characters
    for char in [round.character1, round.character2]:
        if not char or char.lower() == "unknown":
            continue
        
        # Create filename with underscores instead of spaces
        filename = os.path.join('rounds', f"{char.replace(' ', '_')}.json")
        
        # Initialize or load existing rounds
        rounds_data = []
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    rounds_data = json.load(f)
            except json.JSONDecodeError:
                # If file is corrupted, start fresh
                rounds_data = []
        
        # Check if round already exists
        if not any(r.get('round_id') == round_id for r in rounds_data):
            # Convert Round object to dictionary and add round_id
            round_dict = round.__dict__
            round_dict['round_id'] = round_id
            
            # Add to rounds data and save
            rounds_data.append(round_dict)
            with open(filename, 'w') as f:
                json.dump(rounds_data, f, indent=2) 

def remove_round_from_character(round_id: str, character: str) -> None:
    """
    Remove a round from a single character's JSON file by its round_id.
    Skips if the character is empty, 'unknown', or the file doesn't exist.
    """
    import json
    import os

    if not character or character.lower() == 'unknown':
        return

    filename = os.path.join('rounds', f"{character.replace(' ', '_')}.json")
    if not os.path.exists(filename):
        return

    # Attempt to read existing rounds
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        data = []

    # Filter out the round with the old id
    new_data = [r for r in data if r.get('round_id') != round_id]

    # Write back if there's any difference
    if len(new_data) != len(data):
        with open(filename, 'w') as f:
            json.dump(new_data, f, indent=2)

def delete_round(round_obj) -> None:
    """
    Delete a round from all character-specific JSON files.
    Accepts either a Round instance or a dict with equivalent fields
    ('video_id', 'start_time', 'character1', 'character2').
    """
    # If it's a Round object, reuse calculate_round_id and access attributes normally.
    # Otherwise, build the round_id and characters from dictionary keys.
    if isinstance(round_obj, Round):
        round_id = calculate_round_id(round_obj)
        chars = [round_obj.character1, round_obj.character2]
    else:
        round_id = f"{round_obj['video_id']}_{int(round_obj['start_time'])}"
        chars = [round_obj.get('character1'), round_obj.get('character2')]

    for char in chars:
        remove_round_from_character(round_id, char)

def relabel_round(round_obj: Round, character1: str, character2: str) -> None:
    """
    Relabel a round from (round_obj.character1, round_obj.character2)
    to (character1, character2). If the new labels differ, remove the
    existing round data from the old characters' .json files,
    update the round object, and re-save under the correct character
    files. Also copy the original screenshot into
    'misclassified_screenshots' with a filename of 
    '{character1}_vs_{character2}.jpg'.
    If such a file already exists, add '_{number}' before the extension
    to keep it unique.

    Args:
        round_obj: The Round object to relabel.
        character1: The new first character label.
        character2: The new second character label.
    """
    import os
    import shutil

    # Check if no relabel is needed
    if round_obj.character1 == character1 and round_obj.character2 == character2:
        return

    # Remove this round from any old characters' .json
    delete_round(round_obj)

    # Update the round with the new labels
    round_obj.character1 = character1
    round_obj.character2 = character2

    # Re-save under the new labels
    save_round(round_obj)

    # Copy the debug screenshot to misclassified_screenshots
    debug_screenshot_path = make_debug_screenshot_filepath(round_obj.video_id, round_obj.start_time)
    print(f"Debug screenshot path: {debug_screenshot_path}")
    if os.path.exists(debug_screenshot_path):
        os.makedirs('misclassified_screenshots', exist_ok=True)

        # Construct a base filename like "char1_vs_char2.jpg"
        base_name = f"{character1.replace(' ', '_')}_vs_{character2.replace(' ', '_')}.jpg"
        output_path = os.path.join('misclassified_screenshots', base_name)

        # If there's a name collision, insert _1, _2, etc.
        if os.path.exists(output_path):
            file_root, file_ext = os.path.splitext(output_path)
            counter = 1
            while os.path.exists(output_path):
                output_path = f"{file_root}_{counter}{file_ext}"
                counter += 1

        # Copy the file over
        shutil.copyfile(debug_screenshot_path, output_path)