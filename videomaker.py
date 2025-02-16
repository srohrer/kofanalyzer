from dataclasses import dataclass
from datetime import datetime
import json
from typing import List, Dict
from kof_dataclasses import Round, KOFXV_CHARACTERS
from videoutil import TimeInterval, stitch_video_segments, make_debug_screenshot_filepath
from videodownloader import get_or_download_video
import re
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
from roundsaver import delete_round, relabel_round

def local_path_to_video_id(local_path: str) -> str:
    """
    Extracts the video_id from the local path (e.g. videos/12345_hq.mp4 -> 12345).
    Corrects for Windows paths by converting backslashes to forward slashes first.
    """
    # Convert backslashes to forward slashes
    normalized_path = local_path.replace("\\", "/")
    match = re.search(r'/([^/]+?)(?:_hq)?\.mp4$', normalized_path)
    if match:
        return match.group(1)
    return local_path  # fallback if no match found

def load_rounds_from_json(json_path: str, earliest_date_str: str) -> Dict[str, List[dict]]:
    """
    Reads a JSON file of Rounds and returns a structure like:
        { local_video_path: [ {'interval': TimeInterval, 'raw_round': {..}}, ... ], ... }

    Only includes rounds whose upload_date >= earliest_date_str.
    """
    earliest_dt = datetime.fromisoformat(earliest_date_str)
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_rounds = json.load(f)

    if not isinstance(raw_rounds, list):
        raw_rounds = [raw_rounds]

    grouped_data = {}
    for r in raw_rounds:
        upload_str = r.get("upload_date")
        if not upload_str:
            continue

        dt_str = upload_str.replace("Z", "")
        try:
            upload_dt = datetime.fromisoformat(dt_str)
        except ValueError:
            continue

        if upload_dt >= earliest_dt:
            video_id = r.get("video_id", "")
            if not video_id:
                continue

            start_time = r.get("start_time", 0.0)
            end_time = r.get("end_time", 0.0)
            interval = TimeInterval(start=start_time, end=end_time)

            if video_id not in grouped_data:
                grouped_data[video_id] = {
                    "video_id": video_id,
                    "time_intervals": []
                }

            # Store both the interval and the full "raw_round" data
            grouped_data[video_id]["time_intervals"].append({
                "interval": interval,
                "raw_round": r
            })

    # Sort by interval.start
    for video_data in grouped_data.values():
        video_data["time_intervals"].sort(key=lambda x: x["interval"].start)

    # Convert grouped_data into { local_path: [ {interval: TimeInterval, raw_round: {...}}, ... ] }
    result = {}
    for video_dict in grouped_data.values():
        local_path = get_or_download_video(video_dict["video_id"], hq=True)
        intervals = video_dict["time_intervals"]
        result[local_path] = intervals

    return result

def make_video(charname: str, use_gui=False) -> str:
    """
    Make a video of the character with the given name.
    If use_gui=True, a Tkinter GUI is displayed to review/accept/reject rounds,
    as well as to delete or relabel rounds.
    """
    # 1. Load the time intervals (dictionary { local_video_path: [ {interval, raw_round}, ...], ... })
    rounds_dict = load_rounds_from_json(f"rounds/{charname}.json", "2023-01-01T00:00:00")

    # 2. Flatten all rounds into a single list to display/iterate over
    all_rounds = []
    for video_path, items in rounds_dict.items():
        for item in items:
            interval = item["interval"]
            raw_round = item["raw_round"]

            length_seconds = interval.end - interval.start
            video_id = local_path_to_video_id(video_path)

            screenshot_path_start = make_debug_screenshot_filepath(video_id, interval.start)
            screenshot_path_end = make_debug_screenshot_filepath(video_id, interval.start, True)

            # Pull out char1 / char2 if present
            char1 = raw_round.get("character1", "")
            char2 = raw_round.get("character2", "")

            all_rounds.append({
                "video_path": video_path,
                "interval": interval,
                "length_seconds": length_seconds,
                "start_screenshot_path": screenshot_path_start,
                "end_screenshot_path": screenshot_path_end,
                "char1": char1,
                "char2": char2,
                "raw_round": raw_round
            })

    # If not using GUI, just accept *all* intervals and stitch them together
    if not use_gui:
        output_path = f"completedvideos/{charname}.mp4"
        # Transform all_rounds back into the dictionary structure stitch_video_segments needs
        unfiltered_dict = {}
        for rd in all_rounds:
            vp = rd["video_path"]
            if vp not in unfiltered_dict:
                unfiltered_dict[vp] = []
            unfiltered_dict[vp].append(rd["interval"])
        stitch_video_segments(unfiltered_dict, output_path)
        return output_path

    # If using GUI, set up the review interface
    accepted_rounds = [False] * len(all_rounds)
    total_rounds = len(all_rounds)

    root = tk.Tk()
    root.title(f"Review Rounds for {charname}")

    info_label = ttk.Label(root, text=f"0 / {total_rounds} Rounds")
    info_label.pack(pady=5)

    # Main Container - left for images, right for extra controls
    main_frame = ttk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Left pane: screenshots
    images_frame = ttk.Frame(main_frame)
    images_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

    image_label_start = ttk.Label(images_frame)
    image_label_start.grid(row=0, column=0, padx=5)

    image_label_end = ttk.Label(images_frame)
    image_label_end.grid(row=0, column=1, padx=5)

    length_label = ttk.Label(images_frame, text="")
    length_label.grid(row=1, column=0, columnspan=2, pady=(10, 0))

    # Right pane: char selection + buttons
    right_frame = ttk.Frame(main_frame)
    right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

    # Character 1/2 labels + combo boxes
    char1_label = ttk.Label(right_frame, text="Character 1:")
    char1_label.pack(pady=(0, 2))
    char1_var = tk.StringVar()
    char1_dropdown = ttk.Combobox(right_frame, textvariable=char1_var, values=KOFXV_CHARACTERS, state="readonly")
    char1_dropdown.pack(pady=(0, 10))

    char2_label = ttk.Label(right_frame, text="Character 2:")
    char2_label.pack(pady=(0, 2))
    char2_var = tk.StringVar()
    char2_dropdown = ttk.Combobox(right_frame, textvariable=char2_var, values=KOFXV_CHARACTERS, state="readonly")
    char2_dropdown.pack(pady=(0, 10))

    # Action buttons (Accept / Reject)
    button_frame = ttk.Frame(images_frame)
    button_frame.grid(row=2, column=0, columnspan=2, pady=15)

    accept_button = ttk.Button(button_frame, text="Accept")
    reject_button = ttk.Button(button_frame, text="Reject")
    accept_button.pack(side=tk.LEFT, padx=5)
    reject_button.pack(side=tk.LEFT, padx=5)

    # Delete / Relabel buttons
    action_frame = ttk.Frame(right_frame)
    action_frame.pack(pady=(10, 0))

    delete_button = ttk.Button(action_frame, text="Delete")
    relabel_button = ttk.Button(action_frame, text="Relabel")
    delete_button.pack(side=tk.LEFT, padx=5)
    relabel_button.pack(side=tk.LEFT, padx=5)

    # Keep track of the current round index
    current_index = 0
    photo_cache_start = None
    photo_cache_end = None

    def show_round(idx: int):
        """
        Update the UI to show round idx in all_rounds.
        """
        nonlocal photo_cache_start, photo_cache_end

        info_label.config(text=f"{idx+1} / {total_rounds} Rounds")
        round_data = all_rounds[idx]

        # Update screenshots
        start_abs = os.path.abspath(round_data["start_screenshot_path"])
        end_abs = os.path.abspath(round_data["end_screenshot_path"])

        if not os.path.exists(start_abs):
            image_label_start.config(image="", text=f"File not found:\n{start_abs}")
        else:
            try:
                start_img = Image.open(start_abs)
                photo_cache_start = ImageTk.PhotoImage(start_img)
                image_label_start.config(image=photo_cache_start, text="")
            except Exception as e:
                image_label_start.config(image="", text=f"Cannot open image:\n{start_abs}\n{e}")

        if not os.path.exists(end_abs):
            image_label_end.config(image="", text=f"File not found:\n{end_abs}")
        else:
            try:
                end_img = Image.open(end_abs)
                photo_cache_end = ImageTk.PhotoImage(end_img)
                image_label_end.config(image=photo_cache_end, text="")
            except Exception as e:
                image_label_end.config(image="", text=f"Cannot open image:\n{end_abs}\n{e}")

        length_sec = round_data["length_seconds"]
        length_label.config(text=f"Round length: {length_sec:.2f} seconds")

        # Update character comboboxes
        char1_var.set(round_data["char1"])
        char2_var.set(round_data["char2"])

    def go_next_round():
        nonlocal current_index
        current_index += 1
        if current_index >= total_rounds:
            root.destroy()
        else:
            show_round(current_index)

    def on_accept():
        nonlocal current_index
        accepted_rounds[current_index] = True
        go_next_round()

    def on_reject():
        nonlocal current_index
        accepted_rounds[current_index] = False
        go_next_round()

    def on_delete():
        """
        Delete the current round using its raw data object,
        now converted to a Round before passing to roundsaver.
        """
        round_data = all_rounds[current_index]
        round_obj = Round(**round_data["raw_round"])
        delete_round(round_obj)
        go_next_round()

    def on_relabel():
        """
        Relabel the current round to the new character1 and character2,
        converting raw_round to a Round first.
        """
        round_data = all_rounds[current_index]
        round_obj = Round(**round_data["raw_round"])
        new_char1 = char1_var.get()
        new_char2 = char2_var.get()
        relabel_round(round_obj, new_char1, new_char2)
        # Also update the in-memory reference
        round_data["char1"] = new_char1
        round_data["char2"] = new_char2

        on_reject()

    # Hook up the buttons
    accept_button.config(command=on_accept)
    reject_button.config(command=on_reject)
    delete_button.config(command=on_delete)
    relabel_button.config(command=on_relabel)

    # Show the first round or close immediately if none
    if total_rounds > 0:
        show_round(0)
    else:
        root.destroy()

    root.mainloop()

    # 3. Filter out the rejected rounds
    final_rounds_dict = {}
    for i, accepted in enumerate(accepted_rounds):
        if accepted:
            rdata = all_rounds[i]
            vp = rdata["video_path"]
            interval = rdata["interval"]
            if vp not in final_rounds_dict:
                final_rounds_dict[vp] = []
            final_rounds_dict[vp].append(interval)

    output_path = f"completedvideos/{charname}.mp4"
    if final_rounds_dict:
        stitch_video_segments(final_rounds_dict, output_path)
    else:
        raise ValueError("All rounds were rejected. No video created.")

    return output_path

def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python videomaker.py <charname>")
        sys.exit(1)

    charname = sys.argv[1]
    output_path = make_video(charname, use_gui=True)
    print(f"Output video created at: {output_path}")

if __name__ == "__main__":
    main()
