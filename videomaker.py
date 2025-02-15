from dataclasses import dataclass
from datetime import datetime
import json
from typing import List, Dict
from kof_dataclasses import Round
from videoutil import TimeInterval, stitch_video_segments
from videodownloader import get_or_download_video

def load_rounds_from_json(json_path: str, earliest_date_str: str) -> Dict[str, List[TimeInterval]]:
    """
    Reads a JSON file of Rounds (as defined in kof_dataclasses.py) and returns
    a dictionary whose keys are local video file paths (obtained via
    get_or_download_video on the video_id), and whose values are lists of
    TimeIntervals corresponding to that file.

    Only includes rounds whose upload_date >= earliest_date_str.

    Args:
        json_path (str): Path to the JSON file containing rounds (list of Round dataclass fields).
        earliest_date_str (str): Earliest ISO 8601 date (e.g., '2023-01-01T00:00:00')
                                 to include in the returned structure.

    Returns:
        Dict[str, List[TimeInterval]]: A dictionary where each key is the local path
        to the downloaded / cached video file, and each value is a list of
        TimeInterval objects.
    """
    earliest_dt = datetime.fromisoformat(earliest_date_str)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_rounds = json.load(f)
    
    # In case the JSON is not a list of round objects, handle it as a single-element list
    if not isinstance(raw_rounds, list):
        raw_rounds = [raw_rounds]
    
    # Group intermediate data by video_id
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
            grouped_data[video_id]["time_intervals"].append(interval)
    
    # Sort the time intervals for each video by start time
    for video_data in grouped_data.values():
        video_data["time_intervals"].sort(key=lambda x: x.start)
    
    # Convert grouped_data into the final structure:
    # keys: local paths to the video file
    # values: list of TimeIntervals
    result = {}
    for video_dict in grouped_data.values():
        local_path = get_or_download_video(video_dict["video_id"], hq=True)
        intervals = video_dict["time_intervals"]
        result[local_path] = intervals
    
    return result

def make_video(charname: str) -> str:
    """
    Make a video of the character with the given name.
    """
    rounds_dict = load_rounds_from_json(f"rounds/{charname}.json", "2023-01-01T00:00:00")
    output_path = f"completedvideos/{charname}.mp4"
    stitch_video_segments(rounds_dict, output_path)
