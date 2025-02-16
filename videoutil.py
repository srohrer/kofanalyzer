import cv2
from dataclasses import dataclass
from moviepy import VideoFileClip, concatenate_videoclips

@dataclass
class TimeInterval:
    start: float
    end: float

def seek_frame(video: cv2.VideoCapture, end_timestamp: float) -> None:
        # Get current position
        current_pos = video.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        
        # If seeking backwards, use set() instead of grab()
        if end_timestamp < current_pos:
            print("warning: seeking backward")
            video.set(cv2.CAP_PROP_POS_MSEC, end_timestamp * 1000)
            return
        
        # Keep calling grab() until we are at or past the end timestamp
        while current_pos < end_timestamp:
            ret = video.grab()
            if not ret:
                break
            current_pos = video.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

def stitch_video_segments(video_intervals: dict[str, list[TimeInterval]], output_path: str) -> str:
    """
    Stitches together the given time intervals from multiple videos into
    a single output video *with audio* using MoviePy.
    """
    if not video_intervals:
        raise ValueError("video_intervals dictionary is empty.")

    subclips = []
    for video_path, intervals in video_intervals.items():
        for interval in intervals:
            # Load entire video in memory for this snippet. If the videos are large,
            # you may want to call close() on each clip after concatenation
            clip = VideoFileClip(video_path).subclipped(interval.start, interval.end)
            subclips.append(clip)

    if not subclips:
        raise ValueError("No valid intervals found to stitch.")

    # Concatenate all subclips
    final_clip = concatenate_videoclips(subclips)

    # Write out the final clip (includes audio)
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

    for clip in subclips:
        clip.close()
    final_clip.close()

    return output_path

def make_debug_screenshot_filepath(video_id: str, start_time: float, end: bool = False) -> str:
    """
    Create a debug screenshot filepath from a video id and round start time.
    """
    import datetime
    import os

    # Ensure debug directory exists
    os.makedirs('debug', exist_ok=True)

    # Convert seconds to HH:MM:SS format and replace colons with underscores
    timestamp = str(datetime.timedelta(seconds=int(start_time))).replace(":", "_")
    if end:
        return f'debug/{video_id}.{timestamp}_end.jpg'
    else:
        return f'debug/{video_id}.{timestamp}.jpg'