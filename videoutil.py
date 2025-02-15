import cv2
from dataclasses import dataclass

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
    Stitches together the given time intervals from multiple videos into a single output video.

    :param video_intervals: A dictionary where each key is a path to a video file,
                            and each value is a list of TimeInterval objects (with .start and .end in seconds).
    :param output_path: Path to the output stitched video file.
    :return: The path to the stitched video.
    """
    if not video_intervals:
        raise ValueError("video_intervals dictionary is empty.")

    # Pick the first video path to determine output video properties.
    first_video_path, first_intervals = next(iter(video_intervals.items()))
    first_video = cv2.VideoCapture(first_video_path)
    if not first_video.isOpened():
        raise IOError(f"Could not open video file: {first_video_path}")

    # Get the video properties from the first file.
    fps = first_video.get(cv2.CAP_PROP_FPS)
    width = int(first_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first_video.release()

    # Create a VideoWriter for the stitched output video.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Go through each video and each of its intervals to stitch frames.
    for video_path, intervals in video_intervals.items():
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            print(f"warning: Could not open {video_path}, skipping...")
            continue

        for interval in intervals:
            # Seek to the start of the interval
            seek_frame(video, interval.start)

            while True:
                current_pos = video.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if current_pos >= interval.end:
                    break

                ret, frame = video.read()
                if not ret:
                    # Reached end of video or couldn't read frame
                    break

                writer.write(frame)

        video.release()

    # Release the writer once done stitching all intervals.
    writer.release()
    return output_path