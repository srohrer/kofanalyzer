import os
from pytubefix import YouTube
import sys

def get_or_download_video(video_id: str, hq: bool = False) -> str:
    """
    Checks if the YouTube video identified by 'video_id' exists in the local 'videos' folder.
    If not, downloads the video using pytube and saves it in that folder.
    
    Optionally, if 'hq' is True, download the highest quality version (progressive stream)
    and save it as '{video_id}_hq.mp4'.
    
    Args:
        video_id (str): The YouTube video ID.
        hq (bool): If True, store as '{video_id}_hq.mp4' and ensure highest resolution progressive stream.
    
    Returns:
        str: The path to the local video file.
    
    Raises:
        Exception: If the video fails to download or no suitable stream is found.
    """
    # Determine the path to the videos folder (relative to this file's directory)
    current_directory = os.path.dirname(os.path.abspath(__file__))
    videos_folder = os.path.join(current_directory, "videos")

    # Create the videos folder if it doesn't exist
    if not os.path.exists(videos_folder):
        os.makedirs(videos_folder)

    # Decide on filename based on whether we're downloading HQ or not
    filename = f"{video_id}_hq" if hq else video_id
    video_path = os.path.join(videos_folder, f"{filename}.mp4")

    # Check if the video file already exists locally
    if os.path.exists(video_path):
        print(f"Video '{filename}' already exists at '{video_path}'.")
        return video_path

    # Construct the YouTube URL
    youtube_url = f"https://www.youtube.com/watch?v={video_id}"
    
    try:
        print(f"Downloading video '{video_id}' from {youtube_url}...")
        yt = YouTube(youtube_url)
        
        # Select a progressive stream (containing both video and audio) in mp4 format
        # .desc().first() ensures we pick the highest resolution available.
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        if not stream:
            raise Exception("No suitable mp4 progressive stream found!")
        
        # Download the video to the 'videos' folder, naming it accordingly
        stream.download(output_path=videos_folder, filename=f"{filename}.mp4")
        print(f"Downloaded video saved to '{video_path}'.")
        return video_path

    except Exception as e:
        print(f"Error downloading video '{video_id}': {e}")
        raise

# Example usage:
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python videoanalyzer.py <video_id>")
        sys.exit(1)
        
    video_id = sys.argv[1]
    local_video_path = get_or_download_video(video_id)
    print(f"Local video file path: {local_video_path}")
