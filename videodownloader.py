import os
import sys
import subprocess
from pytubefix import YouTube

def merge_video_audio_with_ffmpeg(video_path_temp: str, audio_path_temp: str, final_path: str) -> None:
    """
    Merges a video-only file and an audio-only file into a single output using ffmpeg.

    Args:
        video_path_temp (str): Path to the temporary video file (video-only).
        audio_path_temp (str): Path to the temporary audio file (audio-only).
        final_path (str): Path to the merged output file.

    Raises:
        Exception: If the ffmpeg process fails.
    """
    print(f"Merging using ffmpeg (stream copy) into '{final_path}'...")
    try:
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite without asking
            "-i", video_path_temp,
            "-i", audio_path_temp,
            "-c", "copy",  # Copy both audio & video streams directly
            final_path
        ]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise Exception(f"FFmpeg merge failed: {e}")


def get_or_download_video(video_id: str, hq: bool = False) -> str:
    """
    Checks if the YouTube video identified by 'video_id' exists in the local 'videos' folder.
    If not, downloads the video using pytube/pytubefix and saves it in that folder.

    If 'hq' is True, it downloads separate audio and video streams, then merges them
    using ffmpeg, storing the result as '{video_id}_hq.mp4'.

    Args:
        video_id (str): The YouTube video ID.
        hq (bool): If True, store as '{video_id}_hq.mp4' after merging highest audio and video streams.

    Returns:
        str: The path to the local merged video file.

    Raises:
        Exception: If anything goes wrong in downloading or merging.
    """
    current_directory = os.path.dirname(os.path.abspath(__file__))
    videos_folder = os.path.join(current_directory, "videos")

    # Create the videos folder if it doesn't exist
    if not os.path.exists(videos_folder):
        os.makedirs(videos_folder)

    filename = f"{video_id}_hq" if hq else video_id
    final_path = os.path.join(videos_folder, f"{filename}.mp4")

    # If final file already exists, return it
    if os.path.exists(final_path):
        print(f"Video '{filename}.mp4' already exists at '{final_path}'.")
        return final_path

    # Construct the YouTube URL
    youtube_url = f"https://www.youtube.com/watch?v={video_id}"
    print(f"Preparing to download video '{video_id}' from {youtube_url}...")

    try:
        yt = YouTube(youtube_url)

        if hq:
            # 1) Highest-resolution video-only stream
            video_stream = (
                yt.streams
                .filter(adaptive=True, file_extension="mp4", only_video=True)
                .order_by("resolution")
                .desc()
                .first()
            )
            if not video_stream:
                raise Exception("No suitable high-resolution video stream found!")

            # 2) Best audio-only stream
            audio_stream = (
                yt.streams
                .filter(adaptive=True, file_extension="mp4", only_audio=True)
                .order_by("abr")
                .desc()
                .first()
            )
            if not audio_stream:
                raise Exception("No suitable audio stream found!")

            video_temp = os.path.join(videos_folder, f"{filename}_video.mp4")
            audio_temp = os.path.join(videos_folder, f"{filename}_audio.mp4")

            print(f"Downloading HQ video stream to '{video_temp}'...")
            video_stream.download(output_path=videos_folder, filename=f"{filename}_video.mp4")

            print(f"Downloading HQ audio stream to '{audio_temp}'...")
            audio_stream.download(output_path=videos_folder, filename=f"{filename}_audio.mp4")

            # Merge into final_path
            merge_video_audio_with_ffmpeg(video_temp, audio_temp, final_path)
            os.remove(video_temp)
            os.remove(audio_temp)

            print(f"Merged high-quality video saved to '{final_path}'.")

        else:
            # Use a progressive stream
            stream = (
                yt.streams
                .filter(progressive=True, file_extension="mp4")
                .order_by("resolution")
                .desc()
                .first()
            )
            if not stream:
                raise Exception("No suitable mp4 progressive stream found!")

            print(f"Downloading progressive (lower-resolution) video to '{filename}.mp4'...")
            stream.download(output_path=videos_folder, filename=f"{filename}.mp4")
            print(f"Downloaded video saved to '{final_path}'.")

        return final_path

    except Exception as e:
        print(f"Error downloading video '{video_id}': {e}")
        raise


# Updated main: now takes a video ID and manually calls ffmpeg on paths, similar to HQ logic
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python videoanalyzer.py <video_id>")
        sys.exit(1)

    video_id = sys.argv[1]
    current_directory = os.path.dirname(os.path.abspath(__file__))
    videos_folder = os.path.join(current_directory, "videos")

    # Create the videos folder if it doesn't exist
    if not os.path.exists(videos_folder):
        os.makedirs(videos_folder)

    # Just assume we have already downloaded two separate files named:
    #   <video_id>_video.mp4 and <video_id>_audio.mp4
    # We will merge them into <video_id>_hq.mp4
    video_path_temp = os.path.join(videos_folder, f"{video_id}_hq_video.mp4")
    audio_path_temp = os.path.join(videos_folder, f"{video_id}_hq_audio.mp4")
    final_path = os.path.join(videos_folder, f"{video_id}_hq.mp4")

    print(f"About to merge these paths into '{final_path}':")
    print(f"   Video: {video_path_temp}")
    print(f"   Audio: {audio_path_temp}")

    try:
        merge_video_audio_with_ffmpeg(video_path_temp, audio_path_temp, final_path)
        print(f"Merged HQ video saved to '{final_path}'.")
    except Exception as e:
        print(f"Failed to merge video/audio: {e}")
