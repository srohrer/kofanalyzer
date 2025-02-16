from kof_dataclasses import Round, VideoInfo
from typing import List, Optional
import json
import sys
from anthropic import Anthropic
from dotenv import load_dotenv
import os
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import cv2
import numpy as np
from scenedetect import detect, ContentDetector
import matplotlib.pyplot as plt
import datetime
from videodownloader import get_or_download_video
from fastai.vision.all import load_learner
import pathlib
import time
from portraitreader import extract_character_names
from videoutil import seek_frame, make_debug_screenshot_filepath
from roundsaver import save_round  # <-- Import the new utility function

class KOFAnalyzer:
    def __init__(self, download_screenshots=True):
        # Patch PosixPath for Windows compatibility
        pathlib.PosixPath = pathlib.WindowsPath
        
        # Load environment variables
        load_dotenv()
        self.download_screenshots = download_screenshots
        
        # Load the fastai learner
        self.roundornot = load_learner('roundornot.pkl')
        self.endhead = load_learner('endhead.pkl')
        self.portrait_learner = load_learner('ig_mk2_portraits.pkl')
        
        # Initialize Anthropic client with API key
        anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        if not anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        self.client = Anthropic(api_key=anthropic_api_key)
        
        # Initialize YouTube client
        youtube_api_key = os.getenv('YOUTUBE_API_KEY')
        if not youtube_api_key:
            raise ValueError("YOUTUBE_API_KEY not found in environment variables")
        self.youtube = build('youtube', 'v3', developerKey=youtube_api_key)

    def _get_video_metadata(self, video_id: str) -> tuple[str, str]:
        """
        Get the video title and its upload date from YouTube using the YouTube Data API.

        Args:
            video_id: The YouTube video ID

        Returns:
            A tuple of (video_title, video_upload_date)
        
        Raises:
            HttpError: If there's an error accessing the YouTube API
            ValueError: If the video is not found
        """
        try:
            response = self.youtube.videos().list(
                part='snippet',
                id=video_id
            ).execute()
            
            if not response['items']:
                raise ValueError(f"Video not found with ID: {video_id}")
            
            snippet = response['items'][0]['snippet']
            video_title = snippet['title']
            # 'publishedAt' is an ISO 8601 date/time string like "2023-09-28T15:27:01Z"
            video_upload_date = snippet['publishedAt']
            return video_title, video_upload_date
            
        except HttpError as e:
            raise ValueError(f"Error accessing YouTube API: {str(e)}")

    def analyze_video(self, video_id: str) -> VideoInfo:
        """
        Analyze a KOF video to extract rounds and player information.
        If players can't be determined from the title, they'll be identified per-round.
        """
        # Retrieve video title & upload date
        video_title, video_upload_date = self._get_video_metadata(video_id)

        # Attempt to determine players from title (or fallback)
        # title_players = self._get_players_from_title(video_id)
        title_players = ["Laudandus", "Pinecone"]

        # Analyze rounds
        rounds = self.analyze_rounds(video_id, title_players, video_upload_date)

        # Get character information for each round
        video_path = get_or_download_video(video_id)
        video = cv2.VideoCapture(video_path)
        try:
            for round_obj in rounds:
                self.get_round_characters(round_obj, video)
        finally:
            video.release()

        # Save each round before returning (using the new utility function)
        for round_obj in rounds:
            save_round(round_obj)

        return VideoInfo(
            video_id=video_id,
            upload_date=video_upload_date,
            title_players=title_players,
            rounds=rounds
        )

    def _get_players_from_title(self, video_id: str) -> Optional[List[str]]:
        """
        Try to extract player names from video title.
        Returns None if we can't confidently identify a fixed set of players.
        """

        video_title = self._get_video_title(video_id)

        prompt = f"""Analyze this fighting game video title and extract the player names if possible:
        "{video_title}"
        
        Rules:
        - Return exactly two player names if you can identify them with high confidence
        - Return None if:
          - The title suggests multiple matches (tournament/compilation)
          - You can't confidently extract both player names
          - The title format is unfamiliar or ambiguous
        
        Respond in JSON format: {{"players": ["player1", "player2"]}} or {{"players": null}}
        """

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            temperature=0,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        try:
            # Access the text from the TextBlock
            result = json.loads(response.content[0].text)
            return result["players"]
        except (json.JSONDecodeError, KeyError, IndexError):
            return None

    def _get_video_title(self, video_id: str) -> str:
        """
        Get the video title from YouTube using the YouTube Data API.
        
        Args:
            video_id: The YouTube video ID
            
        Returns:
            The video title as a string
            
        Raises:
            HttpError: If there's an error accessing the YouTube API
            ValueError: If the video is not found
        """
        try:
            response = self.youtube.videos().list(
                part='snippet',
                id=video_id
            ).execute()
            
            if not response['items']:
                raise ValueError(f"Video not found with ID: {video_id}")
                
            return response['items'][0]['snippet']['title']
            
        except HttpError as e:
            raise ValueError(f"Error accessing YouTube API: {str(e)}")
        
    def get_round_characters(self, round: Round, video=None) -> None:
        """
        Update a Round object with character information from the video.
        We attempt up to 5 times to verify that the scene is actually a round:
        1) Seek to the round start_time (in seconds).
        2) Grab a frame, check with the round-or-not model.
           - If "round", proceed to extract characters and break out.
           - If not, skip 10 frames ahead (10/fps seconds) and try again.
        3) If after 5 tries we still don't get a "round," abandon character extraction.
        """
        # Load video if not provided
        if video is None:
            video_path = get_or_download_video(round.video_id)
            video = cv2.VideoCapture(video_path)
            should_release = True
        else:
            should_release = False

        fps = video.get(cv2.CAP_PROP_FPS) if video.isOpened() else 30.0
        time_of_frame = round.start_time

        found_round = False
        frame_skip_seconds = 10.0 / fps  # number of frames to skip converted to seconds

        for attempt in range(5):
            # Seek to the current time offset with our custom seek_frame utility
            seek_frame(video, time_of_frame)
            ret, frame = video.read()
            if not ret or frame is None:
                # If we can't read a frame, just move to the next skip interval
                time_of_frame += frame_skip_seconds
                continue

            # Convert frame to RGB for FastAI model
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Use "roundornot" model to check if this is likely a round frame
            with self.roundornot.no_bar(), self.roundornot.no_logging():
                pred = self.roundornot.predict(frame_rgb)
            pred_label = self.roundornot.dls.vocab[pred[1]]
            prob = float(pred[2][pred[1]])

            if pred_label == "Round":
                # This frame is likely a round
                found_round = True

                # Use new function to build debug screenshot filepath
                filepath = make_debug_screenshot_filepath(round.video_id, round.start_time)
                success = cv2.imwrite(filepath, frame)
                print(f"Verified round frame saved to {filepath}, Success: {success}")

                # Extract character names using the portrait learner
                try:
                    round.character1, round.character2 = extract_character_names(frame, self.portrait_learner)
                except Exception as e:
                    print(f"Error extracting character names: {e}")
                    import traceback
                    traceback.print_exc()
                    round.character1 = "unknown"
                    round.character2 = "unknown"
                break
            else:
                # Skip ahead by 10 frames (i.e. 10/fps seconds)
                time_of_frame += frame_skip_seconds

        # If no valid round frame was found
        if not found_round:
            print(f"No valid round frame found for round starting at {round.start_time} seconds.")
            round.character1 = "unknown"
            round.character2 = "unknown"

        # Release video if we opened it
        if should_release:
            video.release()

    def split_scenes(self, video_id: str) -> List:
        """
        Split video into scenes using scene change detection.
        Returns a list of scenes detected in the video.
        """
        video_path = get_or_download_video(video_id)
        
        if not os.path.exists(video_path):
            return []

        # Create a ContentDetector instance
        detector = ContentDetector(threshold=25.0)
        
        # Perform scene detection
        return detect(video_path, detector)

    def save_training_data(self, video_id: str, title_players: Optional[List[str]]) -> List[Round]:
        """
        Analyze video to detect rounds using scene change detection.
        """
        rounds = []

        # Create screenshots directory if DOWNLOAD_SCREENSHOTS is enabled
        if self.download_screenshots:
            screenshots_dir = os.path.abspath("screenshots")
            os.makedirs(screenshots_dir, exist_ok=True)
        
        # Set screenshot probability based on whether players are identified
        screenshot_probability = 0.10 if title_players else 0.01
        
        # Get scenes from video
        scenes = self.split_scenes(video_id)
        video_path = get_or_download_video(video_id)

        # Process scenes to identify rounds
        for i, scene in enumerate(scenes):
            start_time = scene[0].get_seconds()
            end_time = scene[1].get_seconds()
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue
            
            if self.download_screenshots:
                # Save screenshots before scene start with adjusted probability
                for offset in [-2.0, -1.5, -1.0, -0.5]:
                    if np.random.random() < screenshot_probability:
                        frame_time = max(0, start_time + offset)
                        seek_frame(cap, frame_time)
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            timestamp = str(datetime.timedelta(seconds=int(frame_time))).replace(":", "_")
                            filename = f"{video_id}_{timestamp}_before_{abs(offset):.1f}s.jpg"
                            filepath = os.path.join(screenshots_dir, filename)
                            cv2.imwrite(filepath, frame)
                
                # Save screenshots after scene start with adjusted probability
                for offset in [-2.0, -1.5, -1.0, -0.5]:
                    if np.random.random() < (screenshot_probability * 0.25):
                        frame_time = start_time + offset
                        seek_frame(cap, frame_time)
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            timestamp = str(datetime.timedelta(seconds=int(frame_time))).replace(":", "_")
                            filename = f"{video_id}_{timestamp}_after_{offset:.1f}s.jpg"
                            filepath = os.path.join(screenshots_dir, filename)
                            cv2.imwrite(filepath, frame)
            
            # Get frame from middle of scene for round analysis
            mid_frame_time = (start_time + end_time) / 2
            seek_frame(cap, mid_frame_time)
            ret, frame = cap.read()
            if ret and frame is not None:
                # Create debug directory if it doesn't exist
                os.makedirs('debug', exist_ok=True)
                # Convert seconds to HH:MM:SS format
                timestamp = str(datetime.timedelta(seconds=int(mid_frame_time)))
                cv2.imwrite(f'debug/{video_id}.{timestamp}.jpg', frame)
            cap.release()

        return []

    def analyze_rounds(self, video_id: str, title_players: Optional[List[str]], video_upload_date: str) -> List[Round]:
        """
        Analyze video to detect rounds using scene detection and frame classification,
        without using cap.set for round-end screenshot capture. We gather frames as we go,
        then save the middle frame if we detect a round end.
        """
        # Retrieve scenes & prepare
        scenes = self.split_scenes(video_id)
        video_path = get_or_download_video(video_id)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_scenes = len(scenes)
        in_round = False
        current_round_start = None
        rounds = []

        if title_players:
            player1, player2 = title_players
        else:
            player1 = player2 = "Unknown"

        start_time = time.time()

        # We'll track the current frame index ourselves while reading sequentially
        # so we don't need cap.set.
        current_frame_index = 0

        with open(f"{video_id}.txt", "w") as f:
            for i, scene in enumerate(scenes):
                # Some progress tracking
                elapsed_time = time.time() - start_time
                progress = i / total_scenes
                remaining = (elapsed_time / (progress + 1e-8)) * (1 - progress) if progress > 0 else 0
                bar_length = 100
                filled_length = int(bar_length * progress)
                bar = '=' * filled_length + '-' * (bar_length - filled_length)
                print(
                    f'\r|{bar}| {progress:.2%} [{i}/{total_scenes} '
                    f'{int(elapsed_time):02d}:{int(remaining):02d}<{int(remaining):02d}]',
                    end=''
                )

                boundary_time_secs = scene[0].get_seconds()
                boundary_frame = int(boundary_time_secs * fps)

                before_probs_and_frames = []
                after_probs_and_frames = []

                # Collect frames for "before" end detection if currently in a round
                if in_round:
                    # We'll read the 30 frames leading up to boundary_frame
                    # but in sequential order — skipping frames we don't need.
                    desired_frames = list(range(boundary_frame - 30, boundary_frame))
                    desired_frames = [df for df in desired_frames if df >= 0]  # no negatives
                    before_probs_and_frames, current_frame_index = self.get_frame_probs(
                        cap=cap,
                        start_index=current_frame_index,
                        frame_nums=desired_frames,
                        model_type="end"
                    )

                # Collect frames for "after" round detection if not currently in a round
                if not in_round:
                    desired_frames = list(range(boundary_frame + 1, boundary_frame + 31, 3))
                    after_probs_and_frames, current_frame_index = self.get_frame_probs(
                        cap=cap,
                        start_index=current_frame_index,
                        frame_nums=desired_frames,
                        model_type="round"
                    )

                # Pull out just the probabilities
                before_probs = [pf[0] for pf in before_probs_and_frames]
                after_probs = [pf[0] for pf in after_probs_and_frames]

                # Calculate means
                mean_round_after = np.mean(after_probs) if after_probs else 0.0
                mean_not_round_after = 1.0 - mean_round_after if after_probs else 0.0
                mean_end_before = np.mean(before_probs) if before_probs else 0.0

                # Debug
                timestamp = str(datetime.timedelta(seconds=int(boundary_time_secs)))
                f.write(f"Scene start: {timestamp}\n")
                f.write(f"In round: {in_round}\n")
                f.write(f"Mean round after: {mean_round_after:.3f}\n")
                f.write(f"Mean not round after: {mean_not_round_after:.3f}\n")
                f.write(f"Mean end before: {mean_end_before:.3f}\n\n")

                for pval in after_probs:
                    f.write(f"probs: round {pval:.3f}\n")
                for pval in before_probs:
                    f.write(f"probs: end {pval:.3f}\n")
                f.write("\n\n")

                # Round detection logic
                if not in_round:
                    # Start new round if enough round-likelihood "after" boundary
                    if mean_round_after >= 0.85:
                        in_round = True
                        current_round_start = boundary_time_secs
                        rounds.append(Round(
                            start_time=boundary_time_secs,
                            end_time=0.0,
                            player1=player1,
                            player2=player2,
                            winner="unknown",
                            video_id=video_id,
                            upload_date=video_upload_date
                        ))
                else:
                    # If round is too long (over 99s), consider it invalid/ended
                    if boundary_time_secs - current_round_start > 99:
                        in_round = False
                        # Remove the pending round
                        if rounds:
                            rounds.pop()
                        # Check if we should start a new round here
                        if mean_round_after >= 0.85:
                            in_round = True
                            current_round_start = boundary_time_secs
                            rounds.append(Round(
                                start_time=boundary_time_secs,
                                end_time=0.0,
                                player1=player1,
                                player2=player2,
                                winner="unknown",
                                video_id=video_id,
                                upload_date=video_upload_date
                            ))
                    # End the round if the "end" threshold is met
                    elif mean_end_before >= 0.8 and rounds:
                        rounds[-1].end_time = boundary_time_secs

                        # We want to save a screenshot from the middle of
                        # before_probs_and_frames
                        if before_probs_and_frames:
                            mid_idx = len(before_probs_and_frames) // 2
                            _, middle_frame = before_probs_and_frames[mid_idx]
                            if middle_frame is not None:
                                from videoutil import make_debug_screenshot_filepath
                                filepath = make_debug_screenshot_filepath(
                                    rounds[-1].video_id,
                                    rounds[-1].start_time,
                                    True
                                )
                                success = cv2.imwrite(filepath, middle_frame)
                                print(f"\nSaved round-end debug screenshot to {filepath}, success: {success}")

                        in_round = False

        # If we ended mid-round, close at final frame
        if in_round and rounds:
            video_end = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
            rounds[-1].end_time = video_end

        cap.release()
        return rounds

    def get_frame_probs(
        self,
        cap: cv2.VideoCapture,
        start_index: int,
        frame_nums: List[int],
        model_type: str = "round"
    ) -> tuple[List[tuple[float, Optional[np.ndarray]]], int]:
        """
        Given an *already open* VideoCapture "cap", read frames in ascending order from the
        current position until we reach each target frame in frame_nums. For each frame read,
        compute model probabilities. Return a list of (prob, frame) tuples and the updated
        capture index.

        Args:
            cap: Already opened cv2.VideoCapture object streaming forward only.
            start_index: The current frame index in this cap.
            frame_nums: A sorted list of integer frame indices to read from this capture.
            model_type: "round" uses self.roundornot, "end" uses self.endhead.

        Returns:
            (results, new_index):
                - results: List of (prob, frame) for each requested frame
                - new_index: The new capture position after reading
        """
        if not cap.isOpened():
            return [], start_index

        if not frame_nums:
            return [], start_index

        # Ensure frames are in ascending order
        frame_nums = sorted(frame_nums)

        # Decide which model to use
        model = self.endhead if model_type == "end" else self.roundornot

        results = []
        current_pos = start_index

        # We'll read frames sequentially, discarding any in-between
        # until we reach each frame_num in frame_nums.
        next_target_idx = 0

        while True:
            # If we've satisfied all requested frames, break
            if next_target_idx >= len(frame_nums):
                break

            # If the current position is already past the final requested frame, break
            if current_pos > frame_nums[-1]:
                break

            ret, frame = cap.read()
            if not ret or frame is None:
                # If no more frames, break early
                break

            # We read a new frame, so increment position
            current_pos += 1

            # Check if this frame is one of the targets
            while next_target_idx < len(frame_nums) and current_pos - 1 == frame_nums[next_target_idx]:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with model.no_bar(), model.no_logging():
                    pred = model.predict(frame_rgb)
                pred_label = model.dls.vocab[pred[1]]
                prob = float(pred[2][pred[1]])

                # For "round" detection, store prob of 'Round'
                # For "end" detection, store prob of 'End'
                if model_type == "round":
                    # Probability it's "Round"
                    if pred_label == "Round":
                        results.append((prob, frame))
                    else:
                        # Probability it's "NotRound"
                        results.append((1.0 - prob, frame))
                else:
                    # Probability it's "End"
                    if pred_label == "End":
                        results.append((prob, frame))
                    else:
                        # Probability it's "NotEnd"
                        results.append((1.0 - prob, frame))

                next_target_idx += 1

        return results, current_pos

    def get_frames_at_time(self, video_path: str, time: float, num_frames: int) -> List[np.ndarray]:
        """
        Get frames from a video at a specific time.
        
        Args:
            video_path: Path to the video file
            time: Time in seconds to get frames from
            num_frames: Number of frames to get
            
        Returns:
            List of frames as numpy arrays
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        seek_frame(cap, time)
        ret, frame = cap.read()
        if ret:
            for i in range(num_frames):
                frames.append(frame)
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
        
        cap.release()
        return frames

    def save_round(self, round: Round) -> None:
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
        round_id = f"{round.video_id}_{int(round.start_time)}"
        
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

# Example usage
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python videoanalyzer.py <video_id1> [video_id2 ...] OR python videoanalyzer.py all")
        sys.exit(1)
        
    analyzer = KOFAnalyzer(download_screenshots=False)
    results = {}
    
    if sys.argv[1].lower() == "all":
        # Process all videos in the videos folder
        videos_dir = os.path.abspath("videos")
        if not os.path.exists(videos_dir):
            print(f"Error: videos directory not found at {videos_dir}")
            sys.exit(1)
            
        video_files = [f for f in os.listdir(videos_dir) if f.endswith(('.mp4', '.mkv', '.avi'))]
        if not video_files:
            print("No video files found in videos directory")
            sys.exit(1)
            
        for video_file in video_files:
            video_id = os.path.splitext(video_file)[0]  # Remove file extension
            try:
                video_info = analyzer.analyze_video(video_id)
                results[video_id] = video_info.rounds
            except Exception as e:
                print(f"Error processing video {video_id}: {str(e)}")
                continue
    else:
        # Process specific video IDs provided as arguments
        video_ids = sys.argv[1:]
        for video_id in video_ids:
            try:
                video_info = analyzer.analyze_video(video_id)
                results[video_id] = video_info.rounds
            except Exception as e:
                print(f"Error processing video {video_id}: {str(e)}")
                continue
    
    # Print results for each video
    for video_id, rounds in results.items():
        print(f"\nResults for video {video_id}:")
        for round in rounds:
            timestamp = str(datetime.timedelta(seconds=round.start_time))
            timestamp2 = str(datetime.timedelta(seconds=round.end_time))
            print(timestamp + " - " + timestamp2)
            print(round.character1 + " vs " + round.character2)
