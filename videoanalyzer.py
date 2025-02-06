from dataclasses import dataclass
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

@dataclass
class Player:
    name: str
    character: str

@dataclass
class Round:
    start_time: float
    end_time: float
    player1: Player
    player2: Player
    winner: str  # "p1" or "p2"

@dataclass
class VideoInfo:
    video_id: str
    title_players: Optional[List[str]]  # Players identified from title, if any
    rounds: List[Round]

class KOFAnalyzer:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
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

    def analyze_video(self, video_id: str) -> VideoInfo:
        """
        Analyze a KOF video to extract rounds and player information.
        If players can't be determined from the title, they'll be identified per-round.
        """
        # Get video title and try to extract players
        # title_players = self._get_players_from_title(video_id)
        title_players = ["Laudandus", "Pinecone"]
        
        # Get all rounds with their timing and player info
        rounds = self._analyze_rounds(video_id, title_players)
        
        return VideoInfo(
            video_id=video_id,
            title_players=title_players,
            rounds=rounds
        )

    def _get_players_from_title(self, video_id: str) -> Optional[List[str]]:
        """
        Try to extract player names from video title.
        Returns None if we can't confidently identify a fixed set of players.
        """
        # Get video title using your video platform API
        # This is a placeholder - you'll need to implement this
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
        
    def get_round_details(self, start_time: float, end_time: float, screenshot, title_players: Optional[List[str]]) -> Round:
        # Get the player names from the title
        if title_players:
            player1 = Player(name=title_players[0], character="Benimaru")
            player2 = Player(name=title_players[1], character="Kyo")
        else:
            player1 = Player(name="Laudandus", character="Duo Lon")
            player2 = Player(name="Pinecone", character="Sylvie")

        return Round(
            start_time=start_time,
            end_time=end_time,
            player1=player1,
            player2=player2,
            winner="p1"
        )

    def _analyze_rounds(self, video_id: str, title_players: Optional[List[str]]) -> List[Round]:
        """
        Analyze video to detect rounds using scene change detection.
        """
        rounds = []
        video_path = f"videos/{video_id}.mp4"
        
        # Create a ContentDetector instance
        detector = ContentDetector(threshold=40.0)
        
        # Continue with original scene detection
        scenes = detect(video_path, detector)
        
        round_is_going = False
        current_round_info = None
        
        for i, scene in enumerate(scenes):
            start_time = scene[0].get_seconds()
            end_time = scene[1].get_seconds()
            
            # Load a frame from middle of scene for analysis
            cap = cv2.VideoCapture(video_path)
            mid_frame_time = (start_time + end_time) / 2
            cap.set(cv2.CAP_PROP_POS_MSEC, mid_frame_time * 1000)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                continue
            
            if not round_is_going:
                # Ask Anthropic API if this is a new round
                start_check_time = start_time + 1;
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_MSEC, start_check_time * 1000)
                is_new_round, round_info = self._ask_if_new_round(frame)
                if is_new_round:
                    round_is_going = True
                    current_round_info = round_info
                    current_round_info['start_time'] = start_time
            else:
                # Check if this is the end of a round
                end_check_time = max(0, end_time - 2)
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_MSEC, end_check_time * 1000)
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    is_end_of_round, winner = self._ask_if_end_of_round(frame)
                    if is_end_of_round:
                        current_round_info['end_time'] = end_time
                        current_round_info['winner'] = winner
                        rounds.append(Round(**current_round_info))
                        round_is_going = False

        return rounds

    def _ask_if_new_round(self, frame) -> (bool, dict):
        # Placeholder for calling Anthropic API to determine if this is a new round
        # Return a tuple (is_new_round, round_info)
        # round_info should be a dictionary with keys: 'player1', 'player2', 'character1', 'character2'
        pass

    def _ask_if_end_of_round(self, frame) -> (bool, str):
        # Placeholder for calling Anthropic API to determine if this is the end of a round
        # Return a tuple (is_end_of_round, winner)
        # winner should be "p1" or "p2"
        pass

# Example usage
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python videoanalyzer.py <video_id>")
        sys.exit(1)
        
    video_id = sys.argv[1]
    analyzer = KOFAnalyzer()
    video_info = analyzer.analyze_video(video_id)
    for round in video_info.rounds:
        # Convert start_time to a timestamp
        timestamp = str(datetime.timedelta(seconds=round.start_time))
        print(timestamp)

