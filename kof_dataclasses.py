from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Round:
    start_time: float
    end_time: float
    player1: str
    player2: str
    character1: str = "unknown"
    character2: str = "unknown"
    winner: str = "unknown"
    start_probs: List[float] = None  # Add default None
    end_probs: List[float] = None    # Add default None
    video_id: str = ""               # Add default empty string
    upload_date: str = ""            # ← New field for the upload date

@dataclass
class VideoInfo:
    video_id: str
    upload_date: str                  # ← New field for the upload date
    title_players: Optional[List[str]]  # Players identified from title, if any
    rounds: List[Round]