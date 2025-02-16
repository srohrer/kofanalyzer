from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Round:
    round_id: Optional[int] = None  # ← New optional field for a round ID
    start_time: float = 0.0
    end_time: float = 0.0
    player1: str = ""
    player2: str = ""
    character1: str = "unknown"
    character2: str = "unknown"
    winner: str = "unknown"
    start_probs: Optional[List[float]] = None
    end_probs: Optional[List[float]] = None
    video_id: str = ""
    upload_date: str = ""

@dataclass
class VideoInfo:
    video_id: str
    upload_date: str                  # ← New field for the upload date
    title_players: Optional[List[str]]  # Players identified from title, if any
    rounds: List[Round]

KOFXV_CHARACTERS = [
    "Andy Bogard",
    "Angel",
    "Antonov",
    "Ash Crimson",
    "Athena Asamiya",
    "B. Jenet",
    "Benimaru Nikaido",
    "Billy Kane",
    "Blue Mary",
    "Chizuru Kagura",
    "Chris",
    "Clark Still",
    "Darli Dagger",
    "Dolores",
    "Duo Lon",
    "Elisabeth Blanctorche",
    "Gato",
    "Geese Howard",
    "Goenitz",
    "Haohmaru",
    "Heidern",
    "Hinako",
    "Iori Yagami",
    "Isla",
    "Joe Higashi",
    "K'",
    "Kim Kaphwan",
    "King",
    "King of Dinosaurs",
    "Krohnen",
    "Kukri",
    "Kula Diamond",
    "Kyo Kusanagi",
    "Leona Heidern",
    "Luong",
    "Mai Shiranui",
    "Mature",
    "Maxima",
    "Meitenkun",
    "Najd",
    "Nakoruru",
    "Omega Rugal",
    "Orochi Chris",
    "Orochi Shermie",
    "Orochi Yashiro",
    "Ralf Jones",
    "Ramon",
    "Robert Garcia",
    "Rock Howard",
    "Ryo Sakazaki",
    "Ryuji Yamazaki",
    "Shermie",
    "Shingo Yabuki",
    "Shun'ei",
    "Sylvie",
    "Terry Bogard",
    "Vanessa",
    "Vice",
    "Whip",
    "Yashiro Nanakase",
    "Yuri Sakazaki",
]