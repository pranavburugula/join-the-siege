from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class ClassifierInput:
    files: Optional[List[Path]]
    dir_path: Optional[Path] = None
