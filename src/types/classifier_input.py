from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class ClassifierInput:
    files: List[Path]
