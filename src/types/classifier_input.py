from dataclasses import dataclass
from pathlib import Path


@dataclass
class ClassifierInput:
    file: Path
