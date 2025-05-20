from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from src.types.document_type import DocumentType


@dataclass
class ClassifierOutput:
    output_per_file: Dict[Path, DocumentType]
