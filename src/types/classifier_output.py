from dataclasses import dataclass

from src.types.document_type import DocumentType


@dataclass
class ClassifierOutput:
    output_class: DocumentType
