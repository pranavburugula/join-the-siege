from pathlib import Path
from typing import Dict
from src.classifier import Classifier
from src.types.document_type import DocumentType
from src.types.classifier_input import ClassifierInput
from src.types.classifier_output import ClassifierOutput


class FilenameClassifier(Classifier):
    def classify(self, input: ClassifierInput) -> ClassifierOutput:
        output_per_file: Dict[Path, DocumentType] = {}
        for file in input.files:
            filename = str(file).lower()

            result_class: DocumentType = DocumentType.UNKNOWN
            
            if "drivers_license" in filename:
                result_class = DocumentType.DRIVERS_LICENSE

            if "bank_statement" in filename:
                result_class = DocumentType.BANK_STATEMENT

            if "invoice" in filename:
                result_class = DocumentType.INVOICE

            output_per_file[file] = result_class

        return ClassifierOutput(output_per_file)
