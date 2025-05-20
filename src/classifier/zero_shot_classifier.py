import logging
from pathlib import Path
from typing import List
from src.classifier import Classifier

from transformers.pipelines import pipeline

from src.feature_extraction.ocr_extractor import OCRExtractor
from src.types.classifier_input import ClassifierInput
from src.types.classifier_output import ClassifierOutput
from src.types.document_type import DocumentType


_log = logging.getLogger(__name__)

class ZeroShotClassifier(Classifier):
    """Use zero-shot BERT classification to classify documents.
    
    Initializes any HuggingFace Transformers model in aa zero-shot classification
    pipeline. By default, uses the DeBERTav3-zeroshot model (https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0).
    """

    def __init__(self, model_name: str = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"):
        self._model_pipeline =  pipeline(
            "zero-shot-classification",
            model=model_name,
        )
    
    def classify(self, input: ClassifierInput) -> ClassifierOutput:
        _log.info(f"Classifying file {input.file}")

        # Use OCR to get text from file
        try:
            _log.info("Extracting text from file")
            text = OCRExtractor.extract_text(input.file)
        except Exception:
            _log.exception(f"Failed to run OCR extraction on file {input.file}. Returning UNKNOWN")
            return ClassifierOutput(output_class=DocumentType.UNKNOWN)


        if not text:
            _log.warning(f"No text extracted from file {input.file}. Returning UNKNOWN")
            return ClassifierOutput(output_class=DocumentType.UNKNOWN)

        _log.info(f"Got {len(text)} chars from file. Invoking zero-shot classification pipeline")

        
        # Call HF zero-shot pipeline
        result = self._model_pipeline(
            text,
            candidate_labels=[doc_type.value for doc_type in DocumentType],
        )

        if not result or 'scores' not in result or 'labels' not in result:
            _log.error(f"Unknown error prevented model from generating outputs. Returning UNKNOWN")
            return ClassifierOutput(output_class=DocumentType.UNKNOWN)


        # Get the label with the highest confidence score to return as classification
        scores: List[float] = result['scores']
        labels: List[str] = result['labels']

        max_score_index = scores.index(max(scores))

        pred_label = labels[max_score_index]
        pred_score = scores[max_score_index]

        _log.info(f"Classified doc as {pred_label} with score {pred_score}")

        return ClassifierOutput(output_class=DocumentType(pred_label))


if __name__ == "__main__":
    file_path = Path('files', 'bank_statement_1.pdf')

    _log.info(f"Classifying file {file_path}")

    classifier = ZeroShotClassifier()
    _log.info(f"Got {classifier.classify(ClassifierInput(file_path))}")
