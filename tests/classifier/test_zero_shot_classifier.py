from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

from src.classifier.zero_shot_classifier import ZeroShotClassifier
from src.types.classifier_input import ClassifierInput
from src.types.classifier_output import ClassifierOutput
from src.types.document_type import DocumentType


class TestZeroShotClassifier(TestCase):
    @patch('src.classifier.zero_shot_classifier.pipeline')
    def setUp(self, mock_pipeline):
        self.mock_model = MagicMock()
        mock_pipeline.return_value = self.mock_model

        self.classifier = ZeroShotClassifier()


    @patch('src.classifier.zero_shot_classifier.OCRExtractor')
    def test_zero_shot_classifier(self, mock_ocr_extractor):
        test_text = "test text"
        mock_ocr_extractor.extract_all_documents.return_value = {Path("test.pdf"): test_text}

        test_model_result = {
            'scores': [0, 0.4, 0.6],
            'labels': ['unknown', 'drivers_license', 'bank_statement']
        }
        self.mock_model.return_value = test_model_result

        expected = ClassifierOutput(
            output_per_file={
                Path("test.pdf"): DocumentType.BANK_STATEMENT
            }
        )

        input = ClassifierInput(
            files=[Path("test.pdf")]
        )
        actual: ClassifierOutput = self.classifier.classify(input)

        self.assertEqual(expected, actual)

    @patch('src.classifier.zero_shot_classifier.OCRExtractor')
    def test_ocr_exception(self, mock_ocr_extractor):
        mock_ocr_extractor.extract_all_documents.side_effect = Exception("test exception")

        input = ClassifierInput(
            files=[Path("test.pdf")]
        )

        expected = ClassifierOutput(
            output_per_file={
                Path("test.pdf"): DocumentType.UNKNOWN
            }
        )

        actual: ClassifierOutput = self.classifier.classify(input)

        self.assertEqual(expected, actual)
    
    @patch('src.classifier.zero_shot_classifier.OCRExtractor')
    def test_empty_result(self, mock_ocr_extractor):
        test_text = ""
        mock_ocr_extractor.extract_all_documents.return_value = {Path("test.pdf"): test_text}
        
        input = ClassifierInput(
            files=[Path("test.pdf")]
        )
        
        expected = ClassifierOutput(
            output_per_file={
                Path("test.pdf"): DocumentType.UNKNOWN
            }
        )

        actual: ClassifierOutput = self.classifier.classify(input)

        self.assertEqual(expected, actual)
    
    @patch('src.classifier.zero_shot_classifier.OCRExtractor')
    def test_missing_keys_in_result(self, mock_ocr_extractor):
        test_text = "test text"
        mock_ocr_extractor.extract_all_documents.return_value = {Path("test.pdf"): test_text}

        test_model_result = {
            'scores': [0.4, 0.6],
        }
        self.mock_model.return_value = test_model_result

        input = ClassifierInput(
            files=[Path("test.pdf")]
        )

        expected = ClassifierOutput(
            output_per_file={
                Path("test.pdf"): DocumentType.UNKNOWN
            }
        )

        actual: ClassifierOutput = self.classifier.classify(input)

        self.assertEqual(expected, actual)