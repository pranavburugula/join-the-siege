from pathlib import Path
from unittest import TestCase
from src.classifier.filename_classifier import FilenameClassifier
from src.types.document_type import DocumentType
from src.types.classifier_input import ClassifierInput


class TestFilenameClassifier(TestCase):
    def test_classify(self):
        test_input = ClassifierInput(file=Path('test_file.pdf'))

        classifier = FilenameClassifier()
        expected = DocumentType.UNKNOWN
        actual = classifier.classify(test_input)

        self.assertEqual(actual.output_class, expected)
