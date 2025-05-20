from pathlib import Path
from unittest import TestCase

from src.feature_extraction.ocr_extractor import OCRExtractor


class TestOCRExtractor(TestCase):
    def test_extract_text(self):
        file_path = Path('files/drivers_license_1.jpg')
        
        text = OCRExtractor.extract_text(file_path).lower()

        self.assertTrue('driver' in text and 'license' in text)
    
    def test_extract_text_file_not_found(self):
        file_path = Path('some/path')

        with self.assertRaises(FileNotFoundError):
            OCRExtractor.extract_text(file_path)
    
    def test_unsupported_file_type(self):
        file_path = Path('requirements.txt')

        with self.assertRaises(ValueError):
            OCRExtractor.extract_text(file_path)
    
    def test_extract_all_documents(self):
        dir_path = Path('files')

        result = OCRExtractor.extract_all_documents(dir_path)

        self.assertEqual(len(result), 9)
    
    def test_extract_all_documents_file_not_found(self):
        dir_path = Path('some/path')

        with self.assertRaises(FileNotFoundError):
            OCRExtractor.extract_all_documents(dir_path)
    
    def test_extract_all_documents_path_not_a_directory(self):
        file_path = Path('files/drivers_license_1.jpg')

        with self.assertRaises(ValueError):
            OCRExtractor.extract_all_documents(file_path)
    
