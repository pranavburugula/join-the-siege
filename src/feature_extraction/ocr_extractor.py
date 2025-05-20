import logging
from pathlib import Path
from typing import Dict, List

import pytesseract
from PIL import Image

from src.constants import SUPPORTED_IMAGE_TYPES


_log = logging.getLogger(__name__)


class OCRExtractor:
    @classmethod
    def extract_text(cls, file_path: Path) -> str:
        """Extracts text from a single file.

        Args:
            file_path (Path): Path to the file.

        Returns:
            str: Extracted text.
        """

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        _log.info(f"Extracting text from {file_path.name}")

        if file_path.suffix.lower() in SUPPORTED_IMAGE_TYPES:
            return cls._run_image_ocr_single_file(file_path)
        elif file_path.suffix.lower() == '.pdf':
            return cls._run_pdf_ocr_single_file(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

    @classmethod
    def extract_all_documents(cls, dir_path: Path) -> Dict[Path, str]:
        """Extracts text from all documents in a directory.

        Args:
            dir_path (Path): Path to the directory.

        Returns:
            Dict[Path, str]: Dictionary with file paths as keys and extracted text as values.
        """

        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        if not dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {dir_path}")

        _log.info(f"Extracting text from all documents in {dir_path.name}")

        result: Dict[Path, str] = {}
        file_paths: List[Path] = list(dir_path.glob('**/*'))

        _log.info(f"Found {len(file_paths)} files to extract")

        for file_path in file_paths:
            cls.extract_text(file_path)
            result[file_path] = cls.extract_text(file_path)

        _log.info(f"Extracted text from {len(result)} files")

        return result
    
    @classmethod
    def _run_image_ocr_single_file(cls, file_path: Path) -> str:
        _log.info(f"Using image OCR extractor")

        try:
            image = Image.open(file_path)
            text: str = pytesseract.image_to_string(image)
            return text
        except Exception:
            _log.error(f"Error extracting text from image {file_path.name}")
            raise
    
    @classmethod
    def _run_pdf_ocr_single_file(cls, file_path: Path) -> str:
        _log.info(f"Using PDF OCR extractor")

        # TODO - implement
        return ""
