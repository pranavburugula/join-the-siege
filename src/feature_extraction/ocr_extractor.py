import logging
from pathlib import Path
from typing import Dict, List, Optional

import pymupdf4llm
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
    def extract_all_documents(cls, dir_path: Optional[Path] = None, paths_list: Optional[List[Path]] = None) -> Dict[Path, str]:
        """Extracts text from all documents in a directory.

        Args:
            dir_path (Path): Path to the directory.

        Returns:
            Dict[Path, str]: Dictionary with file paths as keys and extracted text as values.
        """

        file_paths: List[Path] = []

        if dir_path:
            if not dir_path.exists():
                raise FileNotFoundError(f"Directory not found: {dir_path}")
            elif not dir_path.is_dir():
                raise ValueError(f"Path is not a directory: {dir_path}")
            
            _log.info(f"Extracting text from all documents in {dir_path.name}")

            file_paths = list(dir_path.glob('**/*'))
        elif paths_list:
            file_paths = paths_list
        else:
            raise ValueError("Either dir_path or paths_list must be provided")


        result: Dict[Path, str] = {}

        _log.info(f"Found {len(file_paths)} files to extract")

        for file_path in file_paths:
            result[file_path] = cls.extract_text(file_path)

        _log.info(f"Extracted text from {len(result)} files")

        return result
    
    @classmethod
    def _run_image_ocr_single_file(cls, file_path: Path) -> str:
        _log.debug(f"Using image OCR extractor")

        try:
            image = Image.open(file_path)
            text: str = pytesseract.image_to_string(image)
            return text
        except Exception:
            _log.exception(f"Error extracting text from image {file_path.name}")
            raise
    
    @classmethod
    def _run_pdf_ocr_single_file(cls, file_path: Path) -> str:
        _log.debug(f"Using PDF OCR extractor")

        markdown_doc: str = pymupdf4llm.to_markdown(file_path)

        return markdown_doc


if __name__ == '__main__':
    # Enter a test path
    file_path = Path('files', 'invoice_2.pdf')

    _log.info(f"Extracting text from {file_path.name}")

    _log.info(OCRExtractor.extract_text(file_path))
