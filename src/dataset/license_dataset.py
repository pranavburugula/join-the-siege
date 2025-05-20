import logging
from pathlib import Path
import random
from typing import List

from src.types.document_type import DOCUMENT_TO_INT_LABEL, DocumentType
from src.constants import DATASET_DIR

from torch.utils.data import Dataset

_log = logging.getLogger(__name__)


class LicenseDataset(Dataset):

    def __init__(self, num_samples=100, base_dir=Path(DATASET_DIR, 'licenses', 'data')):
        super(LicenseDataset).__init__()

        self._data_base_dir: Path = base_dir

        all_files: List[Path] = list(self._data_base_dir.glob('**/*'))
        self._file_paths: List[Path] = random.choices(all_files, k=min(len(all_files),num_samples))
        
    
    def __len__(self):
        return len(self._file_paths)
    
    def __getitem__(self, index):
        return str(self._file_paths[index]), DOCUMENT_TO_INT_LABEL[DocumentType.DRIVERS_LICENSE]
