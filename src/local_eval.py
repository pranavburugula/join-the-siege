import logging
from pathlib import Path
from typing import List
from src.classifier import Classifier
from src.classifier.zero_shot_classifier import ZeroShotClassifier
from src.dataset.invoice_dataset import InvoiceDataset
from src.dataset.license_dataset import LicenseDataset
from src.dataset.statements_dataset import StatementsDataset

from src.types.classifier_input import ClassifierInput
from src.types.classifier_output import ClassifierOutput
from src.types.document_type import DOCUMENT_TO_INT_LABEL

import torch
from torch.utils.data import ConcatDataset, DataLoader
from torcheval.metrics.functional import multiclass_accuracy


_log = logging.getLogger(__name__)

def main():
    # Set up eval datasets
    invoice_dataset = InvoiceDataset()
    license_dataset = LicenseDataset()
    bank_statements_dataset = StatementsDataset()
    dataset = ConcatDataset([invoice_dataset, license_dataset, bank_statements_dataset])

    dataloader = DataLoader(dataset, batch_size=1,shuffle=True)

    classifier: Classifier = ZeroShotClassifier()

    # Populate predictions over each batch, and calculate accuracy metric
    predictions: List[int] = []
    labels: List[int] = []
    i = 0
    for batch_files, batch_labels in dataloader:
        if i % 5 == 0:
            _log.info(f"Batch {i}/{len(dataloader)}")
        i += 1

        # Invoke classifier with input file
        input = ClassifierInput(files=[Path(file) for file in batch_files])
        output: ClassifierOutput = classifier.classify(input)

        predictions += [DOCUMENT_TO_INT_LABEL[output_class] for output_class in output.output_per_file.values()]
        labels += batch_labels
    
    accuracy = multiclass_accuracy(torch.tensor(predictions), torch.tensor(labels))
    _log.info(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    main()
