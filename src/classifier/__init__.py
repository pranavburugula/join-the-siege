from abc import ABC, abstractmethod

from src.types.classifier_input import ClassifierInput
from src.types.classifier_output import ClassifierOutput


class Classifier(ABC):
    """Base document type classifier.
    
    Subclasses must provide an implementation for the `classify` method, which is
    called during runtime.
    """

    @abstractmethod
    def classify(self, input: ClassifierInput) -> ClassifierOutput:
        """Takes an input payload with filename and classifies its type."""
        pass