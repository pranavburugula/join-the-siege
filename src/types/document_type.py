from enum import Enum


class DocumentType(Enum):
    UNKNOWN = "other"
    DRIVERS_LICENSE = "drivers_license"
    BANK_STATEMENT = "bank_statement"
    INVOICE = "invoice"

DOCUMENT_TO_INT_LABEL = {doc_type: idx for idx, doc_type in enumerate(DocumentType)}