from enum import Enum


class DocumentType(Enum):
    UNKNOWN = "unknown"
    DRIVERS_LICENSE = "drivers_license"
    BANK_STATEMENT = "bank_statement"
    INVOICE = "invoice"