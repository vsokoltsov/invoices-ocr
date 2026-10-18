from enum import StrEnum

OCR_CONFUSION_PAIRS = frozenset({
    frozenset({"0", "O"}),
    frozenset({"1", "I"}),
    frozenset({"1", "L"}),
    frozenset({"2", "Z"}),
    frozenset({"3", "B"}),
    frozenset({"4", "A"}),
    frozenset({"5", "S"}),
    frozenset({"6", "G"}),
    frozenset({"7", "Z"}),
    frozenset({"8", "B"}),
    frozenset({"8", "S"}),
    frozenset({"9", "I"}),
    frozenset({"9", "O"}),
})

class MatchType(StrEnum):
    EXACT = "exact"
    NORMALIZED = "normalized"
    OCR_EQUIVALENT = "ocr_equivalent"
    PARTIAL = "partial"

class EntityType(StrEnum):
    INVOICE_NUMBER = "INVOICE_NUMBER"
    INVOICE_DATE = "INVOICE_DATE"
    INVOICE_DUE_DATE = "INVOICE_DUE_DATE"

    SELLER_NAME = "SELLER_NAME"
    SELLER_ADDRESS = "SELLER_ADDRESS"

    CLIENT_NAME = "CLIENT_NAME"
    CLIENT_ADDRESS = "CLIENT_ADDRESS"

    TAX = "TAX"
    TOTAL = "TOTAL"
    DISCOUNT = "DISCOUNT"

    PAYMENT_DUE_DATE = "PAYMENT_DUE_DATE"
    ACCOUNT_NUMBER = "ACCOUNT_NUMBER"
    BANK_NAME = "BANK_NAME"
    PAYMENT_METHOD = "PAYMENT_METHOD"