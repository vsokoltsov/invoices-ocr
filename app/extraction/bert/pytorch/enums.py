from enum import StrEnum

class InvoiceField(StrEnum):
    INVOICE_CLIENT_ADDRESS = "invoice.client_address"
    INVOICE_CLIENT_NAME = "invoice.client_name"
    INVOICE_SELLER_ADDRESS = "invoice.seller_address"
    INVOICE_SELLER_NAME = "invoice.seller_name"
    INVOICE_NUMBER = "invoice.invoice_number"
    INVOICE_DATE = "invoice.invoice_date"
    ACCOUNT_NUMBER = "payment_instructions.account_number"
    SUBTOTAL_TAX = "subtotal.tax"
    SUBTOTAL_TOTAL = "subtotal.total"

class InvoiceBIOLabel(StrEnum):
    CLIENT_ADDRESS = "CLIENT_ADDRESS"
    CLIENT_NAME = "CLIENT_NAME"
    SELLER_ADDRESS = "SELLER_ADDRESS"
    SELLER_NAME = "SELLER_NAME"
    INVOICE_NUMBER = "INVOICE_NUMBER"
    INVOICE_DATE = "INVOICE_DATE"
    ACCOUNT_NUMBER = "ACCOUNT_NUMBER"
    TAX = "TAX"
    TOTAL = "TOTAL"

FIELD_TO_ENTITY: dict[str, str] = {
    InvoiceField.INVOICE_CLIENT_ADDRESS.value: InvoiceBIOLabel.CLIENT_ADDRESS.value,
    InvoiceField.INVOICE_CLIENT_NAME.value: InvoiceBIOLabel.CLIENT_NAME.value,
    InvoiceField.INVOICE_SELLER_ADDRESS.value: InvoiceBIOLabel.SELLER_ADDRESS.value,
    InvoiceField.INVOICE_SELLER_NAME.value: InvoiceBIOLabel.SELLER_NAME.value,
    InvoiceField.INVOICE_NUMBER.value: InvoiceBIOLabel.INVOICE_NUMBER.value,
    InvoiceField.INVOICE_DATE.value: InvoiceBIOLabel.INVOICE_DATE.value,
    InvoiceField.ACCOUNT_NUMBER.value: InvoiceBIOLabel.ACCOUNT_NUMBER.value,
    InvoiceField.SUBTOTAL_TAX.value: InvoiceBIOLabel.TAX.value,
    InvoiceField.SUBTOTAL_TOTAL.value: InvoiceBIOLabel.TOTAL.value
}
