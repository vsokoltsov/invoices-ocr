from pydantic import BaseModel
from app.extraction.bert.fields.field import Field
from app.extraction.bert.fields.account import AccountNumberField
from app.extraction.bert.fields.identifier import IdentifierField
from app.extraction.bert.fields.address import AddressField
from app.extraction.bert.fields.money import MoneyField
from app.extraction.bert.enums import EntityType

class InvoiceData(BaseModel):
    invoice_number: IdentifierField | None = None
    invoice_date: Field | None = None
    invoice_due_date: Field | None = None
    seller_name: Field | None = None
    seller_address: AddressField | None = None
    client_name: Field | None = None
    client_address: AddressField | None = None


class SubtotalData(BaseModel):
    tax: MoneyField | None = None
    total: MoneyField | None = None
    discount: MoneyField | None = None


class PaymentInstructions(BaseModel):
    due_date: Field | None = None
    account_number: AccountNumberField | None = None
    bank_name: Field | None = None
    payment_method: Field | None = None


class InvoiceTarget(BaseModel):
    invoice: InvoiceData
    subtotal: SubtotalData
    payment_instructions: PaymentInstructions
    items: list[dict]

    def extract_fields(self) -> list[tuple[EntityType, Field]]:
        fields = [
            ( EntityType.INVOICE_NUMBER, self.invoice.invoice_number ),
            ( EntityType.INVOICE_DATE, self.invoice.invoice_date ),
            ( EntityType.INVOICE_DUE_DATE, self.invoice.invoice_due_date ),
            ( EntityType.SELLER_NAME, self.invoice.seller_name ),
            ( EntityType.SELLER_ADDRESS, self.invoice.seller_address ),
            ( EntityType.CLIENT_NAME, self.invoice.client_name ),
            ( EntityType.CLIENT_ADDRESS, self.invoice.client_address ),
            ( EntityType.TAX, self.subtotal.tax ),
            ( EntityType.DISCOUNT, self.subtotal.discount ),
            ( EntityType.TOTAL, self.subtotal.total ),
            ( EntityType.PAYMENT_DUE_DATE, self.payment_instructions.due_date ),
            ( EntityType.BANK_NAME, self.payment_instructions.bank_name ),
            ( EntityType.ACCOUNT_NUMBER, self.payment_instructions.account_number ),
            ( EntityType.PAYMENT_METHOD, self.payment_instructions.payment_method )
        ]
        return [
            (entity, value)
            for entity, value in fields
            if (
                value is not None
                and value.normalized()
            )
        ]