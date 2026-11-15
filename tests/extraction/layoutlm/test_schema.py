import pytest

from app.extraction.bert.pytorch.schema import InvoiceLabelSchema as BertLabelSchema
from app.extraction.enums import FIELD_TO_ENTITY
from app.extraction.layoutlm.schema import InvoiceLabelSchema


def test_default_schema_uses_shared_fields_and_bio_labels():
    schema = InvoiceLabelSchema()
    assert schema.num_labels == 19
    assert schema.field_to_entity == FIELD_TO_ENTITY
    assert schema.labels == (
        "O",
        "B-CLIENT_ADDRESS",
        "I-CLIENT_ADDRESS",
        "B-CLIENT_NAME",
        "I-CLIENT_NAME",
        "B-SELLER_ADDRESS",
        "I-SELLER_ADDRESS",
        "B-SELLER_NAME",
        "I-SELLER_NAME",
        "B-INVOICE_NUMBER",
        "I-INVOICE_NUMBER",
        "B-INVOICE_DATE",
        "I-INVOICE_DATE",
        "B-ACCOUNT_NUMBER",
        "I-ACCOUNT_NUMBER",
        "B-TAX",
        "I-TAX",
        "B-TOTAL",
        "I-TOTAL",
    )
    assert schema.label2id["O"] == 0
    assert schema.id2label[18] == "I-TOTAL"
    assert schema.field_to_entity is not FIELD_TO_ENTITY
    bert_schema = BertLabelSchema()
    assert schema.field_to_entity == bert_schema.field_to_entity
    assert schema.label2id == bert_schema.label2id


def test_custom_schema_keeps_order_and_copies_input():
    mapping = {"invoice.seller_name": "SELLER", "invoice.client_name": "CUSTOMER"}
    schema = InvoiceLabelSchema(mapping)
    mapping.clear()
    assert schema.labels == ("O", "B-SELLER", "I-SELLER", "B-CUSTOMER", "I-CUSTOMER")
    assert schema.entity_to_field["SELLER"] == "invoice.seller_name"
    assert all(
        schema.id2label[index] == label for label, index in schema.label2id.items()
    )


@pytest.mark.parametrize(
    "mapping",
    [
        {"a": "NAME", "b": "NAME"},
        {"": "NAME"},
        {"invoice.name ": "NAME"},
        {"invoice.name": " NAME"},
        {"invoice.name": ""},
    ],
)
def test_invalid_schema_is_rejected(mapping):
    with pytest.raises(ValueError):
        InvoiceLabelSchema(mapping)
