import pytest

from app.extraction.bert.pytorch.schema import InvoiceLabelSchema


def test_schema_builds_reversible_bio_labels_and_copies_mapping():
    mapping = {"invoice.seller_name": "SELLER_NAME"}
    schema = InvoiceLabelSchema(mapping)
    mapping["invoice.client_name"] = "CLIENT_NAME"

    assert schema.labels == ("O", "B-SELLER_NAME", "I-SELLER_NAME")
    assert schema.num_labels == 3
    assert schema.entity_to_field == {"SELLER_NAME": "invoice.seller_name"}
    assert schema.field_to_entity == {"invoice.seller_name": "SELLER_NAME"}
    assert all(
        schema.id2label[index] == label for label, index in schema.label2id.items()
    )


@pytest.mark.parametrize(
    "mapping",
    [
        {"invoice.name": "NAME", "invoice.other": "NAME"},
        {"": "NAME"},
        {"invoice.name": " NAME"},
        {"invoice.name": ""},
    ],
)
def test_schema_rejects_ambiguous_or_blank_names(mapping):
    with pytest.raises(ValueError):
        InvoiceLabelSchema(mapping)


def test_default_schema_includes_account_number_and_total():
    schema = InvoiceLabelSchema()
    assert (
        schema.entity_to_field["ACCOUNT_NUMBER"]
        == "payment_instructions.account_number"
    )
    assert schema.entity_to_field["TOTAL"] == "subtotal.total"
