from torch.utils.data import Dataset


class InvoiceDataset(Dataset):
    """Convert OCR spans into BIO labels for consecutive token windows."""

    def __init__(
        self,
        records,
        tokenizer,
        label2id,
        field_to_entity,
        max_length=512,
        *,
        include_decoder_metadata=False,
    ):
        self.examples = []

        for record in records:
            encoded = tokenizer(
                record["text"],
                truncation=True,
                max_length=max_length,
                return_offsets_mapping=True,
                return_overflowing_tokens=True,
                stride=0,
            )

            for window, offsets in enumerate(encoded["offset_mapping"]):
                labels = []
                previous_entity = None

                for start, end in offsets:
                    if start == end:
                        labels.append(-100)
                        previous_entity = None
                        continue

                    overlaps_ignored_region = any(
                        start < region["end"] and end > region["start"]
                        for region in record.get("ignored_regions", [])
                    )

                    if overlaps_ignored_region:
                        labels.append(-100)
                        previous_entity = None
                        continue

                    overlapping = [
                        (index, entity)
                        for index, entity in enumerate(record["entities"])
                        if start < entity["end"] and end > entity["start"]
                    ]

                    if not overlapping:
                        labels.append(label2id["O"])
                        previous_entity = None
                        continue

                    if len(overlapping) != 1:
                        labels.append(-100)
                        previous_entity = None
                        continue

                    entity_index, entity = overlapping[0]

                    # Ignore tokens that cross annotation boundaries.
                    if start < entity["start"] or end > entity["end"]:
                        labels.append(-100)
                        previous_entity = None
                        continue

                    prefix = "I" if previous_entity == entity_index else "B"

                    entity_type = field_to_entity[entity["label"]]
                    bio_label = f"{prefix}-{entity_type}"

                    labels.append(label2id[bio_label])
                    previous_entity = entity_index

                example = {
                    key: encoded[key][window]
                    for key in tokenizer.model_input_names
                    if key in encoded
                }
                example["labels"] = labels
                if include_decoder_metadata:
                    example["decoder_metadata"] = {
                        "offsets": offsets,
                        "words": record["text_with_boxes"],
                    }

                self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]
