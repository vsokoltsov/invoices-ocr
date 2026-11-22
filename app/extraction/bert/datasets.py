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

                # Score and supervise one token per OCR word, independent of subword count.
                words = record.get("text_with_boxes")
                if words is not None:
                    starts = {
                        int(word["start"]) for word in words if word["text"].strip()
                    }
                    labels = [
                        label if start in starts and start < end else -100
                        for label, (start, end) in zip(labels, offsets, strict=True)
                    ]
                else:
                    word_ids = encoded.word_ids(batch_index=window)
                    labels = [
                        label
                        if word_id is not None
                        and (index == 0 or word_id != word_ids[index - 1])
                        else -100
                        for index, (label, word_id) in enumerate(
                            zip(labels, word_ids, strict=True)
                        )
                    ]
                first = next(
                    (index for index, label in enumerate(labels) if label != -100), None
                )
                if first is not None:
                    name = next(
                        name
                        for name, value in label2id.items()
                        if value == labels[first]
                    )
                    if name.startswith("I-"):
                        labels[first] = label2id["B-" + name[2:]]

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
