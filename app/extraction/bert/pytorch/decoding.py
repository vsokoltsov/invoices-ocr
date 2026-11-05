from dataclasses import dataclass

import torch


@dataclass
class OCRWordDecoder:
    """Pool entity-type probabilities over complete OCR words in one window.

    Preserve an explicit B label at a word start. Leave partial words and tokens
    that cross OCR-word boundaries unchanged. No gold labels are used.
    """

    label2id: dict[str, int]

    def __post_init__(self):
        self.id2label = {index: label for label, index in self.label2id.items()}
        self.entity_types = [
            label[2:] for label in self.label2id if label.startswith("B-")
        ]
        if "O" not in self.label2id or any(
            f"I-{entity}" not in self.label2id for entity in self.entity_types
        ):
            raise ValueError("Decoder requires O and paired B/I labels")

    @torch.inference_mode()
    def decode(self, logits: torch.Tensor, offsets, words) -> list[int]:
        """Return BIO IDs for [tokens, labels] logits and global OCR offsets."""
        if logits.ndim != 2 or logits.shape != (len(offsets), len(self.label2id)):
            raise ValueError("Logit shape must match token offsets and label count")
        probabilities = logits.softmax(dim=-1).cpu()
        original = probabilities.argmax(dim=-1).tolist()
        predictions = original.copy()
        offsets = [(int(start), int(end)) for start, end in offsets]
        types = torch.stack(
            [probabilities[:, self.label2id["O"]]]
            + [
                probabilities[:, self.label2id[f"B-{entity}"]]
                + probabilities[:, self.label2id[f"I-{entity}"]]
                for entity in self.entity_types
            ],
            dim=-1,
        )
        for word in sorted(words, key=lambda word: int(word["start"])):
            start, end = int(word["start"]), int(word["end"])
            indices = [
                index
                for index, (left, right) in enumerate(offsets)
                if left < right and left < end and right > start
            ]
            if not indices:
                continue
            first, last = indices[0], indices[-1]
            if (
                offsets[first][0] != start
                or offsets[last][1] != end
                or any(offsets[i][0] < start or offsets[i][1] > end for i in indices)
            ):
                continue
            chosen = int(types[indices].mean(dim=0).argmax().item())
            if chosen == 0:
                for index in indices:
                    predictions[index] = self.label2id["O"]
                continue
            entity = self.entity_types[chosen - 1]
            begin, inside = (
                self.label2id[f"{prefix}-{entity}"] for prefix in ("B", "I")
            )
            previous = (
                predictions[first - 1]
                if first > 0 and offsets[first - 1][0] < offsets[first - 1][1]
                else self.label2id["O"]
            )
            predictions[first] = (
                begin
                if previous not in (begin, inside) or original[first] == begin
                else inside
            )
            for index in indices[1:]:
                predictions[index] = inside
        return predictions
