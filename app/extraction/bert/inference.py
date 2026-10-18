from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase
)
import torch
from app.extraction.bert.enums import EntityType

@dataclass(frozen=True, slots=True)
class PredictedEntity:
    entity: EntityType
    start: int
    end: int
    text: str
    confidence: float

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(
                "Entity start cannot be negative"
            )

        if self.end <= self.start:
            raise ValueError(
                f"Invalid entity span: "
                f"[{self.start}, {self.end})"
            )

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "Confidence must be between 0 and 1"
            )

@dataclass(slots=True)
class DocumentModelPredictor:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    id2label: Mapping[int, str]
    max_length: int = 512
    window_stride: int = 64

    def __post_init__(self) -> None:
        if not self.tokenizer.is_fast:
            raise ValueError(
                "DocumentsModelPredictor requires "
                "a fast tokenizer because it uses "
                "offset mappings"
            )

        if self.max_length <= 0:
            raise ValueError(
                "max_length must be positive"
            )

        if not 0 <= self.window_stride < self.max_length:
            raise ValueError(
                "window_stride must be between 0 "
                "and max_length - 1"
            )

    @property
    def device(self) -> torch.device:
        return next(
            self.model.parameters()
        ).device

    def predict(
        self,
        text: str,
    ) -> list[PredictedEntity]:
        text = str(text)

        if not text.strip():
            return []

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            stride=self.window_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=True,
            return_tensors="pt",
        )

        offset_chunks = encoding.pop(
            "offset_mapping"
        ).tolist()

        encoding.pop(
            "overflow_to_sample_mapping",
            None,
        )

        model_inputs = {
            name: values.to(self.device)
            for name, values in encoding.items()
        }

        self.model.eval()

        with torch.inference_mode():
            logits = self.model(
                **model_inputs
            ).logits.cpu()

        token_predictions = (
            self._merge_chunk_predictions(
                logits=logits,
                offset_chunks=offset_chunks,
            )
        )

        return self._tokens_to_entities(
            text=text,
            token_predictions=(
                token_predictions
            ),
        )

    def _merge_chunk_predictions(
        self,
        *,
        logits: torch.Tensor,
        offset_chunks: list[
            list[list[int]]
        ],
    ) -> list[
        tuple[int, int, str, float]
    ]:
        """
        Merge predictions for tokens repeated in
        overlapping tokenizer chunks.

        Returns:

            [
                (
                    token_start,
                    token_end,
                    predicted_label,
                    confidence,
                ),
                ...
            ]
        """
        logits_by_offset: dict[
            tuple[int, int],
            list[torch.Tensor],
        ] = defaultdict(list)

        for chunk_logits, offsets in zip(
            logits,
            offset_chunks,
        ):
            for token_logits, offset in zip(
                chunk_logits,
                offsets,
            ):
                start, end = offset

                # Ignore special and padding tokens.
                if start == end:
                    continue

                logits_by_offset[
                    (start, end)
                ].append(token_logits)

        predictions = []

        for (
            start,
            end,
        ), repeated_logits in sorted(
            logits_by_offset.items(),
            key=lambda item: item[0],
        ):
            # A token may be present in multiple
            # overlapping chunks. Average its logits.
            mean_logits = torch.stack(
                repeated_logits
            ).mean(dim=0)

            probabilities = torch.softmax(
                mean_logits,
                dim=-1,
            )

            predicted_id = int(
                probabilities.argmax().item()
            )

            confidence = float(
                probabilities[
                    predicted_id
                ].item()
            )

            label = self.id2label[
                predicted_id
            ]

            predictions.append(
                (
                    start,
                    end,
                    label,
                    confidence,
                )
            )

        return predictions

    def _tokens_to_entities(
        self,
        *,
        text: str,
        token_predictions: list[
            tuple[int, int, str, float]
        ],
    ) -> list[PredictedEntity]:
        entities: list[PredictedEntity] = []

        current_entity: EntityType | None = None
        current_start: int | None = None
        current_end: int | None = None
        current_confidences: list[float] = []

        def finish_current_entity() -> None:
            nonlocal current_entity
            nonlocal current_start
            nonlocal current_end
            nonlocal current_confidences

            if (
                current_entity is None
                or current_start is None
                or current_end is None
            ):
                return

            confidence = (
                sum(current_confidences)
                / len(current_confidences)
            )

            entities.append(
                PredictedEntity(
                    entity=current_entity,
                    start=current_start,
                    end=current_end,
                    text=text[
                        current_start:current_end
                    ],
                    confidence=confidence,
                )
            )

            current_entity = None
            current_start = None
            current_end = None
            current_confidences = []

        for (
            token_start,
            token_end,
            label,
            confidence,
        ) in token_predictions:
            if label == "O":
                finish_current_entity()
                continue

            if "-" not in label:
                raise ValueError(
                    f"Invalid BIO label: {label}"
                )

            prefix, entity_name = label.split(
                "-",
                1,
            )

            if prefix not in {"B", "I"}:
                raise ValueError(
                    f"Invalid BIO prefix: {prefix}"
                )

            entity_type = EntityType(
                entity_name
            )

            starts_new_entity = (
                prefix == "B"
                or current_entity is None
                or current_entity != entity_type
            )

            if starts_new_entity:
                finish_current_entity()

                current_entity = entity_type
                current_start = token_start
                current_end = token_end
                current_confidences = [
                    confidence
                ]
            else:
                current_end = token_end
                current_confidences.append(
                    confidence
                )

        finish_current_entity()

        return entities