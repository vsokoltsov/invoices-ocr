import json
import re
import io
from dataclasses import dataclass, asdict
import pandas as pd
import unicodedata
from rapidfuzz import fuzz
import contextlib
from app.extraction.bert.pytorch.spans import OCRSpanMatcher, NumericField, Normalizer
from app.extraction.bert.pytorch.enums import FIELD_TO_ENTITY

ACCOUNT_NUMBER_MIN_SCORE = 80
MIN_NUMBER_SCORE = 80

@dataclass
class InvoiceRecordBuilder:
    span_matcher: OCRSpanMatcher
    numeric: NumericField
    normalizer: Normalizer

    @staticmethod
    def _flatten_field_paths(
        data: dict,
        prefix: str = "",
    ) -> list[dict]:
        """Return dotted paths for non-empty string values in a nested JSON object."""
        paths = []

        for key, value in data.items():
            path = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                paths.extend(
                    InvoiceRecordBuilder._flatten_field_paths(value, path)
                )
            elif isinstance(value, list):
                for idx, item in enumerate(value):
                    if isinstance(item, dict):
                        paths.extend(
                            InvoiceRecordBuilder._flatten_field_paths(item, f"{path}[{idx}]")
                        )
            elif isinstance(value, str) and value.strip():
                paths.append({path: value})

        return paths

    def find_unmatched_fields(self, row):
        """Log and return missing, mismatched, and review-required base fields."""
        ocr_text = row["ocred_text"]
        json_data = row["json_data"]

        if isinstance(json_data, str):
            json_data = json.loads(json_data)

        result_dict, expected_dict = {}, {}
        result = self.span_matcher.find_spans(ocr_text, json_data)
        json_fields = self._flatten_field_paths(json_data)

        for item in result.entities:
            if item.label in result_dict:
                result_dict[item.label] += f"\n{item.text}"
            else:
                result_dict[item.label] = item.text

        for item in json_fields:
            for key, value in item.items():
                if key in expected_dict:
                    expected_dict[key] += f"\n{value}"
                else:
                    expected_dict[key] = value

        issues = []

        for key, value in expected_dict.items():
            orig_value = result_dict.get(key)

            if "items" in key:
                continue

            if not orig_value:
                issues.append({
                    "field": key,
                    "status": "missing",
                    "expected": value,
                    "observed": None
                })
                continue
            
            if self.numeric.is_numeric_field(key):
                is_percentage = unicodedata.normalize(
                    "NFKC", value
                ).strip().endswith("%")
                expected_number = self.normalizer.normalize_number(
                    value,
                    percentage=is_percentage
                )
                observed_number = self.normalizer.normalize_number(
                    orig_value, percentage=is_percentage
                )

                score = fuzz.ratio(
                    s1=str(expected_number), 
                    s2=str(observed_number)
                )
                is_match = (
                    expected_number is not None
                    and observed_number is not None
                    and expected_number == observed_number
                )

                if not is_match: #score < 80:
                    issues.append({
                        "field": key,
                        "status": "numeric_mismatch",
                        "expected": value,
                        "observed": orig_value
                    })
                continue

            if key.endswith(("_name", "_number")):
                score = fuzz.ratio(
                    s1=self.normalizer.normalize(value, path=key), 
                    s2=self.normalizer.normalize(orig_value, path=key)
                )
                field_spans = [entity for entity in result.entities if entity.label == key]
                if key.endswith(".account_number"):
                    requires_review = score < ACCOUNT_NUMBER_MIN_SCORE
                elif key.endswith("_name"):
                    requires_review = any(
                        entity.needs_review for entity in field_spans
                    )
                else:
                    requires_review = self.normalizer.normalize(
                        value, path=key
                    ) != self.normalizer.normalize(
                        orig_value, path=key
                    )
                if requires_review:
                    issues.append({
                        "field": key,
                        "status": "needs_review",
                        "expected": value,
                        "observed": orig_value,
                        "score": score,
                        "spans": field_spans
                    })
                continue

            # A subset of address words must not count as a complete match.
            ratio = fuzz.ratio(
                s1=self.normalizer.normalize(value, path=key),
                s2=self.normalizer.normalize(orig_value, key)
            )

            if ratio < MIN_NUMBER_SCORE:
                issues.append({
                    "field": key,
                    "status": "text_mismatch",
                    "expected": value,
                    "observed": orig_value,
                    "score": ratio
                })
                continue

        return issues


    def find_ignored_account_regions(self, text, json_data):
        account = (
            json_data.get("payment_instructions", {})
            .get("account_number")
        )

        if account is not None and str(account).strip():
            return []

        # Assumes this dataset's IBAN value is one whitespace-delimited token.
        pattern = re.compile(
            r"\bIBAN\s*:\s*(?P<value>[A-Za-z0-9][A-Za-z0-9/.-]*)",
            flags=re.IGNORECASE,
        )

        return [
            {
                "start": match.start("value"),
                "end": match.end("value"),
            }
            for match in pattern.finditer(text)
        ]


    def build(self, frame: pd.DataFrame):
        FIELD_PATHS = FIELD_TO_ENTITY.keys()

        LABELS = ["O"]

        for field in FIELD_PATHS:
            LABELS.extend([f"B-{field}", f"I-{field}"])

        records = []
        rejected = []

        for row_index, row in frame.iterrows():
            target = row["json_data"]

            if isinstance(target, str):
                target = json.loads(target)

            text = row["ocred_text"]

            # Keep parser diagnostics without flooding notebook output.
            with contextlib.redirect_stdout(io.StringIO()) as log:
                issues = self.find_unmatched_fields(row)

            if not isinstance(issues, list):
                raise TypeError(
                    "find_unmatched_fields() must return a list."
                )

            issues = [
                issue
                for issue in issues
                if issue["field"] in FIELD_PATHS
            ]

            result = self.span_matcher.find_spans(text, target)

            entities = [
                entity
                for entity in result.entities
                if entity.label in FIELD_PATHS
            ]

            entities.sort(
                key=lambda entity: (entity.start, entity.end)
            )

            for entity in entities:
                start, end = entity.start, entity.end

                if not (
                    0 <= start < end <= len(text)
                    and text[start:end] == entity.text
                ):
                    raise ValueError(
                        f"Invalid span in invoice {row_index}: {entity}"
                    )

                if entity.needs_review:
                    issues.append({
                        "field": entity.label,
                        "status": "needs_review",
                    })

            for left, right in zip(entities, entities[1:]):
                if right.start < left.end:
                    issues.append({
                        "field": right.label,
                        "status": "overlap",
                    })

            # The parser's similarity threshold can accept partial addresses.
            # Require complete normalized address coverage for this baseline.
            for field in (
                "invoice.client_address",
                "invoice.seller_address",
            ):
                key = field.split(".")[1]
                expected = target.get("invoice", {}).get(key, "")

                observed = " ".join(
                    entity.text
                    for entity in entities
                    if entity.label == field
                )

                if self.normalizer.normalize(expected) != self.normalizer.normalize(observed):
                    issues.append({
                        "field": field,
                        "status": "partial_address",
                    })

            if issues:
                rejected.append({
                    "row_index": row_index,
                    "issues": issues,
                    "log": log.getvalue(),
                })
            else:
                records.append({
                    "row_index": row_index,
                    "text": text,
                    "entities": [asdict(entity) for entity in entities],
                    "ignored_regions": self.find_ignored_account_regions(
                        text=text,
                        json_data=target
                    )
                })

        return records, rejected
