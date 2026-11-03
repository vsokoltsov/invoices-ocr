import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from typing import ClassVar

from rapidfuzz import fuzz

from app.extraction.bert.pytorch.annotations import (
    OCRSpanEntity,
    OCRSpanResult,
)
from app.extraction.bert.pytorch.enums import InvoiceField

ACCOUNT_NUMBER_MIN_SCORE = 80

NUMERIC_FIELD_NAMES = {
    "quantity",
    "total_price",
    "tax",
    "discount",
    "total",
}


@dataclass
class Normalizer:
    def normalize_with_offsets(self, text: str) -> tuple[str, list[int]]:
        """Normalize text for matching while mapping each output character to OCR.

        Punctuation and whitespace form one separator, preserving word boundaries.
        Unicode expansions map back to the original character that produced them.
        """
        normalized_chars = []
        original_offsets = []

        for index, char in enumerate(text):
            char = unicodedata.normalize("NFKC", char).casefold()
            for normalized_char in char:
                if normalized_char.isalnum():
                    normalized_chars.append(normalized_char)
                    original_offsets.append(index)
                elif normalized_chars and normalized_chars[-1] != " ":
                    normalized_chars.append(" ")
                    original_offsets.append(index)

        if normalized_chars and normalized_chars[-1] == " ":
            normalized_chars.pop()
            original_offsets.pop()

        return "".join(normalized_chars), original_offsets

    def normalize_identifier(self, value: str) -> str:
        """Ignore identifier spacing and separators while preserving letters/digits."""
        value = unicodedata.normalize("NFKC", str(value)).casefold()
        return re.sub(r"[\s./\-]", "", value)

    def normalize(self, value: str, path: str = "") -> str:
        """Normalize text, or an identifier when its field path is supplied."""
        if path.rsplit(".", 1)[-1] in {"invoice_number", "account_number"}:
            return self.normalize_identifier(value)
        normalized, _ = self.normalize_with_offsets(str(value))
        return normalized

    def normalize_number(
        self, value: str, *, percentage: bool = False
    ) -> Decimal | None:
        """Parse invoice amounts with up to two decimal places and valid grouping.

        Accept comma/dot decimals, grouped thousands, and surrounding currency
        symbols. Reject ambiguous single separators (1,000), malformed grouping,
        arbitrary letters, and multiple amounts instead of silently merging them.
        With percentage=True, require a trailing percent sign and return percentage
        points (10% becomes Decimal('10'), not Decimal('0.1')).
        """
        value = unicodedata.normalize("NFKC", str(value)).strip()
        if percentage:
            if not value.endswith("%"):
                return None
            value = value[:-1].strip()
        value = value.strip("$€£¥ ").strip()
        value = value.replace("’", "'")
        sign = ""
        if value.startswith(("+", "-")):
            sign, value = value[0], value[1:]

        if not value or not re.fullmatch(r"[0-9.,'\s]+", value):
            return None

        integer_part = value
        fractional_part = ""
        separators = {char for char in value if char in ".,"}
        if separators:
            decimal_separator = value[max(value.rfind("."), value.rfind(","))]
            before, after = value.rsplit(decimal_separator, 1)
            if re.fullmatch(r"[0-9]{1,2}", after):
                integer_part, fractional_part = before, after
                if decimal_separator in integer_part:
                    return None
            elif len(separators) == 1 and value.count(decimal_separator) >= 2:
                # Repeated three-digit groups have an unambiguous integer reading.
                pass
            else:
                return None

        if not re.fullmatch(r"[0-9]+", integer_part):
            if not re.fullmatch(
                r"[0-9]{1,3}([.,'\s])[0-9]{3}(?:\1[0-9]{3})*", integer_part
            ):
                return None
        integer_digits = re.sub(r"[^0-9]", "", integer_part)
        canonical = sign + integer_digits
        if fractional_part:
            canonical += "." + fractional_part

        try:
            return Decimal(canonical)
        except InvalidOperation:
            return None


@dataclass
class NumericField:
    NUMERIC_FIELD_NAMES: ClassVar[set[str]] = {
        "quantity",
        "total_price",
        "tax",
        "discount",
        "total",
    }
    normalizer: Normalizer

    def is_numeric_field(self, path: str) -> bool:
        field_name = path.rsplit(".", 1)[-1].split("[", 1)[0]
        return field_name in self.NUMERIC_FIELD_NAMES

    def number_pattern(self, value: str) -> re.Pattern:
        """Build a regex for a numeric field, allowing decimal/grouping variants.

        The rightmost dot or comma is treated as the decimal separator when the
        value has digits after it. Group separators and spaces between integer
        digits are accepted, while the decimal boundary and fractional digits are
        kept significant (so 13.99 cannot match 139,9).
        """
        is_percentage = unicodedata.normalize("NFKC", str(value)).strip().endswith("%")
        amount = self.normalizer.normalize_number(value, percentage=is_percentage)
        if amount is None:
            raise ValueError(f"Invalid or ambiguous invoice amount: {value!r}")
        number_text = format(amount, "f")

        sign = r"[-+]?"
        if number_text.startswith(("-", "+")):
            sign = re.escape(number_text[0])
            number_text = number_text[1:].strip()

        decimal_pos = max(number_text.rfind("."), number_text.rfind(","))
        if decimal_pos >= 0 and number_text[decimal_pos + 1 :].isdigit():
            integer_part = number_text[:decimal_pos]
            fractional_part = number_text[decimal_pos + 1 :]
        else:
            integer_part = number_text
            fractional_part = ""

        integer_digits = re.findall(r"\d", integer_part)
        if not integer_digits:
            raise ValueError(f"Numeric field has no integer digits: {value!r}")

        # Permit OCR differences in thousands separators, e.g. 1 234,20 / 1234.20.
        integer_pattern = r"[\s.,'’]*".join(map(re.escape, integer_digits))
        number_pattern_text = sign + integer_pattern

        if fractional_part:
            number_pattern_text += r"[.,]" + re.escape(fractional_part)

        if is_percentage:
            number_pattern_text += r"\s*%"

        return re.compile(rf"(?<!\d){number_pattern_text}(?!\d)")


@dataclass
class OCRSummarySpan:
    normalizer: Normalizer

    # Define a complete monetary amount
    # The regex:
    #   * (?<![\w.,]) - The preceding character must not be a word character, comma or dot
    #   * [+-]? - Permit an optional sign
    #   * \d{1,3}(?:[ \u00a0.'’]\d{3})+ - Permit grouped integer digits, such as `2 731``
    #   * `|`` between the alternatives - Choose either grouped digits or the ungrouped alternative
    #   * \d+ - Permit ungrouped digits, such as `2731`
    #   * [,.]\d{2} - Require a decimal separator followed by exactly two digits
    #   * (?![\d.,]) - Don’t stop immediately before another digit, comma or dot
    #   * \u00a0 - represents a nonbreaking space, which OCR can use as a thousands separator.
    # Examples this pattern can capture:
    #   * 273,18
    #   * 273.18
    #   * 3 004,96
    #   * 3004.96
    MONEY_PATTERN: ClassVar[re.Pattern] = re.compile(
        r"(?<![\w.,])"
        r"[+-]?"
        r"(?:\d{1,3}(?:[ \u00a0.'’]\d{3})+|\d+)"
        r"[,.]\d{2}"
        r"(?![\d.,])"
    )

    def find(
        self,
        text: str,
        json_data: dict,
        lookahead: int = 200,
    ) -> dict[str, OCRSpanEntity]:
        """Align JSON subtotal values to complete tokens in the summary.

        Search monetary amounts after the final Total label and percentages
        between the preceding SUMMARY label and Total. Accept only unique
        numerically equal candidates.

        Args:
            text: the original OCR string
            json_data: The expected annotations
            lookahead: how many characters after Total to search
        Returns:
            Vlaue maps field paths to span dictionaries
        """

        # Find all occurrences of the word Total
        anchors = list(
            re.finditer(
                r"\bTotal\b",
                text,
                flags=re.IGNORECASE,
            )
        )

        if not anchors:
            return {}

        # Select the last occurrence
        # This reduces the chance of selecting an earlier occurrence in a descriptio
        anchor = anchors[-1]
        # Set the search boundary to at most 200 characters after the anchor.
        # min() prevents the boundary from exceeding the document length
        region_end = min(len(text), anchor.end() + lookahead)

        # Extract amounts after `Total`
        # These matches already contain offsets into the original full text.
        # Search the original text, beginning immediately after Total and stopping at region_end.
        # Example:
        # ["2 731,78", "273,18", "3 004,96"]
        amounts = list(
            self.MONEY_PATTERN.finditer(
                text,
                anchor.end(),
                region_end,
            )
        )

        # Require the expected three-amount structure.
        # If OCR loses an amount—or the 200-character region contains an
        # extra amount—the function returns {}
        # It does not partially process that row
        if len(amounts) != 3:
            return {}

        # Find the preceding `SUMMARY` section
        # Search only the text before the selected Total for `SUMMARY`.
        # This prepares a second search area for percentage targets such as "10%"
        summary_anchors = list(
            re.finditer(
                r"\bSUMMARY\b",
                text[: anchor.start()],
                flags=re.IGNORECASE,
            )
        )

        # Start with no percentage candidates.
        percentages = []

        # Only search percentages if a preceding summary anchor was found.
        if summary_anchors:
            # This matches values such as:
            #   * 10%
            #   * 10 %
            #   * 7.5%
            #   * 7,5%
            # The decimal portion and whitespace before % are optional.
            percentage_pattern = re.compile(r"(?<![\w.,])\d+(?:[,.]\d+)?\s*%")

            # Search between the last preceding `SUMMARY` and the selected `Total`.
            percentages = list(
                percentage_pattern.finditer(
                    text,
                    summary_anchors[-1].end(),
                    anchor.start(),
                )
            )

        result = {}

        # Process the two expected fields
        # Run the following logic once for tax, then once for total.
        for field in ("tax", "total"):
            # Read the expected field safely.
            # For the example:
            #   First iteration
            #       field = "tax"
            #       expected = "273.18"
            #   Second iteration
            #       field = "total"
            #       expected = "3004.96"
            # If subtotal or the field is absent, use "".
            expected = str(json_data.get("subtotal", {}).get(field, "")).strip()

            if not expected:
                continue

            # Determine whether this particular target is a percentage
            is_percentage = expected.endswith("%")

            # Convert the expected value into a Decimal:
            # Examples:
            #   "273.18" => Decimal("273.18")
            #   "3004.96" => Decimal("3004.96")
            #   "10%" => Decimal("10")  # when percentage=True
            expected_number = self.normalizer.normalize_number(
                expected,
                percentage=is_percentage,
            )

            # Skip a value that the number parser cannot interpret safely.
            if expected_number is None:
                continue

            # Choose monetary or percentage candidates
            #   For "273.18", use the three monetary amounts.
            #   For "10%", use percentage matches from the preceding summary section.
            available = percentages if is_percentage else amounts

            # candidate.group() returns the actual matched OCR substring.
            # For tax = "273.18", the comparisons are:
            #   Decimal("2731.78") == Decimal("273.18")  # False
            #   Decimal("273.18")  == Decimal("273.18")  # True
            #   Decimal("3004.96") == Decimal("273.18")  # False
            # Therefore, only the match for 273,18 remains.
            # For total = "3004.96", only 3 004,96 remains.
            candidates = [
                candidate
                for candidate in available
                if self.normalizer.normalize_number(
                    candidate.group(),
                    percentage=is_percentage,
                )
                == expected_number
            ]

            # Require one unambiguous match
            # Skip the field when:
            # - There are zero numerically equal candidates.
            # - The same expected number occurs in multiple candidate positions.
            if len(candidates) != 1:
                continue

            # The function doesn’t return a review record itself
            # the missing annotation is detected by your verification code.
            # For tax:
            #   path = "subtotal.tax"
            #   observed = "273,18"
            candidate = candidates[0]
            path = f"subtotal.{field}"
            observed = candidate.group()

            # Build the annotation
            #   * start: original-text starting offset.
            #   * end: exclusive ending offset.
            #   * label: field name.
            #   * text: original OCR substring.
            #   * score: diagnostic text similarity.
            #   * match_status: records how the annotation was accepted.
            #   * needs_review=False: it passed this function’s uniqueness and numeric checks.
            result[path] = OCRSpanEntity(
                start=candidate.start(),
                end=candidate.end(),
                label=path,
                text=observed,
                score=fuzz.ratio(
                    self.normalizer.normalize(expected),
                    self.normalizer.normalize(observed),
                ),
                match_status="summary_numeric_equal",
                needs_review=False,
            )

        return result


@dataclass
class OCRSpanMatcher:
    normalizer: Normalizer
    summary_span: OCRSummarySpan
    numeric: NumericField

    def find_normalized_span(
        self, text: str, value: str, label: str
    ) -> OCRSpanEntity | None:
        normalized_text, offsets = self.normalizer.normalize_with_offsets(text)
        normalized_value, _ = self.normalizer.normalize_with_offsets(value)

        if not normalized_value or not normalized_text:
            return None

        match = fuzz.partial_ratio_alignment(normalized_value, normalized_text)
        if not match or match.dest_start == match.dest_end:
            return None

        start = offsets[match.dest_start]
        end = offsets[match.dest_end - 1] + 1
        matched_ocr_text = text[start:end]

        # Reject a mapped range if it doesn't normalize back to the requested value.
        checked_text, _ = self.normalizer.normalize_with_offsets(matched_ocr_text)
        if checked_text != normalized_value:
            return None

        return OCRSpanEntity(
            start=start,
            end=end,
            label=label,
            text=matched_ocr_text,
        )

    def find_name_span(
        self, text: str, expected: str, label: str
    ) -> OCRSpanEntity | None:
        """Accept exact names or ordered partial names with the first name word.

        Partial candidates must contain at least two expected words and no inserted
        words inside their boundaries. Other fuzzy candidates require review.
        This checks text alignment; it cannot establish seller/client context.
        """
        exact = self.find_normalized_span(text, expected, label)
        if exact is not None:
            return OCRSpanEntity(
                start=exact.start,
                end=exact.end,
                label=exact.label,
                text=exact.text,
                needs_review=False,
                score=exact.score,
                match_status="exact",
                expected=exact.expected,
            )

        normalized_text, offsets = self.normalizer.normalize_with_offsets(text)
        normalized_expected = self.normalizer.normalize(expected)
        if not normalized_text or not normalized_expected:
            return None
        match = fuzz.partial_ratio_alignment(normalized_expected, normalized_text)
        if match is None or match.dest_start == match.dest_end:
            return None

        candidate_tokens = list(
            re.finditer(r"\S+", normalized_text[match.dest_start : match.dest_end])
        )
        expected_words = normalized_expected.split()
        blocks = [
            block
            for block in SequenceMatcher(
                None,
                expected_words,
                [token.group() for token in candidate_tokens],
                autojunk=False,
            ).get_matching_blocks()
            if block.size
        ]

        start = offsets[match.dest_start]
        end = offsets[match.dest_end - 1] + 1
        partial = False
        if blocks:
            first, last = blocks[0], blocks[-1]
            matched_words = sum(block.size for block in blocks)
            # Ignore unrelated words at the edges, but reject interior insertions.
            if (
                first.a == 0
                and matched_words >= 2
                and last.b + last.size - first.b == matched_words
            ):
                left = match.dest_start + candidate_tokens[first.b].start()
                right = (
                    match.dest_start + candidate_tokens[last.b + last.size - 1].end()
                )
                start, end = offsets[left], offsets[right - 1] + 1
                partial = True

        observed = text[start:end]
        observed_normalized = self.normalizer.normalize(observed)
        return OCRSpanEntity(
            start=start,
            end=end,
            label=label,
            text=observed,
            needs_review=not partial,
            match_status="partial" if partial else "candidate",
            expected=expected,
            score=fuzz.ratio(normalized_expected, observed_normalized),
        )

    def find_description_spans(
        self, text: str, description: str, label: str
    ) -> tuple[list[OCRSpanEntity], list[str]]:
        """Find description lines within an item region.

        Return matched OCR spans and lines that could not be matched uniquely.
        """
        norm_text, offsets = self.normalizer.normalize_with_offsets(text)
        spans: list[OCRSpanEntity] = []
        unmatched = []

        for fragment in description.splitlines():
            norm_fragment, _ = self.normalizer.normalize_with_offsets(fragment)
            if not norm_fragment:
                continue
            match = fuzz.partial_ratio_alignment(norm_fragment, norm_text)
            if match is None or match.dest_start == match.dest_end:
                unmatched.append(fragment)
                continue

            start = offsets[match.dest_start]
            end = offsets[match.dest_end - 1] + 1
            spans.append(
                OCRSpanEntity(
                    start=start,
                    end=end,
                    label=label,
                    text=text[start:end],
                    score=match.score,
                )
            )

        return spans, unmatched

    def find_spans(self, text, json_data, prefix="") -> OCRSpanResult:
        entities: list[OCRSpanEntity] = []
        stack = [(json_data, "")]
        search_from = {"quantity": 0, "total_price": 0}
        norm_text, offsets = self.normalizer.normalize_with_offsets(text)
        summary_spans = self.summary_span.find(text, json_data)

        while stack:
            current, path = stack.pop()

            if isinstance(current, dict):
                for key, value in current.items():
                    child_path = f"{path}.{key}" if path else key
                    stack.append((value, child_path))
                continue

            if isinstance(current, list):
                for idx in range(len(current) - 1, -1, -1):
                    stack.append((current[idx], f"{path}[{idx}]"))
                continue

            if not isinstance(current, str) or not current.strip():
                continue

            if "items" in path:
                continue
            if path in {
                InvoiceField.SUBTOTAL_TAX.value,
                InvoiceField.SUBTOTAL_TOTAL.value,
            }:
                entity = summary_spans.get(path)
                if entity is not None:
                    entities.append(entity)
                continue
            if path.endswith("_name"):
                entity = self.find_name_span(text, current, path)
                if entity is not None:
                    entities.append(entity)
                continue

            if "account_number" in path:
                norm_value, _ = self.normalizer.normalize_with_offsets(current)
                if not norm_value or not norm_text:
                    continue
                match = fuzz.partial_ratio_alignment(norm_value, norm_text)
                if not match or match.dest_start == match.dest_end:
                    continue
                start = offsets[match.dest_start]
                end = offsets[match.dest_end - 1] + 1
                score = fuzz.ratio(
                    self.normalizer.normalize(current, path),
                    self.normalizer.normalize(text[start:end], path),
                )
                entities.append(
                    OCRSpanEntity(
                        start=start,
                        end=end,
                        label=path,
                        text=text[start:end],
                        needs_review=score < ACCOUNT_NUMBER_MIN_SCORE,
                        score=score,
                    )
                )
                continue

            if self.numeric.is_numeric_field(path):
                pattern = self.numeric.number_pattern(current)
                cursor_key = path.rsplit(".", 1)[-1].split("[", 1)[0]
                start_at = search_from.get(cursor_key, 0)
                match = pattern.search(text, start_at)
                if not match:
                    continue
                start, end = match.span()
                entities.append(
                    OCRSpanEntity(
                        start=start,
                        end=end,
                        label=path,
                        text=text[start:end],
                    )
                )
                search_from[cursor_key] = end

                continue

            if path.endswith("_address"):
                norm_text, offsets = self.normalizer.normalize_with_offsets(text)
                for line in current.splitlines():
                    line = line.strip()
                    if not line:
                        continue

                    entity = self.find_normalized_span(text, line, path)
                    if entity is not None:
                        entity.needs_review = False
                        entity.match_status = "exact"
                        entities.append(entity)
                        continue
                    norm_line, _ = self.normalizer.normalize_with_offsets(line)
                    match = fuzz.partial_ratio_alignment(norm_line, norm_text)
                    if not match or match.dest_end <= match.dest_start:
                        continue

                    start = offsets[match.dest_start]
                    end = offsets[match.dest_end - 1] + 1
                    house_number = re.match(r"\d+\b", line.strip())
                    if house_number:
                        number = house_number.group()

                        # Search within the fuzzy candidate, preserving original offsets.
                        candidate_text = text[start:end]
                        anchor = re.search(
                            rf"(?<!\w){re.escape(number)}(?!\w)", candidate_text
                        )

                        if anchor:
                            start += anchor.start()

                    matched_text = text[start:end]
                    entities.append(
                        OCRSpanEntity(
                            start=start,
                            end=end,
                            label=path,
                            text=matched_text,
                        )
                    )
                continue

            norm_current, _ = self.normalizer.normalize_with_offsets(current)
            if not norm_current or not norm_text:
                continue
            match = fuzz.partial_ratio_alignment(norm_current, norm_text)
            if not match or match.dest_start == match.dest_end:
                continue
            start = offsets[match.dest_start]
            end = offsets[match.dest_end - 1] + 1
            needs_review = (path.endswith("_name") or path.endswith("_number")) and (
                self.normalizer.normalize(current, path)
                != self.normalizer.normalize(text[start:end], path)
            )
            entities.append(
                OCRSpanEntity(
                    start=start,
                    end=end,
                    label=path,
                    text=text[start:end],
                    needs_review=needs_review,
                    score=match.score,
                )
            )

        entities.sort(key=lambda entity: entity.start)

        return OCRSpanResult(
            text=text,
            entities=entities,
        )


# def normalize_with_offsets(text: str) -> tuple[str, list[int]]:
#     """Normalize text for matching while mapping each output character to OCR.

#     Punctuation and whitespace form one separator, preserving word boundaries.
#     Unicode expansions map back to the original character that produced them.
#     """
#     normalized_chars = []
#     original_offsets = []

#     for index, char in enumerate(text):
#         char = unicodedata.normalize("NFKC", char).casefold()
#         for normalized_char in char:
#             if normalized_char.isalnum():
#                 normalized_chars.append(normalized_char)
#                 original_offsets.append(index)
#             elif normalized_chars and normalized_chars[-1] != " ":
#                 normalized_chars.append(" ")
#                 original_offsets.append(index)

#     if normalized_chars and normalized_chars[-1] == " ":
#         normalized_chars.pop()
#         original_offsets.pop()

#     return "".join(normalized_chars), original_offsets


# def find_normalized_span(text: str, value: str, label: str) -> dict | None:
#     normalized_text, offsets = normalize_with_offsets(text)
#     normalized_value, _ = normalize_with_offsets(value)

#     if not normalized_value:
#         return None

#     match = fuzz.partial_ratio_alignment(
#         normalized_value,
#         normalized_text
#     )
#     if not match:
#         return

#     start = offsets[match.dest_start]
#     end = offsets[match.dest_end - 1] + 1
#     matched_ocr_text = text[start:end]

#     # Reject a mapped range if it doesn't normalize back to the requested value.
#     checked_text, _ = normalize_with_offsets(matched_ocr_text)
#     if checked_text != normalized_value:
#         return None

#     return {
#         "start": start,
#         "end": end,
#         "label": label,
#         "text": matched_ocr_text,
#     }

# def find_fuzzy_description_candidate(
#     text: str,
#     fragment: str,
#     item_start: int,
#     item_end: int,
#     min_score: float = 85,
# ) -> dict | None:
#     """Suggest an OCR span for one unmatched description fragment."""
#     item_text = text[item_start:item_end]

#     normalized_text, offsets = normalize_with_offsets(item_text)
#     normalized_fragment, _ = normalize_with_offsets(fragment)

#     if not normalized_text or not normalized_fragment:
#         return None

#     match = fuzz.partial_ratio_alignment(
#         normalized_fragment,
#         normalized_text,
#         score_cutoff=min_score,
#     )
#     if match is None:
#         return None

#     start = item_start + offsets[match.dest_start]
#     end = item_start + offsets[match.dest_end - 1] + 1

#     return {
#         "start": start,
#         "end": end,
#         "text": text[start:end],
#         "expected": fragment,
#         "score": match.score,
#     }


# def find_summary_spans(
#     text: str,
#     json_data: dict,
#     lookahead: int = 200,
# ) -> dict[str, dict]:
#     """Align JSON subtotal values to complete tokens in the summary.

#     Search monetary amounts after the final Total label and percentages
#     between the preceding SUMMARY label and Total. Accept only unique
#     numerically equal candidates.

#     Args:
#         text: the original OCR string
#         json_data: The expected annotations
#         lookahead: how many characters after Total to search
#     Returns:
#         Vlaue maps field paths to span dictionaries
#     """

#     # Find all occurrences of the word Total
#     anchors = list(re.finditer(
#         r"\bTotal\b",
#         text,
#         flags=re.IGNORECASE,
#     ))

#     if not anchors:
#         return {}

#     # Select the last occurrence
#     # This reduces the chance of selecting an earlier occurrence in a descriptio
#     anchor = anchors[-1]
#     # Set the search boundary to at most 200 characters after the anchor.
#     # min() prevents the boundary from exceeding the document length
#     region_end = min(len(text), anchor.end() + lookahead)

#     # Define a complete monetary amount
#     # The regex:
#     #   * (?<![\w.,]) - The preceding character must not be a word character, comma or dot
#     #   * [+-]? - Permit an optional sign
#     #   * \d{1,3}(?:[ \u00a0.'’]\d{3})+ - Permit grouped integer digits, such as `2 731``
#     #   * `|`` between the alternatives - Choose either grouped digits or the ungrouped alternative
#     #   * \d+ - Permit ungrouped digits, such as `2731`
#     #   * [,.]\d{2} - Require a decimal separator followed by exactly two digits
#     #   * (?![\d.,]) - Don’t stop immediately before another digit, comma or dot
#     #   * \u00a0 - represents a nonbreaking space, which OCR can use as a thousands separator.
#     # Examples this pattern can capture:
#     #   * 273,18
#     #   * 273.18
#     #   * 3 004,96
#     #   * 3004.96
#     money_pattern = re.compile(
#         r"(?<![\w.,])"
#         r"[+-]?"
#         r"(?:\d{1,3}(?:[ \u00a0.'’]\d{3})+|\d+)"
#         r"[,.]\d{2}"
#         r"(?![\d.,])"
#     )

#     # Extract amounts after `Total`
#     # These matches already contain offsets into the original full text.
#     # Search the original text, beginning immediately after Total and stopping at region_end.
#     # Example:
#     # ["2 731,78", "273,18", "3 004,96"]
#     amounts = list(money_pattern.finditer(
#         text,
#         anchor.end(),
#         region_end,
#     ))

#     # Require the expected three-amount structure.
#     # If OCR loses an amount—or the 200-character region contains an
#     # extra amount—the function returns {}
#     # It does not partially process that row
#     if len(amounts) != 3:
#         return {}

#     # Find the preceding `SUMMARY` section
#     # Search only the text before the selected Total for `SUMMARY`.
#     # This prepares a second search area for percentage targets such as "10%"
#     summary_anchors = list(re.finditer(
#         r"\bSUMMARY\b",
#         text[:anchor.start()],
#         flags=re.IGNORECASE,
#     ))

#     # Start with no percentage candidates.
#     percentages = []

#     # Only search percentages if a preceding summary anchor was found.
#     if summary_anchors:
#         # This matches values such as:
#         #   * 10%
#         #   * 10 %
#         #   * 7.5%
#         #   * 7,5%
#         # The decimal portion and whitespace before % are optional.
#         percentage_pattern = re.compile(
#             r"(?<![\w.,])\d+(?:[,.]\d+)?\s*%"
#         )

#         # Search between the last preceding `SUMMARY` and the selected `Total`.
#         percentages = list(percentage_pattern.finditer(
#             text,
#             summary_anchors[-1].end(),
#             anchor.start(),
#         ))

#     result = {}

#     # Process the two expected fields
#     # Run the following logic once for tax, then once for total.
#     for field in ("tax", "total"):
#         # Read the expected field safely.
#         # For the example:
#         #   First iteration
#         #       field = "tax"
#         #       expected = "273.18"
#         #   Second iteration
#         #       field = "total"
#         #       expected = "3004.96"
#         # If subtotal or the field is absent, use "".
#         expected = str(
#             json_data.get("subtotal", {}).get(field, "")
#         ).strip()

#         if not expected:
#             continue

#         # Determine whether this particular target is a percentage
#         is_percentage = expected.endswith("%")

#         # Convert the expected value into a Decimal:
#         # Examples:
#         #   "273.18" => Decimal("273.18")
#         #   "3004.96" => Decimal("3004.96")
#         #   "10%" => Decimal("10")  # when percentage=True
#         expected_number = normalize_number(
#             expected,
#             percentage=is_percentage,
#         )

#         # Skip a value that the number parser cannot interpret safely.
#         if expected_number is None:
#             continue

#         # Choose monetary or percentage candidates
#         #   For "273.18", use the three monetary amounts.
#         #   For "10%", use percentage matches from the preceding summary section.
#         available = percentages if is_percentage else amounts

#         # candidate.group() returns the actual matched OCR substring.
#         # For tax = "273.18", the comparisons are:
#         #   Decimal("2731.78") == Decimal("273.18")  # False
#         #   Decimal("273.18")  == Decimal("273.18")  # True
#         #   Decimal("3004.96") == Decimal("273.18")  # False
#         # Therefore, only the match for 273,18 remains.
#         # For total = "3004.96", only 3 004,96 remains.
#         candidates = [
#             candidate
#             for candidate in available
#             if normalize_number(
#                 candidate.group(),
#                 percentage=is_percentage,
#             ) == expected_number
#         ]

#         # Require one unambiguous match
#         # Skip the field when:
#         # - There are zero numerically equal candidates.
#         # - The same expected number occurs in multiple candidate positions.
#         if len(candidates) != 1:
#             continue

#         # The function doesn’t return a review record itself
#         # the missing annotation is detected by your verification code.
#         # For tax:
#         #   path = "subtotal.tax"
#         #   observed = "273,18"
#         candidate = candidates[0]
#         path = f"subtotal.{field}"
#         observed = candidate.group()

#         # Build the annotation
#         #   * start: original-text starting offset.
#         #   * end: exclusive ending offset.
#         #   * label: field name.
#         #   * text: original OCR substring.
#         #   * score: diagnostic text similarity.
#         #   * match_status: records how the annotation was accepted.
#         #   * needs_review=False: it passed this function’s uniqueness and numeric checks.
#         result[path] = {
#             "start": candidate.start(),
#             "end": candidate.end(),
#             "label": path,
#             "text": observed,
#             "score": fuzz.ratio(
#                 self.normalizer.normalize(expected),
#                 normalize(observed),
#             ),
#             "match_status": "summary_numeric_equal",
#             "needs_review": False,
#         }

#     return result


# def find_spans(text, json_data, prefix=""):
#     entities = []
#     stack = [(json_data, "")]
#     search_from = {"quantity": 0, "total_price": 0}
#     norm_text, offsets = normalize_with_offsets(text)
#     summary_spans = find_summary_spans(text, json_data)

#     while stack:
#         current, path = stack.pop()

#         if isinstance(current, dict):
#             for key, value in current.items():
#                 child_path = f"{path}.{key}" if path else key
#                 stack.append((value, child_path))
#             continue

#         if isinstance(current, list):
#             for idx in range(len(current) - 1, -1, -1):
#                 stack.append((current[idx], f"{path}[{idx}]"))
#             continue

#         if not isinstance(current, str) or not current.strip():
#             continue

#         if "items" in path:
#             continue
#         if path in {"subtotal.tax", "subtotal.total"}:
#             entity = summary_spans.get(path)
#             if entity is not None:
#                 entities.append(entity)
#             continue
#         if path.endswith("_name"):
#             entity = find_name_span(text, current, path)
#             if entity is not None:
#                 entities.append(entity)
#             continue

#         if "account_number" in path:
#             norm_value, _ = normalize_with_offsets(current)
#             match = fuzz.partial_ratio_alignment(norm_value,norm_text)
#             if not match:
#                 continue
#             start = offsets[match.dest_start]
#             end = offsets[match.dest_end-1] + 1
#             score = fuzz.ratio(normalize(current, path), normalize(text[start:end], path))
#             entities.append({
#                 "start": start,
#                 "end": end,
#                 "label": path,
#                 "text": text[start:end],
#                 "needs_review": score < ACCOUNT_NUMBER_MIN_SCORE,
#                 "score": score,
#             })
#             continue

#         if is_numeric_field(path):
#             pattern = number_pattern(current)
#             cursor_key = path.rsplit(".", 1)[-1].split("[", 1)[0]
#             start_at = search_from.get(cursor_key, 0)
#             match = pattern.search(text, start_at)
#             if not match:
#                 continue
#             start, end = match.span()
#             entities.append({
#                 "start": start,
#                 "end": end,
#                 "label": path,
#                 "text": text[start:end],
#             })
#             search_from[cursor_key] = end

#             continue

#         if path.endswith("_address"):
#             norm_text, offsets = normalize_with_offsets(text)
#             for line in current.splitlines():
#                 line = line.strip()
#                 if not line:
#                     continue

#                 entity = find_normalized_span(text, line, path)
#                 if entity is not None:
#                     entity["needs_review"] = False
#                     entity["match_status"] = "exact"
#                     entities.append(entity)
#                     continue
#                 norm_line, _ = normalize_with_offsets(line)
#                 match = fuzz.partial_ratio_alignment(norm_line,norm_text)
#                 if not match or match.dest_end <= match.dest_start:
#                     continue

#                 start = offsets[match.dest_start]
#                 end = offsets[match.dest_end-1] + 1
#                 house_number = re.match(r"\d+\b", line.strip())
#                 if house_number:
#                     number = house_number.group()

#                     # Search within the fuzzy candidate, preserving original offsets.
#                     candidate_text = text[start:end]
#                     anchor = re.search(rf"(?<!\w){re.escape(number)}(?!\w)", candidate_text)

#                     if anchor:
#                         start += anchor.start()

#                 matched_text = text[start:end]
#                 entities.append({
#                     "start": start,
#                     "end": end,
#                     "label": path,
#                     "text": matched_text,
#                 })

#             continue

#         norm_current, _ = normalize_with_offsets(current)
#         match = fuzz.partial_ratio_alignment(norm_current,norm_text)
#         if not match:
#             continue
#         start = offsets[match.dest_start]
#         end = offsets[match.dest_end-1] + 1
#         entities.append({
#             "start": start,
#             "end": end,
#             "label": path,
#             "text": text[start:end],
#             "needs_review": (
#                 path.endswith("_name") or path.endswith("_number")
#             ) and self.normalizer.normalize(current, path) != self.normalizer.normalize(text[start:end], path),
#             "score": match.score,
#         })

#     entities.sort(key=lambda entity: entity["start"])

#     return {
#         "text": text,
#         "entities": entities,
#     }

# def normalize_identifier(value: str) -> str:
#     """Ignore identifier spacing and separators while preserving letters/digits."""
#     value = unicodedata.normalize("NFKC", str(value)).casefold()
#     return re.sub(r"[\s./\-]", "", value)


# def normalize(value: str, path: str = "") -> str:
#     """Normalize text, or an identifier when its field path is supplied."""
#     if path.rsplit(".", 1)[-1] in {"invoice_number", "account_number"}:
#         return normalize_identifier(value)
#     normalized, _ = normalize_with_offsets(str(value))
#     return normalized

# def flatten_field_paths(
#     data: dict,
#     prefix: str = "",
# ) -> list[dict]:
#     """Return dotted paths for non-empty string values in a nested JSON object."""
#     paths = []

#     for key, value in data.items():
#         path = f"{prefix}.{key}" if prefix else key

#         if isinstance(value, dict):
#             paths.extend(flatten_field_paths(value, path))
#         elif isinstance(value, list):
#             for idx, item in enumerate(value):
#                 if isinstance(item, dict):
#                     paths.extend(flatten_field_paths(item, f"{path}[{idx}]"))
#         elif isinstance(value, str) and value.strip():
#             paths.append({path: value})

#     return paths

# def iter_leaf_fields(data, path=""):
#     """Yield (field_path, value), keeping list items under the same path."""
#     if isinstance(data, dict):
#         for key, value in data.items():
#             child_path = f"{path}.{key}" if path else key
#             yield from iter_leaf_fields(value, child_path)

#     elif isinstance(data, list):
#         for item in data:
#             yield from iter_leaf_fields(item, path)

#     elif data is not None and str(data).strip():
#         yield path, data

# def normalize_number(value: str, *, percentage: bool = False) -> Decimal | None:
#     """Parse invoice amounts with up to two decimal places and valid grouping.

#     Accept comma/dot decimals, grouped thousands, and surrounding currency
#     symbols. Reject ambiguous single separators (1,000), malformed grouping,
#     arbitrary letters, and multiple amounts instead of silently merging them.
#     With percentage=True, require a trailing percent sign and return percentage
#     points (10% becomes Decimal('10'), not Decimal('0.1')).
#     """
#     value = unicodedata.normalize("NFKC", str(value)).strip()
#     if percentage:
#         if not value.endswith("%"):
#             return None
#         value = value[:-1].strip()
#     value = value.strip("$€£¥ ").strip()
#     value = value.replace("’", "'")
#     sign = ""
#     if value.startswith(("+", "-")):
#         sign, value = value[0], value[1:]

#     if not value or not re.fullmatch(r"[0-9.,'\s]+", value):
#         return None

#     integer_part = value
#     fractional_part = ""
#     separators = {char for char in value if char in ".,"}
#     if separators:
#         decimal_separator = value[max(value.rfind("."), value.rfind(","))]
#         before, after = value.rsplit(decimal_separator, 1)
#         if re.fullmatch(r"[0-9]{1,2}", after):
#             integer_part, fractional_part = before, after
#             if decimal_separator in integer_part:
#                 return None
#         elif len(separators) == 1 and value.count(decimal_separator) >= 2:
#             # Repeated three-digit groups have an unambiguous integer reading.
#             pass
#         else:
#             return None

#     if not re.fullmatch(r"[0-9]+", integer_part):
#         if not re.fullmatch(r"[0-9]{1,3}([.,'\s])[0-9]{3}(?:\1[0-9]{3})*", integer_part):
#             return None
#     integer_digits = re.sub(r"[^0-9]", "", integer_part)
#     canonical = sign + integer_digits
#     if fractional_part:
#         canonical += "." + fractional_part

#     try:
#         return Decimal(canonical)
#     except InvalidOperation:
#         return None

# def find_fuzzy_identifier(text: str, expected: str, min_score: float = 90):
#     expected = expected.strip()
#     if not expected:
#         return None

#     # Try same-length OCR windows: identifiers should not gain or lose characters.
#     width = len(expected)
#     candidates = []

#     for start in range(len(text) - width + 1):
#         candidate = text[start : start + width]
#         score = fuzz.ratio(expected, candidate)

#         if score >= min_score:
#             candidates.append((score, start, start + width, candidate))

#     # Return only an unambiguous best candidate.
#     candidates.sort(reverse=True)
#     if not candidates:
#         return None

#     best = candidates[0]
#     if len(candidates) > 1 and candidates[1][0] == best[0]:
#         return None

#     score, start, end, candidate = best
#     return {
#         "start": start,
#         "end": end,
#         "text": candidate,       # OCR text, unchanged
#         "score": score,
#         "expected": expected,    # keep expected JSON separately
#     }


# def find_unmatched_fields(row):
#     """Log and return missing, mismatched, and review-required base fields."""
#     ocr_text = row["ocred_text"]
#     json_data = row["json_data"]

#     if isinstance(json_data, str):
#         json_data = json.loads(json_data)

#     result_dict, expected_dict = {}, {}
#     result = find_spans(ocr_text, json_data)
#     json_fields = flatten_field_paths(json_data)

#     for item in result['entities']:
#         if item['label'] in result_dict:
#             result_dict[item['label']] += f"\n{item['text']}"
#         else:
#             result_dict[item['label']] = item['text']

#     for item in json_fields:
#         for key, value in item.items():
#             if key in expected_dict:
#                 expected_dict[key] += f"\n{value}"
#             else:
#                 expected_dict[key] = value

#     issues = []

#     for key, value in expected_dict.items():
#         orig_value = result_dict.get(key)

#         if "items" in key:
#             continue

#         if not orig_value:
#             issues.append({"field": key, "status": "missing", "expected": value,
#                            "observed": None})
#             continue

#         if is_numeric_field(key):
#             is_percentage = unicodedata.normalize("NFKC", value).strip().endswith("%")
#             expected_number = normalize_number(value, percentage=is_percentage)
#             observed_number = normalize_number(orig_value, percentage=is_percentage)

#             score = fuzz.ratio(str(expected_number), str(observed_number))
#             is_match = (
#                 expected_number is not None
#                 and observed_number is not None
#                 and expected_number == observed_number
#             )

#             if not is_match: #score < 80:
#                 issues.append({"field": key, "status": "numeric_mismatch", "expected": value,
#                                "observed": orig_value})
#             continue

#         if key.endswith(("_name", "_number")):
#             score = fuzz.ratio(normalize(value, key), normalize(orig_value, key))
#             field_spans = [entity for entity in result["entities"] if entity["label"] == key]
#             if key.endswith(".account_number"):
#                 requires_review = score < ACCOUNT_NUMBER_MIN_SCORE
#             elif key.endswith("_name"):
#                 requires_review = any(entity.get("needs_review", True) for entity in field_spans)
#             else:
#                 requires_review = normalize(value, key) != normalize(orig_value, key)
#             if requires_review:
#                 issues.append({"field": key, "status": "needs_review", "expected": value,
#                                "observed": orig_value, "score": score,
#                                "spans": field_spans})
#             continue

#         # A subset of address words must not count as a complete match.
#         ratio = fuzz.ratio(normalize(value, key), normalize(orig_value, key))

#         if ratio < 80:
#             issues.append({"field": key, "status": "text_mismatch", "expected": value,
#                            "observed": orig_value, "score": ratio})
#             continue

#     return issues


# def find_ignored_account_regions(text, json_data):
#     account = (
#         json_data.get("payment_instructions", {})
#         .get("account_number")
#     )

#     if account is not None and str(account).strip():
#         return []

#     # Assumes this dataset's IBAN value is one whitespace-delimited token.
#     pattern = re.compile(
#         r"\bIBAN\s*:\s*(?P<value>[A-Za-z0-9][A-Za-z0-9/.-]*)",
#         flags=re.IGNORECASE,
#     )

#     return [
#         {
#             "start": match.start("value"),
#             "end": match.end("value"),
#         }
#         for match in pattern.finditer(text)
#     ]

# def prepare_records(frame):
#     FIELD_PATHS = [
#         "invoice.client_name",
#         "invoice.seller_name",
#         "invoice.client_address",
#         "invoice.seller_address",
#         "invoice.invoice_number",
#         "invoice.invoice_date",
#         "payment_instructions.account_number",
#         "subtotal.tax",
#         "subtotal.total",
#     ]

#     LABELS = ["O"]

#     for field in FIELD_PATHS:
#         LABELS.extend([f"B-{field}", f"I-{field}"])

#     label2id = {label: index for index, label in enumerate(LABELS)}
#     id2label = dict(enumerate(LABELS))

#     records = []
#     rejected = []

#     for row_index, row in frame.iterrows():
#         target = row["json_data"]

#         if isinstance(target, str):
#             target = json.loads(target)

#         text = row["ocred_text"]

#         # Keep parser diagnostics without flooding notebook output.
#         with contextlib.redirect_stdout(io.StringIO()) as log:
#             issues = find_unmatched_fields(row)

#         if not isinstance(issues, list):
#             raise TypeError(
#                 "find_unmatched_fields() must return a list."
#             )

#         issues = [
#             issue
#             for issue in issues
#             if issue["field"] in FIELD_PATHS
#         ]

#         result = find_spans(text, target)

#         entities = [
#             entity
#             for entity in result["entities"]
#             if entity["label"] in FIELD_PATHS
#         ]

#         entities.sort(
#             key=lambda entity: (entity["start"], entity["end"])
#         )

#         for entity in entities:
#             start, end = entity["start"], entity["end"]

#             if not (
#                 0 <= start < end <= len(text)
#                 and text[start:end] == entity["text"]
#             ):
#                 raise ValueError(
#                     f"Invalid span in invoice {row_index}: {entity}"
#                 )

#             if entity.get("needs_review", False):
#                 issues.append({
#                     "field": entity["label"],
#                     "status": "needs_review",
#                 })

#         for left, right in zip(entities, entities[1:]):
#             if right["start"] < left["end"]:
#                 issues.append({
#                     "field": right["label"],
#                     "status": "overlap",
#                 })

#         # The parser's similarity threshold can accept partial addresses.
#         # Require complete normalized address coverage for this baseline.
#         for field in (
#             "invoice.client_address",
#             "invoice.seller_address",
#         ):
#             key = field.split(".")[1]
#             expected = target.get("invoice", {}).get(key, "")

#             observed = " ".join(
#                 entity["text"]
#                 for entity in entities
#                 if entity["label"] == field
#             )

#             if normalize(expected) != normalize(observed):
#                 issues.append({
#                     "field": field,
#                     "status": "partial_address",
#                 })

#         if issues:
#             rejected.append({
#                 "row_index": row_index,
#                 "issues": issues,
#                 "log": log.getvalue(),
#             })
#         else:
#             records.append({
#                 "row_index": row_index,
#                 "text": text,
#                 "entities": entities,
#                 "ignored_regions": find_ignored_account_regions(text, target)
#             })

#     return records, rejected
