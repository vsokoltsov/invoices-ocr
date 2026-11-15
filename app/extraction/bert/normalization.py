import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


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
