import re
from app.extraction.bert.fields.field import Field
from app.extraction.bert.spans import CandidateSpan

class MoneyField(Field):
    def digits(self) -> str:
        return "".join(
            re.findall(r"\d", self)
        )

    def find_matches(
        self,
        text: str,
    ) -> list[CandidateSpan]:
        digits = self.digits()

        if not digits:
            return []

        pattern = (
            r"(?<!\d)"
            + r"[\s,.'’]*".join(
                re.escape(digit)
                for digit in digits
            )
            + r"(?!\d)"
        )

        return [
            CandidateSpan(
                start=match.start(),
                end=match.end(),
            )
            for match in re.finditer(
                pattern,
                text,
            )
        ]
