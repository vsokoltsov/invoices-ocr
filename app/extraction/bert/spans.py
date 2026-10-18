from dataclasses import dataclass
from app.extraction.bert.enums import EntityType

@dataclass(frozen=True, slots=True)
class CandidateSpan:
    start: int
    end: int

    def overlaps(
        self,
        other: "CandidateSpan",
    ) -> bool:
        return (
            self.start < other.end
            and self.end > other.start
        )

@dataclass(frozen=True, slots=True)
class EntitySpan:
    """
    Represents entity span for each particular entity
    """

    start: int
    end: int
    entity: str
    value: str

    def __post_init__(self):
        if self.start < 0:
            raise ValueError(
                "Span start cannot be negative"
            )

        if self.end <= self.start:
            raise ValueError(
                f"Invalid span: [{self.start}, {self.end})"
            )

        if not self.entity:
            raise ValueError(
                "Entity type cannot be empty"
            )

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(
        self,
        other: "EntitySpan",
    ) -> bool:
        return (
            self.start < other.end 
            and self.end > other.start
        )

    def overlaps_range(
        self,
        start: int,
        end: int,
    ) -> bool:
        return (
            self.start < end
            and self.end > start
        )

@dataclass(frozen=True, slots=True)
class FinancialSummarySpans:
    net: CandidateSpan
    tax: CandidateSpan
    total: CandidateSpan

    def get(
        self,
        entity: EntityType,
    ) -> CandidateSpan | None:
        match entity:
            case EntityType.TAX:
                return self.tax

            case EntityType.TOTAL:
                return self.total

            case _:
                return None