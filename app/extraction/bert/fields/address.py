from app.extraction.bert.fields.field import Field

class AddressField(Field):
    def alignment_parts(self) -> tuple["AddressField", ...]:
        return tuple(
            type(self)(line.strip())
            for line in self.splitlines()
            if line.strip()
        )