import pytest
import torch

from app.extraction.bert.pytorch.decoding import OCRWordDecoder


@pytest.mark.parametrize(
    "labels,offsets,words,expected",
    [
        ([1, 0, 2], [(0, 2), (2, 3), (3, 6)], [(0, 6)], [1, 2, 2]),
        ([1, 1], [(0, 2), (2, 6)], [(0, 6)], [1, 2]),
        # Adjacent entities with an explicit B at each word must remain separate.
        ([1, 1], [(0, 2), (3, 6)], [(0, 2), (3, 6)], [1, 1]),
        ([1, 2], [(0, 2), (3, 6)], [(0, 2), (3, 6)], [1, 2]),
        ([0, 0, 1], [(0, 2), (2, 3), (3, 6)], [(0, 6)], [0, 0, 0]),
        # A word split across windows keeps its token predictions.
        ([1, 0], [(2, 3), (3, 6)], [(0, 6)], [1, 0]),
        # A token crossing two OCR words must not be pooled into either word.
        ([1], [(0, 6)], [(0, 2), (3, 6)], [1]),
        ([0, 1, 0], [(0, 0), (0, 6), (0, 0)], [(0, 6)], [0, 1, 0]),
    ],
)
def test_word_decoder_repairs_fragments_without_merging_entities(
    labels, offsets, words, expected
):
    decoder = OCRWordDecoder({"O": 0, "B-TOTAL": 1, "I-TOTAL": 2})
    logits = torch.full((len(labels), 3), -8.0)
    for index, label in enumerate(labels):
        logits[index, label] = 8
    original = logits.clone()
    assert (
        decoder.decode(
            logits, offsets, [{"start": start, "end": end} for start, end in words]
        )
        == expected
    )
    torch.testing.assert_close(logits, original)


def test_word_decoder_uses_combined_type_probability():
    decoder = OCRWordDecoder({"O": 0, "B-TOTAL": 1, "I-TOTAL": 2})
    # O wins individual-label argmax; TOTAL wins after B/I aggregation.
    logits = torch.tensor([[0.4, 0.3, 0.3]]).log()
    assert decoder.decode(logits, [(0, 2)], [{"start": 0, "end": 2}]) == [1]
    with pytest.raises(ValueError, match="Logit shape"):
        decoder.decode(logits, [], [])


def test_word_decoder_requires_complete_bio_schema():
    with pytest.raises(ValueError, match="paired B/I"):
        OCRWordDecoder({"O": 0, "B-TOTAL": 1})
