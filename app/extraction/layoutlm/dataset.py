import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers.image_processing_utils import BaseImageProcessor
from transformers.tokenization_utils_tokenizers import TokenizersBackend

from app.extraction.enums import IGNORE_INDEX
from app.extraction.layoutlm.converter import AttachmentToImageConverter


class InvoiceDataset(Dataset):
    """Cache complete-word windows and retain page coordinates for evaluation."""

    def __init__(
        self,
        pages: list[dict],
        tokenizer: TokenizersBackend,
        image_processor: BaseImageProcessor,
        converter: AttachmentToImageConverter,
        ignore_index: int = IGNORE_INDEX,
        max_length: int = 512,
    ):
        if not 4 <= max_length <= 512:
            raise ValueError("max_length must be between 4 and 512")

        self.pages = pages
        self.samples = []
        self.max_length = max_length

        capacity = max_length - tokenizer.num_special_tokens_to_add(pair=False)

        for page_index, page in enumerate(tqdm(pages, desc="Tokenize pages")):
            if not (len(page["words"]) == len(page["boxes"]) == len(page["labels"])):
                raise ValueError("Each word must have one box and one label")
            if not page["words"]:
                raise ValueError("Page has no OCR words")
            encoded = tokenizer(
                page["words"],
                boxes=page["boxes"],
                add_special_tokens=False,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            image = converter.load_page_image(page).convert("RGB")

            pixel_values = image_processor(
                images=image,
                return_tensors="pt",
            )["pixel_values"][0]

            pieces = [[] for _ in page["words"]]

            for token_id, word_id in zip(
                encoded["input_ids"],
                encoded.word_ids(),
            ):
                pieces[word_id].append(token_id)

            if any(not tokens or len(tokens) > capacity for tokens in pieces):
                raise ValueError(
                    f"Empty or overlong word: "
                    f"{page['document_id']}, page={page['page']}"
                )

            start = 0

            while start < len(pieces):
                end = start
                token_count = 0

                while end < len(pieces) and token_count + len(pieces[end]) <= capacity:
                    token_count += len(pieces[end])
                    end += 1

                input_ids = [tokenizer.cls_token_id]
                boxes = [[0, 0, 0, 0]]
                labels = [ignore_index]
                word_ids = [-1]

                for word_id in range(start, end):
                    tokens = pieces[word_id]
                    continuation_count = len(tokens) - 1

                    input_ids.extend(tokens)
                    boxes.extend([page["boxes"][word_id]] * len(tokens))
                    labels.extend(
                        [page["labels"][word_id]] + [ignore_index] * continuation_count
                    )
                    word_ids.extend([word_id] + [-1] * continuation_count)

                input_ids.append(tokenizer.sep_token_id)
                boxes.append([1000, 1000, 1000, 1000])
                labels.append(ignore_index)
                word_ids.append(-1)

                attention_mask = [1] * len(input_ids)
                padding = max_length - len(input_ids)

                sample = {
                    "input_ids": (input_ids + [tokenizer.pad_token_id] * padding),
                    "attention_mask": attention_mask + [0] * padding,
                    "bbox": boxes + [[0, 0, 0, 0]] * padding,
                    "labels": labels + [ignore_index] * padding,
                    "word_ids": word_ids + [-1] * padding,
                    "page_index": page_index,
                }
                sample = {
                    key: torch.tensor(value, dtype=torch.long)
                    for key, value in sample.items()
                }
                sample["pixel_values"] = pixel_values

                self.samples.append(sample)

                start = end

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
