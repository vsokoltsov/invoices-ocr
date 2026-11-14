# Train DocILE line-item fields

```bash
uv run python -m commands.cli train bert-pytorch \
  --dataset-name docile \
  --dataset-path docile_df.parquet \
  --line-items-only \
  --num-epoch 20 \
  --checkpoint-dir data/interim/docile_items_checkpoint \
  --result-filename data/interim/docile_items_training \
  --diagnostic-dir data/interim/docile_items_diagnostics
```

The flag selects four entity types: ITEM_DESCRIPTION, ITEM_QUANTITY,
ITEM_UNIT_PRICE_GROSS, and ITEM_AMOUNT_GROSS. The complete OCR text remains the
input. Invoice fields and table headers are not positive item labels. Net prices
and other item fields are outside this experiment's target schema.

The builder selects OCR words by annotation boxes and pages, preserves global
text offsets, and retains `line_item_id` in spans and annotation sources. Different
items cannot merge during span preparation, even when they share a field type.
Words assigned to competing item IDs are reported as overlaps. Missing spans,
unresolved overlaps, and invalid offsets reject a document. Documents with no
selected item annotations are also rejected. The preparation log reports accepted
and rejected document counts.

This experiment learns field types, not item grouping. Ground-truth item IDs are
metadata and are not model inputs or classification targets. Entity F1 therefore
measures field extraction, not complete-item reconstruction.

Without a target flag, DocILE training runs two separate models: body and items.
Use `--body-only` for invoice-level fields or `--line-items-only` for item fields.
The flags are mutually exclusive. Environment defaults are
`BERTPYTORCH_LINE_ITEMS_ONLY=false` and `BERTPYTORCH_BODY_ONLY=false`.
Kaggle supports only body training because it lacks the item-box alignment used
by this item builder.

The default two-model run saves histories with `_body.csv` and `_items.csv`
suffixes. Checkpoints and diagnostics use `body/` and `items/` subdirectories.
Single-model modes use the configured paths directly. Both models use the same
document partition, fixed before task-specific rejection. Accepted subsets may
differ, but documents cannot cross training and evaluation partitions between
models. A split manifest is saved with each DocILE checkpoint.

OCR-word decoding is the default for DocILE validation, best-checkpoint
selection, final test evaluation, and diagnostics. Loss is still computed from
raw logits. Kaggle keeps token argmax because it has no OCR word metadata.

The saved checkpoint contains the item schema and original validation inputs.
Run `debug bert-pytorch` with this checkpoint to compare decoders for item fields.
Keep item and invoice experiments in separate output and checkpoint directories.
