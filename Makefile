install-kernel:
	uv run python -m ipykernel install --user --name=ocr --display-name="Optic Character Recognition for invoices"

lint:
	uv run ruff check .

lint-fix:
	uv run ruff check --fix .

typecheck:
	uv run ty check .

check: lint typecheck
