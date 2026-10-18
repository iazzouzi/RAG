.DEFAULT_GOAL := run

install:
	@uv add -r requirements.txt
	@[ -f data/vllm-0.10.1.zip ] && [ ! -d data/raw ] && unzip data/vllm-0.10.1.zip -d data/raw || true

run: install
	@uv run python -m src || true

debug:
	@uv run python -m pdb src || true

clean:
	@find . -type d -name "__pycache__" -exec rm -r {} +
	@find . -type d -name ".mypy_cache" -exec rm -r {} +

lint:
	@uv run python -m flake8 src
	@uv run python -m mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

.PHONY: install run debug clean lint