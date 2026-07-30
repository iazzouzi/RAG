.DEFAULT_GOAL := run

install:
	@uv sync

run:
	@uv run python -m src index --max_chunk_size 1300 || true

debug:
	@uv run python -m pdb src || true

clean:
	@find . -type d -name "__pycache__" -exec rm -r {} +
	@find . -type d -name ".mypy_cache" -exec rm -r {} +

lint:
	@uv run python -m flake8 src
	@uv run python -m mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

.PHONY: install run debug clean lint