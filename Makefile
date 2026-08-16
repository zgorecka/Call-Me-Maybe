PYTHON ?= python3
UV ?= uv
APP_MODULE = src

.PHONY: install run debug clean lint lint-strict

install:
	$(UV) sync --group dev

run:
	$(UV) run python -m $(APP_MODULE)

debug:
	$(UV) run python -m pdb src

clean:
	find . -type d \( -name "__pycache__" -o -name ".mypy_cache" -o -name ".pytest_cache" \) -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".ruff_cache" -prune -exec rm -rf {} +

lint:
	$(UV) run flake8 .
	$(UV) run mypy . --exclude=llm_sdk/ --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	$(UV) run flake8 .
	$(UV) run mypy . --strict
