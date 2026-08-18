VENV = .venv
VENV_PYTHON	= $(VENV)/bin/python

MYPY_FLAGS = --warn-return-any \
			 --warn-unused-ignores \
			 --ignore-missing-imports \
			 --disallow-untyped-defs \
			 --check-untyped-defs

.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint:
	$(VENV_PYTHON) -m flake8 --exclude=".venv,llm_sdk" .
	$(VENV_PYTHON) -m mypy --exclude="(^|/)llm_sdk(/|$$)" --exclude="$(VENV)" .

lint-strict:
	$(VENV_PYTHON) -m flake8 --exclude=".venv,llm_sdk" .
	$(VENV_PYTHON) -m mypy --exclude="(^|/)llm_sdk(/|$$)" --exclude="$(VENV)" --strict .