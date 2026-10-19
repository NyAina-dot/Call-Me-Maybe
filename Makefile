VENV = $(HOME)/goinfre/CallMeMaybe/.venv
VENV_PYTHON	= $(VENV)/bin/python

HF_HOME= $(HOME)/goinfre/CallMeMaybe/.llm
UV_CACHE_DIR= $(HOME)/goinfre/CallMeMaybe/.uv_cache
UV_PROJECT_ENVIRONMENT=$(VENV)

export HF_HOME
export UV_CACHE_DIR
export UV_PROJECT_ENVIRONMENT

MYPY_FLAGS = --warn-return-any \
			 --warn-unused-ignores \
			 --ignore-missing-imports \
			 --disallow-untyped-defs \
			 --check-untyped-defs

.PHONY: install run debug clean lint lint-strict

install:
	@mkdir -p $(HF_HOME) $(UV_CACHE_DIR) $(UV_PROJECT_ENVIRONMENT)
	uv sync

run:
	uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calls.json

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