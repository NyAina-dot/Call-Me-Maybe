VENV = $(abspath ..)/Call_Me_Maybe/.venv
VENV_PYTHON = $(VENV)/bin/python

HF_HOME = $(abspath ..)/Call_Me_Maybe/.llm
UV_CACHE_DIR = $(abspath ..)/Call_Me_Maybe/.uv_cache
UV_PROJECT_ENVIRONMENT = $(VENV)

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
	@mkdir -p $(HF_HOME) $(UV_CACHE_DIR)
	PATH=$$(echo $$PATH | tr ':' '\n' | grep -v "\.pyenv" | tr '\n' ':' | sed 's/:$$//') \
	uv sync --python 3.13 --python-preference managed

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