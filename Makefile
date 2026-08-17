PYTHON		= python3
VENV		= .venv
VENV_PYTHON	= $(VENV)/bin/python
VENV_PIP	= $(VENV)/bin/pip
# MAIN		=

MYPY_FLAGS	= --warn-return-any \
			  --warn-unused-ignores \
			  --ignore-missing-imports \
			  --disallow-untyped-defs \
			  --check-untyped-defs

# SRC			= 

.PHONY: install run debug clean lint lint-strict

install:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements.txt

run:
# 	$(VENV_PYTHON) $(MAIN) $(CONFIG)

debug:
# 	$(VENV_PYTHON) -m pdb $(MAIN) $(CONFIG)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint:
	$(VENV_PYTHON) -m flake8 --exclude $(VENV) .
	$(VENV_PYTHON) -m mypy --exclude $(VENV) .

lint-strict:
	$(VENV_PYTHON) -m flake8 --exclude $(VENV) .
	$(VENV_PYTHON) -m mypy --exclude $(VENV) --strict .
