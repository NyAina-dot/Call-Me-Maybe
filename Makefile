.PHONY: install run debug clean lint lint-strict

install:
	uv sync --python 3.13

run:
	uv run python -m src $(ARGS) \
		$(if $(FUNCTIONS_DEFINITION),--functions_definition "$(FUNCTIONS_DEFINITION)") \
		$(if $(INPUT),--input "$(INPUT)") \
		$(if $(OUTPUT),--output "$(OUTPUT)")

debug:
	uv run python -m pdb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint:
	uv run python -m flake8 --exclude=".venv,llm_sdk" .
	uv run python -m mypy --exclude="(^|/)llm_sdk(/|$$)" .

lint-strict:
	uv run python -m flake8 --exclude=".venv,llm_sdk" .
	uv run python -m mypy --exclude="(^|/)llm_sdk(/|$$)" --strict .