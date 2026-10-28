import argparse
import json
from pydantic import BaseModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
            description=(
                "Translate natural language prompts into "
                "structured function calls "
                "using constrained decoding"
            )
        )
    parser.add_argument(
        '--functions_definition',
        default="data/input/functions_definition.json",
        help=(
            "Path to the JSON file describing the available "
            "functions (default: "
            "data/input/functions_definition.json)."
        )
    )
    parser.add_argument(
        '--input',
        default="data/input/function_calling_tests.json",
        help=(
            "Path to the JSON file containing the prompts to "
            "process (default: "
            "data/input/function_calling_tests.json)."
        )
    )
    parser.add_argument(
        '--output',
        default="data/output/function_calling_results.json",
        help=(
            "Path where the results JSON will be written "
            "(default: data/output/function_calling_results.json)."
        )
    )
    return parser.parse_args()


def load_file(path: str, model: type[BaseModel]) -> list[BaseModel]:
    """Load a JSON file containing a list and validate each element.

    Args:
        path: Path to the JSON file.
        model: Pydantic model class used to validate each element.

    Returns:
        The list of validated models.
    """
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return [model.model_validate(item) for item in data]
