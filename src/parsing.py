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


def validate_file_content(path: str, model: BaseModel) -> None:
    with open(path, "r") as file:
        data = json.load(file)
    for i in range(len(data)):
        model.model_validate(data[i])
