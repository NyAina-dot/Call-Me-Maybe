"""Run constrained function calling from the command line."""

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from .utils import Tokenizer
from .parsing import parse_args, load_file
from .models import FuncDefModel, InputModel, ResultModel
from pydantic import ValidationError
import json
from .constraint import FunctionSelector, ParameterGenerator
from pathlib import Path


if __name__ == "__main__":
    try:
        model = Small_LLM_Model()
        arg = parse_args()

        funcdef_content = load_file(
            arg.functions_definition, FuncDefModel
        )
        input_content = load_file(arg.input, InputModel)

        tokenizer = Tokenizer(model)
        selector = FunctionSelector(model, tokenizer)
        generator = ParameterGenerator(model, tokenizer)

        results = []

        for test in input_content:
            prompt = test.prompt

            function_name = selector.choose_function_name(
                prompt, funcdef_content
            )

            function = next(
                (
                    func for func in funcdef_content
                    if func.name == function_name
                ),
                None
            )

            if function is None:
                raise ValueError(
                    f"Unknown function: {function_name}"
                )

            parameters = generator.generate_parameters(
                prompt, function
            )

            result = ResultModel(
                prompt=prompt,
                name=function_name,
                parameters=parameters
            )

            results.append(result)

        output_path = Path(arg.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(
                [result.model_dump() for result in results],
                file,
                indent=4,
                ensure_ascii=False
            )

        print(f"Successfully processed {len(results)} prompts.")

    except (
        ValidationError,
        KeyboardInterrupt,
        OSError,
        json.JSONDecodeError,
        UnicodeDecodeError,
        ValueError
    ) as e:
        print(f"Error: {e}")
