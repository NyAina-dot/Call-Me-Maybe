from .utils import Tokenizer
from .models import FuncDefModel
from llm_sdk import Small_LLM_Model
from typing import Any
import json


class FunctionSelector:
    def __init__(self, model: Small_LLM_Model, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def build_prompt(
        self, prompt: str, functions: list[FuncDefModel]
    ) -> None:
        lines = [
            "You are a function-calling assistant. Given a user request, "
            "select the single best function from the list below.",
            "",
            "Available functions:",
        ]
        for function in functions:
            lines.append(f"- {function.name}: {function.description}")
        lines += ["", f"Request: {prompt}", "Function: "]
        return "\n".join(lines)

    def choose_function_name(
            self, prompt: str, functions: list[FuncDefModel]
    ) -> str:
        prompt_ids = self.tokenizer.text_to_token_ids(
            self.build_prompt(prompt, functions)
        )
        tree = self.tokenizer.function_names_token_tree(functions)
        generated_ids: list[int] = []
        while not self.tokenizer.is_complete(tree, generated_ids):
            logits = self.model.get_logits_from_input_ids(
                prompt_ids + generated_ids
            )
            allowed = self.tokenizer.get_allowed_tokens(tree, generated_ids)
            next_id = max(allowed, key=lambda i: logits[i])
            generated_ids.append(next_id)
        return self.tokenizer.token_ids_to_text(generated_ids)


class ParameterGenerator:
    def __init__(
        self, model: Small_LLM_Model, tokenizer: Tokenizer
    ):
        self.model = model
        self.tokenizer = tokenizer

    def build_prompt(
        self, prompt: str, function: FuncDefModel
    ) -> str:
        parameters = {
            name: parameter.type
            for name, parameter in function.parameters.items()
        }
        return (
            "You are a function-calling assistant. Extract the parameters "
            "for the requested function from the user request.\n\n"
            f"Function: {function.name}\n"
            f"Description: {function.description}\n"
            f"Parameter types: {json.dumps(parameters)}\n\n"
            f"Request: {prompt}\n"
            "Arguments: "
        )

    def generate_parameters(
        self, prompt: str, function: FuncDefModel
    ) -> dict[str, float | int | str | bool]:
        tokens_ids = self.tokenizer.text_to_token_ids(
            self.build_prompt(prompt, function)
        )
        result: dict[str, float | int | str | bool] = {}
        parameter_names = list(function.parameters)

        for index, name in enumerate(parameter_names):
            parameter = function.parameters[name]
            prefix = json.dumps(name) + ": "
            tokens_ids.extend(self.tokenizer.text_to_token_ids(prefix))

            if parameter.type in ("number", "integer"):
                value = self.generate_number(
                    tokens_ids,
                    integer=(parameter.type == "integer")
                )
            elif parameter.type == "string":
                value = self.generate_string(tokens_ids)
            elif parameter.type == "boolean":
                value = self.generate_boolean(tokens_ids)
            else:
                raise ValueError(
                    f"Unsupported parameter type: {parameter.type}"
                )
            result[name] = value
            tokens_ids.extend(
                self.tokenizer.text_to_token_ids(json.dumps(value))
            )
            if index < len(parameter_names) - 1:
                tokens_ids.extend(
                    self.tokenizer.text_to_token_ids(", ")
                )
        return result

    def generate_number(
        self, context_ids: list[int], integer: bool
    ) -> float | int:
        char_numbers = "0123456789-"
        if not integer:
            char_numbers += "."
        allowed_tokens = []
        for token_id, token_text in enumerate(self.tokenizer.id_to_text):
            if token_text and all(char in char_numbers for char in token_text):
                allowed_tokens.append(token_id)
        generated_ids: list[int] = []
        for _ in range(20):
            logits = self.model.get_logits_from_input_ids(
                context_ids + generated_ids
            )
            masked_logits = self.tokenizer.mask_logits(logits, allowed_tokens)
            next_token_id = max(
                range(len(masked_logits)),
                key=lambda i: masked_logits[i]
            )
            token_text = self.tokenizer.id_to_text[next_token_id]
            generated_ids.append(next_token_id)
        return 1

    def generate_string(self, context_ids: list[int]) -> str:
        return "Bonjour"

    def generate_boolean(self, context_ids: list[int]) -> bool:
        return False
