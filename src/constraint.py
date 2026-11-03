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
        prompt: str, function: FuncDefModel
    ) -> dict[str, float | int | str | bool]:
        pass
