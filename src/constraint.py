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
        negative_numbers = self.get_negative_number_flags(prompt)
        number_index = 0

        for index, name in enumerate(parameter_names):
            parameter = function.parameters[name]
            prefix = json.dumps(name) + ": "
            tokens_ids.extend(self.tokenizer.text_to_token_ids(prefix))

            if parameter.type in ("number", "integer"):
                value = self.generate_number(
                    tokens_ids,
                    integer=(parameter.type == "integer"),
                    require_negative=(
                        number_index < len(negative_numbers)
                        and negative_numbers[number_index]
                    ),
                )
                number_index += 1
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

    def get_negative_number_flags(self, text: str) -> list[bool]:
        """Return the signs of numeric literals in text, in appearance order."""
        flags: list[bool] = []
        index = 0

        while index < len(text):
            starts_number = text[index].isdigit() or (
                text[index] == "."
                and index + 1 < len(text)
                and text[index + 1].isdigit()
            )
            starts_negative_number = text[index] == "-" and (
                index + 1 < len(text)
                and (
                    text[index + 1].isdigit()
                    or (
                        text[index + 1] == "."
                        and index + 2 < len(text)
                        and text[index + 2].isdigit()
                    )
                )
            )

            if not (starts_number or starts_negative_number):
                index += 1
                continue

            number = "-" if starts_negative_number else ""
            if starts_negative_number:
                index += 1

            while index < len(text):
                candidate = number + text[index]
                if not self.is_number_prefix(candidate, False):
                    break
                number += text[index]
                index += 1

            flags.append(number.startswith("-"))

        return flags

    def is_number_prefix(
        self, text: str, integer: bool = False
    ) -> bool:
        """Return True if text can still become a valid number."""
        if text.startswith("-"):
            body = text[1:]
        else:
            body = text

        if integer:
            if body == "":
                return True
            if body.isascii() and body.isdigit():
                return True
            return False

        if body == "":
            return True

        left, dot, right = body.partition(".")

        if not (left.isascii() and left.isdigit()):
            return False

        if right != "":
            if not (right.isascii() and right.isdigit()):
                return False

        return True

    def get_allowed_number_tokens(
        self, generated_text: str, integer: bool,
        require_negative: bool = False,
    ) -> list[int]:
        allowed_tokens = []

        for token_id, token_text in enumerate(self.tokenizer.id_to_text):
            if not token_text:
                continue

            candidate = generated_text + token_text

            if (
                self.is_number_prefix(candidate, integer)
                and (not require_negative or generated_text or
                     candidate.startswith("-"))
            ):
                allowed_tokens.append(token_id)

        return allowed_tokens

    def is_complete_number(self, generated_text: str, integer: bool) -> bool:
        """Return whether generated_text is a complete JSON number."""
        if not generated_text:
            return False

        body = (
            generated_text[1:]
            if generated_text.startswith("-")
            else generated_text
        )
        if not body:
            return False

        if integer:
            return body.isascii() and body.isdigit()

        left, dot, right = body.partition(".")
        if not (left.isascii() and left.isdigit()):
            return False
        if not dot:
            return True
        return right.isascii() and right.isdigit()

    def generate_number(
        self, context_ids: list[int], integer: bool,
        require_negative: bool = False,
    ) -> float | int:
        generated_ids: list[int] = []
        generated_text = ""

        for _ in range(20):
            logits = self.model.get_logits_from_input_ids(
                context_ids + generated_ids
            )

            allowed_tokens = self.get_allowed_number_tokens(
                generated_text, integer, require_negative
            )
            number_is_complete = self.is_complete_number(
                generated_text, integer
            )
            termination_tokens = [
                token_id
                for token_id, token_text in enumerate(self.tokenizer.id_to_text)
                if number_is_complete and token_text in (",", "}")
            ]
            allowed_tokens.extend(termination_tokens)

            if not allowed_tokens:
                raise ValueError(
                    f"Invalid or incomplete generated number: "
                    f"{generated_text!r}"
                )

            masked_logits = self.tokenizer.mask_logits(logits, allowed_tokens)

            next_token_id = max(
                range(len(masked_logits)),
                key=lambda i: masked_logits[i]
            )

            token_text = self.tokenizer.id_to_text[next_token_id]

            if token_text in (",", "}"):
                return int(generated_text) if integer else float(generated_text)

            generated_ids.append(next_token_id)
            generated_text += token_text

        if not self.is_complete_number(generated_text, integer):
            raise ValueError(
                f"Number generation limit reached with an incomplete number: "
                f"{generated_text!r}"
            )

        return int(generated_text) if integer else float(generated_text)

    def generate_string(self, context_ids: list[int]) -> str:
        return "Bonjour"

    def generate_boolean(self, context_ids: list[int]) -> bool:
        return False
