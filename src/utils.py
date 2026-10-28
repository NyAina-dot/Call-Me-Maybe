from typing import Any
from llm_sdk import Small_LLM_Model
from .models import FuncDefModel
import json


class Tokenizer:
    """Wrap the SDK tokenizer and expose an id -> text table."""

    def __init__(
        self, model: Small_LLM_Model,
    ):
        """Load the vocabulary and build the id -> text table once.

        An empty string marks an unusable id (special token, padding,
        or partial UTF-8 byte), to be masked to -inf during decoding.
        """
        self.model = model
        with open(
            self.model.get_path_to_vocab_file(),
            encoding="utf-8"
        ) as file:
            vocab: dict[str, int] = json.load(file)
        self.vocab_size: int = len(
            self.model.get_logits_from_input_ids([0])
        )
        self.id_to_text: list[str] = [""] * self.vocab_size

        for token_id in vocab.values():
            text = self.model.decode([token_id])
            if "\ufffd" not in text:
                self.id_to_text[token_id] = text

    def text_to_token_ids(self, text: str) -> list[int]:
        """Encode a text into a flat list of token ids."""
        encoded = self.model.encode(text)
        return encoded[0].tolist()

    def token_ids_to_text(self, token_ids: list[int]) -> str:
        """Decode a list of token ids back into text."""
        return self.model.decode(token_ids)

    def function_names_to_tokens(
        self,
        functions: list[FuncDefModel]
    ) -> dict[str, list[int]]:
        """Map each function name to its canonical token ids."""
        result: dict[str, list[int]] = {}

        for function in functions:
            name = function.name
            result[name] = self.text_to_token_ids(name)

        return result

    def function_names_token_tree(
        self, functions: list[FuncDefModel]
    ) -> dict[int, dict]:
        """Build a trie of token ids from the canonical encoding of each name."""
        tree: dict[int, dict] = {}
        names_tokens = self.function_names_to_tokens(functions)

        for token_ids in names_tokens.values():
            node = tree

            for token_id in token_ids:
                node = node.setdefault(token_id, {})

            node[None] = {}

        return tree

    def get_allowed_tokens(
        self,
        tree: dict[int | None, Any],
        generated_tokens: list[int],
    ) -> list[int]:
        """Return the token ids allowed after the generated tokens.

        Args:
            tree: Trie built by function_names_token_tree.
            generated_tokens: Token ids already generated for the name.

        Returns:
            Ids of the tokens that continue a valid function name.

        Raises:
            ValueError: If generated_tokens leaves the trie, or if no
                continuation is possible.
        """
        current = tree

        for token_id in generated_tokens:
            if token_id not in current:
                raise ValueError(
                    f"Token {token_id} is not a valid continuation"
                )
            current = current[token_id]

        allowed = [key for key in current if key is not None]

        if not allowed:
            raise ValueError("No token can continue this function name")

        return allowed
