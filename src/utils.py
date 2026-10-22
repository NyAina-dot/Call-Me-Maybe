from llm_sdk import Small_LLM_Model
from .models import FuncDefModel


class Tokenizer:
    def __init__(self, model: Small_LLM_Model):
        self.model = model
        # for n in range(151642):
        with open(self.model.get_path_to_vocab_file()):
            pass
            

    def text_to_token_ids(self, text: str) -> list[int]:
        encoded = self.model.encode(text)
        return encoded[0].tolist()

    def token_ids_to_text(self, token_ids: list[int]) -> str:
        return self.model.decode(token_ids)

    def function_names_to_tokens(
        self,
        functions: list[FuncDefModel]
    ) -> dict[str, list[int]]:
        result = {}

        for function in functions:
            name = function.name
            result[name] = self.text_to_token_ids(name)

        return result
