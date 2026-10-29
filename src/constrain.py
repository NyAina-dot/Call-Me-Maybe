from .utils import Tokenizer
from .models import FuncDefModel


class FunctionSelector:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
    
    def choose_function_name(
        self, prompt: str, functions: list[FuncDefModel]
    ):
        input_ids = self.tokenizer.text_to_token_ids(prompt)
        for _ in range(20):
            logits = self.tokenizer.model.get_logits_from_input_ids(
                input_ids
            )
            next_token_id = max(
                range(len(logits)),
                key = lambda i: logits[i]
            )
            input_ids.append(next_token_id)
            genrated_name = self.tokenizer.token_ids_to_text(
                input_ids
            )
