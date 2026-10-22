from llm_sdk import Small_LLM_Model
from .parsing import parse_args, validate_file_content
from .models import FuncDefModel, InputModel
from pydantic import ValidationError
import sys
from json import JSONDecodeError


if __name__ == "__main__":
    try:
        model = Small_LLM_Model()
        arg = parse_args()
        validate_file_content(arg.functions_definition, FuncDefModel)
        validate_file_content(arg.input, InputModel)
        print()
        print(model.get_path_to_vocab_file())
        print()
    except (
        ValidationError,
        KeyboardInterrupt,
        OSError,
        FileNotFoundError,
        PermissionError,
        JSONDecodeError,
        UnicodeDecodeError,
        Exception
    ) as e:
        print(e)
