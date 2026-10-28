from llm_sdk import Small_LLM_Model
from .parsing import parse_args, load_file
from .models import FuncDefModel, InputModel
from pydantic import ValidationError
from json import JSONDecodeError


if __name__ == "__main__":
    try:
        model = Small_LLM_Model()
        arg = parse_args()
        funcdef_content = load_file(
            arg.functions_definition, FuncDefModel
        )
        input_content = load_file(arg.input, InputModel)
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
