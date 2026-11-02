from llm_sdk import Small_LLM_Model
from .utils import Tokenizer
from .parsing import parse_args, load_file
from .models import FuncDefModel, InputModel
from pydantic import ValidationError
from json import JSONDecodeError
from .constraint import FunctionSelector


if __name__ == "__main__":
    try:
        model = Small_LLM_Model()
        arg = parse_args()
        funcdef_content = load_file(
            arg.functions_definition, FuncDefModel
        )
        input_content = load_file(arg.input, InputModel)
        tokenizer = Tokenizer(model)
        constraint = FunctionSelector(model, tokenizer)
        prompt = "Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'"
        print()
        print(constraint.build_prompt(prompt, funcdef_content))
        print()
        result = constraint.choose_function_name(prompt, funcdef_content)
        print(result)

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
