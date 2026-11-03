from llm_sdk import Small_LLM_Model
from .utils import Tokenizer
from .parsing import parse_args, load_file
from .models import FuncDefModel, InputModel
from pydantic import ValidationError
from json import JSONDecodeError
from .constraint import FunctionSelector, ParameterGenerator


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
        print()
        p = ParameterGenerator(model, tokenizer)
        for i in range(len(funcdef_content)):
            if funcdef_content[i].name == result:
                index = i
                break
            i += 1
        print(
            p.build_prompt(prompt, funcdef_content[index])
        )

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
