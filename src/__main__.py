# from llm_sdk import Small_LLM_Model
from .parsing import parse_args, validate_file_content
from .models import FuncDefModel, InputModel
import json


if __name__ == "__main__":
    arg = parse_args()
    validate_file_content(arg.functions_definition, FuncDefModel)
    validate_file_content(arg.input, InputModel)
