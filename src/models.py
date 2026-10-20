import sys
from typing import Literal
try:
    from pydantic import BaseModel, ConfigDict, Field
except ImportError as e:
    print(e)
    sys.exit(1)


class ParamType(BaseModel):
    model_config = ConfigDict(extra='ignore')
    type: Literal["number", "string", "boolean", "integer"]


class FuncDefModel(BaseModel):
    model_config = ConfigDict(extra='ignore')
    name: str = Field(min_length=1)
    description: str
    parameters: dict[str, ParamType]
    returns: ParamType


class InputModel(BaseModel):
    model_config = ConfigDict(extra='ignore')
    prompt: str
