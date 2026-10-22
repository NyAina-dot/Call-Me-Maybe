from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class ParamType(BaseModel):
    model_config = ConfigDict(extra='forbid')
    type: Literal["number", "string", "boolean", "integer"]


class FuncDefModel(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str = Field(min_length=1)
    description: str
    parameters: dict[str, ParamType]
    returns: ParamType


class InputModel(BaseModel):
    model_config = ConfigDict(extra='forbid')
    prompt: str


class ResultModel(BaseModel):
    model_config = ConfigDict(extra='forbid')
    prompt: str
    name: str = Field(min_length=1)
    parameters: dict[str, float | int | str | bool]
