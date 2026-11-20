"""Pydantic models for function definitions and generated results."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class ParamType(BaseModel):
    """Describe the JSON type of a function parameter or return value."""

    model_config = ConfigDict(extra='forbid')
    type: Literal["number", "string", "boolean", "integer"]


class FuncDefModel(BaseModel):
    """Describe a callable function and its parameters."""

    model_config = ConfigDict(extra='forbid')
    name: str = Field(min_length=1)
    description: str
    parameters: dict[str, ParamType]
    returns: ParamType


class InputModel(BaseModel):
    """Represent one natural-language input prompt."""

    model_config = ConfigDict(extra='forbid')
    prompt: str


class ResultModel(BaseModel):
    """Represent a generated function call."""

    model_config = ConfigDict(extra='forbid')
    prompt: str
    name: str = Field(min_length=1)
    parameters: dict[str, float | int | str | bool]
