from pydantic import BaseModel, ConfigDict


class ValueType(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str


class Function(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    parameters: dict[str, ValueType]
    returns: ValueType


class Prompt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str
