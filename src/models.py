from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, RootModel


class ValueType(BaseModel):
    """Represents the type of a function parameter or return value.

    Attributes:
        type: The declared JSON-compatible type name, such as "string",
            "number", or "boolean".
    """

    model_config = ConfigDict(extra="forbid")
    type: Literal["string", "number", "boolean"]


class Function(BaseModel):
    """Represents a callable function definition available to the model.

    Attributes:
        name: Unique function name used during selection.
        description: Human-readable description of the function's purpose.
        parameters: Mapping from parameter names to their declared value types.
        returns: Declared return type of the function.
    """

    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, ValueType]
    returns: ValueType


class Prompt(BaseModel):
    """Represents a single natural-language prompt for function calling.

    Attributes:
        prompt: User request that needs to be mapped to a function call.
    """

    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1)
    @field_validator("prompt")
    @classmethod
    def prompt_cannot_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Prompt cannot be empty")
        return value

class PromptList(RootModel[list[Prompt]]):
    """List of function-calling prompts."""

    pass

class FunctionList(RootModel[list[Function]]):
    """List of available function definitions."""

    pass