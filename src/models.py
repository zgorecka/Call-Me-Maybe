from pydantic import BaseModel, ConfigDict, TypeAdapter

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

def main() -> None:
    data = [
    {
        "name": "fn_add_numbers",
        "description": "Add two numbers together and return their sum.",
        "parameters": {
        "a": {
            "type": "number"
        },
        "b": {
            "type": "number"
        }
        },
        "returns": {
        "type": "number"
        }
    },
    {
        "name": "fn_greet",
        "description": "Generate a greeting message for a person by name.",
        "parameters": {
        "name": {
            "type": "string"
        }
        },
        "returns": {
        "type": "string"
        }
    }
    ]

    functions_adapter = TypeAdapter(list[Function])
    functions = functions_adapter.validate_python(data)
    print(functions)
    print(type(functions))
    print(type(functions[0]))
    print(type(functions[0].parameters["a"]))


if __name__ == "__main__":
    main()