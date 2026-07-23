import json
from pathlib import Path
from typing import Any
from pydantic import TypeAdapter, ValidationError
from src.models import Function, Prompt

def load_json(path: Path) -> Any:
    try:
        with path.open(mode='r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError as e:
        print(e)
    except json.JSONDecodeError as e:
        print(e)

def load_function_def(path: Path) -> list[Function]:
    data = load_json(path)
    functions_adapter = TypeAdapter(list[Function])
    try:
        functions = functions_adapter.validate_python(data)
        return 
    except ValidationError as e:
        print(e)

def load_prompt(path: Path) -> list[Prompt]:
    data = load_json(path)
    functions_adapter = TypeAdapter(list[Prompt])
    try:
        functions = functions_adapter.validate_python(data)
        return data
    except ValidationError as e:
        print(e)

if __name__ == "__main__":
    load_function_def('data/input/functions_definition1.json')