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
    except ValidationError as e:
        print(e)
    return functions


def load_prompt(path: Path) -> list[Prompt]:
    data = load_json(path)
    prompt_adapter = TypeAdapter(list[Prompt])
    try:
        prompts = prompt_adapter.validate_python(data)
    except ValidationError as e:
        print(e)
    return prompts


def generete_json(results: list, path: Path):
    print(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)
