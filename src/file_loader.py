import json
from pathlib import Path
from typing import Any
from pydantic import ValidationError
from src.models import Function, Prompt, FunctionList, PromptList


def load_json(path: Path) -> object:
    """Load and parse a JSON file.

    Args:
        path: Path to the JSON file to read.

    Returns:
        The parsed JSON object (usually a dict or list).

    Raises:
        ValueError: If the file cannot be read, is not UTF-8, or
            contains invalid JSON.
    """

    try:
        with path.open("r", encoding="utf-8") as file:
            data: object = json.load(file)
            return data

    except OSError as error:
        raise ValueError(
            f"Could not read file {path}: {error}"
        ) from error

    except UnicodeDecodeError as error:
        raise ValueError(
            f"File {path} is not valid UTF-8"
        ) from error

    except json.JSONDecodeError as error:
        raise ValueError(
            f"File {path} contains invalid JSON: "
            f"line {error.lineno}, column {error.colno}"
        ) from error


def load_function_def(path: Path) -> list[Function]:
    """Load and validate function definitions from a JSON file.

    Args:
        path: Path to the functions definition JSON file.

    Returns:
        A list of validated `Function` objects.

    Raises:
        ValueError: If the JSON cannot be parsed, validation fails,
            or the resulting function list is empty.
    """

    data = load_json(path)

    try:
        validated = FunctionList.model_validate(data)
    except ValidationError as error:
        raise ValueError(
            f"Invalid function definition file {path}:\n{error}"
        ) from error

    if not validated.root:
        raise ValueError(
            "Function definition list cannot be empty"
        )

    return validated.root


def load_prompt(path: Path) -> list[Prompt]:
    """Load and validate user prompts from a JSON file.

    Args:
        path: Path to the prompts JSON file.

    Returns:
        A list of validated `Prompt` objects.

    Raises:
        ValueError: If the JSON cannot be parsed, validation fails,
            or the resulting prompt list is empty.
    """

    data = load_json(path)

    try:
        validated = PromptList.model_validate(data)
    except ValidationError as error:
        raise ValueError(
            f"Invalid prompt file {path}:\n{error}"
        ) from error

    if not validated.root:
        raise ValueError(
            "Prompt list cannot be empty"
        )

    return validated.root


def generate_json(
    results: list[dict[str, object]],
    path: Path,
) -> None:
    """Serialize results and write them to a JSON file.

    Args:
        results: List of dictionaries representing function-call results.
        path: Destination file path for the JSON output.

    Returns:
        None

    Raises:
        ValueError: If the file cannot be written or the results are
            not JSON-serializable.
    """

    try:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open("w", encoding="utf-8") as file:
            json.dump(
                results,
                file,
                indent=4,
                ensure_ascii=False,
                allow_nan=False,
            )

    except OSError as error:
        raise ValueError(
            f"Could not write output file {path}: {error}"
        ) from error

    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Results cannot be serialized to JSON: {error}"
        ) from error
