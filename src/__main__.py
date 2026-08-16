from src.file_loader import load_function_def, load_prompt, generate_json
from pathlib import Path
import argparse
from src.llm import select_call
from llm_sdk import Small_LLM_Model
import sys


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the project entry point.

    Returns:
        Parsed CLI arguments containing function definitions,
        input prompts, and output destination paths.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=Path("data/input/functions_definition.json"),
        help="Path to functions definition JSON file"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/input/function_calling_tests.json"),
        help="Path to input prompts JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/function_calls.json"),
        help="Path to output JSON file"
    )

    return parser.parse_args()


def main() -> None:
    """Run the function-calling pipeline for all input prompts.

    The function loads the available schema and prompt set, creates a model
    instance, resolves the selected function calls for each prompt, and writes
    the structured results to the output JSON file.
    """

    args = parse_arguments()

    functions = load_function_def(
        path=args.functions_definition
    )
    prompts = load_prompt(
        path=args.input
    )

    model = Small_LLM_Model()

    results: list[dict[str, object]] = []

    for prompt_data in prompts:
        result = select_call(
            model=model,
            functions=functions,
            user_prompt=prompt_data.prompt,
        )

        results.append(result)

    generate_json(
        results=results,
        path=args.output,
    )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
