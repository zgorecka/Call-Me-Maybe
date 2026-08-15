from src.file_loader import load_function_def, load_prompt, generete_json
from pathlib import Path
import argparse
from src.llm import select_call
from llm_sdk.llm_sdk import Small_LLM_Model


def parse_arguments() -> argparse.Namespace:
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
    # args = parse_arguments()
    # results: list[dict[str, int | float | str | bool]] = []
    # function_list = load_function_def(path=args.functions_definition)
    # prompts = load_prompt(path=args.input)
    # model = Small_LLM_Model()

    # for prompt in prompts:
    #     res = {}
    #     selected_function = select_function(
    #         model, function_list, prompt.prompt
    #         )
    #     params = select_parameters(model, selected_function, prompt.prompt)
    #     res["prompt"] = prompt.prompt
    #     res["name"] = selected_function.name
    #     res["parameters"] = params
    #     results.append(res)

    # print(results)
    # generete_json(results=results, path=args.output)

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

    generete_json(
        results=results,
        path=args.output,
    )


if __name__ == "__main__":
    main()
