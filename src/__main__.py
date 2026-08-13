from src.file_loader import load_function_def, load_prompt, generete_json
from pathlib import Path
import argparse
from src.llm import select_function, select_parameters
from llm_sdk import Small_LLM_Model

user_prompt = "Replace all vowels in 'Programming is fun' with asterisks"

def parse_arguments() -> argparse.Namespace:
    pass


def main() -> None: 
    results = []
    path = Path('data/input/functions_definition.json')
    function_list = load_function_def(path)
    path = Path('data/input/function_calling_tests.json')
    prompts = load_prompt(path)
    model = Small_LLM_Model()
    #print("prompt: ", selection_prompt)
    #print(prompts[1].prompt)
    #selected_func = select_function(model, function_list, user_prompt)
    #print(selected_func.name)

    #for prompt in prompts:
    #    selection_prompt = build_selection_prompt(fun, prompt.prompt)
    #    print(select_function(model, fun, selection_prompt))
    #param_prompt = build_parameter_prompt(fun[0], prompt)
    #params = select_parameters(model, selected_func, user_prompt) 
    #print("sdsdff ", params)
    #print(string_candidates(prompt))

    for prompt in prompts:
        #print(prompt)
        res = {}
        selected_function = select_function(model, function_list, prompt.prompt)
        params = select_parameters(model, selected_function, prompt.prompt)
        res["prompt"] = prompt.prompt
        res["name"] = selected_function.name
        res["parameters"] = params
        results.append(res)

    print(results)
    generete_json(results)

if __name__ == "__main__":
    main()