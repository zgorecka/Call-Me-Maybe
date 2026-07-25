from src.file_loader import load_function_def, load_prompt
from pathlib import Path
import argparse
from src.llm import select_function, build_selection_prompt, build_parameter_prompt, extract_parameter
from llm_sdk import Small_LLM_Model

prompt = "Reverse the string 'hello'"

def parse_arguments() -> argparse.Namespace:
    pass


def main() -> None:        
    path = Path('data/input/functions_definition.json')
    fun = load_function_def(path)
    path = Path('data/input/function_calling_tests.json')
    prompts = load_prompt(path)
    model = Small_LLM_Model()
    #print("prompt: ", selection_prompt)
    #print(prompts[1].prompt)
    #selected_func = select_function(model, fun, prompt)
    #print(selected_func)

    #for prompt in prompts:
    #    selection_prompt = build_selection_prompt(fun, prompt.prompt)
    #    print(select_function(model, fun, selection_prompt))
    param_prompt = build_parameter_prompt(fun[2], prompt)
    print(param_prompt)
    params = extract_parameter(model, fun[0], param_prompt)

if __name__ == "__main__":
    main()