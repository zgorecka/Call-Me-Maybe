from src.file_loader import load_function_def, load_prompt
from pathlib import Path
import argparse
from src.llm import select_function, build_selection_prompt
from llm_sdk import Small_LLM_Model

prompt = "What is the sum of 265 and 345?"

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

    for prompt in prompts:
        print(select_function(model, fun, prompt.prompt))

if __name__ == "__main__":
    main()