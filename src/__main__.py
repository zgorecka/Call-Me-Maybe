from src.file_loader import load_function_def, load_prompt
from pathlib import Path
import argparse
from src.llm import select_function, build_selection_prompt
from llm_sdk import Small_LLM_Model

prompt = "Reverse the string 'hello'"

def parse_arguments() -> argparse.Namespace:
    pass


def main() -> None:        
    path = Path('data/input/functions_definition.json')
    fun = load_function_def(path)
    path = Path('data/input/function_calling_tests.json')
    model = Small_LLM_Model()
    selection_prompt = build_selection_prompt(fun, prompt)
    print("prompt: ", selection_prompt)
    selected_func = select_function(model, fun, selection_prompt)
    print(selected_func)

if __name__ == "__main__":
    main()