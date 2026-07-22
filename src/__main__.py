from src.file_loader import load_function_def, load_prompt
from pathlib import Path
import argparse
from src.llm import select_function
from llm_sdk import Small_LLM_Model

selection_prompt = (
    "<|im_start|>system\n"
    "Select the function that best matches the user request. "
    "Return only the function name.\n"
    "\n"
    "Available functions:\n"
    "Name: fn_add_numbers\n"
    "Description: Add two numbers together.\n"
    "\n"
    "Name: fn_greet\n"
    "Description: Generate a greeting.\n"
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "What is the sum of 2 and 3?\n"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)

def parse_arguments() -> argparse.Namespace:
    pass

def main() -> None:        
    path = Path('data/input/functions_definition.json')
    fun = load_function_def(path)
    path = Path('data/input/function_calling_tests.json')
    model = Small_LLM_Model()
    selected_func = select_function(model, fun, selection_prompt)
    print(selected_func)

if __name__ == "__main__":
    main()