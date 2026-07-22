from src.file_loader import load_function_def, load_prompt
from pathlib import Path
import argparse
from src.llm import prep_funtion_names

def parse_arguments() -> argparse.Namespace:
    pass

def main() -> None:        
    path = Path('data/input/functions_definition.json')
    fun = load_function_def(path)
    path = Path('data/input/function_calling_tests.json')
    load_prompt(path)
    print(fun)
    names = prep_funtion_names(fun)
    print(names)

if __name__ == "__main__":
    main()