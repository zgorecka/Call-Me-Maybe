from llm_sdk import Small_LLM_Model
import numpy as np
from src.models import Function
import re

def build_selection_prompt(functions: list[Function], user_prompt: str) -> str:
    lines = [
        "<|im_start|>system\n",
        "Select the function that best matches the user request.\n",
        "Return only the exact function name.\n",
        "Do not explain your choice.\n",
        "",
        "Available functions:\n",
    ]

    for function in functions:
        lines.append("")
        lines.append(f"Name: {function.name}\n")
        lines.append(f"Description: {function.description}\n")
        lines.append("Parameters:\n")

        for param_name, param_data in function.parameters.items():
            lines.append(
                f"- {param_name}: {param_data.type}\n"
            )
    lines.append("<|im_end|>\n")
    lines.append("<|im_start|>user\n")
    lines.append(user_prompt)
    lines.append("<|im_end|>\n")
    lines.append("<|im_start|>assistant\n")

    res = "".join(lines)
    return res

def is_number(text: str) -> bool:
    if not isinstance(text, str) or not text.isascii():
        return False

    try:
        value = float(text)
        return np.isfinite(value)
    except ValueError:
        return False


def is_valid_prefix(text: str) -> bool:
    if not isinstance(text, str) or not text.isascii():
        return False

    # Dozwolone początki liczby
    if text in ("", "-"):
        return True

    unsigned = text.removeprefix("-")

    # Liczba nie może zaczynać się od kropki
    if not unsigned or unsigned.startswith("."):
        return False

    # Maksymalnie jedna kropka
    if unsigned.count(".") > 1:
        return False

    # Wyłącznie cyfry ASCII i kropka
    return all(char in "0123456789." for char in unsigned)
def build_parameter_prompt(function: Function, user_prompt: str) -> str:
    lines = [
        "<|im_start|>system\n",
        "Extract the parameter values required by the selected function.\n",
        "Return only the values in the exact parameter order shown below.\n",
        "Separate consecutive values with a single space.\n",
        "Do not include parameter names, explanations, or additional text.\n",
        "Selected function:\n",
    ]

    lines.append(f"Name: {function.name}\n")
    lines.append("Parameters in required order:\n")

    for param_name, param_data in function.parameters.items():
            lines.append(
                f"- {param_name}: {param_data.type}\n"
            )
    lines. append("<|im_end|>\n")
    lines.append("<|im_start|>user\n")
    lines.append(user_prompt)
    lines.append("\n<|im_end|>\n")
    lines.append("<|im_start|>assistant\n")

    res = "".join(lines)
    return res

def numeric_candidates(prompt: str) -> list[str]:
    words = prompt.split(" ")
    candidates = []
    for word in words:
        try:
            candidates.append(float(word))
        except ValueError:
            continue
    


def select_function(model: Small_LLM_Model, functions: list[Function], selection_prompt: str) -> str:
    names = []
    for function in functions:
        names.append(function.name)
    function_token_ids = []
    for name in names:
        function_token_ids.append(model.encode(name).tolist()[0])

    input_ids = model.encode(selection_prompt).tolist()[0]

    generated_ids = []
    while True:
        if generated_ids in function_token_ids:
            break
        maching_seq = []
        for seq in function_token_ids:
            if generated_ids == seq[:len(generated_ids)]:
                maching_seq.append(seq)

        next_token_id = model.get_logits_from_input_ids(input_ids + generated_ids)

        mask = np.full_like(
            np.asarray(next_token_id),
            -np.inf,
            dtype=float,
        )

        pos = len(generated_ids)

        allowed_ids =[]

        for seq in maching_seq:
            allowed_ids.append(seq[pos])

        for id in allowed_ids:
            mask[id] = next_token_id[id]

        next_token = np.argmax(mask)
        generated_ids.append(next_token)
    
    result = ""
    for id in generated_ids:
        result += model.decode(id)

    return result

def extract_parameter(model: Small_LLM_Model, function:Function, parameter_prompt: str, max_new_tokens: int = 20) -> str:
    input_ids = model.encode(parameter_prompt).tolist()[0]
    generated_ids = []
    genereted_param = []
    for _ in range(max_new_tokens):
        logits = model.get_logits_from_input_ids(input_ids + generated_ids)

        for _ in range(len(logits)):
            next_token = np.argmax(logits)
            next_token_text = model.decode(next_token)

            current_text = "".join(
                model.decode(token_id)
                for token_id in generated_ids
            ).lstrip()

            if current_text == "":
                next_token_text = next_token_text.lstrip()
        
            if next_token_text == "":
                logits[next_token] = -np.inf
                continue

            if any(char.isspace() for char in next_token_text):
                parts = next_token_text.split(maxsplit=1)

                if not parts:
                    logits[next_token] = -np.inf
                    continue

                token_part = parts[0]
                candidate_text = current_text + token_part

                if is_number(candidate_text):
                    return candidate_text

                logits[next_token] = -np.inf
                continue

            candidate_text = current_text + next_token_text
            #print("candidate: ", candidate_text)

            if is_valid_prefix(candidate_text):
                generated_ids.append(next_token)
                #print(next_token_text)
                break

            logits[next_token] = -np.inf

        else:
            raise ValueError(
                f"Model nie znalazł poprawnego tokenu dla: {current_text!r}"
            )

    raise ValueError(
        f"Przekroczono limit {max_new_tokens} tokenów parametru"
    )





if __name__ == "__main__":
    model = Small_LLM_Model()

    #select_function(model, func, selection_prompt)