from llm_sdk import Small_LLM_Model
import numpy as np
from src.models import Function


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

def is_number():
    pass

def build_parameter_prompt(function: Function, user_prompt: str) -> str:
    lines = [
        "<|im_start|>system\n",
        "Extract the parameter values required by the selected function.\n",
        "Return only the values in the exact parameter order shown below.\n",
        "Separate consecutive values with a single space.\n",
        "Do not include parameter names, explanations, or additional text.\n",
        "Selected function:\n",
    ]

    lines.append(f"Name: {function.name}")
    lines.append("Parameters in required order:\n")

    for param_name, param_data in function.parameters.items():
            lines.append(
                f"- {param_name}: {param_data.type}\n"
            )
    lines. append("<|im_end|>\n")
    lines.append("<|im_start|>user\n")
    lines.append(user_prompt)
    lines.append("\n<|im_end|>\n")
    lines.append("<|im_start|>assistant")

    res = "".join(lines)
    return res

def extract_parameter(model: Small_LLM_Model, function:Function, parameter_prompt: str) -> str:
    input_ids = model.encode(parameter_prompt).tolist()[0]
    generated_ids = []
    end_ids = model.encode("<|im_end|>").tolist()[0]
    #print(end_ids)

    while True:
        next_token_id = model.get_logits_from_input_ids(input_ids + generated_ids)
        next_token = np.argmax(next_token_id)
        print(
            "token:",
            next_token,
            "tekst:",
            repr(model.decode(next_token))
        )
        generated_ids.append(next_token)
        if generated_ids[-len(end_ids):] == end_ids:
            break
    result = ""
    for id in generated_ids:
        result += model.decode(id)
    print(result)
    return result


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


if __name__ == "__main__":
    model = Small_LLM_Model()

    #select_function(model, func, selection_prompt)