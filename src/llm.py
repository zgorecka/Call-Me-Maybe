from llm_sdk import Small_LLM_Model
import numpy as np
from src.models import Function


def build_selection_prompt(functions: list[Function], user_prompt: str) -> str:
    """Build a prompt to ask the model to select the best function."""

    lines = [
        "<|im_start|>system\n",
        "Select the function that best matches the user request.\n",
        "Return only the exact function name.\n",
        "Do not explain your choice.\n",
        "\n",
        "Available functions:\n",
    ]

    for function in functions:
        lines.append("\n")
        lines.append("Name: " + function.name + "\n")
        lines.append("Description: " + function.description + "\n")
        lines.append("Parameters:\n")

        for param_name, param_data in function.parameters.items():
            lines.append(f"- {param_name}: {param_data.type}\n")

    lines.append("<|im_end|>\n")
    lines.append("<|im_start|>user\n")
    lines.append(user_prompt)
    lines.append("<|im_end|>\n")
    lines.append("<|im_start|>assistant\n")

    return "".join(lines)


def is_number(text: str) -> bool:
    """Return True if the text represents a finite number."""

    if not isinstance(text, str) or not text.isascii():
        return False

    try:
        value = float(text)
        return np.isfinite(value)
    except ValueError:
        return False


def is_valid_prefix(text: str) -> bool:
    """Return True if the string can be a prefix of a number."""

    if not isinstance(text, str) or not text.isascii():
        return False

    if text in ("", "-"):
        return True

    unsigned = text.removeprefix("-")

    if not unsigned or unsigned.startswith("."):
        return False

    if unsigned.count(".") > 1:
        return False

    return all(char in "0123456789." for char in unsigned)


def build_parameter_prompt_str2(
        function: Function,
        user_prompt: str,
        params: dict,
        name: str
              ) -> str:
    """Build a prompt for generating a parameter value as text."""

    lines = (
        "<|im_start|>system\n"
        "Generate the value of the current parameter required by the"
        "selected function.\nUse the function description and user request.\n"
        "Also use the parameter name and previously selected arguments.\n"
        "For a string parameter, preserve the complete value, including spaces,"
        "punctuation and parentheses.\n"
        "Return only its value with no extra whitespace.\nEnd response"
        "immediately after the value.\n"
        "Selected function:\n"
        f"Name: {function.name}\n"
        f"Description: {function.description}\n"
        "Parameters in required order:\n"
    )

    for param_name, param_data in function.parameters.items():
        lines += f"- {param_name}: {param_data.type}\n"
    lines += (
        f"Current parameter: {name}\n"
        f"Parameter type: {function.parameters[name].type}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_prompt}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    arg = ""
    for param_name, param_data in params.items():
        arg += f"{param_name}: {param_data} "

    arg += f"{name}: "
    lines += arg

    res = "".join(lines)
    print(res)
    return res

def build_parameter_prompt_str(
    function: Function,
    user_prompt: str,
    params: dict[str, str | int | float | bool],
    name: str,
) -> str:
    """Build a completion prompt for generating a string parameter."""

    lines = (
        "Task: Generete or extract arguments needed to call the selected function.\n"
        "Do not execute the function or calculate its result.\n"
        "Generate the complete value of the current parameter.\n"
        "Preserve all spaces, punctuation, quotes and parentheses "
        "contained in the value.\n"
        "Generate only the value after the equals sign.\n"
        "Finish the value with a newline.\n\n"
        "Selected function:\n"
        f"{function.name}\n\n"
        "Function description:\n"
        f"{function.description}\n\n"
        "Expected parameters:\n"
    )

    for param_name, param_data in function.parameters.items():
        lines += f"{param_name} ({param_data.type})\n"

    lines += (
        "\nUser request:\n"
        f"{user_prompt}\n\n"
        "Parameters:\n"
    )

    for param_name, param_value in params.items():
        lines += f"{param_name}={param_value}\n"

    lines += f"{name}="

    print(lines)
    return lines


def build_parameter_prompt(
        function: Function,
        user_prompt: str,
        params: dict, name
        ) -> str:
    """Build a simplified prompt for generating a parameter value."""

    lines = [
        "<|im_start|>system\n",
        "Generate the value of the current parameter required to perform\n",
        "the selected function. Use the user request and already selected\n",
        "arguments. Do not include parameter names or extra text.\n",
        "Selected function:\n",
    ]

    lines.append("Name: " + function.name + "\n")
    lines.append("Parameters in required order:\n")

    for param_name, param_data in function.parameters.items():
        lines.append(f"- {param_name}: {param_data.type}\n")

    lines.append("<|im_end|>\n")
    lines.append("<|im_start|>user\n")
    lines.append(user_prompt)
    lines.append("\n<|im_end|>\n")
    lines.append("<|im_start|>assistant\n")

    arg = ""
    for param_name, param_data in params.items():
        arg += f"{param_name}: {param_data}, "

    arg += f"{name}: "
    lines.append(arg)

    return "".join(lines)


def build_parameter_prompt2(
        function: Function,
        user_prompt: str,
        params: dict, name
        ) -> str:
    """Alternative prompt format that serializes arguments inline."""

    lines = ["User request: " + user_prompt + "\n"]

    lines.append("Function: " + function.name + "\n")
    lines.append("Current parameter: " + name + "\n")
    lines.append("Expected type: " + str(function.parameters[name]) + "\n")
    arguments = "Arguments: {"
    for param_name, param_data in params.items():
        arguments += f'"{param_name}": {param_data}, '
    arguments += f'"{name}": '
    lines.append(arguments)
    lines.append("\n<|im_end|>\n")
    lines.append("<|im_start|>assistant\n")

    return "".join(lines)


def numeric_candidates(prompt: str) -> list[str]:
    """Extract numeric candidate strings from text."""

    words = prompt.split(" ")
    candidates = []
    for word in words:
        cleaned_word = word.strip(',!?;:()[]{}"\'')
        if cleaned_word.endswith(".") and cleaned_word[:-1].isdigit():
            cleaned_word = cleaned_word[:-1]
        try:
            float(cleaned_word)
        except ValueError:
            continue
        float_num = float(cleaned_word)
        if float_num.is_integer():
            candidates.append(str(int(cleaned_word)))
        else:
            candidates.append(str(float_num))
    return candidates


def select_parameters_str(
        model: Small_LLM_Model,
        function: Function,
        user_prompt: str,
        params: dict,
        name: str
        ) -> str:
    """Generate a string parameter value using the model."""

    parameter_prompt = build_parameter_prompt_str(
        function, user_prompt, params, name
    )

    input_ids = model.encode(parameter_prompt).tolist()[0]
    generated_ids: list[float] = []
    result = ""

    while "\n" not in result:
        next_token_id = model.get_logits_from_input_ids(
            input_ids + generated_ids
            )
        next_token = np.argmax(next_token_id)
        generated_ids.append(int(next_token))
        result += model.decode(next_token)

    return result


def select_parameters_num(
        model: Small_LLM_Model,
        function: Function,
        user_prompt: str,
        params: dict,
        name: str
        ) -> str:
    """Select a numeric parameter by matching generated token sequences."""

    prompt_num = numeric_candidates(user_prompt)
    parameter_prompt = build_parameter_prompt(
        function, user_prompt, params, name
        )
    num_token_ids = []
    for num in prompt_num:
        num_token_ids.append(model.encode(num).tolist()[0])

    input_ids = model.encode(parameter_prompt).tolist()[0]
    generated_ids: list[float] = []

    while generated_ids not in num_token_ids:
        if generated_ids in num_token_ids:
            break

        maching_seq = []
        for seq in num_token_ids:
            if generated_ids == seq[:len(generated_ids)]:
                maching_seq.append(seq)

        next_token_id = model.get_logits_from_input_ids(
            input_ids + generated_ids
            )

        mask = np.full_like(np.asarray(next_token_id), -np.inf, dtype=float)

        pos = len(generated_ids)

        allowed_ids = []

        for seq in maching_seq:
            allowed_ids.append(seq[pos])

        for id_ in allowed_ids:
            mask[id_] = next_token_id[id_]

        next_token = np.argmax(mask)
        generated_ids.append(int(next_token))

    result = model.decode(generated_ids)
    return result

def select_parameters_bool(
    model: Small_LLM_Model,
    function: Function,
    user_prompt: str,
    params: dict[str, str | int | float | bool],
    name: str,
) -> bool:
    """Select a boolean parameter using constrained decoding."""

    candidates = ["true", "false"]

    candidate_token_ids: list[list[int]] = [
        model.encode(candidate).tolist()[0]
        for candidate in candidates
    ]

    parameter_prompt = build_parameter_prompt(
        function,
        user_prompt,
        params,
        name,
    )

    input_ids: list[int] = model.encode(
        parameter_prompt
    ).tolist()[0]

    generated_ids: list[int] = []

    while generated_ids not in candidate_token_ids:
        matching_sequences: list[list[int]] = []

        for sequence in candidate_token_ids:
            if generated_ids == sequence[:len(generated_ids)]:
                matching_sequences.append(sequence)

        if not matching_sequences:
            raise ValueError(
                f"Could not generate boolean parameter: {name}"
            )

        position = len(generated_ids)
        allowed_ids: set[int] = set()

        for sequence in matching_sequences:
            if position < len(sequence):
                allowed_ids.add(sequence[position])

        logits = model.get_logits_from_input_ids(
            input_ids + generated_ids
        )

        mask = np.full_like(
            np.asarray(logits),
            -np.inf,
            dtype=float,
        )

        for token_id in allowed_ids:
            mask[token_id] = logits[token_id]

        next_token = int(np.argmax(mask))
        generated_ids.append(next_token)

    result = model.decode(generated_ids).strip().lower()

    return result == "true"

def select_parameters(
        model: Small_LLM_Model,
        function: Function,
        user_prompt: str
        ) -> dict:
    """Pick values for all required parameters of a function."""

    params: dict[str, str | int | float | bool] = {}

    for name, value_type in function.parameters.items():
        match value_type.type:
            case "number":
                params[name] = select_parameters_num(
                    model, function, user_prompt, params, name
                )
                try:
                    num = float(params[name])
                    if num.is_integer():
                        params[name] = int(num)
                    else:
                        params[name] = float(num)
                except (ValueError, TypeError):
                    continue
            case "string":
                params[name] = select_parameters_str(
                    model, function, user_prompt, params, name
                ).strip()
            case "boolean":
                params[name] = select_parameters_bool(
                    model, function, user_prompt, params, name
                )

    return params


def select_function(
        model: Small_LLM_Model,
        functions: list[Function],
        user_prompt: str
        ) -> Function:
    """Ask the model to select the best function from a list."""

    selection_prompt = build_selection_prompt(functions, user_prompt)
    names = [function.name for function in functions]
    function_token_ids = [model.encode(name).tolist()[0] for name in names]

    input_ids = model.encode(selection_prompt).tolist()[0]

    generated_ids: list[float] = []
    while True:
        if generated_ids in function_token_ids:
            break
        maching_seq = []
        for seq in function_token_ids:
            if generated_ids == seq[:len(generated_ids)]:
                maching_seq.append(seq)

        next_token_id = model.get_logits_from_input_ids(
            input_ids + generated_ids
            )

        mask = np.full_like(np.asarray(next_token_id), -np.inf, dtype=float)
        pos = len(generated_ids)
        allowed_ids = []
        for seq in maching_seq:
            allowed_ids.append(seq[pos])
        for id_ in allowed_ids:
            mask[id_] = next_token_id[id_]
        next_token = np.argmax(mask)
        generated_ids.append(next_token)

    result = "".join(model.decode(i) for i in generated_ids)

    for function in functions:
        if function.name == result:
            return function
    raise ValueError("Selected function name not found in provided list")


def extract_parameter(
        model: Small_LLM_Model,
        function: Function,
        parameter_prompt: str,
        max_new_tokens: int = 20
        ) -> str:
    """Extract a numeric parameter token-by-token and return it as text."""

    input_ids = model.encode(parameter_prompt).tolist()[0]
    generated_ids: list[float] = []
    for _ in range(max_new_tokens):
        logits = model.get_logits_from_input_ids(input_ids + generated_ids)

        for _ in range(len(logits)):
            next_token = np.argmax(logits)
            next_token_text = model.decode(next_token)

            current_text = "".join(
                model.decode(token_id) for token_id in generated_ids
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

            if is_valid_prefix(candidate_text):
                generated_ids.append(next_token)
                break
            logits[next_token] = -np.inf

        else:
            raise ValueError(
                f"Model didn't find right token for: {current_text!r}"
                )

    raise ValueError(
        f"Exceeded token limit ({max_new_tokens}) while extracting parameter"
        )
