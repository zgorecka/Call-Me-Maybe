from llm_sdk.llm_sdk import Small_LLM_Model
import numpy as np
from src.models import Function
import json

_JSON_STRING_START_CACHE: dict[int, list[int]] = {}
_TOKEN_IDS_CACHE: dict[
    tuple[int, tuple[str, ...]],
    list[list[int]],
] = {}


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


def build_call_prompt(
    functions: list[Function],
    user_prompt: str,
) -> str:
    """Build one prompt for function and argument selection."""

    lines = (
        "Select the best function and extract its arguments.\n"
        "Do not execute the function.\n"
        "Generate arguments that perform the complete user request.\n"
        "Copy explicit values exactly.\n"
        "A regex must match one requested occurrence at a time, so that "
        "re.sub can replace every occurrence independently.\n"
        "Do not use .* to connect separate occurrences unless the user "
        "explicitly requests the entire span between them.\n"
        "When a replacement is described as a symbol, return the literal "
        "symbol, for example asterisk means *.\n"
        "Return valid JSON.\n\n"
        "Available functions:\n"
    )

    for function in functions:
        lines += (
            f"\nName: {function.name}\n"
            f"Description: {function.description}\n"
            "Parameters:\n"
        )

        for param_name, param_data in function.parameters.items():
            lines += f"- {param_name}: {param_data.type}\n"

    lines += (
        "\nUser request:\n"
        f"{user_prompt}\n\n"
        "Output:\n"
        "{"
        f'"prompt": {json.dumps(user_prompt, ensure_ascii=False)}, '
        '"name": "'
    )

    return lines


def select_function_from_context(
    model: Small_LLM_Model,
    functions: list[Function],
    context_ids: list[int],
) -> tuple[Function, list[int]]:
    """Select a function using constrained decoding."""

    function_by_name = {
        function.name: function
        for function in functions
    }

    result, generated_ids = select_candidate_from_context(
        model,
        context_ids,
        list(function_by_name),
    )

    return function_by_name[result], generated_ids


def json_string_start_token_ids(
    model: Small_LLM_Model,
    vocab_size: int,
) -> list[int]:
    """Return token IDs that begin a JSON string."""

    cache_key = id(model)

    if cache_key in _JSON_STRING_START_CACHE:
        return _JSON_STRING_START_CACHE[cache_key]

    allowed_ids: list[int] = []

    for token_id in range(vocab_size):
        token_text = model.decode([token_id])

        if token_text.lstrip().startswith('"'):
            allowed_ids.append(token_id)

    if not allowed_ids:
        raise ValueError(
            "Tokenizer does not contain a JSON string start token"
        )

    _JSON_STRING_START_CACHE[cache_key] = allowed_ids
    return allowed_ids


def extract_complete_json_string(
    text: str,
) -> str | None:
    """Extract the first complete JSON string."""

    text = text.lstrip()

    if not text.startswith('"'):
        return None

    for index in range(1, len(text)):
        if text[index] != '"':
            continue

        backslashes = 0
        position = index - 1

        while position >= 0 and text[position] == "\\":
            backslashes += 1
            position -= 1

        if backslashes % 2 == 0:
            return text[:index + 1]

    return None


def encode_candidates(
    model: Small_LLM_Model,
    candidates: list[str],
) -> list[list[int]]:
    """Encode candidate sequences once."""

    key = (id(model), tuple(candidates))

    if key not in _TOKEN_IDS_CACHE:
        _TOKEN_IDS_CACHE[key] = [
            model.encode(candidate).tolist()[0]
            for candidate in candidates
        ]

    return _TOKEN_IDS_CACHE[key]


def select_call(
    model: Small_LLM_Model,
    functions: list[Function],
    user_prompt: str,
) -> dict[str, object]:
    prompt = build_call_prompt(functions, user_prompt)
    context_ids = model.encode(prompt).tolist()[0]

    function, function_ids = select_function_from_context(
        model,
        functions,
        context_ids,
    )

    context_ids.extend(function_ids)
    context_ids.extend(
        model.encode('", "parameters": {').tolist()[0]
    )

    params: dict[str, str | int | float | bool] = {}
    value: str | int | float | bool
    value_ids: list[int]

    parameter_items = list(function.parameters.items())

    for index, (name, type_info) in enumerate(parameter_items):
        if index > 0:
            syntax = f', {json.dumps(name)}: '
        else:
            syntax = f'{json.dumps(name)}: '

        context_ids.extend(model.encode(syntax).tolist()[0])

        if type_info.type == "number":
            value, value_ids = select_parameters_num(
                model,
                user_prompt,
                context_ids,
            )

        elif type_info.type == "boolean":
            value, value_ids = select_parameters_bool(
                model,
                context_ids,
            )

        elif type_info.type == "string":
            value, value_ids = select_parameters_str(
                model,
                context_ids,
            )

        else:
            raise ValueError(
                f"Unsupported parameter type: {type_info.type}"
            )

        params[name] = value
        context_ids.extend(value_ids)

    context_ids.extend(model.encode("}}").tolist()[0])

    return {
        "prompt": user_prompt,
        "name": function.name,
        "parameters": params,
    }


def select_candidate_from_context(
    model: Small_LLM_Model,
    context_ids: list[int],
    candidates: list[str],
) -> tuple[str, list[int]]:
    """Select one candidate using constrained decoding."""

    if not candidates:
        raise ValueError("Candidate list cannot be empty")

    candidate_token_ids = encode_candidates(
        model,
        candidates,
    )

    generated_ids: list[int] = []

    while generated_ids not in candidate_token_ids:
        matching_sequences = [
            sequence
            for sequence in candidate_token_ids
            if generated_ids == sequence[:len(generated_ids)]
        ]

        if not matching_sequences:
            raise ValueError(
                "Generated tokens do not match any candidate"
            )

        position = len(generated_ids)

        allowed_ids = {
            sequence[position]
            for sequence in matching_sequences
            if position < len(sequence)
        }

        if not allowed_ids:
            raise ValueError("No allowed next token")

        if len(allowed_ids) == 1:
            next_token = next(iter(allowed_ids))
        else:
            logits = model.get_logits_from_input_ids(
                context_ids + generated_ids
            )

            next_token = max(
                allowed_ids,
                key=lambda token_id: logits[token_id],
            )

        generated_ids.append(next_token)

    return model.decode(generated_ids), generated_ids


def select_parameters_num(
    model: Small_LLM_Model,
    user_prompt: str,
    context_ids: list[int],
) -> tuple[int | float, list[int]]:
    """Select a number from the user request."""

    candidates = numeric_candidates(user_prompt)

    if not candidates:
        raise ValueError(
            f"No numeric candidates found in: {user_prompt!r}"
        )

    result, generated_ids = select_candidate_from_context(
        model,
        context_ids,
        candidates,
    )

    number = float(result)

    if number.is_integer():
        return int(number), generated_ids

    return number, generated_ids


def select_parameters_bool(
    model: Small_LLM_Model,
    context_ids: list[int],
) -> tuple[bool, list[int]]:
    """Select true or false using constrained decoding."""

    result, generated_ids = select_candidate_from_context(
        model,
        context_ids,
        ["true", "false"],
    )

    return result == "true", generated_ids


def select_parameters_str(
    model: Small_LLM_Model,
    context_ids: list[int],
    max_new_tokens: int = 100,
) -> tuple[str, list[int]]:
    """Generate one complete JSON string."""

    generated_ids: list[int] = []

    for position in range(max_new_tokens):
        logits = model.get_logits_from_input_ids(
            context_ids + generated_ids
        )

        if position == 0:
            allowed_ids = json_string_start_token_ids(
                model,
                len(logits),
            )

            next_token = max(
                allowed_ids,
                key=lambda token_id: logits[token_id],
            )

        else:
            next_token = int(np.argmax(logits))

        generated_ids.append(next_token)

        generated_text = model.decode(generated_ids)
        json_string = extract_complete_json_string(
            generated_text
        )

        if json_string is None:
            continue

        try:
            value = json.loads(json_string)
        except json.JSONDecodeError as error:
            raise ValueError(
                "Model generated an invalid JSON string: "
                f"{json_string!r}"
            ) from error

        normalized_string = json.dumps(
            value,
            ensure_ascii=False,
        )

        normalized_ids = model.encode(
            normalized_string
        ).tolist()[0]

        return value, normalized_ids

    raise ValueError(
        f"String was not finished after {max_new_tokens} tokens"
    )


def build_parameter_prompt_str(
    function: Function,
    user_prompt: str,
    params: dict[str, str | int | float | bool],
    name: str,
) -> str:
    """Build a JSON-completion prompt for one string parameter."""

    lines = (
        "Extract arguments required to call the function.\n"
        "Copy explicit values completely and exactly.\n"
        "Generate machine-readable values when they must be derived.\n"
        "For regex, use concise canonical syntax without unnecessary "
        "capturing groups.\n"
        "Use a character class to match any character from a set.\n"
        "Use word boundaries when matching a complete word.\n"
        "Return exactly one JSON string value.\n\n"
        f"Function: {function.name}\n"
        f"Description: {function.description}\n"
        f"User request: {user_prompt}\n"
        "Output: {"
        f'"prompt": {json.dumps(user_prompt, ensure_ascii=False)}, '
        f'"name": {json.dumps(function.name)}, '
        '"parameters": {'
    )

    for param_name, param_value in params.items():
        lines += (
            f"{json.dumps(param_name)}: "
            f"{json.dumps(param_value, ensure_ascii=False)}, "
        )

    lines += f"{json.dumps(name)}: \""

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
