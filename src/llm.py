from llm_sdk import Small_LLM_Model
import numpy as np
from src.models import Function
import json

_JSON_STRING_START_CACHE: dict[int, list[int]] = {}
_TOKEN_IDS_CACHE: dict[
    tuple[int, tuple[str, ...]],
    list[list[int]],
] = {}


def build_call_prompt(
    functions: list[Function],
    user_prompt: str,
) -> str:
    """Build the prompt used to select a function and extract its arguments.

    Args:
        functions: Available callable functions and their parameter schema.
        user_prompt: Natural-language request from the user.

    Returns:
        A prompt instructing the model to return a valid JSON function call.
    """

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
    """Select the most relevant function using constrained decoding.

    Args:
        model: Local language model used for token generation.
        functions: Available functions to choose from.
        context_ids: Token ids of the prompt prefix before function selection.

    Returns:
        The selected function object and the generated token ids for it.
    """

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
    """Return token ids that can begin a JSON string literal.

    Args:
        model: The language model whose tokenizer vocabulary is inspected.
        vocab_size: Number of tokens in the tokenizer vocabulary.

    Returns:
        Token ids whose decoded text begins with a double quote.

    Raises:
        ValueError: If no valid JSON string start token
        exists in the vocabulary.
    """

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
    """Extract the first complete JSON string literal from text.

    Args:
        text: Decoded generated text that may contain a quoted JSON string.

    Returns:
        The substring representing a complete JSON string literal,
        or None if no valid string literal is present yet.
    """

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
    """Tokenize all candidate strings once and cache the result.

    Args:
        model: Local language model used to tokenize candidate values.
        candidates: Candidate strings for constrained decoding.

    Returns:
        A list of token-id sequences corresponding to the candidates.
    """

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
    """Select a valid candidate from a constrained set.

    Args:
        model: Local language model used to score valid next tokens.
        context_ids: Token ids already generated
                    before the candidate selection.
        candidates: Allowed decoding candidates.

    Returns:
        The decoded winning candidate and the token ids used to generate it.

    Raises:
        ValueError: If the candidate list is empty, no
        valid continuation exists, or generation cannot match a valid prefix.
    """

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
    """Select a numeric value from the user request.

    Args:
        model: Local language model used for constrained numeric selection.
        user_prompt: Prompt containing the natural-language query.
        context_ids: Token ids for the current JSON output prefix.

    Returns:
        The selected integer or float and the generated token ids.

    Raises:
        ValueError: If no numeric candidate can be extracted from the prompt.
    """

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
    """Select a boolean value using constrained decoding.

    Args:
        model: Local language model used to choose between the valid values.
        context_ids: Token ids for the current JSON output prefix.

    Returns:
        The selected boolean and the generated token ids.
    """

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
    """Generate a complete JSON string value for a function parameter.

    Args:
        model: Local language model used to generate the string literal.
        context_ids: Token ids for the current JSON output prefix.
        max_new_tokens: Maximum number of generated tokens to attempt.

    Returns:
        The parsed string value and the token ids used to generate it.

    Raises:
        ValueError: If a valid JSON string cannot be generated within the token
            budget.
    """

    generated_ids: list[int] = []
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than zero")

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

        if not isinstance(value, str):
            raise ValueError(
                f"Generated JSON value is not a string: {value!r}"
            )

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


def numeric_candidates(prompt: str) -> list[str]:
    """Extract numeric candidate strings from a prompt.

    Args:
        prompt: Natural-language text containing numeric values.

    Returns:
        A list of numeric strings found in the input, preserving integer and
        floating-point forms when present.
    """

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
            candidates.append(str(int(float_num)))
        else:
            candidates.append(str(float_num))
    return candidates
