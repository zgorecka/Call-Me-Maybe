from llm_sdk import Small_LLM_Model
import numpy as np

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

def prep_funtion_names(functions: list) -> list:
    names = []
    for function in functions:
        names.append(function["name"])
    
    return names

def select_function(model: Small_LLM_Model, names: list, selection_prompt: str) -> None:
    function_token_ids = []
    for name in names:
        function_token_ids.append(model.encode(name).tolist()[0])
    print(function_token_ids)

    input_ids = model.encode(selection_prompt).tolist()[0]

    

def initModel() -> None:
    prompt = "What is the sum of 2 and 3?"
    model = Small_LLM_Model()
    encoded = model.encode(selection_prompt)
    print(encoded.shape)
    print(encoded.tolist())
    input_ids = encoded.tolist()[0]
    logits = model.get_logits_from_input_ids(input_ids)
    logits_array = np.asarray(logits)
    print(logits_array.shape)
    next_token_id = int(np.argmax(logits))
    print("Token ID:", next_token_id)
    print("Logit:", logits[next_token_id])
    token_text = model.decode([next_token_id])
    print("Token:", token_text)
    for i in range(100):
        input_ids.append(next_token_id)
        logits = model.get_logits_from_input_ids(input_ids)
        next_token_id = int(np.argmax(logits))
        token_text = model.decode([next_token_id])
        print("Token:", token_text)

if __name__ == "__main__":
    model = Small_LLM_Model()

    select_function(model, ['fn_add_numbers', 'fn_greet', 'fn_reverse_string', 'fn_get_square_root', 'fn_substitute_string_with_regex'])