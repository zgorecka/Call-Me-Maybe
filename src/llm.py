from llm_sdk import Small_LLM_Model
import numpy as np

func = [
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": {
        "type": "number"
      },
      "b": {
        "type": "number"
      }
    },
    "returns": {
      "type": "number"
    }
  },
  {
    "name": "fn_greet",
    "description": "Generate a greeting message for a person by name.",
    "parameters": {
      "name": {
        "type": "string"
      }
    },
    "returns": {
      "type": "string"
    }
  },
  {
    "name": "fn_reverse_string",
    "description": "Reverse a string and return the reversed result.",
    "parameters": {
      "s": {
        "type": "string"
      }
    },
    "returns": {
      "type": "string"
    }
  },
  {
    "name": "fn_get_square_root",
    "description": "Calculate the square root of a number.",
    "parameters": {
      "a": {
        "type": "number"
      }
    },
    "returns": {
      "type": "number"
    }
  },
  {
    "name": "fn_substitute_string_with_regex",
    "description": "Replace all occurrences matching a regex pattern in a string.",
    "parameters": {
      "source_string": {
        "type": "string"
      },
      "regex": {
        "type": "string"
      },
      "replacement": {
        "type": "string"
      }
    },
    "returns": {
      "type": "string"
    }
  }
]

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


def select_function(model: Small_LLM_Model, functions: list, selection_prompt: str) -> str:
    names = []
    for function in functions:
        names.append(function["name"])
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