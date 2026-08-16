*This project has been created as part of the 42 curriculum by zgorecka.*

# Call Me Maybe

## Description

Call Me Maybe is a small project focused on function calling in large language models. The goal is to teach an LLM to choose the most relevant function from a known schema and then extract the correct arguments from a user request, without executing the function itself.

The project uses a local causal language model from the Hugging Face ecosystem and applies constrained decoding to keep generation valid, structured, and aligned with the function definitions provided in JSON. In practice, the system reads a list of available functions, selects the best match for the user request, and builds a valid JSON object describing the function name and its parameters.

This repository demonstrates an introductory but practical approach to tool-use behavior in LLMs: rather than letting the model freely generate arbitrary text, it is guided to emit only valid function calls following a known interface.

## Project Goal

The project aims to solve a common challenge in modern agentic AI systems:

- interpret a natural-language request,
- map it to the correct callable function,
- extract the required values from the text,
- return a machine-readable structure that a downstream system could safely use.

The project is intentionally compact and educational, making it suitable as a study of structured generation and constrained decoding in the context of function calling.

## Architecture

The program is organized around a small set of core components:

- `src/__main__.py`: entry point; loads available functions and prompts, then runs inference for each request.
- `src/file_loader.py`: loads JSON definitions and writes generated results to disk.
- `src/models.py`: Pydantic models for function definitions and prompt payloads.
- `src/llm.py`: main decision logic, including function selection and constrained argument generation.
- `llm_sdk/`: local SDK wrapper around a Hugging Face causal language model.

The output is written to `data/output/function_calls.json` and follows a predictable schema like:

```json
[
  {
    "prompt": "What is the product of 3 and 5?",
    "name": "fn_multiply_numbers",
    "parameters": {
      "a": 3,
      "b": 5
    }
  }
]
```

## Algorithm explanation

The central idea is constrained decoding.

Instead of allowing the model to generate unrestricted text, the code narrows the possible next tokens according to the expected function call structure. This is done in several stages.

### 1. Prompt construction

The system first builds a prompt describing all available functions. Each function includes:

- its name,
- a human-readable description,
- its parameter list with types,
- the expected return type.

The user request is appended to this prompt, and a JSON completion template is initiated. The output is expected to look like a structured JSON object with fields such as `prompt`, `name`, and `parameters`.

### 2. Function selection with constrained candidates

The model is not allowed to emit any arbitrary function name. Instead, the project builds a list of candidate function names and uses a prefix-matching constraint:

- the model generates tokens sequentially,
- at each step only the next tokens compatible with the candidate names are considered,
- the highest-scoring valid token is chosen,
- generation stops when a full candidate name is complete.

This is implemented in `select_candidate_from_context()`. The method encodes all possible candidate strings once and then compares the generated prefix against them step by step. If the current prefix no longer matches any candidate, decoding stops with an error instead of producing garbage output.

### 3. Parameter-level decoding

Once the correct function is selected, the model continues generation inside the `parameters` object.

For each parameter:

- the parameter name is inserted into the JSON structure,
- a type-specific decoder is selected:
  - `number`: numeric candidates are extracted from the user request and the model must choose among them,
  - `boolean`: the model chooses between `true` and `false`,
  - `string`: the model generates a valid JSON string and the code validates that it parses correctly.

This step is essential because it prevents the model from inventing values that were never present in the original prompt.

### 4. String handling and validation

For string parameters, the implementation does not rely on raw free-form generation. Instead, it:

- restricts the first token to valid JSON string-start tokens,
- keeps decoding until a complete string literal is formed,
- verifies the generated text with `json.loads()`,
- normalizes the result again with `json.dumps()` to maintain a stable JSON representation.

This reduces malformed output and helps preserve valid JSON syntax even when the user request includes punctuation, escaping, or quotes.

### 5. Why this approach matters

The constrained decoding approach is the core of the project because it bridges natural-language understanding and structured tool use. It improves reliability by ensuring the LLM emits values that fit the API schema and the user request, while also making the output easier to consume by downstream systems or agents.

## Design decisions

Several key implementation choices were made to make the project robust and understandable.

### Schema-first design

The project assumes a known function schema. This is not an open-ended capability; it is a controlled environment where the model acts as an argument extractor and router. That design reduces ambiguity and yields much more reliable function calls than free-form tool generation.

### Local model usage

The model is loaded through a lightweight SDK built around Hugging Face `transformers`. This keeps the project self-contained and executable without external paid APIs. It also makes the system accessible for experimentation in a local development environment.

### Type-aware argument extraction

Each parameter is handled according to its type:

- numbers are taken from the user request,
- booleans are constrained to a binary choice,
- strings are validated as JSON strings.

This choice makes the output deterministic and prevents invalid guesses.

### Validation before acceptance

Generated outputs are checked by parsing JSON and validating the final structure. This is important because the model can sometimes produce nearly-correct text but still fail structurally. Keeping validation in the loop helps detect and reject malformed payloads early.

### Minimal but readable implementation

The code avoids excessive abstraction and keeps the logic easy to follow. This is helpful for a teaching project and for future extension to additional function types or more complex schemas.

## Instructions

### Prerequisites

- Python 3.12 or newer
- `uv` installed and available on the system path
- Internet access for the first model download from Hugging Face

### Installation

From the project root:

```bash
make install
```

This installs the project dependencies and the workspace-local LLM SDK.

### Execution

To run the main pipeline on the bundled sample data:

```bash
make run
```

This loads the function definitions from `data/input/functions_definition.json`, reads the prompts from `data/input/function_calling_tests.json`, generates function calls, and writes the result to `data/output/function_calls.json`.

### Custom execution

You can also pass custom file paths explicitly:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

### Linting

The project includes linting commands:

```bash
make lint
make lint-strict
```

## Example usage

### Example 1: multiplication

Command:

```bash
uv run python -m src --input data/input/function_calling_tests.json
```

Typical output fragment:

```json
{
  "prompt": "What is the product of 3 and 5?",
  "name": "fn_multiply_numbers",
  "parameters": {
    "a": 3,
    "b": 5
  }
}
```

### Example 2: string replacement

Input prompt:

```text
Replace all vowels in 'Programming is fun' with asterisks
```

Expected function call:

```json
{
  "prompt": "Replace all vowels in 'Programming is fun' with asterisks",
  "name": "fn_substitute_string_with_regex",
  "parameters": {
    "source_string": "Programming is fun",
    "regex": "[aeiou]",
    "replacement": "*"
  }
}
```

### Example 3: boolean query

Input prompt:

```text
Is 4 an even number?
```

Typical output:

```json
{
  "prompt": "Is 4 an even number?",
  "name": "fn_is_even",
  "parameters": {
    "n": 4
  }
}
```

## Performance analysis

### Accuracy

The project performs well in a narrow and controlled domain because the model is constrained to a finite set of known functions and parameter types. This makes accuracy depend much more on schema coverage and extraction quality than on free-form generation. In tests based on a curated prompt set, the model is generally able to identify the correct function and fill the parameters correctly when the relevant values are explicit in the prompt.

### Speed

The runtime depends mostly on the size of the local model and the hardware used. On CPU, inference is noticeably slower than on a CUDA-enabled GPU, but the design remains practical for small workloads and experimentation. The code avoids unnecessary repeated work by caching encoded candidate sequences and by minimizing repeated tokenization.

### Reliability

The major reliability improvement comes from constrained decoding. By limiting token choices to valid candidates, the system avoids many common failure modes of LLMs, such as:

- invalid function names,
- incorrect parameter types,
- malformed JSON,
- unsupported hallucinated values.

The trade-off is that the system is intentionally narrow: it works best when the available functions and their expected types are known ahead of time.

## Challenges faced

Several technical difficulties had to be solved during development.

### 1. Invalid JSON generation

A language model can easily output text that is almost valid but not syntactically correct JSON. The solution was to validate the generated string with `json.loads()` and to constrain the output to valid string and number formats.

### 2. Ambiguous numeric extraction

User prompts can include numbers embedded in punctuation, decimals, or natural-language phrases. The project extracts candidate numeric literals from the prompt and forces the model to select from these values instead of inventing a new number.

### 3. Function name ambiguity

Some prompts may match multiple functions semantically. The constrained candidate approach reduces this problem by comparing generated token prefixes against available function names and accepting only valid matches.

### 4. String escaping and regex safety

String arguments often include quotes, backslashes, regex patterns, or special characters. The project handles this by generating JSON strings and normalizing them after parsing, ensuring the final object remains valid and machine-readable.

## Testing strategy

The implementation is validated in a lightweight but meaningful way.

- The repository includes a prompt dataset in `data/input/function_calling_tests.json`.
- Each sample prompt is designed to target a specific function and parameter pattern.
- The project runs the whole pipeline and writes the result to `data/output/function_calls.json`.
- Outputs are inspected structurally to confirm that the final JSON contains:
  - a valid function name,
  - correct parameter names,
  - values of the expected types,
  - a valid JSON object shape.

The project also includes lint targets (`make lint` and `make lint-strict`) to check code quality and type consistency.

## Resources

### Classic references

- Hugging Face Transformers documentation: https://huggingface.co/docs/transformers
- LangChain / function calling and tool-use concepts: https://python.langchain.com/docs/concepts/tool_calling
- OpenAI function calling overview: https://platform.openai.com/docs/guides/function-calling
- JSON Schema basics: https://json-schema.org/understanding-json-schema/
- Introduction to constrained decoding in language models: relevant papers and implementation notes from open NLP and inference research

### AI usage note

AI tools were used as an assistive aid during this project for:

- brainstorming the structure of the constrained decoding pipeline,
- validating the logic of function selection and argument extraction,
- drafting and refining the README content,
- reviewing implementation details for clarity and polish.

The AI was used to support the engineering workflow and documentation process, not to replace the core logic or to generate final model decisions at runtime.

## Conclusion

Call Me Maybe is a compact demonstration of function calling in LLMs. It shows how a small local model can be guided with structured prompts and constrained decoding to map natural-language instructions to valid machine-readable function calls. The goal is not only to produce working examples, but also to make the mechanics transparent and easy to study.

This project is useful as a learning exercise in tool-use, JSON generation, schema-to-text decoding, and the practical trade-offs between flexibility and reliability in modern language models.
