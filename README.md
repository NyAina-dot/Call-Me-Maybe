*This project has been created as part of the 42 curriculum by nyrajaon.*

# Call-Me-Maybe

## Description

Call-Me-Maybe is a command-line function-calling experiment based on a
small causal language model. Given a natural-language prompt, it:

1. Selects a function name from a JSON function catalog.
2. Generates values for that function's parameters.
3. Writes a validated JSON list of function calls.

The project uses constrained decoding: at every generation step, only
tokens allowed by the selected function name or parameter grammar receive
model logits. This reduces malformed output while keeping the model
responsible for choosing among valid continuations.

The default function catalog contains arithmetic, greeting, string
reversal, square-root, and regular-expression substitution examples.
Supported parameter types are `number`, `integer`, `string`, and
`boolean`.

## Algorithm Explanation

### Function-name selection

`FunctionSelector` encodes the function catalog and the user prompt, then
builds a token trie from the canonical token ids of the available function
names. At each step, `Tokenizer.get_allowed_tokens()` returns only the
children of the current trie node. `Tokenizer.mask_logits()` gives every
disallowed token negative infinity, and the implementation selects the
highest-logit allowed token.

Generation stops once the generated token sequence reaches a complete
name. Because the trie only contains the declared function names, it is
not possible to produce a name outside the catalog.

### Parameter generation

`ParameterGenerator` writes each parameter name and separator directly
into the context. It then generates only the parameter value according to
its declared type:

- **Numbers:** a candidate token is retained only if the text remains a
  valid integer or decimal-number prefix. JSON separators may terminate a
  complete number.
- **Strings:** generated text must begin with a quote and remain a valid
  JSON string prefix, including escape and Unicode-escape rules. Closing
  quote tokens are allowed once the string is complete.
- **Booleans:** only token continuations of `true` or `false` are allowed.
  Generation ends as soon as the complete literal is reached.
- **Unknown types:** an unsupported parameter type raises `ValueError`
  instead of silently producing an invalid call.

The generated value is appended to the context before generating the next
parameter. Finally, Pydantic models validate the function definition,
input prompt, and generated result before the JSON is written.

## Design Decisions

- **Local small model:** the project uses
  `Qwen/Qwen3-0.6B` by default through the local Hugging Face SDK wrapper.
  A 0.6B model keeps experimentation accessible while testing constrained
  decoding behavior.
- **CPU/CUDA/MPS selection:** `Small_LLM_Model` prefers CUDA, then MPS,
  and otherwise uses CPU. It uses float16 on GPU and MPS and float32 on
  CPU for compatibility.
- **JSON-first interface:** function definitions and prompts are loaded
  from JSON files, which makes the program easy to automate and keeps
  inputs independent from the decoding code.
- **Pydantic validation:** strict Pydantic models reject extra fields and
  validate the declared parameter types. This provides a clear boundary
  between model output and persisted results.
- **Per-type constraints:** constraints are kept close to the generators
  rather than using one generic parser. This keeps the valid JSON rules
  explicit and avoids accepting malformed numbers or strings.
- **Streaming JSON output:** results are written as each prompt is
  processed. The output directory is created automatically, and progress
  is flushed to the file for reliable partial results.

## Instructions

### Requirements

- Python 3.10 or newer
- `uv`
- Internet access on the first run if the Hugging Face model and vocabulary
  files are not already cached

### Installation

From the repository root:

```bash
uv sync
```

The Makefile can also install dependencies, cache paths, and a dedicated
environment:

```bash
make install
```

### Run

With no arguments, the program reads:

- `data/input/functions_definition.json`
- `data/input/function_calling_tests.json`

and writes:

- `data/output/function_calling_results.json`

```bash
uv run python -m src
```

Custom paths can be supplied:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/custom_results.json
```

The Makefile shortcuts are:

```bash
make run ARGS="--output data/output/custom_results.json"
make run OUTPUT=data/output/custom_results.json
make run FUNCTIONS_DEFINITION=data/input/functions_definition.json \
  INPUT=data/input/function_calling_tests.json \
  OUTPUT=data/output/custom_results.json
```

### Linting

Run the project lint and type checks with:

```bash
make lint
```

For stricter mypy checking:

```bash
make lint-strict
```

## Example Usage

The default input contains prompts such as:

```json
{"prompt": "What is the sum of 265 and 345?"}
```

Running the program against the default files produces an entry with a
result shape similar to:

```json
[
  {
    "prompt": "What is the sum of 265 and 345?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 265,
      "b": 345
    }
  }
]
```

The exact model-generated values depend on the runtime model and hardware,
while the constrained structure remains based on the function catalog.

Another example input:

```json
{"prompt": "Greet shrek"}
```

should select `fn_greet` and generate a string value for the `name`
parameter.

## Performance Analysis

The current implementation prioritizes correctness and predictability over
batch throughput. Function-name generation stops after a small number of
token selections because names come from a trie. Parameter generation is
also bounded: numbers use at most 20 steps, strings use at most 50 steps,
and booleans use at most 5 steps.

Accuracy depends mainly on model quality, prompt clarity, and the exact
function descriptions. Constrained decoding guarantees that selected names
and JSON syntax shapes remain valid, but it cannot guarantee that a model
extracts the user's intent perfectly: for example, it cannot prove that a
generated number was the number requested.

Speed is dominated by model inference, especially the forward pass used
for logits at each decoding step and initialization of the local model.
Masking a logit list is inexpensive by comparison, but each additional
generated token invokes the model again. Reliability is improved by
Pydantic validation, bounded generation loops, and explicit errors when no
valid token or complete value exists.

No stable benchmark numbers are currently published for this project.
Useful future measurements are end-to-end time per prompt, model loading
time, tokens generated per parameter, and function/parameter accuracy over
a fixed test fixture.

## Challenges Faced

- **Valid versus complete numbers:** decimal and integer prefixes were
  deliberately distinguished, and negative numbers were detected in the
  prompt to guide the first number token.
- **JSON string escaping:** unconstrained string generation can create
  invalid quotes or malformed escapes. `is_valid_json_string_prefix()`
  validates escapes and control characters as generation proceeds.
- **Function-name boundaries:** tokenized function names do not necessarily
  align with character boundaries. A token trie solves this by constraining
  the canonical token sequence rather than matching raw strings.
- **Booleans and short literals:** boolean generation is limited to prefixes
  of `true` and `false`, so it can stop once the literal is complete.
- **Model repetition:** bounded string generation and repeated-suffix
  detection prevent a small model from continuing indefinitely for regex
  values.
- **Input and output safety:** strict Pydantic schemas and automatic
  output-directory creation reject malformed fixture structures and avoid
  writing to an absent parent directory.

## Testing Strategy

There is currently no dedicated automated test suite. Validation is based
on fixture-driven CLI runs and static checks:

1. Run the default command and inspect
   `data/output/function_calling_results.json`.
2. Verify each result's `prompt`, selected `name`, and `parameters`.
3. Check that every selected name exists in `functions_definition.json`.
4. Verify that parameter values match their declared Pydantic types.
5. Run custom inputs for numbers, strings, booleans, integers, and
   unsupported parameter types.
6. Run `make lint` to execute Flake8 and mypy for source quality.

For reliable regression coverage, future work should add unit tests for
number prefixes, JSON-string prefixes, boolean prefixes, token trie
completion, Pydantic validation, and end-to-end CLI fixtures.

## Resources

- [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers)
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Qwen3 documentation](https://qwen.readthedocs.io/)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [Python `argparse` documentation](https://docs.python.org/3/library/argparse.html)
- [JSON data interchange format](https://www.json.org/json-en.html)
- [uv package and project manager](https://docs.astral.sh/uv/)
- [Flake8 documentation](https://flake8.pycqa.org/)
- [mypy documentation](https://mypy.readthedocs.io/)

### AI Usage

AI assistance was used as a development aid for:

- Reviewing Python implementation choices and constrained-decoding logic.
- Identifying missing public-method docstrings and editing style issues.
- Drafting and organizing this README.
- Reviewing lint, type-check, and CLI usage considerations.

The runtime behavior is implemented in the repository and uses the
Hugging Face language model itself to infer function names and parameter
values. AI-generated suggestions were reviewed and validated against the
source code and project files before being retained.
