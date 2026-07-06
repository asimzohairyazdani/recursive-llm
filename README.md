# Recursive LLM POC with Ollama

This is a small proof of concept for a "recursive LLM" workflow using two local
Ollama models:

- `mistral` as the fast planner/decomposer
- `gpt-oss:20b` as the deeper solver, critic, and synthesizer

The important idea is not that the model weights are recursive. The recursion
lives in the orchestration loop:

```text
user task
  -> planner decomposes into subtasks
  -> reasoner solves each subtask
  -> reasoner critiques each answer
  -> synthesizer merges the recursive work
  -> final answer + visible trace
```

## Run with Ollama

Make sure Ollama is running and your models are available:

```bash
ollama list
ollama serve
```

Then run:

```bash
python3 recursive_llm_poc.py \
  --planner-model mistral \
  --reasoner-model gpt-oss:20b \
  --max-depth 1 \
  --max-children 3 \
  "Design a customer support chatbot architecture for a bank"
```

If your GPT-OSS model has a different local name, replace `gpt-oss:20b` with the
exact name from `ollama list`.

## Run without Ollama

Use mock mode to verify the script and output format:

```bash
python3 recursive_llm_poc.py --mock "Explain recursive LLMs to my manager"
```

## Streamlit UI

Install the UI dependency:

```bash
python3 -m pip install -r requirements.txt
```

Launch the dashboard:

```bash
streamlit run streamlit_app.py
```

In the sidebar:

- Set `Planner model` to your fast model, for example `mistral`.
- Set `Reasoner model` to your thinking model, for example `gpt-oss:20b`.
- Keep `Max recursion depth` at `1` for a laptop-friendly first run.
- Turn on `Mock mode` if Ollama is not running yet.

The UI shows:

- the final synthesized answer
- expandable recursive trace nodes
- confidence and critique per node
- downloadable JSON trace

## Useful knobs

- `--max-depth`: controls how many recursive decomposition levels to run.
- `--max-children`: controls how many subtasks each planner step creates.
- `--planner-model`: fast model used to split work into subtasks.
- `--reasoner-model`: stronger model used for solving, critique, and final answer.

For a laptop-friendly demo, start with `--max-depth 1`. Increasing depth can
multiply model calls quickly, the tiny recursion goblin that it is.
