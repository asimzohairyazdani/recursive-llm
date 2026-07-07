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

The default **fast batch** mode first builds the plan with Mistral, unloads it
to free unified memory, and then processes GPT-OSS work bottom-up in batches.
At depth 1 with two children, this reduces the workflow from 8 generated model
calls in classic mode to 3 calls:

```text
1 planner call -> 1 batched solve+critique call -> 1 synthesis call
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
  --max-children 2 \
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
- Keep `Execution mode` on `Fast batch`.
- Keep `Max recursion depth` at `1` for a laptop-friendly first run.
- Use `2–3` children, a `4K` context, and `low` GPT-OSS thinking effort.
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
- `--execution-mode batch`: groups sibling solves and critiques into fewer calls.
- `--batch-size`: maximum number of nodes included in one reasoner request.
- `--think-level low`: shortens GPT-OSS reasoning for much faster responses.
- `--num-ctx 4096`: caps context memory usage.
- `--num-predict 1024`: caps generated tokens per request.

## Recommended M4 24 GB setup

Run Ollama with one loaded model and one request at a time. Fast batch mode
provides concurrency inside a single prompt without multiplying GPT-OSS context
memory:

```bash
unset OLLAMA_MODELS
OLLAMA_MAX_LOADED_MODELS=1 \
OLLAMA_NUM_PARALLEL=1 \
OLLAMA_FLASH_ATTENTION=1 \
OLLAMA_KV_CACHE_TYPE=q8_0 \
ollama serve
```

Then launch the app in another terminal:

```bash
streamlit run streamlit_app.py
```

Start with depth 1, 2 children, batch size 4, 4K context, 1024 output tokens,
and low thinking effort. Avoid parallel GPT-OSS requests on this machine:
Ollama allocates additional context memory for each parallel request. Increasing
depth can still multiply planner calls and prompt size—the recursion goblin has
been contained, not domesticated.
