from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import time

import streamlit as st

from recursive_llm_poc import (
    DEFAULT_OLLAMA_URL,
    DEFAULT_PLANNER_MODEL,
    DEFAULT_REASONER_MODEL,
    Node,
    OllamaClient,
    node_to_dict,
    recursive_solve,
    synthesize_final,
)


st.set_page_config(
    page_title="Recursive LLM POC",
    page_icon="🌀",
    layout="wide",
)


def render_node(node: Node, level: int = 0) -> None:
    confidence = "n/a" if node.confidence is None else f"{node.confidence:.2f}"
    icon = "🧠" if level == 0 else "🌿"
    with st.expander(f"{icon} {node.role}: {node.title}", expanded=level < 2):
        metric_cols = st.columns(3)
        metric_cols[0].metric("Model", node.model)
        metric_cols[1].metric("Confidence", confidence)
        metric_cols[2].metric(
            "More recursion?",
            "yes" if node.needs_more_recursion else "no",
        )

        st.markdown("**Answer**")
        st.write(node.answer or "_No answer returned._")

        st.markdown("**Critique**")
        st.write(node.critique or "_No critique returned._")

        if node.children:
            st.markdown("**Children**")
            for child in node.children:
                render_node(child, level + 1)


def run_recursive_demo(
    task: str,
    planner_model: str,
    reasoner_model: str,
    ollama_url: str,
    max_depth: int,
    max_children: int,
    mock: bool,
    log: callable,
) -> tuple[Node, str, float]:
    started_at = time.perf_counter()
    client = OllamaClient(ollama_url, mock=mock, logger=log)
    root = recursive_solve(
        client=client,
        task=task,
        planner_model=planner_model,
        reasoner_model=reasoner_model,
        depth=0,
        max_depth=max_depth,
        max_children=max_children,
    )
    final_answer = synthesize_final(
        client=client,
        planner_model=planner_model,
        reasoner_model=reasoner_model,
        task=task,
        root=root,
    )
    elapsed = time.perf_counter() - started_at
    return root, final_answer, elapsed


def save_run_log(export: dict) -> Path:
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = runs_dir / f"recursive_llm_run_{timestamp}.json"
    path.write_text(json.dumps(export, indent=2), encoding="utf-8")
    return path


st.title("🌀 Recursive LLM POC")
st.caption(
    "A local Ollama demo that decomposes a task, solves subtasks, critiques results, "
    "and synthesizes a final answer."
)

with st.sidebar:
    st.header("Settings")
    planner_model = st.text_input("Planner model", DEFAULT_PLANNER_MODEL)
    reasoner_model = st.text_input("Reasoner model", DEFAULT_REASONER_MODEL)
    ollama_url = st.text_input("Ollama URL", DEFAULT_OLLAMA_URL)
    max_depth = st.slider("Max recursion depth", min_value=0, max_value=3, value=1)
    max_children = st.slider("Max children per node", min_value=1, max_value=5, value=2)
    mock = st.toggle("Mock mode", value=False)

    st.info(
        "For gpt-oss:20b, try depth 1 + 2 children first. That is already several LLM calls."
    )

task = st.text_area(
    "Task",
    value="Explain recursive LLMs to my manager and propose a small local POC.",
    height=120,
)

run_button = st.button("Run recursive LLM", type="primary")

if run_button:
    if not task.strip():
        st.error("Please enter a task.")
        st.stop()

    logs: list[str] = []
    log_box = st.empty()

    def add_log(message: str) -> None:
        elapsed = time.strftime("%H:%M:%S")
        logs.append(f"[{elapsed}] {message}")
        log_box.code("\n".join(logs[-80:]), language="text")

    with st.status("Running recursive loop...", expanded=True) as status:
        add_log("🚀 Starting recursive run")
        add_log(
            f"Settings: planner={planner_model.strip()}, reasoner={reasoner_model.strip()}, "
            f"depth={max_depth}, children={max_children}, mock={mock}"
        )
        try:
            root_node, final_answer_text, elapsed_seconds = run_recursive_demo(
                task=task.strip(),
                planner_model=planner_model.strip(),
                reasoner_model=reasoner_model.strip(),
                ollama_url=ollama_url.strip(),
                max_depth=max_depth,
                max_children=max_children,
                mock=mock,
                log=add_log,
            )
        except Exception as exc:
            add_log(f"❌ Run failed: {exc}")
            status.update(label="Run failed", state="error")
            st.error(str(exc))
            st.stop()
        add_log(f"✅ Finished in {elapsed_seconds:.1f}s")
        status.update(label="Recursive loop complete", state="complete")

    st.success(f"Completed in {elapsed_seconds:.1f}s")

    final_tab, trace_tab, json_tab = st.tabs(
        ["Final answer", "Recursive trace", "JSON export"]
    )

    with final_tab:
        st.subheader("Final Answer")
        st.write(final_answer_text or "_No final answer returned._")

    with trace_tab:
        st.subheader("Recursive Trace")
        render_node(root_node)

    with json_tab:
        export = {
            "task": task.strip(),
            "planner_model": planner_model.strip(),
            "reasoner_model": reasoner_model.strip(),
            "max_depth": max_depth,
            "max_children": max_children,
            "mock": mock,
            "logs": logs,
            "final_answer": final_answer_text,
            "trace": node_to_dict(root_node),
        }
        saved_path = save_run_log(export)
        st.info(f"Saved run log to `{saved_path}`")
        st.download_button(
            "Download trace JSON",
            data=json.dumps(export, indent=2),
            file_name="recursive_llm_trace.json",
            mime="application/json",
        )
        st.json(export)
else:
    st.markdown(
        """
        ### What this demonstrates

        1. **Planner** breaks a complex task into small subtasks.
        2. **Reasoner** solves each node using your stronger local model.
        3. **Critic** scores the answer and flags whether more recursion is needed.
        4. **Synthesizer** merges the recursive work into a final answer.

        Use **Mock mode** first if Ollama is not running.
        """
    )
