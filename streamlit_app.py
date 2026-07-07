from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import time

import streamlit as st

from recursive_llm_poc import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_CTX,
    DEFAULT_NUM_PREDICT,
    DEFAULT_OLLAMA_URL,
    DEFAULT_PLANNER_MODEL,
    DEFAULT_SOLVER_MODEL,
    DEFAULT_CRITIC_MODEL,
    Node,
    OllamaClient,
    node_to_dict,
    recursive_solve,
    run_batched_workflow,
    synthesize_final,
)


st.set_page_config(
    page_title="Recursive LLM POC",
    page_icon="🌀",
    layout="wide",
)


@st.cache_data(ttl=30)
def get_available_models(base_url: str) -> list[str]:
    try:
        import urllib.request
        url = f"{base_url.rstrip('/')}/api/tags"
        request = urllib.request.Request(url)
        with urllib.request.urlopen(request, timeout=2) as response:
            data = json.loads(response.read().decode("utf-8"))
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def render_node(node: Node, level: int = 0) -> None:
    confidence = "n/a" if node.confidence is None else f"{node.confidence:.2f}"
    icon = "🧠" if level == 0 else "🌿"
    with st.expander(f"{icon} {node.role}: {node.title}", expanded=level < 2):
        metric_cols = st.columns(4)
        metric_cols[0].metric("Solver Model", node.solver_model)
        metric_cols[1].metric("Critic Model", node.critic_model)
        metric_cols[2].metric("Confidence", confidence)
        metric_cols[3].metric(
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
    solver_model: str,
    critic_model: str,
    ollama_url: str,
    max_depth: int,
    max_children: int,
    execution_mode: str,
    batch_size: int,
    num_ctx: int,
    num_predict: int,
    think_level: str,
    mock: bool,
    log: callable,
    images: list[str] | None = None,
    token_callback: Callable[[str], None] | None = None,
) -> tuple[Node, str, float, int]:
    started_at = time.perf_counter()
    client = OllamaClient(
        ollama_url,
        mock=mock,
        logger=log,
        num_ctx=num_ctx,
        num_predict=num_predict,
        think_level=think_level,
        token_callback=token_callback,
    )
    if execution_mode == "Fast batch":
        root, final_answer = run_batched_workflow(
            client=client,
            task=task,
            planner_model=planner_model,
            solver_model=solver_model,
            critic_model=critic_model,
            max_depth=max_depth,
            max_children=max_children,
            batch_size=batch_size,
            images=images,
        )
    else:
        root = recursive_solve(
            client=client,
            task=task,
            planner_model=planner_model,
            solver_model=solver_model,
            critic_model=critic_model,
            depth=0,
            max_depth=max_depth,
            max_children=max_children,
            images=images,
        )
        final_answer = synthesize_final(
            client=client,
            synthesizer_model=critic_model,
            task=task,
            root=root,
        )
    elapsed = time.perf_counter() - started_at
    return root, final_answer, elapsed, client.call_count


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
    ollama_url = st.text_input("Ollama URL", DEFAULT_OLLAMA_URL)
    
    # Fetch models dynamically
    available_models = get_available_models(ollama_url)
    
    if available_models:
        default_planner = DEFAULT_PLANNER_MODEL if DEFAULT_PLANNER_MODEL in available_models else (
            "mistral:latest" if "mistral:latest" in available_models else available_models[0]
        )
        planner_model_sel = st.selectbox(
            "Planner model",
            available_models,
            index=available_models.index(default_planner) if default_planner in available_models else 0
        )
        custom_planner = st.checkbox("Use custom planner model name")
        if custom_planner:
            planner_model = st.text_input("Custom planner name", planner_model_sel)
        else:
            planner_model = planner_model_sel
            
        # Build option lists
        model_options = []
        gemma_tag = "gemma4:12b" if "gemma4:12b" in available_models else None
        if not gemma_tag:
            gemma_matches = [m for m in available_models if "gemma4" in m.lower()]
            if gemma_matches:
                gemma_tag = gemma_matches[0]
        if gemma_tag:
            model_options.append(f"Gemma 4 (12B) [{gemma_tag}]")
            
        gpt_oss_tag = "gpt-oss:20b" if "gpt-oss:20b" in available_models else None
        if not gpt_oss_tag:
            gpt_oss_matches = [m for m in available_models if "gpt-oss" in m.lower()]
            if gpt_oss_matches:
                gpt_oss_tag = gpt_oss_matches[0]
        if gpt_oss_tag:
            model_options.append(f"GPT-OSS (20B) [{gpt_oss_tag}]")
            
        for m in available_models:
            if m != gemma_tag and m != gpt_oss_tag and m != default_planner:
                model_options.append(m)
                
        model_options.append("Custom model...")
        
        # Solver selector
        default_solver_idx = 0
        for idx, opt in enumerate(model_options):
            if gemma_tag and gemma_tag in opt:
                default_solver_idx = idx
                break
        solver_sel = st.selectbox("Solver model", model_options, index=default_solver_idx)
        if solver_sel == "Custom model...":
            solver_model = st.text_input("Custom solver name", "gemma4:12b")
        else:
            if "[" in solver_sel and "]" in solver_sel:
                solver_model = solver_sel.split("[")[1].split("]")[0]
            else:
                solver_model = solver_sel

        # Critic selector
        default_critic_idx = 0
        for idx, opt in enumerate(model_options):
            if gpt_oss_tag and gpt_oss_tag in opt:
                default_critic_idx = idx
                break
        critic_sel = st.selectbox("Critic model", model_options, index=default_critic_idx)
        if critic_sel == "Custom model...":
            critic_model = st.text_input("Custom critic name", "gpt-oss:20b")
        else:
            if "[" in critic_sel and "]" in critic_sel:
                critic_model = critic_sel.split("[")[1].split("]")[0]
            else:
                critic_model = critic_sel
    else:
        planner_model = st.text_input("Planner model", DEFAULT_PLANNER_MODEL)
        solver_model = st.text_input("Solver model", DEFAULT_SOLVER_MODEL)
        critic_model = st.text_input("Critic model", DEFAULT_CRITIC_MODEL)

    execution_mode = st.selectbox(
        "Execution mode",
        ("Fast batch", "Classic sequential"),
        help="Fast batch combines sibling solves and critiques into fewer model calls.",
    )
    max_depth = st.slider("Max recursion depth", min_value=0, max_value=3, value=1)
    max_children = st.slider("Max children per node", min_value=1, max_value=5, value=2)
    with st.expander("Performance tuning"):
        batch_size = st.slider(
            "Batch size",
            min_value=1,
            max_value=6,
            value=DEFAULT_BATCH_SIZE,
            disabled=execution_mode != "Fast batch",
        )
        num_ctx = st.select_slider(
            "Context window",
            options=(2048, 4096, 8192, 16384),
            value=DEFAULT_NUM_CTX,
        )
        num_predict = st.select_slider(
            "Output token budget",
            options=(512, 768, 1024, 1536, 2048),
            value=DEFAULT_NUM_PREDICT,
        )
        think_level = st.select_slider(
            "Model thinking effort",
            options=("low", "medium", "high"),
            value="low",
        )
    mock = st.toggle("Mock mode", value=False)

    st.info(
        "M4 24 GB recommendation: Fast batch, depth 1, 2–3 children, 4K context, "
        "and low thinking effort."
    )

import base64

task = st.text_area(
    "Task",
    value="Explain recursive LLMs to my manager and propose a small local POC.",
    height=120,
)

is_vision_supported = "gemma4" in solver_model.lower() or "llava" in solver_model.lower() or "vision" in solver_model.lower()

if is_vision_supported:
    uploaded_file = st.file_uploader(
        "Upload image (optional, Gemma 4 supports vision analysis)",
        type=["png", "jpg", "jpeg"]
    )
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded image", use_column_width=False, width=400)
else:
    uploaded_file = None
    st.warning(f"⚠️ Selected Solver model ({solver_model}) is text-only. Image upload is disabled.")

run_button = st.button("Run recursive LLM", type="primary")

if run_button:
    if not task.strip():
        st.error("Please enter a task.")
        st.stop()

    logs: list[str] = []
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚙️ Execution Logs")
        log_box = st.empty()
        
    with col2:
        st.subheader("📺 Live Model Feed")
        stream_placeholder = st.empty()
        stream_placeholder.info("Waiting for model execution to stream thinking...")

    current_tokens = []
    st.session_state.active_model = "None"

    def on_token(token: str) -> None:
        current_tokens.append(token)
        full_text = "".join(current_tokens)

        if "<think>" in full_text:
            parts = full_text.split("<think>")
            before_think = parts[0]
            after_think_parts = parts[1].split("</think>")

            thinking_content = after_think_parts[0]
            after_think = after_think_parts[1] if len(after_think_parts) > 1 else ""

            formatted = ""
            if before_think.strip():
                formatted += before_think.strip() + "\n\n"

            formatted += f"🤔 *Thinking process:*\n<div style='word-break: break-all; word-wrap: break-word; white-space: pre-wrap; font-style: italic; color: #555555; border-left: 3px solid #ccc; padding-left: 10px; margin: 10px 0;'>{thinking_content}</div>\n\n"

            if after_think.strip():
                formatted += f"💡 *Response:*\n<div style='word-break: break-all; word-wrap: break-word; white-space: pre-wrap;'>{after_think}</div>"
        else:
            formatted = f"<div style='word-break: break-all; word-wrap: break-word; white-space: pre-wrap;'>{full_text}</div>"

        model_header = f"🤖 **Active Model: `{st.session_state.get('active_model', 'Unknown')}`**\n\n---\n\n"
        stream_placeholder.markdown(model_header + formatted, unsafe_allow_html=True)

    def add_log(message: str) -> None:
        elapsed = time.strftime("%H:%M:%S")
        logs.append(f"[{elapsed}] {message}")
        log_box.code("\n".join(logs[-80:]), language="text")
        if "▶️ Calling" in message:
            current_tokens.clear()
            model_name = message.split("Calling ")[1].split(" with")[0]
            st.session_state.active_model = model_name
            stream_placeholder.info(f"Streaming output from {model_name}...")

    base64_images = None
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        base64_str = base64.b64encode(file_bytes).decode("utf-8")
        base64_images = [base64_str]

    with st.status("Running recursive loop...", expanded=True) as status:
        add_log("🚀 Starting recursive run")
        add_log(
            f"Settings: planner={planner_model.strip()}, solver={solver_model.strip()}, "
            f"critic={critic_model.strip()}, mode={execution_mode}, depth={max_depth}, "
            f"children={max_children}, batch={batch_size}, context={num_ctx}, "
            f"think={think_level}, mock={mock}, image_uploaded={uploaded_file is not None}"
        )
        try:
            root_node, final_answer_text, elapsed_seconds, call_count = run_recursive_demo(
                task=task.strip(),
                planner_model=planner_model.strip(),
                solver_model=solver_model.strip(),
                critic_model=critic_model.strip(),
                ollama_url=ollama_url.strip(),
                max_depth=max_depth,
                max_children=max_children,
                execution_mode=execution_mode,
                batch_size=batch_size,
                num_ctx=num_ctx,
                num_predict=num_predict,
                think_level=think_level,
                mock=mock,
                log=add_log,
                images=base64_images,
                token_callback=on_token,
            )
        except Exception as exc:
            add_log(f"❌ Run failed: {exc}")
            status.update(label="Run failed", state="error")
            st.error(str(exc))
            st.stop()
        add_log(f"✅ Finished in {elapsed_seconds:.1f}s using {call_count} LLM calls")
        status.update(label="Recursive loop complete", state="complete")

    metric_cols = st.columns(2)
    metric_cols[0].metric("Elapsed", f"{elapsed_seconds:.1f}s")
    metric_cols[1].metric("LLM calls", call_count)

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
            "solver_model": solver_model.strip(),
            "critic_model": critic_model.strip(),
            "max_depth": max_depth,
            "max_children": max_children,
            "execution_mode": execution_mode,
            "batch_size": batch_size,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "think_level": think_level,
            "llm_calls": call_count,
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
        2. **Solver** solves each subtask node.
        3. **Critic** scores the answer and flags whether more recursion is needed.
        4. **Synthesizer** merges the recursive work into a final answer.

        Use **Mock mode** first if Ollama is not running.
        """
    )
