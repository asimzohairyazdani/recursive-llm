# Recursive LLM POC with Ollama

This is a local proof of concept for a **Recursive LLM** workflow designed to run efficiently on local workstation hardware (e.g. Apple Silicon M-series Macs with 24 GB Unified Memory). 

The orchestration loop recursively decomposes a complex task, solves subtasks, critiques answers, and synthesizes a final response:

```text
User Task + Optional Image
  -> Planner (Mistral) decomposes into subtasks
  -> Solver (Gemma 4) solves subtasks recursively (incorporating vision)
  -> Critic (GPT-OSS or DeepSeek-Cloud) reviews answers
  -> Synthesizer compiles the final response
  -> Final Answer + Structured Visual Trace
```

---

## 🌟 Key Features

### 1. Three-Tier Model Architecture
Uses different specialized models for different stages of the recursion:
*   **Planner (`mistral:latest`)**: Fast 7B model that parses and breaks the main task into structured JSON subtasks.
*   **Solver (`gemma4:12b`)**: 12B parameter model with **Multimodal Vision** capabilities to solve text and image-based tasks.
*   **Critic / Synthesizer (`gpt-oss:20b` or `deepseek-v3.1:671b-cloud`)**: High-powered reasoning models that evaluate answers, score them, and compile the final summary report.

### 2. Multimodal Vision Support
Upload system architecture diagrams, database schemas, or flowcharts directly in the UI. The Vision Solver (`gemma4`) will read the pixels and analyze the visual details recursively.

### 3. Live Model Feed & Streaming
Watch the model work in real-time! The UI features a side-by-side layout:
*   **Left Column**: Live execution logs (timings, model load/unload operations).
*   **Right Column**: Real-time token streaming showing the model's `<think>` reasoning thoughts and answers as they generate, with auto-wrapping.

### 4. VRAM Swapping Optimization
Since three models exceed 24 GB RAM, the orchestrator automatically unloads the planner model from VRAM before loading the solver model, preventing memory thrashing and keeping the workstation responsive.

---

## 🚀 Getting Started

### 1. Run Ollama
Make sure Ollama is running and your models are available:
```bash
ollama list
ollama serve
```

### 2. CLI Execution (Mock mode)
You can test the recursive trace format instantly using mock mode:
```bash
python3 recursive_llm_poc.py --mock "Explain recursive LLMs to my manager"
```

### 3. Run the Dashboard UI
Install requirements:
```bash
python3 -m pip install -r requirements.txt
```

Launch the Streamlit app:
```bash
streamlit run streamlit_app.py
```

---

## ⚙️ Configuration & Hardware Tuning (For 24 GB Macs)

To get the absolute best performance on 24 GB Unified Memory, configure the sidebar settings as follows:

*   **Hybrid Local + Cloud Setup**:
    *   **Solver**: Select `Gemma 4 (12B)` (so it can read the uploaded screenshots).
    *   **Critic**: Select a cloud proxy (like `deepseek-v3.1:671b-cloud`). Since the critic is in the cloud, Gemma 4 never has to unload from local VRAM, resulting in **instant response times** with no swap delay!
*   **Pure Local Setup**:
    *   Set **Solver** and **Critic** both to `Gemma 4 (12B)`. This keeps Gemma loaded in memory for both stages, avoiding model swap delays.
*   **Context Window**: Keep set to **`8192`** when vision is active, as visual tokens require significant memory space.
*   **Thinking Effort**: Keep set to **`low`** for rapid testing, and **`medium` / `high`** for deep visual inspection.

---

## 📺 Demo Video
A full walkthrough demonstration is available in:
`demo/Screen Recording 2026-07-07 at 9.23.35 AM.mov`
