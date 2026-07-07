#!/usr/bin/env python3
"""Small recursive LLM POC using local Ollama models.

The POC uses a fast model as a planner/summarizer and a stronger thinking model
as solver/critic. It prints a visible recursion tree so the orchestration is easy
to demo and reason about.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable


DEFAULT_PLANNER_MODEL = "mistral"
DEFAULT_SOLVER_MODEL = "gemma4:12b"
DEFAULT_CRITIC_MODEL = "gpt-oss:20b"
DEFAULT_REASONER_MODEL = DEFAULT_SOLVER_MODEL # backward compatibility
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_BATCH_SIZE = 4
DEFAULT_NUM_CTX = 8192
DEFAULT_NUM_PREDICT = 1024


@dataclass
class Node:
    title: str
    role: str
    solver_model: str
    critic_model: str
    prompt: str
    answer: str = ""
    critique: str = ""
    confidence: float | None = None
    needs_more_recursion: bool = False
    children: list["Node"] = field(default_factory=list)


def node_to_dict(node: Node) -> dict[str, Any]:
    return {
        "title": node.title,
        "role": node.role,
        "solver_model": node.solver_model,
        "critic_model": node.critic_model,
        "prompt": node.prompt,
        "answer": node.answer,
        "critique": node.critique,
        "confidence": node.confidence,
        "needs_more_recursion": node.needs_more_recursion,
        "children": [node_to_dict(child) for child in node.children],
    }


LogCallback = Callable[[str], None]


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        mock: bool = False,
        logger: LogCallback | None = None,
        num_ctx: int = DEFAULT_NUM_CTX,
        num_predict: int = DEFAULT_NUM_PREDICT,
        think_level: str = "low",
        keep_alive: str = "15m",
        token_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.mock = mock
        self.logger = logger
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.think_level = think_level
        self.keep_alive = keep_alive
        self.token_callback = token_callback
        self.call_count = 0

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        json_format: bool = False,
        num_predict: int | None = None,
        keep_alive: str | int | None = None,
        images: list[str] | None = None,
    ) -> str:
        started_at = time.perf_counter()
        self.call_count += 1
        self.log(f"▶️ Calling {model} with {len(prompt):,} prompt chars")
        if self.mock:
            response = self._mock_generate(model, prompt)
            self.log(
                f"✅ {model} mock response in {time.perf_counter() - started_at:.1f}s "
                f"({len(response):,} chars)"
            )
            return response

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": self.token_callback is not None,
            "keep_alive": self.keep_alive if keep_alive is None else keep_alive,
            "options": {
                "temperature": 0.1,
                "num_ctx": self.num_ctx,
                "num_predict": num_predict or self.num_predict,
            },
        }
        if images:
            payload["images"] = images
        if json_format:
            is_thinking = model.lower().startswith("gpt-oss") or "gemma4" in model.lower() or "deepseek" in model.lower()
            if not is_thinking:
                payload["format"] = "json"
        if model.lower().startswith("gpt-oss") or "gemma4" in model.lower() or "deepseek" in model.lower():
            payload["think"] = self.think_level
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                if self.token_callback is not None:
                    full_response = []
                    eval_count = 0
                    eval_duration = 0
                    load_duration = 0
                    for line in response:
                        if line:
                            chunk = json.loads(line.decode("utf-8"))
                            token = chunk.get("response", "")
                            full_response.append(token)
                            self.token_callback(token)
                            if chunk.get("done", False):
                                eval_count = int(chunk.get("eval_count", 0))
                                eval_duration = int(chunk.get("eval_duration", 0)) / 1_000_000_000
                                load_duration = int(chunk.get("load_duration", 0)) / 1_000_000_000
                    output = "".join(full_response).strip()
                else:
                    data = json.loads(response.read().decode("utf-8"))
                    output = str(data.get("response", "")).strip()
                    eval_count = int(data.get("eval_count", 0))
                    eval_duration = int(data.get("eval_duration", 0)) / 1_000_000_000
                    load_duration = int(data.get("load_duration", 0)) / 1_000_000_000
        except urllib.error.HTTPError as exc:
            try:
                err_body = exc.read().decode("utf-8")
                err_msg = json.loads(err_body).get("error", err_body)
            except Exception:
                err_msg = str(exc)
            raise RuntimeError(
                f"Ollama returned HTTP error {exc.code}: {err_msg}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.base_url}. "
                "Start it with `ollama serve` or rerun with `--mock`."
            ) from exc

        elapsed = time.perf_counter() - started_at
        speed = eval_count / eval_duration if eval_duration else 0.0
        metrics = f", load {load_duration:.1f}s, {speed:.1f} tok/s" if eval_count else ""
        self.log(f"✅ {model} response in {elapsed:.1f}s ({len(output):,} chars{metrics})")
        return output

    def unload(self, model: str) -> None:
        if self.mock:
            self.log(f"⏏️ Mock unload {model}")
            return
        payload = {"model": model, "prompt": "", "stream": False, "keep_alive": 0}
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120):
            pass
        self.log(f"⏏️ Unloaded {model} to free unified memory")

    def log(self, message: str) -> None:
        if self.logger:
            self.logger(message)

    def _mock_generate(self, model: str, prompt: str) -> str:
        prompt_lower = prompt.lower()
        if "return json only" in prompt_lower and '"results"' in prompt_lower:
            identifiers = sorted({int(value) for value in re.findall(r'"id":\s*(\d+)', prompt)})
            if "critique" in prompt_lower:
                return json.dumps(
                    {
                        "results": [
                            {
                                "id": identifier,
                                "critique": f"Mock critique for task {identifier} from {model}.",
                                "confidence": 0.85,
                                "needs_more_recursion": False,
                            }
                            for identifier in identifiers
                        ]
                    }
                )
            else:
                # Solve batch, or old combined batch
                if "critique" in prompt_lower or '"critique"' in prompt_lower:
                    return json.dumps(
                        {
                            "results": [
                                {
                                    "id": identifier,
                                    "answer": f"Mock batched answer {identifier} from {model}.",
                                    "critique": "Useful and sufficiently specific for the POC.",
                                    "confidence": 0.82,
                                    "needs_more_recursion": False,
                                }
                                for identifier in identifiers
                            ]
                        }
                    )
                else:
                    return json.dumps(
                        {
                            "results": [
                                {
                                    "id": identifier,
                                    "answer": f"Mock batched answer {identifier} from {model}.",
                                }
                                for identifier in identifiers
                            ]
                        }
                    )
        if "return json only" in prompt_lower and "final_answer" in prompt_lower:
            return json.dumps(
                {
                    "final_answer": "Mock final answer: recursively decompose the task, solve subparts, critique them, and merge the results with a max-depth guard."
                }
            )
        if "return json only" in prompt_lower and "subtasks" in prompt_lower:
            return json.dumps(
                {
                    "subtasks": [
                        "Clarify the target outcome",
                        "Design the recursive workflow",
                        "Identify risks and stopping criteria",
                    ]
                }
            )
        if "return json only" in prompt_lower and "critique" in prompt_lower:
            return json.dumps(
                {
                    "critique": "The answer is directionally useful. Add clearer evaluation criteria before production use.",
                    "confidence": 0.78,
                    "needs_more_recursion": False,
                }
            )
        return f"Mock answer from {model}: {compact(prompt, 180)}"


def compact(text: str, limit: int = 500) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def parse_json_object(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                print(f"--- FAILED TO PARSE JSON ---\n{raw}\n----------------------------", file=sys.stderr)
                raise
        print(f"--- FAILED TO PARSE JSON (NO BRACES) ---\n{raw}\n----------------------------", file=sys.stderr)
        raise


def plan_subtasks(client: OllamaClient, model: str, task: str, max_children: int) -> list[str]:
    client.log(f"🧩 Planning up to {max_children} subtasks for: {compact(task, 90)}")
    prompt = f"""
You are the planner in a recursive LLM system.
Break the user's task into {max_children} small, independent subtasks.

Rules:
- Return JSON only.
- Do not include hidden chain-of-thought.
- Use concise task titles.

JSON shape:
{{"subtasks": ["...", "..."]}}

User task:
{task}
"""
    raw = client.generate(
        model,
        textwrap.dedent(prompt).strip(),
        json_format=True,
        num_predict=384,
    )
    data = parse_json_object(raw)
    subtasks = data.get("subtasks", [])
    if not isinstance(subtasks, list) or not subtasks:
        raise ValueError(f"Planner did not return subtasks: {raw}")
    planned = [str(item).strip() for item in subtasks[:max_children] if str(item).strip()]
    client.log(f"🧩 Planned {len(planned)} subtasks: {', '.join(planned)}")
    return planned


def solve_task(client: OllamaClient, model: str, task: str, context: str, images: list[str] | None = None) -> str:
    client.log(f"🧠 Solving: {compact(task, 90)}")
    prompt = f"""
You are the solver in a recursive LLM system.
Solve the task directly and concisely.

Rules:
- Do not reveal hidden chain-of-thought.
- Give the answer plus brief reasoning summary.
- Be practical and specific.

Context:
{context or "No prior context."}

Task:
{task}
"""
    return client.generate(model, textwrap.dedent(prompt).strip(), images=images)


def critique_solution(client: OllamaClient, model: str, task: str, answer: str) -> dict[str, Any]:
    client.log(f"🔍 Critiquing: {compact(task, 90)}")
    prompt = f"""
You are the critic in a recursive LLM system.
Evaluate whether the answer is good enough or needs one more recursive pass.

Rules:
- Return JSON only.
- Do not include hidden chain-of-thought.
- Be strict but practical.

JSON shape:
{{
  "critique": "short critique",
  "confidence": 0.0,
  "needs_more_recursion": false
}}

Task:
{task}

Answer:
{answer}
"""
    raw = client.generate(model, textwrap.dedent(prompt).strip(), json_format=True)
    data = parse_json_object(raw)
    critique = {
        "critique": str(data.get("critique", "")).strip(),
        "confidence": float(data.get("confidence", 0.0)),
        "needs_more_recursion": bool(data.get("needs_more_recursion", False)),
    }
    client.log(
        f"🔍 Critique confidence={critique['confidence']:.2f}, "
        f"needs_more_recursion={critique['needs_more_recursion']}"
    )
    return critique


def synthesize_final(
    client: OllamaClient,
    synthesizer_model: str,
    task: str,
    root: Node,
) -> str:
    client.log("🧵 Synthesizing final answer")
    child_summaries = "\n\n".join(
        f"Subtask: {child.title}\nAnswer: {child.answer}\nCritique: {child.critique}"
        for child in root.children
    )
    prompt = f"""
You are the final synthesizer in a recursive LLM system.
Merge the subtask answers into one coherent final answer for the user.

Rules:
- Return JSON only.
- Do not include hidden chain-of-thought.
- Keep it concise, useful, and demo-friendly.

JSON shape:
{{"final_answer": "..."}}

Original task:
{task}

Recursive work:
{child_summaries}
"""
    raw = client.generate(
        synthesizer_model,
        textwrap.dedent(prompt).strip(),
        json_format=True,
    )
    try:
        return str(parse_json_object(raw).get("final_answer", "")).strip()
    except json.JSONDecodeError:
        return raw.strip()


def build_plan_tree(
    client: OllamaClient,
    task: str,
    planner_model: str,
    solver_model: str,
    critic_model: str,
    depth: int,
    max_depth: int,
    max_children: int,
) -> Node:
    node = Node(
        title=task,
        role="root" if depth == 0 else "subtask",
        solver_model=solver_model,
        critic_model=critic_model,
        prompt=task,
    )
    if depth < max_depth:
        for subtask in plan_subtasks(client, planner_model, task, max_children):
            node.children.append(
                build_plan_tree(
                    client=client,
                    task=subtask,
                    planner_model=planner_model,
                    solver_model=solver_model,
                    critic_model=critic_model,
                    depth=depth + 1,
                    max_depth=max_depth,
                    max_children=max_children,
                )
            )
    return node


def nodes_by_depth(root: Node) -> dict[int, list[Node]]:
    grouped: dict[int, list[Node]] = {}

    def visit(node: Node, depth: int) -> None:
        grouped.setdefault(depth, []).append(node)
        for child in node.children:
            visit(child, depth + 1)

    visit(root, 0)
    return grouped


def solve_and_critique_batch(
    client: OllamaClient,
    model: str,
    nodes: list[Node],
) -> None:
    items = []
    for identifier, node in enumerate(nodes):
        child_context = "\n\n".join(
            f"{child.title}: {child.answer}" for child in node.children
        )
        items.append(
            {
                "id": identifier,
                "task": node.title,
                "context": child_context or "No prior context.",
            }
        )

    client.log(f"📦 Batch-solving and critiquing {len(nodes)} tasks")
    prompt = f"""
You are the solver and critic in a recursive LLM system.
For every item, solve the task concisely, then evaluate your answer.

Rules:
- Return JSON only and include every input id exactly once.
- Do not reveal hidden chain-of-thought.
- Answers must be practical and specific.
- Set needs_more_recursion only when the answer has a major unresolved gap.

JSON shape:
{{"results": [{{"id": 0, "answer": "...", "critique": "...", "confidence": 0.0, "needs_more_recursion": false}}]}}

Items:
{json.dumps(items, ensure_ascii=False)}
"""
    raw = client.generate(
        model,
        textwrap.dedent(prompt).strip(),
        json_format=True,
        num_predict=max(client.num_predict, 320 * len(nodes)),
    )
    data = parse_json_object(raw)
    results = data.get("results", [])
    indexed = {}
    for result in results:
        if isinstance(result, dict) and "id" in result:
            try:
                indexed[int(result["id"])] = result
            except (ValueError, TypeError):
                pass

    if len(indexed) != len(nodes) and len(results) == len(nodes):
        client.log("⚠️ Batch IDs were duplicate or missing. Falling back to positional mapping.")
        indexed = {i: results[i] for i in range(len(nodes))}

    if len(indexed) != len(nodes):
        raise ValueError(f"Batch returned {len(indexed)} of {len(nodes)} results: {raw}")

    for identifier, node in enumerate(nodes):
        result = indexed[identifier]
        node.answer = str(result.get("answer", "")).strip()
        node.critique = str(result.get("critique", "")).strip()
        node.confidence = float(result.get("confidence", 0.0))
        node.needs_more_recursion = bool(result.get("needs_more_recursion", False))


def solve_batch(
    client: OllamaClient,
    model: str,
    nodes: list[Node],
    images: list[str] | None = None,
) -> None:
    items = []
    for identifier, node in enumerate(nodes):
        child_context = "\n\n".join(
            f"{child.title}: {child.answer}" for child in node.children
        )
        items.append(
            {
                "id": identifier,
                "task": node.title,
                "context": child_context or "No prior context.",
            }
        )

    client.log(f"📦 Batch-solving {len(nodes)} tasks using {model}")
    prompt = f"""
You are the solver in a recursive LLM system.
For every item, solve the task concisely.

Rules:
- Return JSON only and include every input id exactly once.
- Do not reveal hidden chain-of-thought.
- Answers must be practical and specific.

JSON shape:
{{"results": [{{"id": 0, "answer": "..."}}]}}

Items:
{json.dumps(items, ensure_ascii=False)}
"""
    raw = client.generate(
        model,
        textwrap.dedent(prompt).strip(),
        json_format=True,
        num_predict=max(client.num_predict, 250 * len(nodes)),
        images=images,
    )
    data = parse_json_object(raw)
    results = data.get("results", [])
    indexed = {}
    for result in results:
        if isinstance(result, dict) and "id" in result:
            try:
                indexed[int(result["id"])] = result
            except (ValueError, TypeError):
                pass

    if len(indexed) != len(nodes) and len(results) == len(nodes):
        client.log("⚠️ Batch IDs were duplicate or missing. Falling back to positional mapping.")
        indexed = {i: results[i] for i in range(len(nodes))}

    if len(indexed) != len(nodes):
        raise ValueError(f"Batch solve returned {len(indexed)} of {len(nodes)} results: {raw}")

    for identifier, node in enumerate(nodes):
        result = indexed[identifier]
        node.answer = str(result.get("answer", "")).strip()


def critique_batch(
    client: OllamaClient,
    model: str,
    nodes: list[Node],
) -> None:
    items = []
    for identifier, node in enumerate(nodes):
        items.append(
            {
                "id": identifier,
                "task": node.title,
                "answer": node.answer,
            }
        )

    client.log(f"🔍 Batch-critiquing {len(nodes)} tasks using {model}")
    prompt = f"""
You are the critic in a recursive LLM system.
For every item, evaluate whether the answer is good enough or needs one more recursive pass.

Rules:
- Return JSON only and include every input id exactly once.
- Do not reveal hidden chain-of-thought.
- Be strict but practical.

JSON shape:
{{"results": [{{"id": 0, "critique": "...", "confidence": 0.0, "needs_more_recursion": false}}]}}

Items:
{json.dumps(items, ensure_ascii=False)}
"""
    raw = client.generate(
        model,
        textwrap.dedent(prompt).strip(),
        json_format=True,
        num_predict=max(client.num_predict, 150 * len(nodes)),
    )
    data = parse_json_object(raw)
    results = data.get("results", [])
    indexed = {}
    for result in results:
        if isinstance(result, dict) and "id" in result:
            try:
                indexed[int(result["id"])] = result
            except (ValueError, TypeError):
                pass

    if len(indexed) != len(nodes) and len(results) == len(nodes):
        client.log("⚠️ Batch IDs were duplicate or missing. Falling back to positional mapping.")
        indexed = {i: results[i] for i in range(len(nodes))}

    if len(indexed) != len(nodes):
        raise ValueError(f"Batch critique returned {len(indexed)} of {len(nodes)} results: {raw}")

    for identifier, node in enumerate(nodes):
        result = indexed[identifier]
        node.critique = str(result.get("critique", "")).strip()
        node.confidence = float(result.get("confidence", 0.0))
        node.needs_more_recursion = bool(result.get("needs_more_recursion", False))


def run_batched_workflow(
    client: OllamaClient,
    task: str,
    planner_model: str,
    solver_model: str,
    critic_model: str,
    max_depth: int,
    max_children: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
    images: list[str] | None = None,
) -> tuple[Node, str]:
    client.log("⚡ Fast mode: plan first, then solve bottom-up in batches")
    root = build_plan_tree(
        client=client,
        task=task,
        planner_model=planner_model,
        solver_model=solver_model,
        critic_model=critic_model,
        depth=0,
        max_depth=max_depth,
        max_children=max_children,
    )

    if planner_model != solver_model and planner_model != critic_model and max_depth > 0:
        client.unload(planner_model)

    grouped = nodes_by_depth(root)
    if max_depth == 0:
        if solver_model == critic_model:
            solve_and_critique_batch(client, solver_model, [root])
        else:
            solve_batch(client, solver_model, [root], images=images)
            client.unload(solver_model)
            critique_batch(client, critic_model, [root])
        return root, root.answer

    for depth in range(max(grouped), 0, -1):
        depth_nodes = grouped[depth]
        client.log(f"🌳 Processing depth {depth} ({len(depth_nodes)} nodes)")
        
        if solver_model == critic_model:
            for start in range(0, len(depth_nodes), batch_size):
                solve_and_critique_batch(
                    client,
                    solver_model,
                    depth_nodes[start : start + batch_size],
                )
        else:
            # 1. Batch solve all nodes at this depth
            for start in range(0, len(depth_nodes), batch_size):
                solve_batch(
                    client,
                    solver_model,
                    depth_nodes[start : start + batch_size],
                    images=images,
                )
            
            # 2. Unload solver before running critic
            client.unload(solver_model)
            
            # 3. Batch critique all nodes at this depth
            for start in range(0, len(depth_nodes), batch_size):
                critique_batch(
                    client,
                    critic_model,
                    depth_nodes[start : start + batch_size],
                )
                
            # 4. Unload critic if there are higher depths to process
            if depth > 1:
                client.unload(critic_model)

    final_answer = synthesize_final(
        client=client,
        synthesizer_model=critic_model,
        task=task,
        root=root,
    )
    root.answer = final_answer
    confidences = [child.confidence for child in root.children if child.confidence is not None]
    root.confidence = sum(confidences) / len(confidences) if confidences else None
    root.critique = "Final synthesis produced from the batched recursive results."
    return root, final_answer


def recursive_solve(
    client: OllamaClient,
    task: str,
    planner_model: str,
    solver_model: str,
    critic_model: str,
    depth: int,
    max_depth: int,
    max_children: int,
    context: str = "",
    images: list[str] | None = None,
) -> Node:
    client.log(f"🌱 Enter depth {depth}: {compact(task, 90)}")
    node = Node(
        title=task,
        role="root" if depth == 0 else "subtask",
        solver_model=solver_model,
        critic_model=critic_model,
        prompt=task,
    )

    if depth < max_depth:
        subtasks = plan_subtasks(client, planner_model, task, max_children)
        for subtask in subtasks:
            child = recursive_solve(
                client=client,
                task=subtask,
                planner_model=planner_model,
                solver_model=solver_model,
                critic_model=critic_model,
                depth=depth + 1,
                max_depth=max_depth,
                max_children=max_children,
                context=context,
                images=images,
            )
            node.children.append(child)

    child_context = "\n\n".join(f"{child.title}: {child.answer}" for child in node.children)
    
    # Solve task
    node.answer = solve_task(client, solver_model, task, context or child_context, images=images)
    
    # Unload solver before critique if they are different to optimize RAM
    if solver_model != critic_model:
        client.unload(solver_model)

    # Critique solution
    critique = critique_solution(client, critic_model, task, node.answer)
    node.critique = critique["critique"]
    node.confidence = critique["confidence"]
    node.needs_more_recursion = critique["needs_more_recursion"] and depth < max_depth
    
    # Unload critic after critique if different to prepare for next steps
    if solver_model != critic_model:
        client.unload(critic_model)

    client.log(f"🏁 Exit depth {depth}: {compact(task, 90)}")
    return node


def print_tree(node: Node, indent: str = "") -> None:
    confidence = "n/a" if node.confidence is None else f"{node.confidence:.2f}"
    print(f"{indent}- [{node.role}] {node.title}")
    print(f"{indent}  solver: {node.solver_model} | critic: {node.critic_model} | confidence: {confidence}")
    print(f"{indent}  answer: {compact(node.answer, 260)}")
    if node.critique:
        print(f"{indent}  critique: {compact(node.critique, 220)}")
    for child in node.children:
        print_tree(child, indent + "  ")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recursive LLM POC using local Ollama models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("task", nargs="*", help="Task/question to solve recursively.")
    parser.add_argument("--planner-model", default=DEFAULT_PLANNER_MODEL)
    parser.add_argument("--solver-model", default=DEFAULT_SOLVER_MODEL)
    parser.add_argument("--critic-model", default=DEFAULT_CRITIC_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--max-children", type=int, default=3)
    parser.add_argument(
        "--execution-mode",
        choices=("batch", "classic"),
        default="batch",
        help="Batch mode greatly reduces local model calls.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX)
    parser.add_argument("--num-predict", type=int, default=DEFAULT_NUM_PREDICT)
    parser.add_argument(
        "--think-level",
        choices=("low", "medium", "high"),
        default="low",
        help="Model thinking effort; low is fastest.",
    )
    parser.add_argument("--mock", action="store_true", help="Run without calling Ollama.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    task = " ".join(args.task).strip()
    if not task:
        task = "Design a practical recursive LLM proof of concept for local Ollama models."

    if args.max_depth < 0:
        print("--max-depth must be >= 0", file=sys.stderr)
        return 2
    if args.max_children < 1:
        print("--max-children must be >= 1", file=sys.stderr)
        return 2
    if args.batch_size < 1:
        print("--batch-size must be >= 1", file=sys.stderr)
        return 2

    client = OllamaClient(
        args.ollama_url,
        mock=args.mock,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
        think_level=args.think_level,
    )
    if args.execution_mode == "batch":
        root, final_answer = run_batched_workflow(
            client=client,
            task=task,
            planner_model=args.planner_model,
            solver_model=args.solver_model,
            critic_model=args.critic_model,
            max_depth=args.max_depth,
            max_children=args.max_children,
            batch_size=args.batch_size,
        )
    else:
        root = recursive_solve(
            client=client,
            task=task,
            planner_model=args.planner_model,
            solver_model=args.solver_model,
            critic_model=args.critic_model,
            depth=0,
            max_depth=args.max_depth,
            max_children=args.max_children,
        )
        final_answer = synthesize_final(
            client=client,
            synthesizer_model=args.critic_model,
            task=task,
            root=root,
        )

    print("\n=== Recursive Trace ===")
    print_tree(root)
    print("\n=== Final Answer ===")
    print(final_answer)
    print(f"\nLLM calls: {client.call_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
