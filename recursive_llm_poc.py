#!/usr/bin/env python3
"""Small recursive LLM POC using local Ollama models.

The POC uses a fast model as a planner/summarizer and a stronger thinking model
as solver/critic. It prints a visible recursion tree so the orchestration is easy
to demo and reason about.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable


DEFAULT_PLANNER_MODEL = "mistral"
DEFAULT_REASONER_MODEL = "gpt-oss:20b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


@dataclass
class Node:
    title: str
    role: str
    model: str
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
        "model": node.model,
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
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.mock = mock
        self.logger = logger

    def generate(self, model: str, prompt: str) -> str:
        started_at = time.perf_counter()
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
            "stream": False,
            "options": {"temperature": 0.2},
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.base_url}. "
                "Start it with `ollama serve` or rerun with `--mock`."
            ) from exc

        output = str(data.get("response", "")).strip()
        self.log(
            f"✅ {model} response in {time.perf_counter() - started_at:.1f}s "
            f"({len(output):,} chars)"
        )
        return output

    def log(self, message: str) -> None:
        if self.logger:
            self.logger(message)

    def _mock_generate(self, model: str, prompt: str) -> str:
        prompt_lower = prompt.lower()
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
            return json.loads(raw[start : end + 1])
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
    raw = client.generate(model, textwrap.dedent(prompt).strip())
    data = parse_json_object(raw)
    subtasks = data.get("subtasks", [])
    if not isinstance(subtasks, list) or not subtasks:
        raise ValueError(f"Planner did not return subtasks: {raw}")
    planned = [str(item).strip() for item in subtasks[:max_children] if str(item).strip()]
    client.log(f"🧩 Planned {len(planned)} subtasks: {', '.join(planned)}")
    return planned


def solve_task(client: OllamaClient, model: str, task: str, context: str) -> str:
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
    return client.generate(model, textwrap.dedent(prompt).strip())


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
    raw = client.generate(model, textwrap.dedent(prompt).strip())
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
    planner_model: str,
    reasoner_model: str,
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
    raw = client.generate(reasoner_model, textwrap.dedent(prompt).strip())
    try:
        return str(parse_json_object(raw).get("final_answer", "")).strip()
    except json.JSONDecodeError:
        return raw.strip()


def recursive_solve(
    client: OllamaClient,
    task: str,
    planner_model: str,
    reasoner_model: str,
    depth: int,
    max_depth: int,
    max_children: int,
    context: str = "",
) -> Node:
    client.log(f"🌱 Enter depth {depth}: {compact(task, 90)}")
    node = Node(
        title=task,
        role="root" if depth == 0 else "subtask",
        model=reasoner_model,
        prompt=task,
    )

    if depth < max_depth:
        subtasks = plan_subtasks(client, planner_model, task, max_children)
        for subtask in subtasks:
            child = recursive_solve(
                client=client,
                task=subtask,
                planner_model=planner_model,
                reasoner_model=reasoner_model,
                depth=depth + 1,
                max_depth=max_depth,
                max_children=max_children,
                context=context,
            )
            node.children.append(child)

    child_context = "\n\n".join(f"{child.title}: {child.answer}" for child in node.children)
    node.answer = solve_task(client, reasoner_model, task, context or child_context)
    critique = critique_solution(client, reasoner_model, task, node.answer)
    node.critique = critique["critique"]
    node.confidence = critique["confidence"]
    node.needs_more_recursion = critique["needs_more_recursion"] and depth < max_depth
    client.log(f"🏁 Exit depth {depth}: {compact(task, 90)}")
    return node


def print_tree(node: Node, indent: str = "") -> None:
    confidence = "n/a" if node.confidence is None else f"{node.confidence:.2f}"
    print(f"{indent}- [{node.role}] {node.title}")
    print(f"{indent}  model: {node.model} | confidence: {confidence}")
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
    parser.add_argument("--reasoner-model", default=DEFAULT_REASONER_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--max-children", type=int, default=3)
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

    client = OllamaClient(args.ollama_url, mock=args.mock)
    root = recursive_solve(
        client=client,
        task=task,
        planner_model=args.planner_model,
        reasoner_model=args.reasoner_model,
        depth=0,
        max_depth=args.max_depth,
        max_children=args.max_children,
    )
    final_answer = synthesize_final(
        client=client,
        planner_model=args.planner_model,
        reasoner_model=args.reasoner_model,
        task=task,
        root=root,
    )

    print("\n=== Recursive Trace ===")
    print_tree(root)
    print("\n=== Final Answer ===")
    print(final_answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
