from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from datasets import load_dataset
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm

from prompts import SYSTEM_PROMPT, build_react_initial_prompt


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(gold)


def extract_final_answer(text: str) -> str:
    matches = re.findall(r"Answer:\s*(.+)", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return text.strip().splitlines()[-1].strip()


class WikipediaClient:
    SEARCH_URL = "https://en.wikipedia.org/w/api.php"
    SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": os.getenv(
                    "WIKIPEDIA_USER_AGENT",
                    "react-mini-project/1.0 (academic project; contact: local-user)",
                ),
                "Accept": "application/json",
            }
        )

    def search(self, query: str, top_k: int = 3) -> str:
        response = self.session.get(
            self.SEARCH_URL,
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "utf8": 1,
                "srlimit": top_k,
            },
            timeout=20,
        )
        if response.status_code == 403:
            return (
                "Wikipedia search was blocked with HTTP 403. "
                "Try setting WIKIPEDIA_USER_AGENT to a descriptive value and rerun."
            )
        response.raise_for_status()
        data = response.json()
        results = data.get("query", {}).get("search", [])
        if not results:
            return "No search results found."

        lines = []
        for idx, result in enumerate(results, start=1):
            snippet = re.sub(r"<[^>]+>", "", result.get("snippet", ""))
            title = result.get("title", "")
            lines.append(f"{idx}. {title}: {snippet}")
        return "\n".join(lines)

    def lookup(self, title: str) -> str:
        response = self.session.get(self.SUMMARY_URL + quote(title), timeout=20)
        if response.status_code == 404:
            return f"No summary found for {title}."
        if response.status_code == 403:
            return f"Wikipedia lookup was blocked with HTTP 403 for {title}."
        response.raise_for_status()
        data = response.json()
        extract = data.get("extract")
        if not extract:
            return f"No summary found for {title}."
        return extract


class GeminiClient:
    def __init__(self, model: str) -> None:
        self.model = model
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        self.client = genai.Client(api_key=api_key)

    def complete(self, prompt: str) -> str:
        max_retries = int(os.getenv("LLM_MAX_RETRIES", "6"))
        base_delay = float(os.getenv("LLM_RETRY_DELAY_SECONDS", "12"))

        for attempt in range(max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                    ),
                )
                return (response.text or "").strip()
            except Exception as exc:
                error_text = str(exc)
                if not any(token in error_text for token in ("429", "RESOURCE_EXHAUSTED", "rate limit")):
                    raise
                if attempt == max_retries:
                    raise
                delay = base_delay * (attempt + 1)
                time.sleep(delay)

        raise RuntimeError("Model completion failed after retries.")


@dataclass
class ExampleResult:
    question: str
    gold_answer: str
    prediction: str
    correct: bool
    raw_output: str
    trace: list[dict[str, str]]


def load_hotpotqa_sample(sample_size: int, seed: int) -> list[dict[str, str]]:
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")
    bridge_questions = [row for row in dataset if row.get("type") == "bridge"]
    random.Random(seed).shuffle(bridge_questions)

    sample = []
    for row in bridge_questions[:sample_size]:
        sample.append(
            {
                "question": row["question"],
                "answer": row["answer"],
            }
        )
    return sample


def parse_react_step(text: str) -> tuple[str | None, str | None]:
    action_match = re.search(r"Action:\s*(Search|Lookup)\[(.+?)\]", text, flags=re.IGNORECASE | re.DOTALL)
    if action_match:
        action_name = action_match.group(1).capitalize()
        action_arg = action_match.group(2).strip()
        return action_name, action_arg

    answer_match = re.search(r"Answer:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if answer_match:
        return "Answer", answer_match.group(1).strip()

    return None, None


def run_react(
    model: GeminiClient,
    wiki: WikipediaClient,
    question: str,
    gold_answer: str,
    max_steps: int,
) -> ExampleResult:
    transcript = build_react_initial_prompt(question)
    trace: list[dict[str, str]] = [{"role": "user", "content": transcript}]
    raw_outputs: list[str] = []
    prediction = ""

    for _ in range(max_steps):
        output = model.complete(transcript)
        raw_outputs.append(output)
        trace.append({"role": "assistant", "content": output})
        action_name, action_arg = parse_react_step(output)

        if action_name == "Answer":
            prediction = action_arg or ""
            break

        if action_name == "Search":
            try:
                observation = wiki.search(action_arg or "")
            except requests.RequestException as exc:
                observation = f"Search request failed: {exc}"
        elif action_name == "Lookup":
            try:
                observation = wiki.lookup(action_arg or "")
            except requests.RequestException as exc:
                observation = f"Lookup request failed: {exc}"
        else:
            prediction = extract_final_answer(output)
            break

        observation_block = f"Observation: {observation}"
        trace.append({"role": "tool", "content": observation_block})
        transcript = f"{transcript}\n{output}\n{observation_block}"

    if not prediction:
        prediction = extract_final_answer(raw_outputs[-1] if raw_outputs else "")

    return ExampleResult(
        question=question,
        gold_answer=gold_answer,
        prediction=prediction,
        correct=exact_match(prediction, gold_answer),
        raw_output="\n\n".join(raw_outputs),
        trace=trace,
    )


def write_outputs(results: list[ExampleResult], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    traces_path = output_dir / f"traces_{timestamp}.jsonl"
    summary_path = output_dir / f"summary_{timestamp}.json"

    with traces_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    accuracy = sum(result.correct for result in results) / max(len(results), 1)
    summary: dict[str, Any] = {
        "created_at": timestamp,
        "num_examples": len(results),
        "accuracy": accuracy,
        "correct": sum(result.correct for result in results),
        "mode": "react_workflow_demo",
    }

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return traces_path, summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scaled-down ReAct experiment on HotpotQA.")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model name to use.")
    parser.add_argument("--sample-size", type=int, default=1, help="Number of HotpotQA examples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--max-react-steps", type=int, default=10, help="Maximum ReAct action steps.")
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional delay between model calls to reduce burstiness.",
    )
    return parser.parse_args()


def main() -> None:
    env_path = Path(__file__).with_name(".env")
    load_dotenv(dotenv_path=env_path, override=True)
    args = parse_args()
    model = GeminiClient(model=args.model)
    wiki = WikipediaClient()

    examples = load_hotpotqa_sample(sample_size=args.sample_size, seed=args.seed)
    results: list[ExampleResult] = []

    for example in tqdm(examples, desc="Running experiment"):
        question = example["question"]
        gold_answer = example["answer"]

        results.append(run_react(model, wiki, question, gold_answer, args.max_react_steps))
        if args.sleep_seconds:
            time.sleep(args.sleep_seconds)

    traces_path, summary_path = write_outputs(results, Path("outputs"))
    print(f"Wrote traces to {traces_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
