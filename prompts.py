"""Prompt templates for the scaled-down ReAct workflow demo."""

SYSTEM_PROMPT = """You are a careful question answering system.
Follow the requested format exactly.
When you are unsure, say so rather than inventing facts."""


REACT_INSTRUCTIONS = """You are solving a multi-hop question by alternating reasoning and actions.

Available actions:
1. Search[query]
   - Use this to find the most relevant Wikipedia page titles for a query.
2. Lookup[title]
   - Use this to retrieve a short Wikipedia summary for a page title.

Rules:
- Use at most one action at a time.
- After receiving an observation, decide the next best step.
- If you have enough evidence, finish with:
  Answer: <short answer>
- Do not invent observations.
- If a search result is noisy, refine the query.
- Every turn must start with exactly one line beginning with `Thought:`.
- If you are taking an action, the next line must be exactly one `Action:` line.
- Once you know the answer, stop searching and produce `Thought:` followed by `Answer:`.
- Keep the thought short and focused on the next decision.

Your response must be in one of these formats:

Thought: <brief reasoning>
Action: Search[...]

Thought: <brief reasoning>
Action: Lookup[...]

Thought: <brief reasoning>
Answer: <short answer>
"""


REACT_EXAMPLE = """Question: Which scientist developed the laws of motion that were later used by the author of Principia?
Thought: I should identify the author of Principia and the scientist associated with the laws of motion.
Action: Search[author of Principia]
Observation: Top results include Isaac Newton and Philosophiae Naturalis Principia Mathematica.
Thought: I have enough evidence to answer directly.
Answer: Isaac Newton
"""


def build_react_initial_prompt(question: str) -> str:
    return (
        f"{REACT_INSTRUCTIONS}\n\n"
        f"Example:\n{REACT_EXAMPLE}\n"
        f"Now solve this question.\n"
        f"Question: {question}"
    )
