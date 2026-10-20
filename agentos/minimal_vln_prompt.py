from __future__ import annotations

import argparse
from pathlib import Path


PROMPT_TEMPLATE = '''You are controlling a robot in a real indoor environment (a photorealistic 3D scan of a building). You interact only through these tools:

- observe(): look through the robot's cameras. It returns exactly three RGB images in LEFT, FRONT, RIGHT order.

- step(actions): execute exactly one primitive action. The actions list must contain one item: 0 = STOP (permanently ends the episode — declares you have reached the goal), 1 = move forward 0.25 m, 2 = turn left 15 degrees, 3 = turn right 15 degrees.

Your task is to follow this navigation instruction to its endpoint:

"{instruction}"

Rules:

- Alternate observing and stepping after every primitive: inspect the left, front, and right views, decide where the instruction wants you to go next, execute one action, then look again.

- Every response must contain exactly one tool call: observe() or step(). Do not write narration, explanations, plans, summaries, questions, or any text before or after the tool call.

- Make one short visual decision per observation and act immediately. Do not reconsider the same frame repeatedly.

- You have a budget of {budget} movement actions.

- You succeed only if you issue action 0 (STOP) while within 3 meters of the instruction's endpoint. STOP is permanent — issue it only when you believe you are at the goal.

- To turn around, repeatedly alternate observe() with one step([2]) or step([3]); never batch actions.

- Work autonomously until you stop; nobody can answer questions.
'''

INTERACTIVE_PROMPT_TEMPLATE = '''You are controlling a robot in a real indoor environment (a photorealistic 3D scan of a building). You interact only through these tools:

- observe(): look through the robot's cameras. It returns exactly three RGB images in LEFT, FRONT, RIGHT order.

- step(actions): execute exactly one primitive action. The actions list must contain one item: 0 = STOP (permanently ends the episode), 1 = move forward 0.25 m, 2 = turn left 15 degrees, 3 = turn right 15 degrees.

Wait for the user to provide a navigation instruction. Do not call a tool before receiving one. After receiving a navigation instruction:

- Call observe() first.
- Alternate observe() and step() after every primitive.
- Use exactly one tool call per response, with no narration before or after it.
- Make one short visual decision per observation and act immediately.
- Use at most {budget} movement actions.
- Issue action 0 only when you believe you have reached the requested endpoint.
- Work autonomously until you stop; do not ask follow-up questions while navigating.
'''


def build_prompt(instruction: str, budget: int = 500) -> str:
    normalized = instruction.strip()
    if not normalized:
        raise ValueError("instruction must not be empty")
    if budget < 1:
        raise ValueError("budget must be positive")
    return PROMPT_TEMPLATE.format(instruction=normalized, budget=budget)


def build_interactive_prompt(budget: int = 500) -> str:
    if budget < 1:
        raise ValueError("budget must be positive")
    return INTERACTIVE_PROMPT_TEMPLATE.format(budget=budget)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the paper-compatible minimal VLN system prompt.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--instruction")
    mode.add_argument("--interactive", action="store_true")
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prompt = build_interactive_prompt(args.budget) if args.interactive else build_prompt(args.instruction, args.budget)
    args.output.write_text(prompt, encoding="utf-8")


if __name__ == "__main__":
    main()
