"""LLM-based grading for evaluation transcripts."""

import json
import os

from srun.llm import llm


_EVALUATOR_PROMPT = open(
    os.path.join(os.path.dirname(__file__), "prompts", "evaluator.txt")
).read()


def grade(hidden_goal, success_criteria, transcript_text):
    """Grade a single run transcript. Returns (score, reasoning)."""
    prompt = _EVALUATOR_PROMPT.format(
        hidden_goal=hidden_goal,
        success_criteria=success_criteria,
        transcript=transcript_text,
    )
    if not llm.client:
        return 0, "No LLM client configured"

    try:
        from srun.config import config
        kwargs = {"model": config.model, "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.0, "max_tokens": 500, "stream": False}
        resp = llm.client.chat.completions.create(**kwargs)
        summary = resp.choices[0].message.content.strip()
    except Exception as e:
        return 0, f"Grader API error: {e}"

    if not summary:
        return 0, "Grader LLM returned no response"

    try:
        if "```json" in summary:
            start = summary.index("```json") + 7
            end = summary.index("```", start)
            summary = summary[start:end]
        elif "{" in summary:
            start = summary.index("{")
            end = summary.rindex("}") + 1
            summary = summary[start:end]
        result = json.loads(summary)
        return int(result.get("score", 0)), result.get("reasoning", "No reasoning")
    except (json.JSONDecodeError, ValueError, KeyError):
        return 0, f"Failed to parse grader: {summary[:200]}"
