"""Multi-round evaluation runner — synthetic user exercises srun agent."""

import json
import os
import sys
import tempfile
import shutil
import statistics
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from srun.llm import llm
from srun.context import state
from srun.executors.python_exec import PythonExecutor
from srun.executors.shell_exec import ShellExecutor
from srun.executors.r_exec import RExecutor
from srun.repl import _exec_inline

from .transcript import clean_transcript, format_transcript
from .grader import grade

_USER_AGENT_PROMPT = open(
    os.path.join(os.path.dirname(__file__), "prompts", "user_agent.txt")
).read()

_REFLECTION_PROMPT = """You are reviewing an AI agent's performance. Analyze the transcript below.

TRANSCRIPT:
{transcript}

HIDDEN GOAL: {hidden_goal}
FINAL SCORE: {score}/5

Write a structured review with these sections:

## What went well
- Specific things the agent did correctly.
- Approaches or patterns that should be remembered and repeated.

## What went wrong
- Specific mistakes, context losses, wrong datasets, failed commands.
- Root causes — WHY did these things go wrong?

## What to improve
- Concrete, actionable changes that would prevent the failures seen.
- E.g., prompt changes, tool additions, architecture fixes, test additions.
- Be specific: reference exact rounds and behaviors from the transcript.

## Key takeaways
- 1-2 sentences summarizing the most important lesson from this run.

Keep the review under 300 words. Be direct and actionable."""


def load_usecase(name):
    path = os.path.join(os.path.dirname(__file__), "usecases", f"{name}.json")
    with open(path) as f:
        return json.load(f)


def setup_workspace(usecase):
    """Create temp directory with files from usecase setup."""
    tmpdir = tempfile.mkdtemp(prefix="srun_eval_")
    for rel_path, content in usecase.get("setup", {}).get("files", {}).items():
        full_path = os.path.join(tmpdir, rel_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)
    return tmpdir


def run_one(usecase, rounds):
    """Run a single evaluation round. Returns (score, reasoning, review, clean_transcript_list, raw_transcript)."""
    clean_transcript_list = []
    raw_transcript = []

    py = PythonExecutor()
    sh = ShellExecutor()
    r = RExecutor()
    state.reset_session()

    workspace = setup_workspace(usecase)
    original_cwd = os.getcwd()
    os.chdir(workspace)

    try:
        for i in range(rounds):
            step = usecase["script"][i]
            hint = step["hint"]

            ctx = format_transcript(clean_transcript_list)
            prompt_text = _USER_AGENT_PROMPT.format(
                persona=usecase["persona"],
                hidden_goal=usecase["hidden_goal"],
                transcript=ctx if ctx else "(no conversation yet)",
                round_num=i + 1,
                hint=hint,
            )

            user_msg, _ = llm.run(prompt_text)
            if not user_msg:
                user_msg = hint

            summary, commands = llm.run(
                user_msg,
                exec_callback=_exec_inline(py, sh, r),
            )

            clean = clean_transcript(
                i + 1,
                user_msg,
                commands,
                llm._last_output,
                llm._agent_text if hasattr(llm, "_agent_text") else "",
            )
            clean_transcript_list.append(clean)
            raw_transcript.append({
                "round": i + 1,
                "prompt": user_msg,
                "commands": [c if isinstance(c, str) else c.get("command", "") for c in (commands or [])],
                "output": llm._last_output,
                "agent_text": getattr(llm, "_agent_text", ""),
            })

            if i >= 3 and not commands and summary:
                break
    finally:
        os.chdir(original_cwd)
        shutil.rmtree(workspace, ignore_errors=True)

    # Grade
    transcript_text = format_transcript(clean_transcript_list)
    success_key = "grade_5"
    score, reasoning = grade(
        usecase["hidden_goal"],
        usecase["success"].get(success_key, "Complete task"),
        transcript_text,
    )

    # Reflection report
    review = _generate_review(transcript_text, usecase["hidden_goal"], score)

    return score, reasoning, review, clean_transcript_list, raw_transcript


def _generate_review(transcript_text, hidden_goal, score):
    """Generate a structured review of the agent's performance."""
    if not llm.client:
        return "No LLM client configured for review generation."
    try:
        from srun.config import config
        prompt = _REFLECTION_PROMPT.format(
            transcript=transcript_text,
            hidden_goal=hidden_goal,
            score=score,
        )
        kwargs = {"model": config.model, "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.0, "max_tokens": 600, "stream": False}
        resp = llm.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Review generation failed: {e}"


def save_trajectory(usecase_name, run_num, score, reasoning, review,
                     clean_transcript_list, raw_transcript):
    """Save trajectory + report to the output directory."""
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{usecase_name}_run{run_num}_{timestamp}"

    # Save full report
    report = {
        "usecase": usecase_name,
        "run": run_num,
        "timestamp": timestamp,
        "score": score,
        "reasoning": reasoning,
        "review": review,
        "rounds": len(clean_transcript_list),
    }
    with open(os.path.join(out_dir, f"{base}_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Save clean transcript (LLM-readable)
    with open(os.path.join(out_dir, f"{base}_transcript.json"), "w") as f:
        json.dump(clean_transcript_list, f, indent=2)

    # Save raw transcript (debug)
    with open(os.path.join(out_dir, f"{base}_raw.json"), "w") as f:
        json.dump(raw_transcript, f, indent=2)

    return os.path.join(out_dir, f"{base}_report.json")


def evaluate(usecase_name, runs=5, rounds=10):
    """Run multiple evaluations and compute average score."""
    usecase = load_usecase(usecase_name)
    scores = []
    reasonings = []
    reports = []

    print(f"\n{'='*60}")
    print(f"Evaluating: {usecase['name']}")
    print(f"Runs: {runs}, Max rounds: {rounds}")
    print(f"{'='*60}")

    for run in range(runs):
        print(f"\n--- Run {run + 1}/{runs} ---")
        score, reasoning, review, clean_t, raw_t = run_one(usecase, rounds)
        scores.append(score)
        reasonings.append(reasoning)

        report_path = save_trajectory(
            usecase_name, run + 1, score, reasoning, review, clean_t, raw_t
        )
        reports.append(report_path)
        print(f"Score: {score}/5 — {reasoning[:100]}...")
        print(f"Saved: {report_path}")
        state.reset_session()

    avg = statistics.mean(scores)
    stdev = statistics.stdev(scores) if len(scores) > 1 else 0

    print(f"\n{'='*60}")
    print(f"RESULTS: {usecase['name']}")
    print(f"{'='*60}")
    for i, (s, r) in enumerate(zip(scores, reasonings)):
        print(f"  Run {i + 1}: {s}/5 — {r}")
    print(f"\n  Average: {avg:.1f}/5 (±{stdev:.1f})")
    print(f"  Scores: {scores}")
    print(f"  Reports: {reports}")

    return {
        "usecase": usecase_name,
        "runs": runs,
        "scores": scores,
        "average": avg,
        "stdev": stdev,
        "reasonings": reasonings,
        "reports": reports,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate srun agent with synthetic user")
    parser.add_argument("--usecase", choices=["sales", "join", "sensor", "all"],
                        default="sales", help="Use case to evaluate")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per use case")
    parser.add_argument("--rounds", type=int, default=10, help="Max rounds per run")
    args = parser.parse_args()

    if args.usecase == "all":
        results = {}
        for uc in ["explorer", "cleanup", "revenue"]:
            results[uc] = evaluate(uc, args.runs, args.rounds)
        print(f"\n{'='*60}")
        print("OVERALL")
        print(f"{'='*60}")
        for uc, r in results.items():
            print(f"  {uc}: {r['average']:.1f}/5 (±{r['stdev']:.1f})")
    else:
        evaluate(args.usecase, args.runs, args.rounds)
