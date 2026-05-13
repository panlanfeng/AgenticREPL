"""Multi-round evaluation runner — synthetic user exercises srun agent."""

import json
import os
import sys
import tempfile
import shutil
import statistics

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
    """Run a single evaluation round. Returns (score, reasoning, raw_transcript)."""
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

            # Build user prompt with conversation context
            ctx = format_transcript(clean_transcript_list)
            prompt_text = _USER_AGENT_PROMPT.format(
                persona=usecase["persona"],
                hidden_goal=usecase["hidden_goal"],
                transcript=ctx if ctx else "(no conversation yet)",
                round_num=i + 1,
                hint=hint,
            )

            # Get synthetic user message
            user_msg, _ = llm.run(prompt_text)
            if not user_msg:
                user_msg = hint  # fallback

            # Send to srun agent
            summary, commands = llm.run(
                user_msg,
                exec_callback=_exec_inline(py, sh, r),
            )

            # Record turn
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
                "commands": commands,
                "output": llm._last_output,
                "agent_text": getattr(llm, "_agent_text", ""),
            })

            # Check if agent finished (no more commands, task seems done)
            if i >= 3 and not commands and summary:
                break
    finally:
        os.chdir(original_cwd)
        shutil.rmtree(workspace, ignore_errors=True)

    # Grade
    transcript_text = format_transcript(clean_transcript_list)
    success_key = "grade_5"  # use grade_5 as the success description
    score, reasoning = grade(
        usecase["hidden_goal"],
        usecase["success"].get(success_key, "Complete task"),
        transcript_text,
    )
    return score, reasoning, raw_transcript


def evaluate(usecase_name, runs=5, rounds=10):
    """Run multiple evaluations and compute average score."""
    usecase = load_usecase(usecase_name)
    scores = []
    reasonings = []

    print(f"\n{'='*60}")
    print(f"Evaluating: {usecase['name']}")
    print(f"Runs: {runs}, Max rounds: {rounds}")
    print(f"{'='*60}")

    for run in range(runs):
        print(f"\n--- Run {run + 1}/{runs} ---")
        score, reasoning, transcript = run_one(usecase, rounds)
        scores.append(score)
        reasonings.append(reasoning)
        print(f"Score: {score}/5 — {reasoning[:100]}...")
        # Reset LLM state between runs
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

    return {
        "usecase": usecase_name,
        "runs": runs,
        "scores": scores,
        "average": avg,
        "stdev": stdev,
        "reasonings": reasonings,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate srun agent with synthetic user")
    parser.add_argument("--usecase", choices=["explorer", "cleanup", "revenue", "all"],
                        default="explorer", help="Use case to evaluate")
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
