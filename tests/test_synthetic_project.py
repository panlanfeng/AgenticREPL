"""End-to-end test: synthetic user manages a project with srun.

Simulates a realistic session: navigate, view files, run tasks, make mistakes.
"""

import os
import sys
import time
import subprocess
import json

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from srun.repl import execute, print_result
from srun.dispatch import dispatcher
from srun.context import state
from srun.executors.python_exec import PythonExecutor
from srun.executors.shell_exec import ShellExecutor
from srun.executors.r_exec import RExecutor

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic_project")


class UserSession:
    """Simulates an interactive user session in a project."""

    def __init__(self, workdir):
        self.workdir = workdir
        self.py = PythonExecutor()
        self.sh = ShellExecutor()
        self.r = RExecutor()
        self.total_time = 0
        self.llm_calls = 0
        self.results = []

    def run(self, command, description=""):
        state.vars.clear()
        state.active_df = None
        state.last_dispatch_error = None

        cat = dispatcher.classify(command)
        start = time.perf_counter()
        result = execute(cat, command, self.py, self.sh, self.r)
        elapsed_ms = (time.perf_counter() - start) * 1000

        self.total_time += elapsed_ms
        if result.get("llm_used"):
            self.llm_calls += 1

        entry = {
            "input": command[:60],
            "description": description,
            "category": cat,
            "success": result["success"],
            "llm_used": result.get("llm_used", False),
            "elapsed_ms": elapsed_ms,
            "generated_code": result.get("generated_code", ""),
            "fixed_code": result.get("fixed_code", ""),
        }
        self.results.append(entry)
        return result

    def stats(self):
        return {
            "total_commands": len(self.results),
            "success_rate": sum(1 for r in self.results if r["success"]) / max(len(self.results), 1),
            "llm_calls": self.llm_calls,
            "total_time_ms": int(self.total_time),
            "results": self.results,
        }


def write_stats(stats, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)


class TestSyntheticProject:
    """Synthetic user manages a project using srun."""

    @pytest.mark.slow
    def test_full_session(self):
        session = UserSession(PROJECT_ROOT)
        cwd = os.getcwd()
        try:
            os.chdir(PROJECT_ROOT)

            # --- SECTION 1: Orientation (all fast, no LLM) ---
            session.run("ls", "list project root")
            session.run("ls data/", "list data directory")
            session.run("cat README.md", "view README")
            session.run("wc -l data/sales.csv", "count sales records")

            # --- SECTION 2: Data exploration ---
            session.run("cat data/sales.csv", "view sales data")
            session.run("cat data/config.json", "view config")
            session.run("grep -r TODO .", "search for TODOs")

            # --- SECTION 3: Shell operations ---
            session.run("cd data", "enter data dir")
            session.run("ls", "list data contents")
            session.run("cd ..", "go back to project root")
            session.run("echo 'cache/' > .gitignore", "create gitignore")
            session.run("ll", "alias: ll -> ls -la")

            # --- SECTION 4: Python tasks ---
            session.run("python src/app.py", "run the Python app")

            # --- SECTION 5: Typo / error recovery (LLM) ---
            session.run("cat data/sales.csv sort by region filter by amount > 100",
                        "LLM: pseudo-code filter")
            session.run("cdd data", "typo: cdd -> cd")
            session.run("grep --nocolor TODO .", "typo: --nocolor flag")
            session.run("ls all inverse order", "NL: list files reversed")

        finally:
            os.chdir(cwd)

        stats = session.stats()
        write_stats(stats, os.path.join(PROJECT_ROOT, ".srun", "session_stats.json"))

        print(f"\n{'='*60}")
        print(f"Session completed: {stats['total_commands']} commands")
        print(f"Success rate: {stats['success_rate']:.0%}")
        print(f"LLM calls: {stats['llm_calls']}")
        print(f"Total time: {stats['total_time_ms']}ms")
        print(f"{'='*60}")

        # Per-command breakdown
        for r in stats["results"]:
            tag = "+LLM" if r["llm_used"] else "FAST"
            status = "OK" if r["success"] else "FAIL"
            fix_info = ""
            if r["generated_code"]:
                fix_info = f" gen={r['generated_code'][:40]}"
            elif r["fixed_code"]:
                fix_info = f" fix={r['fixed_code'][:40]}"
            print(f"  [{tag}] {r['description']:35s} {r['elapsed_ms']:5.0f}ms [{status}]{fix_info}")

        assert stats["success_rate"] >= 0.75, f"Success rate too low: {stats['success_rate']:.0%}"

    def test_fast_commands_only(self):
        """All fast commands should work without any LLM calls."""
        session = UserSession(PROJECT_ROOT)
        backup = os.getcwd()
        try:
            os.chdir(PROJECT_ROOT)
            session.run("ls")
            session.run("ls data/")
            session.run("cat data/sales.csv")
            session.run("wc -l data/sales.csv")
            session.run("echo hello")
            session.run("cat data/config.json")
            session.run("grep TODO README.md")
        finally:
            os.chdir(backup)

        assert session.llm_calls == 0, f"Fast commands triggered {session.llm_calls} LLM calls"
        assert session.total_time < 500, f"Fast commands took {session.total_time:.0f}ms, expected <500ms"

    def test_cd_persists(self):
        """cd should change the working directory persistently."""
        session = UserSession(PROJECT_ROOT)
        backup = os.getcwd()
        try:
            os.chdir(PROJECT_ROOT)
            data_dir = os.path.join(PROJECT_ROOT, "data")
            session.run("cd data")
            assert os.getcwd() == data_dir, f"cd data failed: {os.getcwd()}"
            session.run("cd ..")
            assert os.getcwd() == PROJECT_ROOT, f"cd .. failed: cwd={os.getcwd()}"
        finally:
            os.chdir(backup)

    def test_python_tasks(self):
        """Running Python scripts should produce expected output."""
        session = UserSession(PROJECT_ROOT)
        backup = os.getcwd()
        try:
            os.chdir(PROJECT_ROOT)
            result = session.run("python src/app.py")
            output = result.get("output", "")
            assert "Loaded" in output, f"Output: {output}"
            assert "East" in output
            assert "Sales by Region" in output
        finally:
            os.chdir(backup)

    def test_file_creation(self):
        """Create and verify a file."""
        session = UserSession(PROJECT_ROOT)
        tmp_file = os.path.join(PROJECT_ROOT, "data", "tmp_test.txt")
        cwd = os.getcwd()
        try:
            os.chdir(PROJECT_ROOT)
            session.run(f"echo 'test content' > {tmp_file}")
            assert os.path.isfile(tmp_file), f"File not created: {tmp_file}"
            with open(tmp_file) as f:
                assert "test content" in f.read()
        finally:
            if os.path.isfile(tmp_file):
                os.remove(tmp_file)
            os.chdir(cwd)

    def test_search(self):
        """Grep should find expected content."""
        session = UserSession(PROJECT_ROOT)
        cwd = os.getcwd()
        try:
            os.chdir(PROJECT_ROOT)
            result = session.run("grep -r TODO .")
            output = result.get("output", "")
            assert "error handling" in output.lower() or "TODO" in output
        finally:
            os.chdir(cwd)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
