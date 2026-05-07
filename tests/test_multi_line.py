"""Tests for executing multi-line shell files via srun."""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from srun.repl import _run_file
from srun.executors.python_exec import PythonExecutor
from srun.executors.shell_exec import ShellExecutor
from srun.executors.r_exec import RExecutor
from srun.context import state


class TestMultiLineFile:
    def setup_method(self):
        self.py = PythonExecutor()
        self.sh = ShellExecutor()
        self.r = RExecutor()
        state.vars.clear()
        state.active_df = None
        state.last_dispatch_error = None
        state.session_log = []
        state._conversation = []

    def test_file_execution_fast_only(self):
        """Execute a script with no typos — all fast, no LLM."""
        from srun.llm import llm
        prev_hit = llm.cache_stats["hit_tokens"]
        prev_miss = llm.cache_stats["miss_tokens"]

        path = os.path.join(os.path.dirname(__file__), "data", "test_multi_line_fast.sh")
        cwd = os.getcwd()
        try:
            _run_file(path, self.py, self.sh, self.r)
        finally:
            os.chdir(cwd)

        assert llm.cache_stats["hit_tokens"] == prev_hit
        assert llm.cache_stats["miss_tokens"] == prev_miss

    @pytest.mark.slow
    @pytest.mark.llm
    def test_file_execution_with_typos(self):
        """Execute a script with typos — LLM fixes trigger, all should pass."""
        path = os.path.join(os.path.dirname(__file__), "data", "test_multi_line.sh")
        project_root = os.path.dirname(os.path.dirname(__file__))
        cwd = os.getcwd()
        try:
            os.chdir(project_root)
            with open(path) as f:
                lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]
            
            from srun.repl import execute
            from srun.dispatch import dispatcher
            
            results = []
            for line in lines:
                cat = dispatcher.classify(line)
                result = execute(cat, line, self.py, self.sh, self.r)
                results.append(result)
            
            failed = [l for l, r in zip(lines, results) if not r["success"]]
            llm_used = sum(1 for r in results if r.get("llm_used"))
            assert llm_used >= 1, f"Expected LLM fixes for typo commands, got {llm_used}"
            pass_rate = (len(lines) - len(failed)) / len(lines)
            assert pass_rate >= 0.7, f"Too many failures: {failed}"
        finally:
            os.chdir(cwd)
