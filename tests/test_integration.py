"""Integration tests — full execute() pipeline."""

import pytest
import time
from srun.repl import execute
from srun.dispatch import dispatcher
from srun.context import state
from srun.executors.python_exec import PythonExecutor
from srun.executors.shell_exec import ShellExecutor
from srun.executors.r_exec import RExecutor


class TestIntegration:
    def setup_method(self):
        self.py = PythonExecutor()
        self.sh = ShellExecutor()
        self.r = RExecutor()
        state.vars.clear()
        state.active_df = None
        state.last_dispatch_error = None

    def test_normal_shell_fast_no_llm(self):
        cat = dispatcher.classify("echo ok")
        start = time.perf_counter()
        result = execute(cat, "echo ok", self.py, self.sh, self.r)
        elapsed = (time.perf_counter() - start) * 1000
        assert result["success"]
        assert not result["llm_used"]
        assert elapsed < 100, f"Normal shell too slow: {elapsed:.0f}ms"

    def test_quick_fix_fast_no_llm(self):
        cat = dispatcher.classify("ll")
        start = time.perf_counter()
        result = execute(cat, "ll", self.py, self.sh, self.r)
        elapsed = (time.perf_counter() - start) * 1000
        assert result["success"]
        assert not result["llm_used"]
        assert elapsed < 100, f"Quick fix too slow: {elapsed:.0f}ms"

    def test_python_expression_fast(self):
        cat = dispatcher.classify("3+5*2")
        start = time.perf_counter()
        result = execute(cat, "3+5*2", self.py, self.sh, self.r)
        elapsed = (time.perf_counter() - start) * 1000
        assert result["success"]
        assert not result["llm_used"]
        assert elapsed < 20, f"Python expression too slow: {elapsed:.0f}ms"

    def test_shell_error_triggers_repair_flow(self):
        cat = dispatcher.classify("ls /nonexistent_dir_xyz_123")
        result = execute(cat, "ls /nonexistent_dir_xyz_123", self.py, self.sh, self.r)
        assert result["success"]  # LLM repair should produce a fixed command

    @pytest.mark.slow
    @pytest.mark.llm
    def test_llm_dispatch_pseudocode(self):
        cat = dispatcher.classify(
            "cat tests/data/test.csv sort by student name filter by scores > 80"
        )
        result = execute(
            cat,
            "cat tests/data/test.csv sort by student name filter by scores > 80",
            self.py,
            self.sh,
            self.r,
        )
        assert result["success"]
        assert result["llm_used"]

    @pytest.mark.slow
    @pytest.mark.llm
    def test_llm_repair_typo(self):
        cat = dispatcher.classify("grep --nonexist Alice tests/data/test.csv")
        result = execute(
            cat,
            "grep --nonexist Alice tests/data/test.csv",
            self.py,
            self.sh,
            self.r,
        )
        assert result["success"]
        assert result["llm_used"]

    @pytest.mark.slow
    @pytest.mark.llm
    def test_shell_error_repair_general(self):
        cat = dispatcher.classify("ls all inverse order")
        assert cat == "shell"
        result = execute(
            cat, "ls all inverse order", self.py, self.sh, self.r
        )
        assert result["success"]
        assert result["llm_used"]
