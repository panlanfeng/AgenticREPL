"""Integration tests — full execute() pipeline."""

import pytest
import time
from srun.repl import execute, _log_turn
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
        state.reset_session()
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
        cat = dispatcher.classify("grep --nonexist Alice tests/data/test.csv")
        result = execute(cat, "grep --nonexist Alice tests/data/test.csv", self.py, self.sh, self.r)
        assert result.get("llm_used") or result.get("summary") is not None, "Repair flow should be triggered"

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

    @pytest.mark.slow
    @pytest.mark.llm
    def test_multi_round_server_cache(self):
        from srun.llm import llm
        llm.reset_cache()

        commands = [
            "ls all inverse order",
            "grep --nocolor root /etc/hosts",
            "find all csv files",
            "ls all reverse order",
            "cat /etc/hosts sort by line number",
        ]

        prev_hit = 0
        for i, cmd in enumerate(commands):
            cat = dispatcher.classify(cmd)
            result = execute(cat, cmd, self.py, self.sh, self.r)
            _log_turn(cmd, result, 0)
            stats = llm.cache_stats
            assert stats["hit_tokens"] >= prev_hit, (
                f"Call #{i+1} '{cmd}': hit tokens ({stats['hit_tokens']}) "
                f"should be >= previous ({prev_hit}). Stats: {stats}"
            )
            prev_hit = stats["hit_tokens"]

        assert llm.cache_stats["total_tokens"] > 0, "No LLM tokens"
        assert llm.cache_stats["hit_tokens"] > 0, "No cache hits at all"
        assert prev_hit >= 128 * (len(commands) - 1), (
            f"Expected at least 128*{len(commands)-1}={128*(len(commands)-1)} hit tokens "
            f"(one system msg per call). Got {prev_hit}"
        )
