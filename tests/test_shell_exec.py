"""Tests for shell executor."""

import pytest
from srun.executors.shell_exec import ShellExecutor


class TestShellExecutor:
    def setup_method(self):
        self.sh = ShellExecutor()

    def test_cat_file(self, test_csv):
        ok, out, *rest = self.sh.execute(f"cat {test_csv}")
        assert ok
        assert "Alice" in out

    def test_pipe(self, test_csv):
        ok, out, *rest = self.sh.execute(f"cat {test_csv} | sort")
        assert ok
        lines = out.strip().split("\n")
        assert "Alice" in lines[0]

    def test_echo(self):
        ok, out, *rest = self.sh.execute("echo hello")
        assert ok
        assert "hello" in out

    def test_command_not_found(self):
        ok, out, *rest = self.sh.execute("nonexistent_cmd_xyz")
        assert not ok

    def test_ls(self):
        ok, out, *rest = self.sh.execute("ls")
        assert ok

    def test_timeout(self):
        ok, out, *rest = self.sh.execute("sleep 2")
        assert ok

    def test_execute_timeout_kills_process_group(self):
        ok, out, *rest = self.sh.execute("sleep 3", timeout=1)
        assert not ok
        assert "timed out" in out

    def test_execute_no_timeout_when_disabled(self):
        ok, out, *rest = self.sh.execute("sleep 2", timeout=0)
        assert ok

    def test_background_start_poll_stop(self):
        tid = self.sh.start_background("sleep 0.2 && echo bg-done")
        status = self.sh.poll_background(tid)
        assert tid in status
        assert "running" in status or "finished" in status
        import time
        time.sleep(0.6)
        status = self.sh.poll_background(tid)
        assert "finished" in status
        assert "bg-done" in status
        self.sh.stop_all_background()

    def test_background_stop_kills(self):
        tid = self.sh.start_background("sleep 30")
        status = self.sh.stop_background(tid)
        assert "finished" in status
        # Process group must actually be gone — poll should no longer report running
        status = self.sh.poll_background(tid)
        assert "finished" in status
        self.sh.stop_all_background()

    def test_background_poll_unknown_task(self):
        out = self.sh.poll_background("bg999")
        assert "No background task" in out
