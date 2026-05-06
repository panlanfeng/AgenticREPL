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
