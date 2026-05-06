"""Tests for repair.py — quick fixes and LLM repair."""

import pytest
from srun.repair import apply_quick_fix, QUICK_FIXES


class TestQuickFixes:
    def test_ll_alias(self):
        result = apply_quick_fix("ll", "")
        assert result == "ls -la"

    def test_la_alias(self):
        result = apply_quick_fix("la", "")
        assert result == "ls -a"

    def test_l_alias(self):
        result = apply_quick_fix("l", "")
        assert result == "ls -CF"

    def test_ls_all(self):
        result = apply_quick_fix("ls all", "")
        assert result == "ls -la"

    def test_ls_everything(self):
        result = apply_quick_fix("ls everything", "")
        assert result == "ls -la"

    def test_cd_dot_dot_no_space(self):
        result = apply_quick_fix("cd..", "")
        assert result == "cd .."

    def test_no_match(self):
        result = apply_quick_fix("random command", "")
        assert result is None

    def test_grep_recursive_missing_dir(self):
        result = apply_quick_fix('grep -r "pattern"', "")
        assert result == 'grep -r "pattern" .'

    @pytest.mark.parametrize(
        "inp,expected",
        [
            ("ll", "ls -la"),
            ("LL", "ls -la"),
            ("Ll", "ls -la"),
        ],
    )
    def test_case_insensitive(self, inp, expected):
        result = apply_quick_fix(inp, "")
        assert result == expected


class TestQuickFixPatterns:
    def test_all_patterns_have_valid_regex(self):
        import re

        for pattern, replacement in QUICK_FIXES:
            re.compile(pattern, re.IGNORECASE)
