"""Tests for dispatch.py — input classification."""

import pytest
from srun.dispatch import dispatcher


class TestDispatClassification:
    SHELL_INPUTS = [
        "ls -la",
        "cat file.csv",
        "grep Alice file.csv | sort",
        "echo hello world",
        "git status",
        "cd /tmp",
        "ll",
        "la",
    ]

    PYTHON_INPUTS = [
        "import pandas as pd",
        "df.groupby('region').mean()",
        "print('hello')",
        "100/4",
        "3 + 5 * 2",
        "x = 42",
        "df['col'].head()",
    ]

    UNKNOWN_INPUTS = [
        "cat file.csv sort by student name filter by scores > 80",
        "find all csv files",
        "groupby region mean",
        "show me the data",
        "filter where age > 30",
        "sort by name descending",
    ]

    @pytest.mark.parametrize("inp", SHELL_INPUTS)
    def test_shell_classification(self, inp):
        assert dispatcher.classify(inp) == "shell"

    @pytest.mark.parametrize("inp", PYTHON_INPUTS)
    def test_python_classification(self, inp):
        assert dispatcher.classify(inp) == "python"

    @pytest.mark.parametrize("inp", UNKNOWN_INPUTS)
    def test_unknown_classification(self, inp):
        assert dispatcher.classify(inp) == "unknown"

    def test_empty_input(self):
        assert dispatcher.classify("") == "empty"
        assert dispatcher.classify("   ") == "empty"
