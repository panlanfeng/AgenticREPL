"""Tests for python executor including AST column resolution."""

import pytest
from srun.executors.python_exec import PythonExecutor
from srun.context import state


class TestPythonExecutor:
    def setup_method(self):
        self.py = PythonExecutor()
        state.vars.clear()
        state.active_df = None

    def test_simple_expression(self):
        ok, out, *_ = self.py.execute("100/4")
        assert ok
        assert "25" in out

    def test_arithmetic(self):
        ok, out, *_ = self.py.execute("3 + 5 * 2")
        assert ok
        assert "13" in out

    def test_print(self):
        ok, out, *_ = self.py.execute('print("hello")')
        assert ok
        assert "hello" in out

    def test_variable_assignment(self):
        ok, out, *_ = self.py.execute("x = 42")
        assert ok
        ok, out, *_ = self.py.execute("x")
        assert ok
        assert "42" in out

    def test_column_resolution(self):
        import pandas as pd

        state.add_var(
            "df",
            {
                "type": "DataFrame",
                "columns": ["name", "age", "score"],
                "rows": 3,
            },
        )
        state.set_active("df")

        code = "df.groupby(age).mean()"
        resolved = self.py._resolve_columns(code)
        assert "age" in resolved
        assert resolved != code

    def test_syntax_error(self):
        ok, output, *rest = self.py.execute("df.groupby(!!!")
        assert not ok

    def test_pandas_dataframe_schema_tracking(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        self.py.namespace["df"] = df
        self.py._track_vars()
        assert "df" in state.vars
        assert state.vars["df"]["columns"] == ["x", "y"]
        assert state.vars["df"]["rows"] == 3
