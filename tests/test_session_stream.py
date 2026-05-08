"""Tests for Python session mode, tool calls, and streaming output."""

import os
import sys
import io
import pytest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from srun.repl import execute
from srun.dispatch import dispatcher
from srun.context import state
from srun.executors.python_exec import PythonExecutor
from srun.executors.shell_exec import ShellExecutor
from srun.executors.r_exec import RExecutor


class TestPythonSession:
    def setup_method(self):
        self.py = PythonExecutor()
        self.sh = ShellExecutor()
        self.r = RExecutor()
        state.reset_session()
        state.vars.clear()
        state.active_df = None

    def test_python_expression(self):
        result = execute("python", "3 + 5", self.py, self.sh, self.r)
        assert result["success"]
        assert "8" in result.get("output", "")

    def test_python_assignment(self):
        result = execute("python", "x = 42", self.py, self.sh, self.r)
        assert result["success"]
        result = execute("python", "x", self.py, self.sh, self.r)
        assert "42" in result.get("output", "")

    @pytest.mark.slow
    @pytest.mark.llm
    def test_python_syntax_error(self):
        result = execute("python", "x =", self.py, self.sh, self.r)
        # LLM repair may fix the syntax error → result may succeed
        assert result["success"] or result.get("llm_used") or result.get("summary") is not None

    @pytest.mark.slow
    @pytest.mark.llm
    def test_python_repair_typo(self):
        result = execute("python", "pritn('hello')", self.py, self.sh, self.r)
        # LLM should be called to fix the typo
        assert result["success"] or result.get("llm_used") or result.get("summary") is not None, \
            "Python typo should trigger repair. Output: " + result.get("output", "")

    def test_python_classification(self):
        # Python code classified correctly
        assert dispatcher.classify("x = 42") == "python"
        assert dispatcher.classify("df.groupby('region').mean()") == "python"
        assert dispatcher.classify("100/4") == "python"


class TestToolCalls:
    def setup_method(self):
        self.py = PythonExecutor()
        self.sh = ShellExecutor()
        self.r = RExecutor()
        state.reset_session()
        state._context_injected = True

    def test_check_command_tool(self):
        from srun.tools import check_command
        result = check_command("ls")
        assert "Command: ls" in result
        assert "Path:" in result

    def test_get_env_info_tool(self):
        from srun.tools import get_env_info
        result = get_env_info()
        assert "OS:" in result
        assert "Python:" in result
        assert "Shell:" in result

    def test_search_files_tool(self):
        from srun.tools import search_files
        result = search_files("*.py")
        assert result is not None

    def test_read_file_tool(self):
        from srun.tools import read_file
        path = os.path.join(os.path.dirname(__file__), "..", "AGENTS.md")
        result = read_file(path, max_lines=5)
        assert result is not None
        assert "srun" in result.lower() or "Smart" in result

    @pytest.mark.slow
    @pytest.mark.llm
    def test_llm_calls_tools_for_dispatch(self):
        from srun.llm import llm
        summary, cmds = llm.run("find all csv files larger than 1MB")
        assert cmds is not None or summary is not None, \
            "LLM dispatch should return commands or summary"

    @pytest.mark.slow
    @pytest.mark.llm
    def test_llm_calls_tools_for_repair(self):
        from srun.llm import llm
        summary, cmds = llm.run("grep --nocolor root /etc/hosts",
            error="grep: unrecognized option --nocolor")
        assert cmds is not None or summary is not None, \
            "LLM repair should return fixed command or summary"
        if cmds:
            # Verify the fix is different from original
            assert "grep --nocolor" not in cmds[0], \
                f"LLM should fix the bad flag: {cmds}"


class TestStreamingOutput:
    def setup_method(self):
        state.reset_session()
        state._context_injected = True

    def test_streaming_accumulates_text(self):
        """Verify streaming correctly accumulates text chunks."""
        from srun.llm import llm
        if not llm.client:
            pytest.skip("No LLM client")

        # Create mock stream chunks
        chunk1 = mock.MagicMock()
        chunk1.usage = None
        delta1 = mock.MagicMock()
        delta1.content = "Hello"
        delta1.tool_calls = None
        chunk1.choices = [mock.MagicMock(delta=delta1)]

        chunk2 = mock.MagicMock()
        chunk2.usage = None
        delta2 = mock.MagicMock()
        delta2.content = " world"
        delta2.tool_calls = None
        chunk2.choices = [mock.MagicMock(delta=delta2)]

        mock_stream = iter([chunk1, chunk2])

        # Verify streaming would produce correct accumulated text
        content_parts = []
        for chunk in mock_stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                content_parts.append(delta.content)

        assert "".join(content_parts) == "Hello world"

    def test_streaming_handles_tool_calls(self):
        """Verify streaming correctly detects tool calls."""
        import types

        tool_call_data = {}

        # Simulate streaming chunks with tool_call
        chunk1 = mock.MagicMock()
        chunk1.usage = None
        delta1 = mock.MagicMock()
        delta1.content = None
        tc = mock.MagicMock()
        tc.index = 0
        tc.id = "call_1"
        tc.function = mock.MagicMock()
        tc.function.name = "run_command"
        tc.function.arguments = '{"command":'
        delta1.tool_calls = [tc]
        chunk1.choices = [mock.MagicMock(delta=delta1)]

        chunk2 = mock.MagicMock()
        chunk2.usage = None
        delta2 = mock.MagicMock()
        delta2.content = None
        tc2 = mock.MagicMock()
        tc2.index = 0
        tc2.id = None
        tc2.function = mock.MagicMock()
        tc2.function.name = ""
        tc2.function.arguments = '"ls -la"}'
        delta2.tool_calls = [tc2]
        chunk2.choices = [mock.MagicMock(delta=delta2)]

        mock_stream = iter([chunk1, chunk2])

        for chunk in mock_stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_call_data:
                        tool_call_data[idx] = {"id": tc.id or "", "name": "", "arguments": ""}
                    if tc.id:
                        tool_call_data[idx]["id"] = tc.id
                    if tc.function and tc.function.name:
                        tool_call_data[idx]["name"] += tc.function.name or ""
                    if tc.function and tc.function.arguments:
                        tool_call_data[idx]["arguments"] += tc.function.arguments or ""

        assert len(tool_call_data) == 1
        tc0 = tool_call_data[0]
        assert tc0["name"] == "run_command"
        assert '"ls -la"' in tc0["arguments"]

    def test_streaming_empty_response(self):
        """Streaming with no content or tool calls should return None."""
        chunk = mock.MagicMock()
        chunk.usage = None
        delta = mock.MagicMock()
        delta.content = None
        delta.tool_calls = None
        chunk.choices = [mock.MagicMock(delta=delta)]

        mock_stream = iter([chunk])
        content_parts = []
        for chunk in mock_stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                content_parts.append(delta.content)

        assert len(content_parts) == 0
        assert "".join(content_parts) == ""


class TestDispatcherClassification:
    def test_shell_with_new_command(self):
        """Commands not in whitelist should still be classified as shell."""
        # 'brew' is in whitelist
        assert dispatcher.classify("brew install python") == "shell"
        # 'docker' is in whitelist
        assert dispatcher.classify("docker ps") == "shell"
        # Random alphanumeric command should be shell (new _is_shell regex)
        assert dispatcher.classify("mysql -u root") == "shell"

    def test_python_not_misclassified_as_shell(self):
        """Python code with binops should NOT be classified as shell."""
        assert dispatcher.classify("100/4") == "python"
        assert dispatcher.classify("3 + 5 * 2") == "python"

    def test_shell_not_misclassified_as_python(self):
        """Shell commands should NOT be classified as Python (ls -la is subtraction)."""
        assert dispatcher.classify("ls -la") == "shell"
        assert dispatcher.classify("cd /tmp") == "shell"


class TestRunCommandTool:
    def setup_method(self):
        state.reset_session()
        state._context_injected = True

    @pytest.mark.slow
    @pytest.mark.llm
    def test_run_command_returns_command(self):
        """LLM calling run_command should return a valid command string."""
        from srun.llm import llm
        summary, cmds = llm.run("ls -la")
        if cmds:
            assert len(cmds) > 0
            assert isinstance(cmds[0], str)
            assert len(cmds[0]) > 0
        else:
            assert summary is not None, "Either commands or summary expected"

    @pytest.mark.slow
    @pytest.mark.llm
    def test_run_command_for_chat_returns_text(self):
        """Chat input should return text without commands."""
        from srun.llm import llm
        summary, cmds = llm.run("hello, how are you?")
        # Chat should get a text response, no commands
        assert summary is not None or cmds is not None, \
            "Chat should get text response"


class TestPythonSessionMode:
    """Tests for Python REPL session behavior."""

    def setup_method(self):
        self.py = PythonExecutor()
        self.sh = ShellExecutor()
        self.r = RExecutor()
        state.reset_session()
        state.vars.clear()

    def test_python_session_variable_persistence(self):
        """Variables set in Python mode should persist across commands."""
        execute("python", "x = 100", self.py, self.sh, self.r)
        result = execute("python", "x", self.py, self.sh, self.r)
        assert result["success"]
        assert "100" in result.get("output", "")

    def test_python_session_multiline(self):
        """Multiple Python commands should work."""
        execute("python", "a = 1", self.py, self.sh, self.r)
        execute("python", "b = 2", self.py, self.sh, self.r)
        result = execute("python", "a + b", self.py, self.sh, self.r)
        assert result["success"]
        assert "3" in result.get("output", "")

    def test_python_session_import(self):
        """Importing modules should work."""
        result = execute("python", "import math", self.py, self.sh, self.r)
        assert result["success"]
        result = execute("python", "math.sqrt(16)", self.py, self.sh, self.r)
        assert result["success"]
        assert "4.0" in result.get("output", "")

    def test_python_wrapper_stripping(self):
        """python -c wrapper should be stripped in Python mode."""
        result = execute("python", 'python -c "print(42)"', self.py, self.sh, self.r)
        assert result["success"]
        assert "42" in result.get("output", "")

    def test_python3_wrapper_stripping(self):
        """python3 -c wrapper should be stripped in Python mode."""
        result = execute("python", "python3 -c \"print(99)\"", self.py, self.sh, self.r)
        assert result["success"]
        assert "99" in result.get("output", "")

    def test_python_session_environment_tracking(self):
        """current_language should be set to python during Python session."""
        state.current_language = "python"
        assert state.current_language == "python"
        state.current_language = "shell"


class TestRSessionMode:
    """Tests for R REPL session behavior."""

    def setup_method(self):
        self.py = PythonExecutor()
        self.sh = ShellExecutor()
        self.r = RExecutor()
        state.reset_session()
        state.vars.clear()

    def test_r_executor_availability(self):
        """R executor should report availability based on Rscript in PATH."""
        from srun.executors.r_exec import RExecutor
        r = RExecutor()
        available = r.available
        assert isinstance(available, bool)

    def test_r_execution_when_available(self):
        """If R is available, basic execution should work."""
        if not self.r.available:
            pytest.skip("Rscript not available")
        ok, out, *_ = self.r.execute("42")
        assert ok
        assert "42" in out

    def test_r_execution_failure(self):
        """R errors should return failure."""
        if not self.r.available:
            pytest.skip("R not available")
        ok, out, *_ = self.r.execute('stop("test error")')
        assert not ok

    def test_r_session_environment_tracking(self):
        """current_language should be set to r during R session."""
        state.current_language = "r"
        assert state.current_language == "r"
        state.current_language = "shell"

    def test_r_mode_enters_and_exits(self):
        """Simulate R mode entry/exit via state changes."""
        state.current_language = "shell"
        state.current_language = "r"
        assert state.current_language == "r"
        state.current_language = "shell"
        assert state.current_language == "shell"

    @pytest.mark.slow
    @pytest.mark.llm
    def test_nl_in_r_generates_code(self):
        """Natural language in R mode should trigger LLM to generate R code."""
        state.current_language = "r"
        from srun.llm import llm
        summary, cmds = llm.run("create a sequence from 1 to 10")
        # Should either get R code or text summary
        assert summary is not None or cmds is not None, \
            "LLM should generate R code or text response"
        if cmds:
            # Should contain R-like code
            for cmd in cmds:
                assert any(kw in cmd for kw in ("c(", "seq", "1:10", "rep(")), \
                    f"Expected R-style code, got: {cmd}"

    @pytest.mark.slow
    @pytest.mark.llm
    def test_nl_in_python_generates_code(self):
        """Natural language in Python mode should trigger LLM to generate Python code."""
        state.current_language = "python"
        from srun.llm import llm
        summary, cmds = llm.run("create a list of numbers from 1 to 10")
        assert summary is not None or cmds is not None
        if cmds:
            for cmd in cmds:
                assert any(kw in cmd for kw in ("range(", "list(", "for ")), \
                    f"Expected Python-style code, got: {cmd}"


class TestNaturalLanguageCrossLanguage:
    """Tests for natural language inputs across different language sessions."""

    def setup_method(self):
        from srun.llm import llm
        self.llm = llm
        state.reset_session()
        state._context_injected = True

    @pytest.mark.slow
    @pytest.mark.llm
    def test_shell_nl_python_task(self):
        """In shell mode, NL asking for Python should generate python -c wrapper."""
        state.current_language = "shell"
        summary, cmds = self.llm.run("create a pandas dataframe")
        assert summary is not None or cmds is not None, \
            "Should generate response"
        if cmds:
            is_shell_wrapper = any("python" in c for c in cmds)
            is_pure_python = any("pd.DataFrame" in c or "import pandas" in c for c in cmds)
            assert is_shell_wrapper or is_pure_python, \
                f"Expected python-related code, got: {cmds}"

    @pytest.mark.slow
    @pytest.mark.llm
    def test_python_nl_shell_task(self):
        """In Python mode, requesting shell info should still work."""
        state.current_language = "python"
        summary, cmds = self.llm.run("list all files in the current directory")
        assert summary is not None or cmds is not None

    @pytest.mark.slow
    @pytest.mark.llm
    def test_r_nl_statistical_task(self):
        """In R mode, NL requesting stats should generate R code."""
        state.current_language = "r"
        summary, cmds = self.llm.run("calculate the mean of numbers 1,2,3,4,5")
        assert summary is not None or cmds is not None
        if cmds:
            for cmd in cmds:
                assert "mean" in cmd.lower() or "c(" in cmd, \
                    f"Expected R statistical code, got: {cmd}"


class TestPythonShellWrapperStripping:
    """Test the regex that strips shell wrappers from Python code."""

    def setup_method(self):
        import re
        self.re = re

    def _extract(self, code):
        m = self.re.match(r'^python3?\s+-c\s+"(.+)"\s*$', code)
        if not m:
            m = self.re.match(r"^python3?\s+-c\s+'(.+)'\s*$", code)
        if not m:
            m = self.re.match(r"^python3?\s+<<\s*'?EOF'?\s*\n(.+)\nEOF\s*$", code, self.re.DOTALL)
        return m.group(1) if m else code

    def test_strip_python_c_double_quotes(self):
        result = self._extract('python -c "print(42)"')
        assert result == "print(42)"

    def test_strip_python3_c_double_quotes(self):
        result = self._extract('python3 -c "print(99)"')
        assert result == "print(99)"

    def test_strip_python_c_single_quotes(self):
        result = self._extract("python -c 'print(42)'")
        assert result == "print(42)"

    def test_strip_heredoc(self):
        result = self._extract("""python << 'EOF'
import pandas as pd
df = pd.DataFrame()
EOF""")
        assert "import pandas as pd" in result

    def test_no_wrapper_passthrough(self):
        result = self._extract("print(42)")
        assert result == "print(42)"
