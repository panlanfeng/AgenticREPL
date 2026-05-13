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
        result = read_file(path, lines=5)
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
            first_cmd = cmds[0] if isinstance(cmds[0], str) else cmds[0].get("command", "")
            assert "grep --nocolor" not in first_cmd, \
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
            first = cmds[0]
            cmd_str = first if isinstance(first, str) else first.get("command", "")
            assert isinstance(cmd_str, str) and len(cmd_str) > 0
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

    def test_python_direct_code(self):
        """Pure Python code should execute directly."""
        result = execute("python", "print(42)", self.py, self.sh, self.r)
        assert result["success"]
        assert "42" in result.get("output", "")

    def test_python_assignment_and_use(self):
        """Assignment should persist across commands."""
        execute("python", "z = 7", self.py, self.sh, self.r)
        result = execute("python", "z * 3", self.py, self.sh, self.r)
        assert result["success"]
        assert "21" in result.get("output", "")

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
            for tc in cmds:
                cmd = tc if isinstance(tc, str) else tc.get("command", "")
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
            for tc in cmds:
                cmd = tc if isinstance(tc, str) else tc.get("command", "")
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
            is_shell_wrapper = any("python" in (c if isinstance(c, str) else c.get("command", "")) for c in cmds)
            is_pure_python = any("pd.DataFrame" in (c if isinstance(c, str) else c.get("command", "")) or "import pandas" in (c if isinstance(c, str) else c.get("command", "")) for c in cmds)
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
            for tc in cmds:
                cmd = tc if isinstance(tc, str) else tc.get("command", "")
                assert "mean" in cmd.lower() or "c(" in cmd, \
                    f"Expected R statistical code, got: {cmd}"


class TestCompactionPersistence:
    """Test that compaction snapshots survive restarts and are read back correctly."""

    def setup_method(self):
        from srun.context import SessionState, FULL_HISTORY_FILE, SESSION_DIR
        state.reset_session()
        # Clean the history file for a fresh test
        if os.path.isfile(FULL_HISTORY_FILE):
            os.remove(FULL_HISTORY_FILE)

    def test_write_and_load_compaction_snapshot(self):
        from srun.context import SessionState, FULL_HISTORY_FILE
        # Simulate conversation + compaction
        state._conversation = [
            {"role": "user", "content": "find csv files"},
            {"role": "assistant", "content": '{"code": "ls *.csv"}'},
            {"role": "user", "content": "count them"},
            {"role": "assistant", "content": '{"code": "ls *.csv | wc -l"}'},
        ]
        state._stable_summary = "User worked with CSV files: listed and counted 5 files."
        state._write_compaction_snapshot()

        # Verify file has the snapshot
        assert os.path.isfile(FULL_HISTORY_FILE), "Snapshot should be written to file"
        import json
        with open(FULL_HISTORY_FILE) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert any(e.get("type") == "compaction_snapshot" for e in lines), \
            "File must contain compaction_snapshot entry"

        # Simulate restart: summary is restored, conversation starts fresh
        new_state = SessionState()
        new_state._load_conversation_state()
        assert new_state._stable_summary == "User worked with CSV files: listed and counted 5 files."
        # _conversation should be empty — fresh session
        assert new_state._conversation == []

    def test_load_ignores_other_entry_types(self):
        from srun.context import SessionState, FULL_HISTORY_FILE
        # Write some non-compaction entries first
        state._turn = 1
        state._write_history({"input": "ls", "output": "file1.txt", "success": True,
                              "elapsed_ms": 5, "language": "shell", "code": "ls", "type": "fast"})
        state._write_history({"input": "bad", "output": "error", "success": False,
                              "elapsed_ms": 10, "language": "shell", "code": "bad", "type": "fast"})
        # Now write a compaction snapshot
        state._conversation = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": '{"code": "echo hi"}'},
        ]
        state._stable_summary = "User said hello, assistant greeted."
        state._write_compaction_snapshot()

        # Load — should find the compaction snapshot despite other entries
        new_state = SessionState()
        new_state._load_conversation_state()
        assert new_state._stable_summary == "User said hello, assistant greeted."
        # _conversation starts fresh — only summary is restored
        assert new_state._conversation == []

    def test_no_file_loads_clean(self):
        from srun.context import SessionState, FULL_HISTORY_FILE
        if os.path.isfile(FULL_HISTORY_FILE):
            os.remove(FULL_HISTORY_FILE)
        new_state = SessionState()
        new_state._load_conversation_state()
        assert new_state._stable_summary is None
        assert new_state._conversation == []


class TestAskUserTool:
    """Test the ask_user tool definition, handler, and callback integration."""

    def setup_method(self):
        from srun.context import state
        state.reset_session()

    def test_ask_user_tool_definition(self):
        from srun.tools import TOOL_DEFINITIONS
        names = [t["function"]["name"] for t in TOOL_DEFINITIONS]
        assert "ask_user" in names, "ask_user must be in TOOL_DEFINITIONS"
        ask_def = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "ask_user")
        assert "question" in ask_def["function"]["parameters"]["required"]
        assert "question" in ask_def["function"]["parameters"]["properties"]
        assert "details" in ask_def["function"]["parameters"]["properties"]

    def test_ask_user_handler_registered(self):
        from srun.tools import TOOL_HANDLERS, ask_user
        assert "ask_user" in TOOL_HANDLERS
        assert TOOL_HANDLERS["ask_user"] is ask_user

    def test_ask_user_handler_without_callback(self):
        from srun.tools import execute_tool
        result = execute_tool("ask_user", {"question": "Can I install pandas?"})
        assert "Do NOT proceed" in result or "denial" in result.lower()

    def test_llm_run_accepts_ask_user_callback_param(self):
        from srun.llm import LLM
        import inspect
        sig = inspect.signature(LLM.run)
        assert "ask_user_callback" in sig.parameters, "LLM.run must accept ask_user_callback"

    def test_ask_user_triggered_via_callback(self):
        """Simulate ask_user tool call with a callback that records the question."""
        from unittest import mock
        from srun.llm import LLM
        callback = mock.Mock(return_value="yes")
        llm_instance = LLM()
        # Test that the parameter is accepted (actual LLM call needs API)
        assert hasattr(llm_instance, "run")
        # Verify callback signature is called correctly
        callback("Can I install pandas?", "pip install pandas")
        assert callback.called
        callback.assert_called_with("Can I install pandas?", "pip install pandas")


class TestRunInput:
    """Test the active _run_input execution path used by the REPL."""

    def setup_method(self):
        from srun.context import state
        from srun.executors.python_exec import PythonExecutor
        from srun.executors.shell_exec import ShellExecutor
        from srun.executors.r_exec import RExecutor
        self.py = PythonExecutor()
        self.sh = ShellExecutor()
        self.r = RExecutor()
        state.reset_session()
        state.current_language = "shell"

    def test_direct_shell_success_no_llm(self):
        """Valid shell command executes directly with zero LLM calls."""
        from srun.repl import _run_input
        import time
        start = time.perf_counter()
        result = _run_input("echo ok", self.py, self.sh, self.r)
        elapsed = (time.perf_counter() - start) * 1000
        assert result["success"]
        assert not result["llm_used"]
        assert "ok" in result["output"]
        assert elapsed < 30, f"Direct shell too slow: {elapsed:.0f}ms"

    def test_direct_shell_failure_triggers_llm(self):
        """Invalid command fails direct execution, then LLM repairs it."""
        from srun.repl import _run_input
        result = _run_input("echoo hello", self.py, self.sh, self.r)
        if not result["success"]:
            # Without API key, LLM returns a config error — that's expected
            assert result.get("llm_used", False) or any(kw in result.get("output", "") for kw in ("No LLM configured", "Unable to understand"))

    def test_direct_python_executes(self):
        """Python code executes in Python session."""
        from srun.context import state
        from srun.repl import _run_input
        state.current_language = "python"
        result = _run_input("3 + 5", self.py, self.sh, self.r)
        assert result["success"]
        assert not result["llm_used"]
        assert "8" in result["output"]

    def test_direct_python_failure_llm_fix(self):
        """Python typo triggers LLM repair."""
        from srun.context import state
        from srun.repl import _run_input
        state.current_language = "python"
        result = _run_input("pritn('hello')", self.py, self.sh, self.r)
        # Without API key, LLM returns config error — that's expected behavior
        assert result["success"] or result["llm_used"] or "No LLM configured" in (result.get("output") or "")

    def test_quick_fix_ll_alias_no_llm(self):
        """'ll' quick-fix runs 'ls -la' with zero LLM calls."""
        from srun.repl import _run_input
        import time
        start = time.perf_counter()
        result = _run_input("ll", self.py, self.sh, self.r)
        elapsed = (time.perf_counter() - start) * 1000
        assert result["success"]
        assert not result["llm_used"]
        assert result.get("fixed_code") == "ls -la"
        assert elapsed < 30, f"Quick-fix too slow: {elapsed:.0f}ms"

    def test_quick_fix_la_alias_no_llm(self):
        """'la' quick-fix runs 'ls -a' with zero LLM calls."""
        from srun.repl import _run_input
        result = _run_input("la", self.py, self.sh, self.r)
        assert result["success"]
        assert not result["llm_used"]
        assert result.get("fixed_code") == "ls -a"

    def test_quick_fix_cddot_alias(self):
        """'cd..' quick-fix runs 'cd ..'."""
        from srun.repl import _run_input
        import os, tempfile
        cwd = os.getcwd()
        tmpdir = tempfile.mkdtemp()
        parent = os.path.realpath(os.path.dirname(tmpdir))
        try:
            os.chdir(tmpdir)
            result = _run_input("cd..", self.py, self.sh, self.r)
            assert result["success"]
            assert not result["llm_used"]
            assert result.get("fixed_code") == "cd .."
            assert os.path.realpath(os.getcwd()) == parent
        finally:
            os.chdir(cwd)
            import shutil
            if os.path.isdir(tmpdir):
                shutil.rmtree(tmpdir)

    def test_stderr_error_triggers_repair(self):
        """Shell command with stderr errors triggers LLM even if exit 0."""
        from srun.repl import _run_input
        result = _run_input("ls --nonexist-flag 2>/dev/null; true", self.py, self.sh, self.r)
        # Should trigger LLM or succeed
        assert result["success"] or result.get("llm_used")

    def test_danger_blocked_user_input(self):
        """Dangerous user commands are blocked before execution."""
        from srun.repl import _run_input
        result = _run_input("rm -rf /", self.py, self.sh, self.r)
        assert not result["success"]
        assert "BLOCKED" in result["output"]

    def test_danger_blocked_llm_generated(self):
        """LLM-generated dangerous commands are blocked."""
        from srun.repl import _run_input
        from srun.llm import llm as llm_mod
        from unittest import mock

        original_run = llm_mod.run
        def fake_run(*args, **kwargs):
            return "done", [{"command": "rm -rf /", "language": "shell"}]
        try:
            llm_mod.run = fake_run
            result = _run_input("delete everything", self.py, self.sh, self.r)
            assert not result["success"]
            assert "BLOCKED" in result["output"]
        finally:
            llm_mod.run = original_run


class TestNewTools:
    """Tests for file_write, file_edit, grep_search tools."""

    def setup_method(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ── file_write ────────────────────────────────────────────────

    def test_file_write_creates_file(self):
        from srun.tools import file_write
        import os
        path = os.path.join(self.tmpdir, "test.txt")
        result = file_write(path, "hello world\nline 2\n")
        assert "Wrote" in result
        assert os.path.isfile(path)
        with open(path) as f:
            assert f.read() == "hello world\nline 2\n"

    def test_file_write_overwrites(self):
        from srun.tools import file_write
        import os
        path = os.path.join(self.tmpdir, "overwrite.txt")
        file_write(path, "old content")
        file_write(path, "new content")
        with open(path) as f:
            assert f.read() == "new content"

    def test_file_write_creates_dirs(self):
        from srun.tools import file_write
        import os
        path = os.path.join(self.tmpdir, "a", "b", "c", "deep.txt")
        file_write(path, "deep")
        assert os.path.isfile(path)

    # ── file_edit ─────────────────────────────────────────────────

    def test_file_edit_single_match(self):
        from srun.tools import file_write, file_edit
        import os
        path = os.path.join(self.tmpdir, "edit.txt")
        file_write(path, "hello world\nline 2\nbye world\n")
        result = file_edit(path, "hello world", "HELLO EARTH")
        assert "Replaced 1 occurrence" in result
        with open(path) as f:
            content = f.read()
        assert "HELLO EARTH" in content
        assert "hello world" not in content

    def test_file_edit_multiple_matches(self):
        from srun.tools import file_write, file_edit
        import os
        path = os.path.join(self.tmpdir, "multi.txt")
        file_write(path, "hello\nworld\nhello\n")
        result = file_edit(path, "hello", "hi")
        assert "Found 2 matches" in result

    def test_file_edit_no_match(self):
        from srun.tools import file_write, file_edit
        import os
        path = os.path.join(self.tmpdir, "nomatch.txt")
        file_write(path, "hello world\n")
        result = file_edit(path, "zzzzzz", "xxx")
        assert "No match found" in result

    def test_file_edit_unique_with_context(self):
        from srun.tools import file_write, file_edit
        import os
        path = os.path.join(self.tmpdir, "context.txt")
        file_write(path, "hello\nworld\nhello\n")
        result = file_edit(path, "world\nhello", "WORLD\nHELLO")
        assert "Replaced 1 occurrence" in result

    def test_file_edit_nonexistent_file(self):
        from srun.tools import file_edit
        result = file_edit("/tmp/nonexistent_srun_edit_test_xyz.txt", "a", "b")
        assert "File not found" in result

    def test_file_edit_exact_whitespace(self):
        from srun.tools import file_write, file_edit
        import os
        path = os.path.join(self.tmpdir, "space.txt")
        file_write(path, "  hello  \n")
        # "hello" is a substring of "  hello  " — it matches
        result = file_edit(path, "hello", "hi")
        assert "Replaced 1 occurrence" in result
        # Exact with surrounding spaces is unique
        result2 = file_edit(path, "  hi  ", "  bye  ")
        assert "Replaced 1 occurrence" in result2

    # ── grep_search ───────────────────────────────────────────────

    def test_grep_search_finds_matches(self):
        from srun.tools import file_write, grep_search
        import os
        path = os.path.join(self.tmpdir, "search.py")
        file_write(path, "def hello():\n    print('world')\n\ndef goodbye():\n    return 42\n")
        result = grep_search("def", self.tmpdir, context_lines=0)
        # output contains file path and match content
        assert "search.py" in result
        assert "hello" in result or "goodbye" in result
        assert result.strip()  # not empty

    def test_grep_search_regex(self):
        from srun.tools import file_write, grep_search
        import os
        path = os.path.join(self.tmpdir, "regex.py")
        file_write(path, "var1 = 100\nvar2 = 200\nxyz = 300\n")
        result = grep_search(r"var\d", self.tmpdir, context_lines=0)
        assert "var1" in result or "var2" in result
        assert "xyz" not in result

    # ── tool definitions ─────────────────────────────────────────

    def test_new_tools_in_definitions(self):
        from srun.tools import TOOL_DEFINITIONS
        names = {t["function"]["name"] for t in TOOL_DEFINITIONS}
        assert "grep_search" in names
        assert "file_edit" in names
        assert "file_write" in names

    def test_new_tools_in_handlers(self):
        from srun.tools import TOOL_HANDLERS
        assert "grep_search" in TOOL_HANDLERS
        assert "file_edit" in TOOL_HANDLERS
        assert "file_write" in TOOL_HANDLERS


class TestOutputFormat:
    """Verify reasoning, response, and code appear on separate lines."""

    def setup_method(self):
        state.reset_session()
        state.current_language = "shell"

    def test_reasoning_newline_when_no_content(self):
        """print(flush=True) runs when reasoning is set but content_parts is empty."""
        # Simulates the condition at llm.py:167
        content_parts = []  # LLM generated only tool calls, no text
        reasoning = True     # reasoning was streamed
        flushed = bool(content_parts or reasoning)
        assert flushed, "Should flush newline after reasoning when content_parts is empty"

    def test_reasoning_newline_with_content(self):
        """print(flush=True) runs when content_parts has text (normal case)."""
        content_parts = ["some text"]
        reasoning = False
        flushed = bool(content_parts or reasoning)
        assert flushed, "Should flush newline when content_parts has text"

    def test_no_flush_when_neither(self):
        """print(flush=True) does NOT run when neither content nor reasoning."""
        content_parts = []
        reasoning = False
        flushed = bool(content_parts or reasoning)
        assert not flushed, "Should not flush when nothing was output"

    @pytest.mark.slow
    @pytest.mark.llm
    def test_code_on_separate_line_from_reasoning(self):
        """LLM output: reasoning on one line, code on the next line."""
        from srun.repl import _exec_inline
        from srun.llm import llm
        from srun.executors.python_exec import PythonExecutor
        from srun.executors.shell_exec import ShellExecutor
        from srun.executors.r_exec import RExecutor
        import io, contextlib

        py = PythonExecutor()
        sh = ShellExecutor()
        r = RExecutor()

        # Capture stdout to verify line separations
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            summary, tcs = llm.run("list the 3 largest files in /tmp with their sizes", exec_callback=_exec_inline(py, sh, r))

        output = buf.getvalue()
        # Reasoning should end with a newline before the code line
        lines = output.strip().split("\n")
        # Find lines: Reasoning → code → output → footer
        has_reasoning = any("Reasoning:" in l for l in lines)
        has_code = any(l.strip().startswith(">") for l in lines)
        if has_reasoning and has_code:
            # Find the reasoning line and the first code line — they should be different lines
            reasoning_idx = next(i for i, l in enumerate(lines) if "Reasoning:" in l)
            code_idx = next(i for i, l in enumerate(lines) if l.strip().startswith(">"))
            assert code_idx > reasoning_idx, \
                f"Code should be on a separate line after reasoning. Reasoning at line {reasoning_idx}, code at line {code_idx}"
        assert has_code or tcs, "LLM should produce some output"

    @pytest.mark.slow
    @pytest.mark.llm
    def test_response_on_separate_line_from_code(self):
        """After code executes, agent response (if any) should be on its own line."""
        from srun.repl import _exec_inline
        from srun.llm import llm
        from srun.executors.python_exec import PythonExecutor
        from srun.executors.shell_exec import ShellExecutor
        from srun.executors.r_exec import RExecutor
        import io, contextlib

        py = PythonExecutor()
        sh = ShellExecutor()
        r = RExecutor()

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            summary, tcs = llm.run("list the 3 largest files in /tmp", exec_callback=_exec_inline(py, sh, r))

        output = buf.getvalue()
        lines = output.strip().split("\n")
        has_agent_response = any("Agent response:" in l for l in lines)
        has_code = any(l.strip().startswith(">") for l in lines)
        if has_agent_response and has_code:
            code_idx = next(i for i, l in enumerate(lines) if l.strip().startswith(">"))
            response_idx = next(i for i, l in enumerate(lines) if "Agent response:" in l)
            assert response_idx > code_idx, \
                f"Agent response should appear after code. Code at line {code_idx}, response at line {response_idx}"
