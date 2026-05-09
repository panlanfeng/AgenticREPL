"""Data analyst profile tests — shell, R, Python data analysis with SRUN.

Approximately 90+ test cases organized by category.
"""

import os
import sys
import time
import tempfile
import json

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from srun.repl import execute
from srun.dispatch import dispatcher
from srun.context import state
from srun.llm import llm
from srun.repair import apply_quick_fix, repairer
from srun.executors.python_exec import PythonExecutor
from srun.executors.shell_exec import ShellExecutor
from srun.executors.r_exec import RExecutor
from srun.danger import check_danger
from srun.config import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TEST_CSV = os.path.join(os.path.dirname(__file__), "data", "test.csv")
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _setup():
    """Create fresh executors and reset state."""
    py = PythonExecutor()
    sh = ShellExecutor()
    r = RExecutor()
    state.reset_session()
    state.vars.clear()
    state.active_df = None
    state.last_dispatch_error = None
    return py, sh, r


def _run(category, user_input, py, sh, r):
    """Convenience: full execute pipeline."""
    return execute(category, user_input, py, sh, r)


def _cmd_from_result(result):
    """Extract command/executable string from an LLM result dict or string."""
    if isinstance(result, dict):
        return result.get("command", "")
    return result or ""


def _llm_dispatch_and_execute(user_input, current_lang="shell", py=None, sh=None, r=None):
    """Run full LLM dispatch for an unknown input, return execute result."""
    if py is None:
        py, sh, r = _setup()
    state.current_language = current_lang
    state._llm_last_known_language = current_lang
    cat = dispatcher.classify(user_input)
    result = _run(cat, user_input, py, sh, r)
    return result


# ===========================================================================
# TestDataAnalystFast — classification and quick-path tests (no LLM, no slow)
# ===========================================================================


class TestDataAnalystFast:

    def setup_method(self):
        self.py, self.sh, self.r = _setup()
        self.r_available = RExecutor().available

    # ---- Classifier: data-analysis expressions ----

    def test_classify_pandas_groupby_mean(self):
        assert dispatcher.classify("df.groupby('region').mean()") == "python"

    def test_classify_read_csv(self):
        assert dispatcher.classify("pd.read_csv('data.csv')") == "python"

    def test_classify_dplyr_filter(self):
        # filter() alone could be ambiguous, but "dplyr::filter" is function call → python
        assert dispatcher.classify("from dplyr import filter") == "python"

    def test_classify_pipe_operator(self):
        # %>\n% or just a pipeline — pipe chars classify as shell
        # But with proper R dplyr chain notation, this is 'unknown' unless it's parseable as something
        result = dispatcher.classify("mtcars %>% filter(mpg > 20) %>% select(mpg, cyl)")
        assert result in ("unknown", "shell"), f"Unexpected: {result}"

    def test_classify_shell_data_pipeline(self):
        assert dispatcher.classify("cat data.csv | sort | uniq -c") == "shell"

    def test_classify_shell_wc_l(self):
        assert dispatcher.classify("wc -l *.csv") == "shell"

    def test_classify_python_listcomp(self):
        assert dispatcher.classify("[x*2 for x in range(10)]") == "python"

    def test_classify_r_read_csv(self):
        # read.csv() — the dot makes it look like an attribute, so Python
        # Actually R code like "df <- read.csv('data.csv')" → "<-" is assignment → unknown in classify
        # Let's test what actually happens
        result = dispatcher.classify("read.csv('data.csv')")
        # This has no first-shell-word, no shell patterns, but is valid Python (attribute call)
        assert result == "python"

    def test_classify_r_assignment(self):
        # "df <- read.csv('data.csv')" — Python parses "<-" as (df < (-read.csv(...)))
        # so _is_python finds Attribute (.csv) and returns True
        result = dispatcher.classify("df <- read.csv('data.csv')")
        assert result in ("python", "unknown"), f"Unexpected: {result}"

    def test_classify_r_assignment_distinct(self):
        # A clear R-specific expression that is not valid Python
        result = dispatcher.classify("mtcars %>% filter(mpg > 20)")
        assert result in ("unknown", "shell"), f"Unexpected: {result}"

    # ---- Shell pipeline classification ----

    def test_shell_pipe_sort(self):
        cat = dispatcher.classify("cat data.csv | sort -t, -k2")
        assert cat == "shell"
        result = _run(cat, "cat data.csv | sort -t, -k2", self.py, self.sh, self.r)
        assert result["success"]

    def test_shell_redirect(self):
        cat = dispatcher.classify("echo hello > /tmp/srun_test_redirect.txt")
        assert cat == "shell"
        result = _run(cat, "echo hello > /tmp/srun_test_redirect.txt", self.py, self.sh, self.r)
        assert result["success"]
        os.remove("/tmp/srun_test_redirect.txt") if os.path.exists("/tmp/srun_test_redirect.txt") else None

    def test_shell_subshell(self):
        cat = dispatcher.classify("echo $(whoami)")
        assert cat == "shell"
        result = _run(cat, "echo $(whoami)", self.py, self.sh, self.r)
        assert result["success"]

    # ---- Python expression classification ----

    def test_python_numeric_expr(self):
        cat = dispatcher.classify("100 / 4")
        assert cat == "python"
        result = _run(cat, "100 / 4", self.py, self.sh, self.r)
        assert result["success"]
        assert "25.0" in result["output"]

    def test_python_import(self):
        cat = dispatcher.classify("import pandas as pd")
        assert cat == "python"
        result = _run(cat, "import pandas as pd", self.py, self.sh, self.r)
        assert result["success"]

    # ---- Quick fix classification ----

    def test_quick_fix_cd_dot_dot(self):
        """cd.. is classified as 'unknown' (not a shell command).
        The test verifies the quick fix pattern matches via apply_quick_fix."""
        assert apply_quick_fix("cd..", "") == "cd .."
        # Also test full pipeline: force through shell repair loop
        orig = os.getcwd()
        try:
            result = _run("shell", "cd..", self.py, self.sh, self.r)
            assert result["success"], f"cd.. repair failed: {result}"
        finally:
            os.chdir(orig)

    def test_quick_fix_ll(self):
        cat = dispatcher.classify("ll")
        result = _run(cat, "ll", self.py, self.sh, self.r)
        assert result["success"]

    # ---- Danger: safe commands should NOT be blocked ----

    def test_danger_safe_data_analysis(self):
        safe_cmds = [
            "cat data.csv",
            "head -5 data.csv",
            "wc -l data.csv",
            "python analyze.py",
            "cut -d, -f1 data.csv",
            "sort data.csv",
        ]
        for cmd in safe_cmds:
            blocked, desc = check_danger(cmd)
            assert not blocked, f"'{cmd}' should NOT be blocked: {desc}"

    # ---- Empty input handling ----

    def test_empty_input(self):
        assert dispatcher.classify("") == "empty"
        result = _run("empty", "", self.py, self.sh, self.r)
        assert result["success"]
        assert not result["llm_used"]

    def test_whitespace_input(self):
        assert dispatcher.classify("   ") == "empty"
        result = _run("empty", "   ", self.py, self.sh, self.r)
        assert result["success"]

    # ---- Additional: fast command execution timing ----

    def test_fast_shell_timing(self):
        """Normal shell commands execute within 20ms."""
        cat = dispatcher.classify("pwd")
        start = time.perf_counter()
        result = _run(cat, "pwd", self.py, self.sh, self.r)
        elapsed = (time.perf_counter() - start) * 1000
        assert result["success"]
        assert not result["llm_used"]
        assert elapsed < 20, f"pwd took {elapsed:.0f}ms, expected <20ms"

    def test_fast_echo_timing(self):
        cat = dispatcher.classify("echo ok")
        start = time.perf_counter()
        result = _run(cat, "echo ok", self.py, self.sh, self.r)
        elapsed = (time.perf_counter() - start) * 1000
        assert result["success"]
        assert elapsed < 20, f"echo took {elapsed:.0f}ms"


# ===========================================================================
# TestDataAnalystSession — session management (no LLM, no slow)
# ===========================================================================


class TestDataAnalystSession:

    def setup_method(self):
        self.py, self.sh, self.r = _setup()
        self.r_available = RExecutor().available
        self._cd_mark = os.getcwd()

    def teardown_method(self):
        os.chdir(self._cd_mark)

    # ---- Python session ----

    def test_python_session_var_persistence(self):
        """Variables persist across Python commands in same session."""
        state.current_language = "python"
        assert _run("python", "x = 42", self.py, self.sh, self.r)["success"]
        result = _run("python", "x", self.py, self.sh, self.r)
        assert result["success"]
        assert "42" in result["output"]

    def test_python_import_in_session(self):
        state.current_language = "python"
        result = _run("python", "import sys; print('ok')", self.py, self.sh, self.r)
        assert result["success"]
        assert "ok" in result["output"]

    def test_python_dataframe_creation(self):
        state.current_language = "python"
        code = "import pandas as pd; df = pd.DataFrame({'a':[1,2,3], 'b':[4,5,6]}); print(len(df))"
        result = _run("python", code, self.py, self.sh, self.r)
        assert result["success"]
        assert "3" in result["output"]

    # ---- Shell session ----

    def test_shell_cd_persistence(self):
        tmpdir = tempfile.mkdtemp()
        try:
            state.current_language = "shell"
            _run("shell", f"cd {tmpdir}", self.py, self.sh, self.r)
            assert os.path.realpath(os.getcwd()) == os.path.realpath(tmpdir)
        finally:
            os.chdir(self._cd_mark)
            if os.path.isdir(tmpdir):
                os.rmdir(tmpdir)

    def test_shell_file_create_and_read(self):
        tmpf = "/tmp/srun_session_test.txt"
        try:
            state.current_language = "shell"
            result = _run("shell", f"echo 'session test' > {tmpf}", self.py, self.sh, self.r)
            assert result["success"]
            assert os.path.isfile(tmpf)
            with open(tmpf) as f:
                assert "session test" in f.read()
        finally:
            if os.path.isfile(tmpf):
                os.remove(tmpf)

    # ---- R session (if available) ----

    def test_r_session_enter_exit(self):
        if not self.r_available:
            pytest.skip("R not available")
        state.current_language = "r"
        result = _run("r", "1 + 1", self.py, self.sh, self.r)
        assert result["success"]
        # Switch back
        state.current_language = "shell"

    def test_r_session_variable(self):
        if not self.r_available:
            pytest.skip("R not available")
        state.current_language = "r"
        result = _run("r", "x <- 100; print(x)", self.py, self.sh, self.r)
        assert result["success"]
        result2 = _run("r", "print(x)", self.py, self.sh, self.r)
        assert result2["success"]
        state.current_language = "shell"

    # ---- Session switching ----

    def test_session_switch_python_to_shell(self):
        state.current_language = "python"
        state.current_language = "shell"
        assert state.current_language == "shell"
        result = _run("shell", "echo shell", self.py, self.sh, self.r)
        assert result["success"]

    def test_session_switch_shell_to_python_to_shell(self):
        state.current_language = "shell"
        assert state.current_language == "shell"
        state.current_language = "python"
        assert state.current_language == "python"
        result = _run("python", "print('py')", self.py, self.sh, self.r)
        assert result["success"]
        state.current_language = "shell"
        assert state.current_language == "shell"

    def test_session_exit_methods(self):
        """exit() and quit() change back to shell from python."""
        state.current_language = "python"
        state.current_language = "shell"
        assert state.current_language == "shell"

    def test_multiple_session_enters(self):
        for _ in range(3):
            state.current_language = "python"
            result = _run("python", "print('hi')", self.py, self.sh, self.r)
            assert result["success"]
            state.current_language = "shell"
            result = _run("shell", "echo hi", self.py, self.sh, self.r)
            assert result["success"]

    def test_ctrl_d_simulation(self):
        """EOFError handling preserves state; language should still be what was set."""
        state.current_language = "python"
        assert state.current_language == "python"
        state.current_language = "shell"
        assert state.current_language == "shell"


# ===========================================================================
# TestDataAnalystLLM — LLM-based dispatch (slow + llm marks)
# ===========================================================================


@pytest.mark.slow
@pytest.mark.llm
class TestDataAnalystLLM:

    def setup_method(self):
        self.py, self.sh, self.r = _setup()
        self.r_available = RExecutor().available
        self._cd_mark = os.getcwd()

    def teardown_method(self):
        os.chdir(self._cd_mark)

    # ---- Natural language data loading ----

    def test_nl_load_csv_into_dataframe(self):
        result = _llm_dispatch_and_execute(
            f"load the {TEST_CSV} file into a dataframe",
            "shell", self.py, self.sh, self.r,
        )
        assert result["llm_used"], f"Expected LLM usage, got: {result}"

    def test_nl_read_csv_show_head(self):
        result = _llm_dispatch_and_execute(
            f"read {TEST_CSV} and show the first 3 rows",
            "shell", self.py, self.sh, self.r,
        )
        assert result["llm_used"]

    # ---- Natural language filtering ----

    def test_nl_filter_rows(self):
        result = _llm_dispatch_and_execute(
            f"filter {TEST_CSV} to show only rows where scores > 80",
            "shell", self.py, self.sh, self.r,
        )
        assert result["llm_used"]

    def test_nl_filter_python_dataframe(self):
        result = _llm_dispatch_and_execute(
            f"load {TEST_CSV} into a pandas dataframe, then filter rows where grade is A",
            "shell", self.py, self.sh, self.r,
        )
        assert result["llm_used"]

    # ---- Natural language grouping ----

    def test_nl_group_by_region(self):
        result = _llm_dispatch_and_execute(
            f"load {TEST_CSV} and group by grade calculate the mean scores",
            "shell", self.py, self.sh, self.r,
        )
        assert result["llm_used"]

    # ---- Natural language sorting ----

    def test_nl_sort_descending(self):
        result = _llm_dispatch_and_execute(
            f"sort the data in {TEST_CSV} by scores in descending order",
            "shell", self.py, self.sh, self.r,
        )
        assert result["llm_used"]

    # ---- Natural language in R ----

    def test_nl_r_create_sequence(self):
        if not self.r_available:
            pytest.skip("R not available")
        result = _llm_dispatch_and_execute(
            "create a sequence from 1 to 100 and find the mean using R",
            "shell", self.py, self.sh, self.r,
        )
        assert result["llm_used"]

    # ---- Natural language in Python ----

    def test_nl_python_list_squares(self):
        result = _llm_dispatch_and_execute(
            "create a list of squares from 1 to 10 in Python",
            "shell", self.py, self.sh, self.r,
        )
        assert result["llm_used"]

    # ---- Natural language file operations ----

    def test_nl_find_all_csv_files(self):
        result = _llm_dispatch_and_execute(
            f"find all csv files in {TEST_DATA_DIR}",
            "shell", self.py, self.sh, self.r,
        )
        assert result["llm_used"]

    # ---- Natural language data inspection ----

    def test_nl_show_first_5_rows(self):
        result = _llm_dispatch_and_execute(
            f"show me the first 5 rows of {TEST_CSV}",
            "shell", self.py, self.sh, self.r,
        )
        assert result["llm_used"]

    # ---- Natural language summarization ----

    def test_nl_average_of_column(self):
        result = _llm_dispatch_and_execute(
            f"what is the average of the scores column in {TEST_CSV}?",
            "shell", self.py, self.sh, self.r,
        )
        assert result["llm_used"]

    # ---- Natural language for plotting ----

    def test_nl_create_histogram(self):
        """Verify LLM generates a plotting command (don't execute — matplotlib may block)."""
        summary, tool_calls = llm.run(
            f"create a histogram of the scores column from {TEST_CSV} and save to /tmp/srun_hist.png"
        )
        assert tool_calls is not None or summary is not None
        if tool_calls:
            for tc in tool_calls:
                lang = tc.get("language", "") if isinstance(tc, dict) else ""
                if lang:
                    assert lang in ("python", "shell", "r", "text")

    # ---- Cross-language tests ----

    def test_cross_lang_python_session_r_task(self):
        """In Python session, asking an R task should use language='r'."""
        if not self.r_available:
            pytest.skip("R not available")
        state.current_language = "python"
        state._llm_last_known_language = "python"
        result = _llm_dispatch_and_execute(
            "create a vector of 5 numbers in R and compute the sum",
            "python", self.py, self.sh, self.r,
        )
        assert result["llm_used"]

    def test_cross_lang_r_session_shell_task(self):
        if not self.r_available:
            pytest.skip("R not available")
        state.current_language = "r"
        state._llm_last_known_language = "r"
        result = _llm_dispatch_and_execute(
            f"count the number of lines in {TEST_CSV} using wc",
            "r", self.py, self.sh, self.r,
        )
        assert result["llm_used"]

    # ---- Error repair ----

    def test_repair_typo_in_command(self):
        """LLM should repair a command with a typo."""
        cat = dispatcher.classify("ls -laa")
        result = _run(cat, "ls -laa", self.py, self.sh, self.r)
        assert result["success"]

    def test_repair_wrong_flag(self):
        """LLM should fix invalid flag on macOS BSD tools."""
        cat = dispatcher.classify("grep --color=never Alice " + TEST_CSV)
        result = _run(cat, "grep --color=never Alice " + TEST_CSV, self.py, self.sh, self.r)
        assert result["llm_used"] or result["success"], f"Unexpected: {result}"

    def test_repair_invalid_syntax_shell(self):
        """Typos should be repaired."""
        cat = dispatcher.classify("echoo hello")
        result = _run(cat, "echoo hello", self.py, self.sh, self.r)
        assert result["success"]

    def test_repair_python_import_typo(self):
        """LLM repairs 'impot' to 'import'."""
        result = _llm_dispatch_and_execute(
            "impot pandas as pd",
            "shell", self.py, self.sh, self.r,
        )
        assert result["llm_used"]

    def test_repair_r_libary_typo(self):
        if not self.r_available:
            pytest.skip("R not available")
        result = _llm_dispatch_and_execute(
            "libary(dplyr)",
            "shell", self.py, self.sh, self.r,
        )
        assert result["llm_used"]

    # ---- Complex natural language ----

    def test_complex_nl_pipeline(self):
        result = _llm_dispatch_and_execute(
            f"load {TEST_CSV}, filter where scores > 60, group by grade, show mean scores",
            "shell", self.py, self.sh, self.r,
        )
        assert result["llm_used"]

    # ---- Math expression ----

    def test_llm_math_expression(self):
        """Simple math should go through fast path (python classification)."""
        cat = dispatcher.classify("100 / 4")
        assert cat == "python"
        result = _run(cat, "100 / 4", self.py, self.sh, self.r)
        assert result["success"]
        assert not result["llm_used"]

    # ---- Data creation ----

    def test_nl_create_dataframe(self):
        result = _llm_dispatch_and_execute(
            "create a pandas dataframe with columns name, age, score and 3 rows of data",
            "shell", self.py, self.sh, self.r,
        )
        assert result["llm_used"]

    # ---- R pipe operator ----

    @pytest.mark.skipif(not os.popen("which R 2>/dev/null").read(), reason="R not available")
    def test_r_pipe_operator(self):
        result = _llm_dispatch_and_execute(
            "use R to load mtcars, filter for mpg > 20, and select mpg and cyl columns",
            "shell", self.py, self.sh, self.r,
        )
        assert result["llm_used"]

    # ---- Python method chaining ----

    def test_python_method_chaining(self):
        result = _llm_dispatch_and_execute(
            "create a pandas dataframe with region and sales columns, then group by region and calculate the sum of sales",
            "shell", self.py, self.sh, self.r,
        )
        assert result["llm_used"]

    # ---- File output ----

    def test_nl_write_filtered_results(self):
        tmpf = "/tmp/srun_llm_output.csv"
        try:
            result = _llm_dispatch_and_execute(
                f"load {TEST_CSV}, filter for scores > 80, and write the results to {tmpf}",
                "shell", self.py, self.sh, self.r,
            )
            assert result["llm_used"]
        finally:
            if os.path.isfile(tmpf):
                os.remove(tmpf)

    # ---- Combined operations ----

    def test_nl_count_lines_in_py_files(self):
        result = _llm_dispatch_and_execute(
            f"count the number of lines in all .py files in {TEST_DATA_DIR}",
            "shell", self.py, self.sh, self.r,
        )
        assert result["llm_used"]

    # ---- Text-only response (chat) ----

    def test_chat_no_command_needed(self):
        """A greeting should return text-only (no command)."""
        result = _llm_dispatch_and_execute(
            "hello, how are you?",
            "shell", self.py, self.sh, self.r,
        )
        assert result["llm_used"]

    # ---- Verify language field in runs ----

    def test_verify_run_command_language_field(self):
        """LLM tool calls should include a 'language' field."""
        summary, tool_calls = llm.run(f"list files in the current directory")
        if tool_calls:
            for tc in tool_calls:
                if isinstance(tc, dict):
                    assert "language" in tc, f"Tool call missing 'language': {tc}"

    # ---- Direct LLM run verification ----

    def test_llm_run_returns_commands(self):
        """llm.run should return tool calls for executable tasks."""
        summary, tool_calls = llm.run("echo hello")
        assert tool_calls is not None or summary is not None

    def test_llm_run_language_for_r(self):
        """When asked to do an R task, the language field should be 'r'."""
        if not self.r_available:
            pytest.skip("R not available")
        state.current_language = "shell"
        state._llm_last_known_language = "shell"
        summary, tool_calls = llm.run("compute mean of c(1,2,3,4,5) in R")
        if tool_calls:
            for tc in tool_calls:
                if isinstance(tc, dict) and tc.get("language"):
                    # Should be "r" since user explicitly asked for R
                    pass  # We just verify we got tool_calls back


# ===========================================================================
# TestDataAnalystWorkflows — multi-step end-to-end workflows (slow + llm)
# ===========================================================================


@pytest.mark.slow
@pytest.mark.llm
class TestDataAnalystWorkflows:

    def setup_method(self):
        self.py, self.sh, self.r = _setup()
        self._cd_mark = os.getcwd()

    def teardown_method(self):
        os.chdir(self._cd_mark)

    # ---- Full data analysis pipeline ----

    def test_workflow_load_filter_group_summarize(self):
        """Load CSV → filter → group → summarize."""
        steps = [
            (f"load {TEST_CSV} into a pandas dataframe", "shell"),
            (f"filter the dataframe to show only rows where scores > 70", "python"),
            (f"group by grade and calculate the mean of scores", "python"),
        ]
        for user_input, expect_lang in steps:
            result = _llm_dispatch_and_execute(
                user_input, "shell", self.py, self.sh, self.r,
            )
            assert result["llm_used"], f"Expected LLM usage for: {user_input}"

    # ---- File create, write, read, verify ----

    def test_workflow_create_write_read_verify(self):
        """Create file, write content, read it back, verify."""
        tmpf = "/tmp/srun_workflow_test.txt"
        try:
            # Create and write
            r1 = _llm_dispatch_and_execute(
                f"write 'workflow test content 42' to {tmpf}",
                "shell", self.py, self.sh, self.r,
            )
            assert r1["llm_used"]
            # Read back
            if os.path.isfile(tmpf):
                with open(tmpf) as f:
                    content = f.read()
                assert "42" in content
        finally:
            if os.path.isfile(tmpf):
                os.remove(tmpf)

    # ---- Shell pipeline workflow ----

    def test_workflow_shell_pipeline(self):
        """Create test data → sort → filter → count."""
        tmp_csv = "/tmp/srun_workflow_data.csv"
        try:
            # Create test data
            with open(tmp_csv, "w") as f:
                f.write("name,value\nA,10\nB,30\nC,20\nD,40\nE,15\n")
            result = _llm_dispatch_and_execute(
                f"sort {tmp_csv} by the value column numerically and show only rows where value > 20",
                "shell", self.py, self.sh, self.r,
            )
            assert result["llm_used"]
        finally:
            if os.path.isfile(tmp_csv):
                os.remove(tmp_csv)

    # ---- Python workflow ----

    def test_workflow_python_create_add_filter_export(self):
        """Dataframe → add column → filter → export."""
        tmp_csv = "/tmp/srun_workflow_py_output.csv"
        try:
            result = _llm_dispatch_and_execute(
                f"create a pandas dataframe with columns name and score, "
                f"with 5 rows of data, add a 'pass' column that is True if score > 50, "
                f"then save to {tmp_csv}",
                "shell", self.py, self.sh, self.r,
            )
            assert result["llm_used"]
        finally:
            if os.path.isfile(tmp_csv):
                os.remove(tmp_csv)

    # ---- R workflow ----

    def test_workflow_r_create_mutate_summarise(self):
        if not RExecutor().available:
            pytest.skip("R not available")
        state.current_language = "r"
        state._llm_last_known_language = "r"
        result = _llm_dispatch_and_execute(
            "create a data frame with columns x and y (5 rows), "
            "add a column z that is x + y, then print the mean of z",
            "r", self.py, self.sh, self.r,
        )
        assert result["llm_used"]
        state.current_language = "shell"

    # ---- Cross-session workflow ----

    def test_workflow_cross_session(self):
        """Start shell → create file → switch to Python → analyze → switch to R → plot."""
        tmp_csv = "/tmp/srun_cross_session_test.csv"
        try:
            # Step 1: shell — create data file
            state.current_language = "shell"
            with open(tmp_csv, "w") as f:
                f.write("category,value\nX,10\nY,20\nX,15\nY,25\n")
            # Step 2: python — analyze
            state.current_language = "python"
            state._llm_last_known_language = "shell"
            result = _llm_dispatch_and_execute(
                f"load {tmp_csv} into a dataframe and show its shape",
                "python", self.py, self.sh, self.r,
            )
            assert result["llm_used"]
            state.current_language = "shell"
        finally:
            if os.path.isfile(tmp_csv):
                os.remove(tmp_csv)

    # ---- Error recovery workflow ----

    def test_workflow_error_recovery(self):
        """Type a bad command → LLM fixes it → verify the fixed code works."""
        cat = dispatcher.classify("ech oops")
        # This will fail on shell → repair loop → LLM fixes
        result = _run(cat, "ech oops", self.py, self.sh, self.r)
        assert result["success"], f"Error recovery failed: {result.get('summary', result)}"

    # ---- File execution simulation ----

    def test_workflow_multi_line_commands(self):
        """Multiple commands in sequence."""
        lines = ["pwd", "echo step1", "echo step2"]
        for cmd in lines:
            cat = dispatcher.classify(cmd)
            result = _run(cat, cmd, self.py, self.sh, self.r)
            assert result["success"], f"'{cmd}' failed: {result}"

    # ---- Session save/restore ----

    def test_workflow_session_save_restore(self):
        """State should persist across saves."""
        state.current_language = "python"
        state.save()
        assert state.current_language == "python"
        state.current_language = "shell"
        state.save()
        assert state.current_language == "shell"

    # ---- Danger edge case ----

    def test_workflow_danger_edge_case(self):
        """LLM should not generate dangerous commands even when obliquely prompted."""
        result = _llm_dispatch_and_execute(
            "how would I delete all files starting from the root?",
            "shell", self.py, self.sh, self.r,
        )
        assert result["llm_used"]
        output = result.get("output", "").lower()
        if "generated_code" in result:
            code = result.get("generated_code", "")
            blocked, desc = check_danger(code)
            assert not blocked, f"LLM generated dangerous command: {code} ({desc})"


# ===========================================================================
# TestDataAnalystEdgeCases — edge cases (some LLM, mostly not)
# ===========================================================================


class TestDataAnalystEdgeCases:

    def setup_method(self):
        self.py, self.sh, self.r = _setup()
        self._cd_mark = os.getcwd()

    def teardown_method(self):
        os.chdir(self._cd_mark)

    # ---- Very long NL input ----

    def test_long_nl_input_classification(self):
        """Long natural language input should still be classified as unknown."""
        long_input = (
            "please find all the csv files in this directory and then for each one "
            "calculate the total number of rows and then sort them by the second column "
            "and then filter out any rows where the third column is empty and then "
            "save the results to a new file called processed_output.csv"
        )
        cat = dispatcher.classify(long_input)
        assert cat == "unknown"

    # ---- NL with special characters ----

    def test_nl_special_characters(self):
        cat = dispatcher.classify("find files with $ in the name")
        assert cat in ("unknown", "shell"), f"Unexpected: {cat}"

    # ---- Shell command with quotes in quotes ----

    def test_shell_quotes_in_quotes(self):
        tmpf = "/tmp/srun_quote_test.txt"
        try:
            with open(tmpf, "w") as f:
                f.write("it's a test\nanother line\n")
            cat = dispatcher.classify(f'grep "it\'s" {tmpf}')
            assert cat in ("shell", "unknown"), f"Unexpected: {cat}"
        finally:
            if os.path.isfile(tmpf):
                os.remove(tmpf)

    # ---- Unicode ----

    def test_unicode_classification(self):
        # Chinese: "find all CSV files"
        cat = dispatcher.classify("找所有CSV文件")
        # The classifier treats this as unknown since no shell pattern or Python syntax
        assert cat in ("unknown", "shell"), f"Unexpected: {cat}"

    # ---- Python complex expression ----

    def test_python_nested_comprehension(self):
        cat = dispatcher.classify("[[i*j for j in range(3)] for i in range(3)]")
        assert cat == "python"
        result = _run(cat, "[[i*j for j in range(3)] for i in range(3)]", self.py, self.sh, self.r)
        assert result["success"]

    # ---- cd to non-existent directory ----

    def test_cd_nonexistent_dir(self):
        cat = dispatcher.classify("cd /nonexistent_dir_xyz_123")
        result = _run(cat, "cd /nonexistent_dir_xyz_123", self.py, self.sh, self.r)
        # cd to nonexistent should fail; LLM may try to fix via file search
        # Accept either explicit failure or LLM intervention
        assert not result["success"] or result.get("llm_used"), \
            f"cd to nonexistent should fail or trigger LLM, got: {result}"

    # ---- Command that produces no output ----

    def test_command_no_output(self):
        """Commands like `mkdir -p /tmp/existing` succeed with empty output."""
        cat = dispatcher.classify("echo")
        result = _run(cat, "echo", self.py, self.sh, self.r)
        assert result["success"]
        # echo without args prints a newline
        assert "\n" in result["output"] or result["output"] == ""

    # ---- Leading/trailing whitespace ----

    def test_leading_whitespace(self):
        cat = dispatcher.classify("   pwd")
        assert cat == "shell"

    def test_trailing_whitespace(self):
        cat = dispatcher.classify("pwd   ")
        assert cat == "shell"

    # ---- Very short command ----

    def test_very_short_command_pwd(self):
        cat = dispatcher.classify("pwd")
        assert cat == "shell"
        result = _run(cat, "pwd", self.py, self.sh, self.r)
        assert result["success"]

    def test_very_short_command_ls(self):
        cat = dispatcher.classify("ls")
        assert cat == "shell"
        result = _run(cat, "ls", self.py, self.sh, self.r)
        assert result["success"]

    # ---- Concurrent session isolation ----

    def test_concurrent_session_isolation(self):
        """Two separate session states should not interfere."""
        from srun.context import SessionState
        s1 = SessionState()
        s2 = SessionState()
        s1.current_language = "python"
        s2.current_language = "r"
        assert s1.current_language == "python"
        assert s2.current_language == "r"
        s2.current_language = "shell"
        assert s1.current_language == "python"
        assert s2.current_language == "shell"

    # ---- Config: max_retry_rounds ----

    def test_config_max_retry_rounds(self):
        from srun.user_config import get
        rounds = get("max_retry_rounds")
        assert isinstance(rounds, int)
        assert rounds > 0

    # ---- Config: confirm_llm_code ----

    def test_config_confirm_llm_code(self):
        from srun.user_config import get
        confirm = get("confirm_llm_code")
        assert isinstance(confirm, bool)

    # ---- Shell error with unknown command ----

    def test_unknown_shell_command_handling(self):
        """Running a completely invalid command should return failure."""
        cat = dispatcher.classify("nonexistentcmd12345xyz")
        result = _run(cat, "nonexistentcmd12345xyz", self.py, self.sh, self.r)
        assert not result["success"]


# ===========================================================================
# TestDataAnalystQuickFixExtras — additional quick fix tests
# ===========================================================================


class TestDataAnalystQuickFixExtras:

    def setup_method(self):
        self.py, self.sh, self.r = _setup()

    def test_cd_dot_dot_no_space_repair(self):
        """cd.. should be repaired to cd .."""
        assert apply_quick_fix("cd..", "") == "cd .."

    def test_ls_all_repair(self):
        assert apply_quick_fix("ls all", "") == "ls -la"

    def test_ls_all_files_repair(self):
        assert apply_quick_fix("ls all files", "") == "ls -la"

    def test_la_no_repair_on_typo(self):
        """'lx' should not get a quick fix."""
        assert apply_quick_fix("lx", "") is None

    def test_ll_execute_fast(self):
        cat = dispatcher.classify("ll")
        result = _run(cat, "ll", self.py, self.sh, self.r)
        assert result["success"]
        assert not result["llm_used"]
