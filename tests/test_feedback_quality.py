"""P1/P2 feedback-quality tests: bounded recoverable output, structured
run_command feedback, conservative JSON extraction, compaction artifacts,
rg-backed search, and whitespace-tolerant file edits."""

import os
import re
import shutil
import subprocess
import types

import pytest

from srun import llm as llm_module
from srun.context import state
from srun.llm import (
    _extract_command_from_text,
    _build_run_command_feedback,
    _truncate_tool_result,
    truncate_tool_output,
)
from srun.executors.shell_exec import ShellExecutor
from srun.tools import file_edit, file_write, read_file, search_files


# ── fake OpenAI stream (same shape as test_agent_loop.py) ─────────

class _FakeToolCall:
    def __init__(self, index):
        self.index = index
        self.id = f"call_{index}"
        self.function = types.SimpleNamespace(
            name="run_command",
            arguments='{"command": "echo hi", "language": "shell"}',
        )


class _FakeDelta:
    def __init__(self, tool_calls):
        self.content = None
        self.reasoning_content = None
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, tool_calls):
        self.delta = _FakeDelta(tool_calls)


class _FakeUsage:
    prompt_cache_miss_tokens = 0
    prompt_cache_hit_tokens = 1000


class _FakeChunk:
    def __init__(self, tool_calls):
        self.usage = _FakeUsage()
        self.choices = [_FakeChoice(tool_calls)]


class _FakeClient:
    def __init__(self):
        self.model = "fake-model"
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=None))


# ── P1-1: truncate_tool_output ─────────────────────────────────────

class TestTruncateToolOutput:
    def test_small_passthrough(self):
        out, truncated = truncate_tool_output("hello world")
        assert not truncated
        assert out == "hello world"

    def test_tail_keeps_marker_and_tail(self):
        text = "\n".join(f"line {i}" for i in range(3000))
        out, truncated = truncate_tool_output(text, max_lines=100, max_chars=100000)
        assert truncated
        assert "truncated" in out
        assert "line 2999" in out
        assert "line 0" not in out

    def test_head_keeps_head(self):
        text = "\n".join(f"line {i}" for i in range(3000))
        out, truncated = truncate_tool_output(
            text, max_lines=100, max_chars=100000, direction="head"
        )
        assert truncated
        assert "line 0" in out
        assert "line 2999" not in out

    def test_huge_single_line_sliced(self):
        out, truncated = truncate_tool_output("x" * 50000, max_lines=10, max_chars=100)
        assert truncated
        assert "truncated" in out
        assert out.strip().endswith("x" * 100)

    def test_empty(self):
        out, truncated = truncate_tool_output("")
        assert out == ""
        assert not truncated


# ── P1-4: structured run_command feedback ─────────────────────────

class TestRunCommandFeedback:
    def test_structured_meta_surfaces_in_tool_message(self, monkeypatch):
        client = _FakeClient()
        monkeypatch.setattr(llm_module.llm, "client", client)
        monkeypatch.setattr(
            llm_module,
            "config_get",
            lambda key: 2 if key == "max_llm_steps" else 12,
        )
        state.reset_session()

        def _exec_meta(cmd, lang):
            return False, "partial output", {
                "stderr": "boom error", "exit_code": 124, "timed_out": True,
            }

        class _Sequenced:
            def __init__(self):
                self.calls = 0

            def __call__(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return iter([_FakeChunk([_FakeToolCall(1)])])
                return iter(())

        client.chat.completions.create = _Sequenced()

        summary, tool_calls, conv = llm_module.llm.run("run it", exec_callback=_exec_meta)
        tool_msgs = [m for m in (conv or []) if m.get("role") == "tool"]
        assert tool_msgs, "expected a tool message in the conversation"
        content = tool_msgs[0]["content"]
        assert "Exit: 124" in content
        assert "timed_out" in content
        assert "stderr:" in content
        assert "boom error" in content
        monkeypatch.setattr(llm_module.llm, "client", None)

    def test_build_feedback_plain(self):
        content = _build_run_command_feedback(True, "ok", {"exit_code": 0})
        assert "Exit: 0" in content
        assert "stdout:" in content

    def test_build_feedback_aborted(self):
        content = _build_run_command_feedback(False, "", {"aborted": True, "exit_code": 130})
        assert "Exit: 130" in content
        assert "flags=aborted" in content

    def test_build_feedback_splits_merged_stderr(self):
        """Shell executor returns out = stdout + stderr merged; rendering must
        show each stream once, not duplicate stderr inside stdout."""
        content = _build_run_command_feedback(
            True, "hello\nwarning!", {"exit_code": 0, "stderr": "warning!"}
        )
        assert "stdout:\nhello" in content
        assert "stderr:\nwarning!" in content
        assert content.count("warning!") == 1

    def test_build_feedback_keeps_merged_when_not_suffix(self):
        """When stderr is not a suffix of out (e.g. python/R), keep out intact."""
        content = _build_run_command_feedback(
            True, "pure stdout", {"exit_code": 0, "stderr": "warn"}
        )
        assert "stdout:\npure stdout" in content

    def test_build_feedback_stderr_only_not_duplicated(self):
        """A command whose whole output is stderr (out == err, e.g. ls missing
        path) must render the message once, in the stderr section only."""
        err = "ls: cannot access '/tmp/x': No such file or directory\n"
        content = _build_run_command_feedback(False, err, {"exit_code": 2, "stderr": err})
        assert "stdout:" not in content
        assert "stderr:" in content
        assert content.count("No such file") == 1


# ── P2-3: conservative _extract_command_from_text ──────────────────

class TestExtractCommandConservative:
    def test_standalone_json(self):
        assert _extract_command_from_text('{"command": "ls -la", "language": "shell"}') == {
            "command": "ls -la", "language": "shell",
        }

    def test_fenced_json(self):
        assert _extract_command_from_text('Here:\n```json\n{"command": "pwd"}\n```') == {
            "command": "pwd", "language": None,
        }

    def test_prose_braces_ignored(self):
        assert _extract_command_from_text("the {command} is ls, run it") is None

    def test_multiple_objects_ignored(self):
        assert _extract_command_from_text('{"a": 1}\n{"command": "ls"}') is None

    def test_empty(self):
        assert _extract_command_from_text("") is None


# ── P1-3: compaction preserves evidence ────────────────────────────

class TestCompactContextArtifacts:
    def test_dropped_tool_outputs_saved_with_path(self, monkeypatch):
        class _FakeCompletions:
            def create(self, **kwargs):
                return types.SimpleNamespace(
                    choices=[
                        types.SimpleNamespace(
                            message=types.SimpleNamespace(content="compact summary")
                        )
                    ],
                    usage=None,
                )

        fake_llm = types.SimpleNamespace(
            client=types.SimpleNamespace(
                model="fake",
                chat=types.SimpleNamespace(completions=_FakeCompletions()),
            ),
            _track_usage=lambda u: None,
        )
        state.reset_session()
        orig_max = state._max_context_tokens
        state._max_context_tokens = 2000
        try:
            for i in range(8):
                state._conversation.append({"role": "user", "content": f"user {i}"})
                state._conversation.append({"role": "assistant", "content": f"assistant {i}"})
                state._conversation.append({
                    "role": "tool", "tool_call_id": f"t{i}",
                    "content": f"tool output {i} " * 50,
                })
            ok = state.compact_context(llm_module=fake_llm)
            assert ok
            assert "Recoverable artifacts" in state._stable_summary
            path = state._stable_summary.split(
                "[Recoverable artifacts from compacted turns: "
            )[1].split(" ")[0]
            assert path.endswith(".txt")
            assert os.path.isfile(path)
            assert "unfinished subtask" not in state._stable_summary  # summary came from fake
        finally:
            state._max_context_tokens = orig_max


# ── P2-1: rg-backed search_files ───────────────────────────────────

class TestSearchFiles:
    def test_respects_gitignore(self, tmp_path, monkeypatch):
        if not shutil.which("rg"):
            pytest.skip("ripgrep not installed")
        monkeypatch.chdir(tmp_path)
        (tmp_path / "keep.py").write_text("")
        (tmp_path / "skip.py").write_text("")
        (tmp_path / ".gitignore").write_text("skip.py\n")
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, capture_output=True)
        result = search_files("*.py")
        assert "keep.py" in result
        assert "skip.py" not in result

    def test_no_match(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        file_write("my_script.py", "")
        result = search_files("*.nonexistent_ext")
        assert "No files matching" in result


# ── P2-2: whitespace-tolerant file_edit ────────────────────────────

class TestFileEditFuzzy:
    def test_fuzzy_indentation_single_match(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        p = tmp_path / "code.py"
        p.write_text("def f():\n    return 42\n")
        result = file_edit("code.py", "  return 42", "  return 43")
        assert "Replaced 1 occurrence" in result
        assert "fuzzy" in result
        assert "return 43" in p.read_text()

    def test_fuzzy_ambiguous_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        p = tmp_path / "code.py"
        p.write_text("def a():\n    return 42\n\ndef b():\n    return 42\n")
        # "return\t42" (tab) never exactly matches the space-separated content,
        # but fuzzy-matches both occurrences → must be rejected as ambiguous.
        result = file_edit("code.py", "return\t42", "return 43")
        assert "fuzzy matches" in result

    def test_exact_still_wins(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        p = tmp_path / "code.py"
        p.write_text("hello world\n")
        result = file_edit("code.py", "hello world", "goodbye")
        assert "exact" in result
        assert "goodbye" in p.read_text()


# ── 大工具结果：只留 tail、落盘、绝对路径提示、read_file 有界 ────

class TestLargeToolOutput:
    """大输出三种场景：run_command 大输出、cat 长文件、read_file 大文件。"""

    @staticmethod
    def _big_text(n=10000):
        return "\n".join(f"line {i}" for i in range(n))

    def test_run_command_large_output_tail_and_dump(self):
        """run_command 大输出：只保留 tail，全文落盘，反馈带绝对路径。"""
        out = self._big_text(10000)  # ~70KB > 50KB 上限
        content = _build_run_command_feedback(True, out, {"exit_code": 0, "stderr": ""})
        assert "truncated" in content
        assert "line 9999" in content       # tail 保留
        assert "line 0" not in content      # head 截掉
        assert "[Full output:" in content
        path = content.split("[Full output: ")[1].split(" — ")[0]
        assert os.path.isfile(path)
        with open(path) as f:
            assert f.read() == out          # 落盘内容完整

    def test_cat_long_file_through_executor(self, tmp_path):
        """模型用 cat 把长文件打进上下文 → 同样被截断并落盘。"""
        big = tmp_path / "big.txt"
        big.write_text(self._big_text(10000) + "\n")
        sh = ShellExecutor()
        ok, out, err, rc, meta = sh.execute(f"cat {big}")
        assert ok
        content = _build_run_command_feedback(ok, out, {"stderr": err, "exit_code": rc})
        assert "truncated" in content
        assert "line 9999" in content
        assert "line 0" not in content

    def test_read_file_large_bounded_preview(self, tmp_path):
        """read_file 大文件：head+tail 有界预览 + 绝对路径，中段不注入上下文。"""
        big = tmp_path / "big.txt"
        big.write_text(self._big_text(5000) + "\n")  # 5000 行 > 2000 行上限
        result = read_file(str(big))
        assert "5000 lines" in result
        assert "line 0" in result            # head
        assert "line 4999" in result         # tail
        assert "line 2500" not in result     # 中段被截
        assert str(big) in result            # 绝对路径提示
        assert len(result) < 60 * 1024

    def test_read_file_offset_paging(self, tmp_path):
        """模型可用 offset+lines 分页读回被省略的段落。"""
        big = tmp_path / "big.txt"
        big.write_text(self._big_text(5000) + "\n")
        result = read_file(str(big), offset=3000, lines=5)
        assert "line 3000" in result
        assert "line 3004" in result
        assert "line 0" not in result
        assert "offset=3005" in result       # 续读提示

    def test_read_file_lines_from_start(self, tmp_path):
        """lines 参数保持向后兼容：从文件开头读 N 行。"""
        big = tmp_path / "big.txt"
        big.write_text(self._big_text(500) + "\n")
        result = read_file(str(big), lines=3)
        assert "line 0" in result and "line 1" in result and "line 2" in result
        assert "line 3" not in result

    def test_read_file_too_large_guard(self, tmp_path):
        big = tmp_path / "huge.bin"
        big.write_bytes(b"x" * (300 * 1024))
        result = read_file(str(big))
        assert "File too large" in result
        assert str(big) in result

    def test_truncate_tool_result_generic_tail_and_dump(self):
        """通用工具结果（如 grep 大输出）：tail 预览 + 落盘 + 绝对路径。"""
        result = _truncate_tool_result("grep_search", self._big_text(10000))
        assert "[Full output saved to" in result
        assert "line 9999" in result
        assert "line 0" not in result
        path = result.split("[Full output saved to ")[1].split(" — ")[0]
        assert os.path.isfile(path)

    def test_truncate_passthrough_small(self):
        assert _truncate_tool_result("grep_search", "small") == "small"

    def test_read_file_not_dumped_again(self, tmp_path):
        """read_file 结果已由工具自身有界化，_truncate_tool_result 直接透传。"""
        big = tmp_path / "big.txt"
        big.write_text(self._big_text(5000) + "\n")
        read_result = read_file(str(big))
        assert _truncate_tool_result("read_file", read_result) == read_result
