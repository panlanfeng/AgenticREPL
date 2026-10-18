"""Tests for deterministic result contracts and the no-LLM checker."""

import json
import os
import subprocess
import time
import types

from srun import llm as llm_module
from srun.context import state
from srun.result_validation import (
    collect_artifacts,
    format_contract,
    normalize_contract,
    validate_contract,
    verify_result_with_llm,
)
from srun.tools import TOOL_DEFINITIONS, TOOL_HANDLERS


class _Delta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.reasoning_content = None
        self.tool_calls = tool_calls


class _Chunk:
    usage = None

    def __init__(self, delta):
        self.choices = [types.SimpleNamespace(delta=delta)]


class _SequencedClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.requests = []
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        self.calls += 1
        self.requests.append(kwargs)
        if not self.responses:
            return iter(())
        return self.responses.pop(0)


def _tool_call(index, name, arguments):
    return types.SimpleNamespace(
        index=index,
        id=f"call_{index}",
        function=types.SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _command(command, output, ok=True, exit_code=0, round=None, captured=True):
    item = {
        "command": command,
        "language": "shell",
        "executed": True,
        "ok": ok,
        "exit_code": exit_code,
        "output": output,
        "captured": captured,
    }
    if round is not None:
        item["round"] = round
    return item


def test_set_result_contract_is_registered_as_a_tool():
    names = [tool["function"]["name"] for tool in TOOL_DEFINITIONS]
    assert "set_result_contract" in names
    assert "set_result_contract" in TOOL_HANDLERS


def test_normalize_contract_keeps_only_well_formed_checks():
    contract = normalize_contract({
        "goal": "create report",
        "expected_format": "csv",
        "checks": [
            {"type": "file_exists", "path": "report.csv"},
            {"path": "missing-type"},
            "not-an-object",
        ],
    }, "create report")
    assert contract["goal"] == "create report"
    assert contract["checks"] == [{"type": "file_exists", "path": "report.csv"}]


def test_format_contract_renders_checks():
    text = format_contract({
        "goal": "create report",
        "expected_format": "csv",
        "checks": [{"type": "file_exists", "path": "report.csv"}],
    })
    assert "Goal: create report" in text
    assert "file_exists" in text


def test_file_exists_and_contains_are_checked_with_code(tmp_path):
    target = tmp_path / "report.csv"
    target.write_text("name,score\nalice,90\n")
    contract = {
        "goal": "create report",
        "checks": [
            {"type": "file_exists", "path": "report.csv"},
            {"type": "file_contains", "path": "report.csv", "text": "alice"},
            {"type": "line_count", "path": "report.csv", "equals": 2},
        ],
    }
    result = validate_contract(contract, [], None, cwd=str(tmp_path))
    assert result["satisfied"] is True
    assert len(result["checks"]) == 3


def test_file_check_failure_returns_retry_instruction(tmp_path):
    contract = {
        "goal": "create report",
        "checks": [{"type": "file_exists", "path": "missing.csv"}],
    }
    result = validate_contract(contract, [], None, cwd=str(tmp_path))
    assert result["satisfied"] is False
    assert "missing.csv" in result["reason"]
    assert "Create or restore" in result["retry_instruction"]


def test_output_and_exit_code_checks_use_execution_evidence():
    commands = [_command("printf 42", "42\n", exit_code=0)]
    contract = {
        "goal": "print 42",
        "checks": [
            {"type": "output_contains", "text": "42", "source": "last"},
            {"type": "exit_code", "equals": "0"},
            {"type": "all_commands_ok"},
        ],
    }
    result = validate_contract(contract, commands, "done")
    assert result["satisfied"] is True


def test_all_output_source_includes_final_summary():
    contract = {
        "goal": "report a value",
        "checks": [{"type": "output_contains", "text": "answer is 42"}],
    }
    result = validate_contract(contract, [], "The answer is 42")
    assert result["satisfied"] is True


def test_last_output_falls_back_to_final_summary_when_no_command_ran():
    contract = {
        "goal": "answer 102",
        "checks": [{"type": "output_equals", "text": "102", "source": "last"}],
    }
    result = validate_contract(contract, [], "102")
    assert result["satisfied"] is True


def test_output_check_failure_is_deterministic():
    commands = [_command("printf wrong", "wrong\n")]
    contract = {
        "goal": "print hello",
        "checks": [{"type": "output_contains", "text": "hello", "source": "last"}],
    }
    result = validate_contract(contract, commands, "done")
    assert result["satisfied"] is False
    assert "does not contain" in result["reason"]


def test_current_round_is_the_default_scope():
    commands = [
        _command("round one", "hello\n", round=1),
        _command("round two", "wrong\n", round=2),
    ]
    contract = {
        "goal": "print hello",
        "checks": [{"type": "output_contains", "text": "hello", "source": "last"}],
    }
    current = validate_contract(contract, commands, "done", current_round=2)
    assert current["satisfied"] is False

    all_rounds = {
        "goal": "print hello",
        "checks": [{"type": "output_contains", "text": "hello", "source": "all", "scope": "all"}],
    }
    historical = validate_contract(all_rounds, commands, "done", current_round=2)
    assert historical["satisfied"] is True


def test_all_commands_ok_ignores_an_earlier_failed_round():
    commands = [
        _command("first attempt", "boom\n", ok=False, exit_code=1, round=1),
        _command("retry", "ok\n", ok=True, exit_code=0, round=2),
    ]
    contract = {"goal": "retry until success", "checks": [{"type": "all_commands_ok"}]}
    result = validate_contract(contract, commands, "done", current_round=2)
    assert result["satisfied"] is True


def test_uncaptured_tty_output_fails_with_a_clear_reason():
    commands = [_command("vim-ish command", "", captured=False)]
    contract = {
        "goal": "print hello",
        "checks": [{"type": "output_contains", "text": "hello", "source": "last"}],
    }
    result = validate_contract(contract, commands, "")
    assert result["satisfied"] is False
    assert "not captured" in result["reason"]


def test_llm_verifier_is_stateless_and_sees_logs_and_artifacts():
    client = _SequencedClient([iter([_Chunk(_Delta(content=json.dumps({
        "satisfied": True,
        "reason": "artifact matches",
        "retry_instruction": "",
    })))])])
    result = verify_result_with_llm(
        client, "model", "create report", {"goal": "create report", "checks": []},
        [_command("printf report", "report\n")], "done", "--- report.csv ---\nname\n",
    )
    assert result["satisfied"] is True
    messages = client.requests[0]["messages"]
    assert len(messages) == 2
    assert "stateless result verifier" in messages[0]["content"]
    assert "Execution log:" in messages[1]["content"]
    assert "Artifacts:" in messages[1]["content"]
    assert "report.csv" in messages[1]["content"]


def test_collect_artifacts_includes_referenced_file(tmp_path):
    target = tmp_path / "report.csv"
    target.write_text("name,score\nalice,90\n")
    contract = {
        "goal": "create report",
        "checks": [{"type": "file_exists", "path": "report.csv"}],
    }
    artifacts = collect_artifacts(cwd=str(tmp_path), contract=contract)
    assert "report.csv" in artifacts
    assert "alice,90" in artifacts


def test_collect_artifacts_ignores_preexisting_repo_changes(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    old = tmp_path / "old.txt"
    old.write_text("committed\n")
    subprocess.run(["git", "add", "old.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=tmp_path, check=True)

    old.write_text("modified before this turn\n")
    old_mtime = time.time() - 120
    os.utime(old, (old_mtime, old_mtime))

    since = time.time()
    new = tmp_path / "new.txt"
    new.write_text("created during this turn\n")

    artifacts = collect_artifacts(
        cwd=str(tmp_path), contract={"goal": "create new file", "checks": []},
        since=since,
    )
    assert "new.txt" in artifacts
    assert "old.txt" not in artifacts


def test_json_field_check():
    commands = [_command("print json", '{"count": 3, "items": ["a", "b"]}')]
    contract = {
        "goal": "print count",
        "checks": [
            {"type": "json_field", "source": "last", "field": "count", "equals": "3"},
            {"type": "json_field", "source": "last", "field": "items.1", "equals": "b"},
        ],
    }
    result = validate_contract(contract, commands, None)
    assert result["satisfied"] is True


def test_json_field_can_assert_absence():
    commands = [_command("print json", '{"count": 3}')]
    contract = {
        "goal": "omit error field",
        "checks": [{"type": "json_field", "source": "last", "field": "error", "exists": False}],
    }
    result = validate_contract(contract, commands, None)
    assert result["satisfied"] is True


def test_empty_checks_are_reported_as_unverifiable():
    result = validate_contract({"goal": "explain something", "checks": []}, [], "answer")
    assert result["satisfied"] is None
    assert result["unverifiable"] is True


def test_llm_run_retries_after_code_check_failure(monkeypatch, tmp_path):
    state.reset_session()
    contract_args = {
        "goal": "print hello",
        "expected_format": "text",
        "checks": [{"type": "output_contains", "text": "hello", "source": "last"}],
    }
    responses = [
        iter([_Chunk(_Delta(tool_calls=[
            _tool_call(0, "set_result_contract", contract_args),
            _tool_call(1, "run_command", {"command": "printf wrong", "language": "shell"}),
        ]))]),
        iter([_Chunk(_Delta(content="first result"))]),
        iter([_Chunk(_Delta(tool_calls=[
            _tool_call(0, "run_command", {"command": "printf hello", "language": "shell"}),
        ]))]),
        iter([_Chunk(_Delta(content="fixed result"))]),
    ]
    client = _SequencedClient(responses)
    monkeypatch.setattr(llm_module.llm, "client", client)
    monkeypatch.setattr(llm_module, "config_get", lambda key: {
        "max_llm_steps": 8,
        "verify_results": True,
        "verify_with_llm": False,
        "max_verify_retries": 1,
    }.get(key, 12))

    executed = []

    def _exec(command, language):
        executed.append(command)
        return True, "hello\n" if command == "printf hello" else "wrong\n"

    summary, commands, conv = llm_module.llm.run("print hello", exec_callback=_exec)

    assert summary == "fixed result"
    assert executed == ["printf wrong", "printf hello"]
    assert [item["command"] for item in commands] == ["printf wrong", "printf hello"]
    assert [item.get("round") for item in commands] == [1, 2]
    assert client.calls == 4
    assert client.requests[0]["tool_choice"] == {
        "type": "function",
        "function": {"name": "set_result_contract"},
    }
    assert llm_module.llm._result_contract["checks"][0]["type"] == "output_contains"
    assert llm_module.llm._last_verification["satisfied"] is True
    assert any(
        "result verification failed" in (message.get("content") or "")
        for message in conv
        if message.get("role") == "user"
    )


def test_commands_are_blocked_until_contract_is_recorded(monkeypatch):
    state.reset_session()
    contract_args = {
        "goal": "print ok",
        "expected_format": "text",
        "checks": [{"type": "output_contains", "text": "ok", "source": "last"}],
    }
    responses = [
        iter([_Chunk(_Delta(tool_calls=[
            _tool_call(0, "run_command", {"command": "printf early", "language": "shell"}),
        ]))]),
        iter([_Chunk(_Delta(tool_calls=[
            _tool_call(0, "set_result_contract", contract_args),
        ]))]),
        iter([_Chunk(_Delta(tool_calls=[
            _tool_call(0, "run_command", {"command": "printf ok", "language": "shell"}),
        ]))]),
        iter([_Chunk(_Delta(content="done"))]),
    ]
    client = _SequencedClient(responses)
    monkeypatch.setattr(llm_module.llm, "client", client)
    monkeypatch.setattr(llm_module, "config_get", lambda key: {
        "max_llm_steps": 8,
        "verify_results": True,
        "verify_with_llm": False,
        "max_verify_retries": 1,
    }.get(key, 12))

    executed = []

    def _exec(command, language):
        executed.append(command)
        return True, "ok\n"

    summary, commands, conv = llm_module.llm.run("print ok", exec_callback=_exec)

    assert summary == "done"
    assert executed == ["printf ok"]
    assert [item["command"] for item in commands] == ["printf ok"]
    assert client.calls == 4
    tool_messages = [message.get("content") or "" for message in conv
                     if message.get("role") == "tool"]
    assert any("result contract missing" in content for content in tool_messages)


def test_llm_verifier_can_reject_even_when_code_checks_pass(monkeypatch):
    state.reset_session()
    contract_args = {
        "goal": "write the requested file",
        "expected_result": "file content is correct",
        "checks": [{"type": "output_contains", "text": "ok", "source": "last"}],
    }
    responses = [
        iter([_Chunk(_Delta(tool_calls=[
            _tool_call(0, "set_result_contract", contract_args),
            _tool_call(1, "run_command", {"command": "printf ok", "language": "shell"}),
        ]))]),
        iter([_Chunk(_Delta(content="first attempt done"))]),
        iter([_Chunk(_Delta(tool_calls=[
            _tool_call(0, "run_command", {"command": "printf ok", "language": "shell"}),
        ]))]),
        iter([_Chunk(_Delta(content="second attempt done"))]),
    ]
    client = _SequencedClient(responses)
    monkeypatch.setattr(llm_module.llm, "client", client)
    monkeypatch.setattr(llm_module, "config_get", lambda key: {
        "max_llm_steps": 8,
        "verify_results": True,
        "verify_with_llm": True,
        "max_verify_retries": 1,
    }.get(key, 12))
    monkeypatch.setattr(llm_module, "collect_artifacts", lambda **kwargs: "artifact snapshot")
    checks = iter([
        {"satisfied": False, "reason": "artifact content is wrong",
         "retry_instruction": "fix the file content", "source": "llm"},
        {"satisfied": True, "reason": "artifact content is correct",
         "retry_instruction": "", "source": "llm"},
    ])
    seen_artifacts = []

    def _verify(*args, **kwargs):
        seen_artifacts.append(args[-1])
        return next(checks)

    monkeypatch.setattr(llm_module, "verify_result_with_llm", _verify)

    executed = []

    def _exec(command, language):
        executed.append(command)
        return True, "ok\n"

    summary, commands, conv = llm_module.llm.run("write the requested file", exec_callback=_exec)

    assert summary == "second attempt done"
    assert executed == ["printf ok", "printf ok"]
    assert [item.get("round") for item in commands] == [1, 2]
    assert seen_artifacts == ["artifact snapshot", "artifact snapshot"]
    assert llm_module.llm._last_verification["satisfied"] is True
    assert llm_module.llm._last_verification["llm"]["source"] == "llm"


def test_llm_run_skips_validation_when_disabled(monkeypatch):
    state.reset_session()
    client = _SequencedClient([iter([_Chunk(_Delta(content="plain answer"))])])
    monkeypatch.setattr(llm_module.llm, "client", client)
    monkeypatch.setattr(llm_module, "config_get", lambda key: {
        "max_llm_steps": 8,
        "verify_results": False,
    }.get(key, 12))

    summary, commands, _ = llm_module.llm.run("hello", exec_callback=lambda *_: (True, "ok"))
    assert summary == "plain answer"
    assert commands is None
    assert llm_module.llm._result_contract is None
