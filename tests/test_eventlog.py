"""Tests for the append-only event log, replay, and derived ATIF trajectory."""

import json

from srun.eventlog import EventLog, build_atif, replay_events
from srun.context import SessionState


def _seed_events(log):
    log.append("turn_start", {"user_input": "create report", "language": "shell"}, turn=1)
    log.append("contract", {"contract": {"goal": "create report", "checks": []}}, turn=1, round=1)
    log.append("llm_conversation", {"messages": [
        {"role": "user", "content": "create report"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "run_command", "arguments": '{"command":"echo ok"}'},
        }]},
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        {"role": "assistant", "content": "done"},
    ]}, turn=1)
    log.append("tool_result", {
        "tool_call_id": "call_1", "command": "echo ok", "language": "shell",
        "ok": True, "exit_code": 0, "captured": True, "output": "ok\n", "round": 1,
    }, turn=1, round=1)
    log.append("artifact_snapshot", {
        "round": 1, "sha256": "abc", "text": "--- report.txt ---\nhello\n",
    }, turn=1, round=1)
    log.append("verification", {
        "round": 1, "verdict": "pass", "satisfied": True,
        "reason": "artifact contains hello", "retry_instruction": "",
        "evidence_refs": [4, 5], "checks": [], "llm": {"source": "llm"},
    }, turn=1, round=1)
    log.append("turn_result", {"entry": {
        "turn": 1, "input": "create report", "success": True, "language": "shell",
    }}, turn=1)


def test_event_log_round_trip_and_hash_chain(tmp_path):
    path = tmp_path / "events.jsonl"
    log = EventLog(str(path), session_id="s1")
    _seed_events(log)
    events, truncated = EventLog(str(path), session_id="s1").read_events()
    assert not truncated
    assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
    assert events[0]["prev_hash"] == ""
    assert events[1]["prev_hash"] == events[0]["hash"]


def test_event_log_repairs_partial_crash_tail(tmp_path):
    path = tmp_path / "events.jsonl"
    log = EventLog(str(path), session_id="s1")
    log.append("turn_start", {"user_input": "one"}, turn=1)
    log.append("turn_end", {"summary": "ok"}, turn=1)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"seq": 3, "type": "partial"')

    repaired = EventLog(str(path), session_id="s1")
    assert repaired.truncated is False
    assert repaired.events[-1]["type"] == "log_repaired"
    events, truncated = repaired.read_events()
    assert not truncated
    assert events[0]["seq"] == 1


def test_two_eventlog_instances_keep_one_hash_chain(tmp_path):
    path = tmp_path / "events.jsonl"
    first = EventLog(str(path), session_id="s1")
    second = EventLog(str(path), session_id="s1")
    first.append("turn_start", {"user_input": "one"}, turn=1)
    second.append("turn_end", {"summary": "two"}, turn=1)
    first.append("turn_start", {"user_input": "three"}, turn=2)

    events, truncated = EventLog(str(path), session_id="s1").read_events()
    assert not truncated
    assert [event["seq"] for event in events] == [1, 2, 3]
    assert events[1]["prev_hash"] == events[0]["hash"]
    assert events[2]["prev_hash"] == events[1]["hash"]


def test_replay_rebuilds_session_state(tmp_path):
    log = EventLog(str(tmp_path / "events.jsonl"), session_id="s1")
    _seed_events(log)
    replayed = replay_events(log.events)
    assert replayed["turn"] == 1
    assert replayed["current_language"] == "shell"
    assert replayed["contract"]["goal"] == "create report"
    assert replayed["last_verification"]["verdict"] == "pass"
    assert replayed["last_verification"]["reason"] == "artifact contains hello"
    assert replayed["commands"][0]["output"] == "ok\n"
    assert [message["role"] for message in replayed["conversation"]] == [
        "user", "assistant", "tool", "assistant",
    ]


def test_atif_export_contains_steps_observations_and_verification(tmp_path):
    log = EventLog(str(tmp_path / "events.jsonl"), session_id="s1")
    _seed_events(log)
    trajectory = build_atif(log.events, "s1", str(log.path))
    assert trajectory["schema_version"] == "ATIF-v1.8"
    assert trajectory["agent"]["name"] == "AgenticREPL"
    assert [step["source"] for step in trajectory["steps"]] == [
        "user", "agent", "agent", "system",
    ]
    assert trajectory["steps"][1]["tool_calls"][0]["function_name"] == "run_command"
    observation = trajectory["steps"][1]["observation"]["results"][0]
    assert observation["source_call_id"] == "call_1"
    assert observation["content"] == "ok"
    assert observation["extra"]["exit_code"] == 0
    verification = trajectory["steps"][-1]
    assert verification["extra"]["verdict"] == "pass"
    assert verification["extra"]["reason"] == "artifact contains hello"
    assert verification["extra"]["evidence_refs"] == [4, 5]


def test_verification_event_is_auditable(tmp_path):
    log = EventLog(str(tmp_path / "events.jsonl"), session_id="s1")
    _seed_events(log)
    verification = [e for e in log.events if e["type"] == "verification"][0]
    payload = verification["payload"]
    assert payload["verdict"] == "pass"
    assert payload["reason"]
    assert payload["evidence_refs"] == [4, 5]
    assert verification["prev_hash"]


def test_session_state_replays_event_log_and_writes_atif(tmp_path):
    log = EventLog(str(tmp_path / "events.jsonl"), session_id="s1")
    _seed_events(log)
    session = SessionState()
    session.event_log = log
    session.events_path = str(log.path)
    session.trajectory_path = str(tmp_path / "trajectory.json")
    session._replay_event_log()
    assert session._turn == 1
    assert session.last_verification["verdict"] == "pass"
    assert session.last_contract["goal"] == "create report"
    session.write_atif()
    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    assert trajectory["schema_version"] == "ATIF-v1.8"
    assert trajectory["steps"]
