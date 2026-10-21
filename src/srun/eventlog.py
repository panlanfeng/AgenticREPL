"""Append-only event log and derived ATIF trajectory export.

The event log is the single source of truth for a session. Every state change
that matters for crash recovery is appended as a hash-chained JSONL event.
`trajectory.json` is a derived ATIF view and can always be rebuilt from the log.
"""

import hashlib
import json
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None


class EventLogCorruption(RuntimeError):
    pass


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _event_hash(event):
    body = {key: value for key, value in event.items() if key != "hash"}
    return hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()


class EventLog:
    def __init__(self, path, session_id="", repair_truncated_tail=True, fsync=False):
        self.path = path
        self.session_id = session_id or os.path.basename(os.path.dirname(path))
        self.lock_path = path + ".lock"
        self.fsync = fsync
        self.last_error = None
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.events, self.truncated = self.read_events()
        self._seq = self.events[-1]["seq"] if self.events else 0
        self._last_hash = self.events[-1]["hash"] if self.events else ""
        if self.truncated and repair_truncated_tail:
            self._rewrite_valid_prefix()
            self.append("log_repaired", {"dropped_partial_tail": True})

    @contextmanager
    def _file_lock(self, exclusive=True):
        os.makedirs(os.path.dirname(self.lock_path) or ".", exist_ok=True)
        handle = open(self.lock_path, "a+")
        try:
            if fcntl:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            yield
        finally:
            if fcntl:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    def _last_event_on_disk(self):
        """Read the last valid event without loading the whole log."""
        if not os.path.isfile(self.path):
            return None
        with open(self.path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            if size == 0:
                return None
            chunk = min(size, 64 * 1024)
            handle.seek(size - chunk)
            lines = handle.read().splitlines()
        for line in reversed(lines):
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(event, dict) and event.get("seq") and event.get("hash"):
                return event
        return None

    def _rewrite_valid_prefix(self):
        with self._file_lock(exclusive=True):
            self._rewrite_valid_prefix_locked()

    def _rewrite_valid_prefix_locked(self):
        if not self.events:
            try:
                os.remove(self.path)
            except FileNotFoundError:
                pass
            self.truncated = False
            return
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            for event in self.events:
                handle.write(_canonical(event) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, self.path)
        self.truncated = False

    def read_events(self):
        """Read and validate the hash chain. A partial final line is treated as
        a crash tail and returned as truncated=True; corruption elsewhere raises."""
        if not os.path.isfile(self.path):
            return [], False
        with self._file_lock(exclusive=False):
            with open(self.path, "r", encoding="utf-8") as handle:
                lines = [line for line in handle if line.strip()]
        events = []
        for index, line in enumerate(lines):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) - 1:
                    return events, True
                raise EventLogCorruption(f"invalid JSON at line {index + 1}")
            expected_prev = events[-1]["hash"] if events else ""
            if event.get("prev_hash", "") != expected_prev:
                raise EventLogCorruption(f"prev_hash mismatch at seq {event.get('seq')}")
            if event.get("hash") != _event_hash(event):
                raise EventLogCorruption(f"hash mismatch at seq {event.get('seq')}")
            if event.get("seq") != len(events) + 1:
                raise EventLogCorruption(f"non-contiguous seq {event.get('seq')}")
            events.append(event)
        return events, False

    def append(self, event_type, payload=None, turn=None, round=None):
        with self._file_lock(exclusive=True):
            last = self._last_event_on_disk()
            self._seq = last["seq"] if last else 0
            self._last_hash = last["hash"] if last else ""
            seq = self._seq + 1
            event = {
                "seq": seq,
                "event_id": uuid.uuid4().hex,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "session_id": self.session_id,
                "type": event_type,
                "turn": turn,
                "round": round,
                "payload": payload or {},
                "prev_hash": self._last_hash,
            }
            event["hash"] = _event_hash(event)
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(_canonical(event) + "\n")
                handle.flush()
                durable = self.fsync or event_type in (
                    "verification", "turn_end", "turn_result", "log_repaired",
                    "session_state", "contract",
                )
                if durable:
                    os.fsync(handle.fileno())
            self.events.append(event)
            self._seq = seq
            self._last_hash = event["hash"]
            return event

    @property
    def last_seq(self):
        return self._seq

    @property
    def last_hash(self):
        return self._last_hash

    def replay(self):
        events, _ = self.read_events()
        return replay_events(events)

    def export_atif(self):
        events, _ = self.read_events()
        return build_atif(events, self.session_id, self.path)

    def write_atif(self, path):
        trajectory = self.export_atif()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(trajectory, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        return path


def replay_events(events):
    """Replay events into the minimum state needed to resume a session."""
    replayed = {
        "conversation": [],
        "summary": None,
        "session_log": [],
        "turn": 0,
        "current_language": "shell",
        "last_lang": None,
        "vars": {},
        "active_df": None,
        "contract": None,
        "verifications": [],
        "retries": [],
        "commands": [],
        "last_verification": None,
        "last_turn_start": None,
        "last_turn_end": None,
        "repairs": [],
        "permission_grants": [],
    }
    for event in events:
        event_type = event.get("type")
        payload = event.get("payload") or {}
        if event_type == "llm_conversation":
            replayed["conversation"].extend(payload.get("messages", []))
        elif event_type == "compaction":
            replayed["summary"] = payload.get("summary")
            replayed["conversation"] = list(payload.get("conversation", []))
        elif event_type == "turn_start":
            replayed["last_turn_start"] = payload
            if payload.get("language"):
                replayed["current_language"] = payload["language"]
        elif event_type == "contract":
            replayed["contract"] = payload.get("contract")
        elif event_type == "tool_result":
            replayed["commands"].append(payload)
        elif event_type == "verification":
            replayed["verifications"].append(payload)
            replayed["last_verification"] = payload
        elif event_type == "retry":
            replayed["retries"].append(payload)
        elif event_type == "session_state":
            if payload.get("current_language"):
                replayed["current_language"] = payload["current_language"]
            replayed["last_lang"] = payload.get("last_lang", replayed["last_lang"])
            replayed["vars"] = payload.get("vars", replayed["vars"])
            replayed["active_df"] = payload.get("active_df", replayed["active_df"])
        elif event_type == "permission_grant":
            grant = payload.get("grant") if isinstance(payload, dict) else None
            if grant and grant not in replayed["permission_grants"]:
                replayed["permission_grants"].append(grant)
        elif event_type == "permission_revoke":
            grant_id = payload.get("grant_id")
            replayed["permission_grants"] = [
                grant for grant in replayed["permission_grants"]
                if grant.get("grant_id") != grant_id
            ]
        elif event_type == "turn_result":
            entry = payload.get("entry") or {}
            replayed["session_log"].append(entry)
            replayed["turn"] = max(replayed["turn"], int(entry.get("turn", 0) or 0))
            if entry.get("language"):
                replayed["current_language"] = entry["language"]
            replayed["last_lang"] = entry.get("language") or replayed["last_lang"]
        elif event_type == "turn_end":
            replayed["last_turn_end"] = payload
        elif event_type == "log_repaired":
            replayed["repairs"].append(payload)
    return replayed


def _agent_step(step_id, timestamp, message="", reasoning=None, tool_calls=None):
    step = {
        "step_id": step_id,
        "timestamp": timestamp,
        "source": "agent",
        "message": message or "",
        "llm_call_count": 1,
    }
    if reasoning:
        step["reasoning_content"] = reasoning
    if tool_calls:
        step["tool_calls"] = tool_calls
    return step


def _convert_tool_calls(tool_calls):
    converted = []
    for call in tool_calls or []:
        function = call.get("function") or {}
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"raw": arguments}
        converted.append({
            "tool_call_id": call.get("id") or "",
            "function_name": function.get("name") or "",
            "arguments": arguments if isinstance(arguments, dict) else {},
        })
    return converted


def build_atif(events, session_id, event_log_path):
    """Build an ATIF-v1.8 trajectory from append-only events.

    ATIF is a derived view: rebuild it after a crash from events.jsonl.
    """
    from .config import config
    from .tools import TOOL_DEFINITIONS

    tool_results = {}
    for event in events:
        if event.get("type") == "tool_result":
            payload = event.get("payload") or {}
            call_id = payload.get("tool_call_id")
            if call_id:
                tool_results[call_id] = payload

    steps = []
    next_step_id = 1
    pending_agent_step = None
    step_by_call_id = {}

    def add_step(step):
        nonlocal next_step_id, pending_agent_step
        step["step_id"] = next_step_id
        next_step_id += 1
        steps.append(step)
        if step.get("source") == "agent":
            pending_agent_step = step
            for call in step.get("tool_calls", []):
                step_by_call_id[call.get("tool_call_id")] = step
        return step

    for event in events:
        event_type = event.get("type")
        payload = event.get("payload") or {}
        timestamp = event.get("timestamp")
        if event_type == "llm_conversation":
            for message in payload.get("messages", []):
                role = message.get("role")
                if role == "user":
                    add_step({
                        "step_id": 0,
                        "timestamp": timestamp,
                        "source": "user",
                        "message": message.get("content") or "",
                    })
                    pending_agent_step = None
                elif role == "system":
                    add_step({
                        "step_id": 0,
                        "timestamp": timestamp,
                        "source": "system",
                        "message": message.get("content") or "",
                    })
                elif role == "assistant":
                    calls = _convert_tool_calls(message.get("tool_calls"))
                    add_step(_agent_step(
                        step_id=0,
                        timestamp=timestamp,
                        message=message.get("content") or "",
                        reasoning=message.get("reasoning_content"),
                        tool_calls=calls,
                    ))
                elif role == "tool":
                    call_id = message.get("tool_call_id")
                    target = step_by_call_id.get(call_id) or pending_agent_step
                    result = {
                        "source_call_id": call_id,
                        "content": message.get("content") or "",
                    }
                    metadata = tool_results.get(call_id)
                    if metadata:
                        result["extra"] = metadata
                    if target is None:
                        target = add_step({
                            "step_id": 0,
                            "timestamp": timestamp,
                            "source": "system",
                            "message": "tool observation",
                        })
                    target.setdefault("observation", {"results": []})["results"].append(result)
        elif event_type == "compaction":
            add_step({
                "step_id": 0,
                "timestamp": timestamp,
                "source": "system",
                "message": "context compacted",
                "extra": {"summary": payload.get("summary")},
            })
        elif event_type == "verification":
            verdict = payload.get("verdict") or (
                "pass" if payload.get("satisfied") is True
                else "fail" if payload.get("satisfied") is False else "unknown"
            )
            add_step({
                "step_id": 0,
                "timestamp": timestamp,
                "source": "system",
                "message": "result verification",
                "observation": {
                    "results": [{
                        "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    }]
                },
                "extra": {
                    "event_type": "verification",
                    "round": payload.get("round"),
                    "verdict": verdict,
                    "reason": payload.get("reason"),
                    "retry_instruction": payload.get("retry_instruction"),
                    "evidence_refs": payload.get("evidence_refs", []),
                    "execution_error": payload.get("execution_error"),
                },
            })
        elif event_type == "error_intent":
            add_step({
                "step_id": 0,
                "timestamp": timestamp,
                "source": "system",
                "message": "execution intent verification",
                "observation": {
                    "results": [{
                        "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    }]
                },
                "extra": {
                    "event_type": "error_intent",
                    "intent": payload.get("intent"),
                    "reason": payload.get("reason"),
                },
            })
        elif event_type == "log_repaired" and payload.get("dropped_partial_tail"):
            add_step({
                "step_id": 0,
                "timestamp": timestamp,
                "source": "system",
                "message": "event log partial tail repaired",
            })

    return {
        "schema_version": "ATIF-v1.8",
        "session_id": session_id,
        "trajectory_id": f"{session_id}-trajectory",
        "agent": {
            "name": "AgenticREPL",
            "version": "0.1.0",
            "model_name": getattr(config, "model", ""),
            "tool_definitions": TOOL_DEFINITIONS,
            "extra": {"source": "events.jsonl"},
        },
        "steps": steps,
        "notes": "Derived from append-only events.jsonl. Rebuild after crash; do not edit as source of truth.",
        "extra": {
            "event_log": event_log_path,
            "last_event_seq": events[-1]["seq"] if events else 0,
            "last_event_hash": events[-1]["hash"] if events else "",
        },
    }
