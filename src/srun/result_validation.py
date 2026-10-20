"""Deterministic result contracts and post-execution checks.

A contract is emitted by the same LLM turn that recognizes intent, through the
``set_result_contract`` tool. The contract is a small list of machine-checkable
checks, so validating it needs no second LLM call.

Code can verify explicit outcomes (a file exists, output contains a value, a
command exits 0, a JSON field equals a value). It cannot reliably judge open-ended
semantic quality such as "explain this well". Contracts should therefore contain
only checks that are actually decidable; an empty check list means unverified.
"""

import json
import hashlib
import os
import re
import subprocess


VERIFY_SYSTEM = """You are a stateless result verifier for an execution agent.
You receive only the user request, the recorded acceptance contract, the execution
log, and an artifact snapshot. You have no conversation history and no tools.

Judge whether the work actually satisfies the user's request. For file changes,
compare the artifact diff/content with the request, not just the agent's claim.
An exit code of 0 is not sufficient by itself. Do not invent evidence. If required
evidence is missing or the artifact does not show the requested change, mark the
result unsatisfied and give one concrete retry instruction.

If an execution error is included, decide what the user intended:
- "code": the user typed a command/code with a typo or invalid syntax; the error is relevant.
- "text": the user wrote a question, greeting, or natural-language request; the shell
  execution error is incidental and should be ignored.

Return ONLY one JSON object:
{
  "satisfied": true or false,
  "intent": "code" or "text" or "unknown",
  "error_relevant": true or false,
  "reason": "brief evidence-based explanation",
  "retry_instruction": "specific next action when unsatisfied; empty when satisfied"
}"""


SUPPORTED_CHECKS = (
    "file_exists",
    "file_not_exists",
    "file_contains",
    "file_equals",
    "line_count",
    "output_contains",
    "output_matches",
    "output_equals",
    "exit_code",
    "all_commands_ok",
    "json_field",
    "command_contains",
)

ACTION_WORDS = re.compile(
    r"\b(create|write|edit|modify|delete|remove|run|execute|test|install|save|"
    r"download|upload|rename|move|copy|fix|build|deploy|commit|push|pull|open|"
    r"read|find|list|show|print|count|sort|filter|group|analyze|convert|compress|"
    r"extract|add|update|change|generate|make|clean|format|check|verify)\b",
    re.IGNORECASE,
)
ACTION_PATHS = re.compile(
    r"(?:^|\s)(?:\./|/|~/)[^\s]+|\b[\w.-]+\.(?:csv|tsv|json|jsonl|py|sh|r|"
    r"md|txt|yaml|yml|toml|log|zip|tar|gz|xlsx?|pdf)\b",
    re.IGNORECASE,
)


def looks_actionable(user_input):
    """Cheap heuristic for whether a turn needs a contract/verifier."""
    text = (user_input or "").strip()
    if not text:
        return False
    return bool(ACTION_WORDS.search(text) or ACTION_PATHS.search(text))


def fallback_contract(user_input):
    """A contract with no checks. It is intentionally not claimed as verified."""
    request = (user_input or "").strip()
    return {
        "goal": request or "Complete the user's request.",
        "expected_result": "",
        "expected_format": "not specified",
        "checks": [],
    }


def normalize_contract(value, user_input):
    """Keep a model-provided contract within the supported check schema."""
    if not isinstance(value, dict):
        return fallback_contract(user_input)
    checks = value.get("checks")
    if not isinstance(checks, list):
        checks = []
    clean_checks = []
    notes = []
    for check in checks:
        if not isinstance(check, dict):
            continue
        check_type = check.get("type")
        if not isinstance(check_type, str) or not check_type.strip():
            continue
        clean = dict(check)
        clean["type"] = check_type.strip().lower()
        if clean["type"] not in SUPPORTED_CHECKS:
            notes.append(f"dropped unsupported check type: {clean['type']}")
            continue
        if clean["type"] in ("output_equals", "file_equals",
                             "output_contains", "file_contains"):
            if not clean.get("text") and clean.get("equals") is not None:
                clean["text"] = clean.pop("equals")
                notes.append(f"moved equals to text for {clean['type']}")
        if clean["type"] == "output_equals" and not clean.get("text"):
            notes.append("dropped output_equals without text")
            continue
        if clean["type"] == "output_matches" and not clean.get("pattern"):
            notes.append("dropped output_matches without pattern")
            continue
        valid_sources = {"all", "last", "summary", "file"}
        if clean["type"].startswith("output_") or clean["type"] == "json_field":
            if clean.get("source") not in valid_sources:
                clean["source"] = (
                    "summary" if clean["type"] == "output_equals"
                    else "last" if clean["type"] == "json_field"
                    else "all"
                )
        if clean["type"] == "output_contains" and "case_sensitive" not in clean:
            clean["case_sensitive"] = False
        if clean.get("scope") not in (None, "current", "all"):
            clean.pop("scope", None)
        clean_checks.append(clean)
    return {
        "goal": str(value.get("goal") or user_input or "").strip(),
        "expected_result": str(value.get("expected_result") or "").strip(),
        "expected_format": str(value.get("expected_format") or "not specified").strip(),
        "checks": clean_checks[:12],
        "normalization_notes": notes,
    }


def format_contract(contract):
    """Render the contract for the tool result and agent context."""
    checks = contract.get("checks") if isinstance(contract, dict) else []
    lines = [
        f"Goal: {contract.get('goal', '')}",
        f"Expected result: {contract.get('expected_result', '')}",
        f"Expected format: {contract.get('expected_format', 'not specified')}",
        "Machine checks:",
    ]
    if checks:
        lines.extend(f"- {json.dumps(check, ensure_ascii=False, sort_keys=True)}"
                     for check in checks)
    else:
        lines.append("- (none; result cannot be verified deterministically)")
    return "\n".join(lines)


def parse_json_object(text):
    """Extract the first JSON object from plain text or a fenced block."""
    if not text:
        return None
    candidates = []
    for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE):
        candidates.append(match.group(1).strip())
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidates.append(stripped[start:end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            return value
    return None


def _completion_text(response):
    """Read text from either a normal ChatCompletion or a stream-like test double."""
    if response is None:
        return ""
    choices = getattr(response, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None)
        if message is not None:
            return getattr(message, "content", "") or ""
    parts = []
    try:
        for chunk in response:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if delta is not None and getattr(delta, "content", None):
                parts.append(delta.content)
    except TypeError:
        return ""
    return "".join(parts)


def _request_json(client, model, system_prompt, user_content, max_tokens=900):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
        max_tokens=max_tokens,
        stream=False,
    )
    return parse_json_object(_completion_text(response))


def _truncate_text(text, limit):
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... ({len(text) - limit} chars omitted)"


def _resolve(path, cwd):
    if not isinstance(path, str) or not path.strip():
        return None
    resolved = os.path.expanduser(path.strip())
    if not os.path.isabs(resolved):
        resolved = os.path.join(cwd, resolved)
    return os.path.abspath(resolved)


def _read_text(path, cwd, max_bytes=1024 * 1024):
    resolved = _resolve(path, cwd)
    if not resolved:
        return None, "check is missing a path"
    if not os.path.isfile(resolved):
        return None, f"file does not exist: {resolved}"
    try:
        if os.path.getsize(resolved) > max_bytes:
            return None, f"file is too large to verify: {resolved}"
        with open(resolved, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read(), None
    except OSError as exc:
        return None, f"could not read {resolved}: {exc}"


def _commands(commands):
    return [command for command in (commands or []) if isinstance(command, dict)]


def _executed_commands(commands):
    return [command for command in _commands(commands) if command.get("executed")]


def _scope_commands(check, context):
    if check.get("scope") == "all":
        return context["all_commands"]
    return context["commands"]


def _all_uncaptured(commands):
    executed = _executed_commands(commands)
    return bool(executed) and all(command.get("captured") is False for command in executed)


def _last_output(commands):
    outputs = [command.get("output") or "" for command in _commands(commands)]
    return outputs[-1] if outputs else ""


def _all_outputs(commands):
    return "\n".join(command.get("output") or "" for command in _commands(commands)
                     if command.get("output"))


def _source_text(check, context):
    source = check.get("source") or "all"
    commands = _scope_commands(check, context)
    if source == "summary":
        return context["summary"], None
    if source == "last":
        last_output = _last_output(commands)
        if last_output:
            return last_output, None
        if context["summary"]:
            return context["summary"], None
        if _all_uncaptured(commands):
            return None, "command output was not captured (TTY-only command)"
        return "", None
    if source == "file":
        return _read_text(check.get("path"), context["cwd"])
    combined = "\n".join(part for part in (
        _all_outputs(commands), context["summary"],
    ) if part)
    if not combined and _all_uncaptured(commands):
        return None, "command output was not captured (TTY-only command)"
    return combined, None


def _contains(text, needle, case_sensitive=True):
    if not isinstance(text, str) or not isinstance(needle, str):
        return False
    if case_sensitive:
        return needle in text
    return needle.lower() in text.lower()


def _as_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _values_equal(found, expected):
    if found == expected:
        return True
    if isinstance(found, bool) and isinstance(expected, str):
        return str(found).lower() == expected.strip().lower()
    if isinstance(found, (int, float)) and isinstance(expected, str):
        return str(found) == expected.strip()
    return False


def _get_field(value, field_path):
    current = value
    for part in str(field_path or "").split("."):
        if not part:
            continue
        if isinstance(current, list):
            index = _as_int(part)
            if index is None or index < 0 or index >= len(current):
                return False, None
            current = current[index]
        elif isinstance(current, dict):
            if part not in current:
                return False, None
            current = current[part]
        else:
            return False, None
    return True, current


def _check_file_exists(check, context):
    resolved = _resolve(check.get("path"), context["cwd"])
    if not resolved:
        return False, "check is missing a path"
    if os.path.isfile(resolved):
        return True, f"file exists: {resolved}"
    return False, f"file does not exist: {resolved}"


def _check_file_not_exists(check, context):
    resolved = _resolve(check.get("path"), context["cwd"])
    if not resolved:
        return False, "check is missing a path"
    if os.path.exists(resolved):
        return False, f"path still exists: {resolved}"
    return True, f"path does not exist: {resolved}"


def _check_file_contains(check, context):
    text, error = _read_text(check.get("path"), context["cwd"])
    if error:
        return False, error
    needle = check.get("text")
    if not isinstance(needle, str) or not needle:
        return False, "file_contains check is missing text"
    case_sensitive = check.get("case_sensitive", True)
    haystack = text if case_sensitive else text.lower()
    target = needle if case_sensitive else needle.lower()
    count = haystack.count(target)
    expected_count = _as_int(check.get("count"))
    if expected_count is not None:
        return count == expected_count, f"found {count} occurrence(s); expected {expected_count}"
    return count > 0, f"found {count} occurrence(s)"


def _check_file_equals(check, context):
    text, error = _read_text(check.get("path"), context["cwd"])
    if error:
        return False, error
    expected = check.get("text")
    if not isinstance(expected, str):
        return False, "file_equals check is missing text"
    return text == expected, "file content matches" if text == expected else "file content differs"


def _check_line_count(check, context):
    text, error = _read_text(check.get("path"), context["cwd"])
    if error:
        return False, error
    count = len(text.splitlines())
    expected = _as_int(check.get("equals"))
    minimum = _as_int(check.get("min"))
    maximum = _as_int(check.get("max"))
    if expected is not None and count != expected:
        return False, f"line count is {count}; expected {expected}"
    if minimum is not None and count < minimum:
        return False, f"line count is {count}; expected at least {minimum}"
    if maximum is not None and count > maximum:
        return False, f"line count is {count}; expected at most {maximum}"
    return True, f"line count is {count}"


def _check_output_contains(check, context):
    text, error = _source_text(check, context)
    if error:
        return False, error
    needle = check.get("text")
    if not isinstance(needle, str) or not needle:
        return False, "output_contains check is missing text"
    case_sensitive = check.get("case_sensitive", False)
    ok = _contains(text, needle, case_sensitive)
    return ok, f"output {'contains' if ok else 'does not contain'} {needle!r}"


def _check_output_matches(check, context):
    text, error = _source_text(check, context)
    if error:
        return False, error
    pattern = check.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        return False, "output_matches check is missing pattern"
    try:
        ok = re.search(pattern, text, re.MULTILINE) is not None
    except re.error as exc:
        return False, f"invalid regex: {exc}"
    return ok, f"output {'matches' if ok else 'does not match'} /{pattern}/"


def _check_output_equals(check, context):
    text, error = _source_text(check, context)
    if error:
        return False, error
    expected = check.get("text")
    if not isinstance(expected, str):
        return False, "output_equals check is missing text"
    ok = text == expected
    return ok, "output matches exactly" if ok else "output differs"


def _check_exit_code(check, context):
    expected = _as_int(check.get("equals"))
    if expected is None:
        return False, "exit_code check is missing integer equals"
    commands = _executed_commands(_scope_commands(check, context))
    command_filter = check.get("command_contains")
    if isinstance(command_filter, str) and command_filter:
        commands = [command for command in commands
                    if command_filter in (command.get("command") or "")]
    if not commands:
        return False, "no executed command matched the exit_code check"
    failed = []
    for command in commands:
        actual = command.get("exit_code")
        if actual is None:
            actual = 0 if command.get("ok") else 1
        if actual != expected:
            failed.append(actual)
    if failed:
        return False, f"exit codes {failed} != {expected}"
    return True, f"all matching commands exited {expected}"


def _check_all_commands_ok(check, context):
    commands = _executed_commands(_scope_commands(check, context))
    if not commands:
        return False, "no command was executed"
    failed = [command.get("command") for command in commands if not command.get("ok")]
    if failed:
        return False, f"{len(failed)} command(s) failed"
    return True, f"all {len(commands)} executed command(s) succeeded"


def _check_json_field(check, context):
    text, error = _source_text(check, context)
    if error:
        return False, error
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        return False, f"source is not valid JSON: {exc}"
    exists, found = _get_field(value, check.get("field"))
    if "exists" in check:
        expected_exists = bool(check.get("exists"))
        if expected_exists != exists:
            return False, f"JSON field existence is {exists}; expected {expected_exists}"
        if not expected_exists:
            return True, f"JSON field {check.get('field')!r} is absent as expected"
    if not exists:
        return False, f"JSON field {check.get('field')!r} is missing"
    if "equals" in check and not _values_equal(found, check.get("equals")):
        return False, f"JSON field value is {found!r}; expected {check.get('equals')!r}"
    if "contains" in check and not _contains(str(found), str(check.get("contains"))):
        return False, f"JSON field value does not contain {check.get('contains')!r}"
    return True, f"JSON field {check.get('field')!r} satisfies the check"


def _check_command_contains(check, context):
    needle = check.get("text")
    if not isinstance(needle, str) or not needle:
        return False, "command_contains check is missing text"
    commands = _commands(_scope_commands(check, context))
    ok = any(needle in (command.get("command") or "") for command in commands)
    return ok, f"command {'contains' if ok else 'does not contain'} {needle!r}"


CHECK_HANDLERS = {
    "file_exists": _check_file_exists,
    "file_not_exists": _check_file_not_exists,
    "file_contains": _check_file_contains,
    "file_equals": _check_file_equals,
    "line_count": _check_line_count,
    "output_contains": _check_output_contains,
    "output_matches": _check_output_matches,
    "output_equals": _check_output_equals,
    "exit_code": _check_exit_code,
    "all_commands_ok": _check_all_commands_ok,
    "json_field": _check_json_field,
    "command_contains": _check_command_contains,
}


def _retry_instruction(check, detail):
    check_type = check.get("type")
    path = check.get("path") or "the requested file"
    text = check.get("text")
    if check_type == "file_exists":
        return f"Create or restore {path}."
    if check_type == "file_not_exists":
        return f"Remove {path}."
    if check_type in ("file_contains", "file_equals"):
        return f"Write the required content into {path}."
    if check_type == "line_count":
        return f"Adjust the line count of {path} to match the check."
    if check_type == "output_contains":
        return f"Make the command output or final response include {text!r}."
    if check_type == "output_equals":
        return f"Make the command output or final response exactly {text!r}."
    if check_type == "output_matches":
        return f"Make the output match the pattern {check.get('pattern')!r}."
    if check_type == "exit_code":
        return f"Make the matching command exit with code {check.get('equals')}."
    if check_type == "all_commands_ok":
        return "Fix the failing command(s) and run them again."
    if check_type == "json_field":
        return f"Make JSON field {check.get('field')!r} satisfy the check."
    if check_type == "command_contains":
        return f"Run a command containing {text!r}."
    return detail


def format_execution_log(commands, summary, max_chars=24000):
    """Render all agent commands and their captured output for the verifier."""
    parts = []
    for command in _commands(commands):
        captured = command.get("captured", True)
        parts.append(
            f"[round {command.get('round', '?')}] "
            f"language={command.get('language', 'unknown')} "
            f"exit={command.get('exit_code', 'unknown')} "
            f"ok={command.get('ok', 'unknown')} "
            f"captured={captured}"
        )
        parts.append(f"command:\n{command.get('command', '')}")
        if captured is False:
            parts.append("output:\n(not captured; TTY-only command)")
        else:
            output = command.get("output") or "(empty)"
            parts.append("output:\n" + _truncate_text(output, 4000))
    if summary:
        parts.append("agent final response:\n" + _truncate_text(summary, 6000))
    return _truncate_text("\n\n".join(parts) if parts else "(no execution log)", max_chars)


def _artifact_preview(path, max_bytes=256 * 1024, max_chars=8000):
    try:
        size = os.path.getsize(path)
        if size > max_bytes:
            return f"({size}B; preview omitted because the file is too large)"
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            content = handle.read()
    except OSError as exc:
        return f"(could not read artifact: {exc})"
    if "\x00" in content:
        return f"({size}B; binary file; preview omitted)"
    try:
        from .tools import _redact_secrets
        content = _redact_secrets(path, content)
    except Exception:
        pass
    return _truncate_text(content, max_chars)


def collect_artifacts(cwd=None, contract=None, commands=None, since=None,
                      max_files=12, max_chars=30000):
    """Collect turn-related git changes and referenced file previews.

    Only files referenced by the contract or modified since ``since`` are treated
    as turn artifacts. This prevents unrelated pre-existing repo changes from
    confusing the verifier.
    """
    cwd = os.path.abspath(cwd or os.getcwd())
    contract_paths = []
    for check in (contract or {}).get("checks", []):
        if isinstance(check, dict) and check.get("path"):
            contract_paths.append(str(check["path"]))

    status_paths = []
    status_lines = []
    is_git = False
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd, capture_output=True, text=True, timeout=5,
        )
        if status.returncode == 0:
            is_git = True
            for line in status.stdout.strip().splitlines():
                raw = line[3:].strip() if len(line) > 3 else ""
                if " -> " in raw:
                    raw = raw.split(" -> ", 1)[1]
                raw = raw.strip().strip('"')
                if raw:
                    status_paths.append(raw)
                    status_lines.append((raw, line))
    except Exception:
        pass

    recent_paths = []
    if since is not None and not is_git:
        skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}
        for root, dirs, files in os.walk(cwd):
            dirs[:] = [name for name in dirs if not name.startswith(".") and name not in skip_dirs]
            for name in files:
                path = os.path.join(root, name)
                try:
                    if os.path.getmtime(path) >= since - 1:
                        recent_paths.append(os.path.relpath(path, cwd))
                except OSError:
                    continue
                if len(recent_paths) >= max_files:
                    break
            if len(recent_paths) >= max_files:
                break

    candidate_paths = contract_paths + status_paths + recent_paths
    contract_resolved = {_resolve(path, cwd) for path in contract_paths}
    selected = []
    seen = set()
    for raw_path in candidate_paths:
        resolved = _resolve(raw_path, cwd)
        if not resolved or resolved in seen or not os.path.isfile(resolved):
            continue
        if since is not None and resolved not in contract_resolved:
            try:
                if os.path.getmtime(resolved) < since - 1:
                    continue
            except OSError:
                continue
        seen.add(resolved)
        selected.append(resolved)
        if len(selected) >= max_files:
            break

    parts = []
    selected_rel = set()
    for path in selected:
        rel = os.path.relpath(path, cwd)
        if rel != ".." and not rel.startswith(".." + os.sep):
            selected_rel.add(rel)
    relevant_status = [line for raw, line in status_lines if raw in selected_rel]
    if relevant_status:
        parts.append("Git status (turn-related):\n" + "\n".join(relevant_status))
    elif status_lines:
        parts.append("Git status (turn-related): (none)")
    else:
        parts.append("Git status (turn-related): (clean or not a git repo)")

    if selected_rel:
        try:
            diff = subprocess.run(
                ["git", "diff", "HEAD", "--no-ext-diff", "--unified=1", "--",
                 *sorted(selected_rel)],
                cwd=cwd, capture_output=True, text=True, timeout=15,
            )
            if diff.returncode == 0 and diff.stdout.strip():
                parts.append("Git diff (turn-related):\n" + _truncate_text(diff.stdout, 18000))
        except Exception:
            pass

    artifacts = []
    for resolved in selected:
        size = os.path.getsize(resolved)
        digest = "skipped"
        if size <= 5 * 1024 * 1024:
            try:
                with open(resolved, "rb") as handle:
                    digest = hashlib.sha256(handle.read()).hexdigest()
            except OSError:
                digest = "unavailable"
        artifacts.append(f"--- {resolved} ({size}B sha256={digest}) ---\n"
                         + _artifact_preview(resolved))
    parts.extend(artifacts)
    return _truncate_text("\n\n".join(parts), max_chars)


def verify_result_with_llm(client, model, user_input, contract, commands,
                           summary, artifacts, error=None, max_tokens=900):
    """Stateless verifier call: fixed system prompt + self-contained evidence."""
    user_content = (
        f"User request:\n{user_input}\n\n"
        f"Acceptance contract:\n{format_contract(contract)}\n\n"
        f"Execution error:\n{error or '(none)'}\n\n"
        f"Execution log:\n{format_execution_log(commands, summary)}\n\n"
        f"Artifacts:\n{artifacts}"
    )
    try:
        value = _request_json(client, model, VERIFY_SYSTEM, user_content, max_tokens=max_tokens)
    except Exception as exc:
        return {
            "satisfied": None,
            "reason": f"LLM verifier unavailable: {exc}",
            "retry_instruction": "",
            "source": "llm",
            "error": str(exc),
        }
    if not isinstance(value, dict):
        return {
            "satisfied": None,
            "reason": "LLM verifier response was not parseable",
            "retry_instruction": "",
            "source": "llm",
            "error": "unparseable verification",
        }
    satisfied = value.get("satisfied")
    if isinstance(satisfied, str):
        decision = satisfied.strip().lower()
        if decision in ("true", "yes", "1", "pass", "passed"):
            satisfied = True
        elif decision in ("false", "no", "0", "fail", "failed"):
            satisfied = False
        else:
            satisfied = None
    if satisfied not in (True, False):
        return {
            "satisfied": None,
            "reason": "LLM verifier omitted a valid satisfied field",
            "retry_instruction": "",
            "source": "llm",
            "error": "missing satisfied field",
        }
    intent = str(value.get("intent") or "unknown").strip().lower()
    if intent not in ("code", "text", "unknown"):
        intent = "unknown"
    error_relevant = value.get("error_relevant")
    if isinstance(error_relevant, str):
        error_relevant = error_relevant.strip().lower() in ("true", "yes", "1")
    return {
        "satisfied": satisfied,
        "intent": intent,
        "error_relevant": bool(error_relevant) if error_relevant is not None else None,
        "reason": str(value.get("reason") or "").strip(),
        "retry_instruction": str(value.get("retry_instruction") or "").strip(),
        "source": "llm",
    }


def combine_verifications(code_result, llm_result):
    """Combine deterministic checks with the stateless LLM judge."""
    code_satisfied = code_result.get("satisfied") if code_result else None
    llm_satisfied = llm_result.get("satisfied") if llm_result else None
    if code_satisfied is False and llm_satisfied is True:
        return {
            "satisfied": True,
            "reason": f"LLM verifier overrode failed code checks: {llm_result.get('reason', '')}".strip(),
            "retry_instruction": "",
            "checks": code_result.get("checks", []),
            "llm": llm_result,
            "code_overridden": True,
            "unverifiable": False,
        }
    if code_satisfied is False:
        return code_result
    if llm_satisfied is False:
        return {
            "satisfied": False,
            "reason": f"LLM verifier: {llm_result.get('reason') or 'result does not satisfy the request'}",
            "retry_instruction": (llm_result.get("retry_instruction")
                                  or code_result.get("retry_instruction", "")),
            "checks": code_result.get("checks", []),
            "llm": llm_result,
            "unverifiable": False,
        }
    if llm_satisfied is True:
        code_reason = code_result.get("reason", "") if code_result else ""
        reason = f"{code_reason}; LLM verifier: {llm_result.get('reason', '')}".strip("; ")
        return {
            "satisfied": True,
            "reason": reason,
            "retry_instruction": "",
            "checks": code_result.get("checks", []),
            "llm": llm_result,
            "unverifiable": code_result.get("unverifiable", False),
        }
    if code_satisfied is True:
        return {
            "satisfied": None,
            "reason": "code checks passed; LLM verification unavailable",
            "retry_instruction": "",
            "checks": code_result.get("checks", []),
            "llm": llm_result,
            "unverifiable": True,
        }
    return code_result


def validate_contract(contract, commands, summary, cwd=None, current_round=None):
    """Validate a contract using only local code. Returns a result dictionary."""
    all_commands = _commands(commands)
    if current_round is not None:
        scoped_commands = [command for command in all_commands
                           if command.get("round") == current_round]
    else:
        rounds = [command.get("round") for command in all_commands
                  if command.get("round") is not None]
        if rounds:
            latest_round = max(rounds)
            scoped_commands = [command for command in all_commands
                               if command.get("round", latest_round) == latest_round]
        else:
            scoped_commands = all_commands
    context = {
        "cwd": os.path.abspath(cwd or os.getcwd()),
        "commands": scoped_commands,
        "all_commands": all_commands,
        "summary": summary or "",
    }
    checks = contract.get("checks") if isinstance(contract, dict) else []
    if not isinstance(checks, list) or not checks:
        return {
            "satisfied": None,
            "reason": "contract has no deterministic checks",
            "retry_instruction": "",
            "checks": [],
            "unverifiable": True,
        }

    results = []
    known = 0
    for index, check in enumerate(checks, 1):
        if not isinstance(check, dict):
            continue
        check_type = check.get("type")
        handler = CHECK_HANDLERS.get(check_type)
        if handler is None:
            results.append({
                "index": index,
                "type": check_type,
                "ok": None,
                "detail": f"unsupported check type: {check_type}",
            })
            continue
        known += 1
        try:
            ok, detail = handler(check, context)
        except Exception as exc:
            ok, detail = False, f"check raised {type(exc).__name__}: {exc}"
        results.append({"index": index, "type": check_type, "ok": bool(ok), "detail": detail})

    if known == 0:
        return {
            "satisfied": None,
            "reason": "contract contains only unsupported checks",
            "retry_instruction": "",
            "checks": results,
            "unverifiable": True,
        }

    failed = [result for result in results if result["ok"] is False]
    if failed:
        reason = "; ".join(f"{item['type']}: {item['detail']}" for item in failed[:3])
        instructions = []
        for item in failed[:3]:
            original = checks[item["index"] - 1]
            instructions.append(_retry_instruction(original, item["detail"]))
        return {
            "satisfied": False,
            "reason": reason,
            "retry_instruction": " ".join(instructions),
            "checks": results,
            "unverifiable": any(result["ok"] is None for result in results),
        }

    return {
        "satisfied": True,
        "reason": f"all {known} deterministic check(s) passed",
        "retry_instruction": "",
        "checks": results,
        "unverifiable": any(result["ok"] is None for result in results),
    }
