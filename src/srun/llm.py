import json
import os
import re
import time
import types
import difflib
import uuid
import httpx
from openai import OpenAI
from .config import config
from .context import state
from .prompts import PROMPT
from .tools import TOOL_DEFINITIONS, execute_tool
from .mcp import mcp
from .user_config import get as config_get


def _extract_command_from_text(text):
    """Extract command and language from JSON in LLM text output. Conservative:
    only trusts fenced ```json blocks or a standalone JSON object, so prose
    containing stray braces cannot hijack execution."""
    if not text:
        return None
    candidates = []
    for m in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE):
        candidates.append(m.group(1).strip())
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped)
    for json_str in candidates:
        try:
            obj = json.loads(json_str)
            if isinstance(obj, dict) and ("command" in obj or "code" in obj):
                return {"command": obj.get("command") or obj.get("code"),
                        "language": obj.get("language")}
        except json.JSONDecodeError:
            continue
    return None


_TOOL_TOKEN_THRESHOLD = 20000  # chars ≈ 5000 tokens; beyond this, dump to file
_TOOL_HEAD_CHARS = 3000       # first ~750 tokens shown in context preview
_TOOL_TAIL_CHARS = 5000       # last ~1250 tokens shown in context preview

# P1-1: bounded, recoverable tool-result feedback (Maka-style).
_MAX_OUTPUT_LINES = 2000        # max lines kept before truncation
_MAX_OUTPUT_CHARS = 50 * 1024   # max chars kept before truncation
_STREAM_IDLE_TIMEOUT_SECONDS = 120  # P2-4: retry when the model stream stalls
_OUTPUT_RECOVERY_HINT = (
    "If the command is safe to re-run, redirect its output to a file "
    "(e.g. `cmd > out.txt 2>&1`) then use read_file on that file for the omitted portion. "
    "If re-running could repeat side effects, do not."
)


def truncate_tool_output(text, max_lines=_MAX_OUTPUT_LINES, max_chars=_MAX_OUTPUT_CHARS, direction="tail"):
    """Keep at most max_lines/max_chars of text from one end, with a marker +
    recovery hint. Never discards everything: an oversized single line is sliced
    to a byte-safe window. Returns (bounded_text, truncated_bool)."""
    if not text:
        return "", False
    total = len(text)
    body = text[:-1] if text.endswith("\n") else text
    lines = body.split("\n")
    if len(lines) <= max_lines and total <= max_chars:
        return text, False

    out = []
    used = 0
    hit_chars = False
    indices = range(len(lines)) if direction == "head" else range(len(lines) - 1, -1, -1)
    for i in indices:
        if len(out) >= max_lines:
            break
        size = len(lines[i]) + (1 if out else 0)
        if used + size > max_chars:
            hit_chars = True
            break
        if direction == "tail":
            out.insert(0, lines[i])
        else:
            out.append(lines[i])
        used += size

    if not out:  # the boundary line alone exceeds the budget — keep a slice of it
        line = lines[0] if direction == "head" else lines[-1]
        out = [line[:max_chars]]
        hit_chars = True

    preview = "\n".join(out)
    removed = max(0, total - used) if hit_chars else max(0, len(lines) - len(out))
    unit = "chars" if hit_chars else "lines"
    marker = f"...{removed} {unit} truncated. {_OUTPUT_RECOVERY_HINT}"
    if direction == "tail":
        return marker + "\n\n" + preview, True
    return preview + "\n\n" + marker, True


def _dump_output_to_file(tool_name, content):
    """Persist a full tool result to the session outputs dir; returns absolute path."""
    os.makedirs(state.outputs_dir, exist_ok=True)
    fname = f"tool_{tool_name}_{uuid.uuid4().hex[:8]}.txt"
    fpath = os.path.join(state.outputs_dir, fname)
    with open(fpath, "w") as f:
        f.write(content)
    return fpath


def _truncate_tool_result(tool_name, result):
    """Truncate large tool results. Dump full output to a file,
    keep head + tail preview in context with an absolute path reference
    (P1-2: the model must not have to guess where the full output lives).
    read_file is never truncated."""
    if tool_name == "read_file":
        return result
    if not result or len(result) <= _TOOL_TOKEN_THRESHOLD:
        return result
    fpath = _dump_output_to_file(tool_name, result)
    head = result[:_TOOL_HEAD_CHARS]
    tail = result[-_TOOL_TAIL_CHARS:]
    omitted = len(result) - _TOOL_HEAD_CHARS - _TOOL_TAIL_CHARS
    if omitted > 0:
        preview = head + f"\n... ({omitted} chars omitted) ...\n" + tail
    else:
        preview = result
    return (
        f"[Full output saved to {fpath} — "
        f"use read_file to access. Preview (first {_TOOL_HEAD_CHARS} + last {_TOOL_TAIL_CHARS} chars):]\n\n{preview}"
    )


def _build_run_command_feedback(ok, out, meta):
    """Structured run_command feedback (P1-4): real exit code, separated
    stdout/stderr, timeout/abort flags, and bounded output with recovery hint."""
    meta = meta or {}
    stderr = meta.get("stderr") or ""
    exit_code = meta.get("exit_code")
    if exit_code is None:
        exit_code = 0 if ok else 1
    flags = []
    if meta.get("timed_out"):
        flags.append("timed_out")
    if meta.get("aborted"):
        flags.append("aborted")
    flag_line = (" flags=" + ",".join(flags)) if flags else ""
    preview, truncated = truncate_tool_output(out or "", direction="tail")
    if truncated:
        fpath = _dump_output_to_file("run_command", out or "")
        preview += f"\n[Full output: {fpath} — use read_file to access]"
    parts = [f"Exit: {exit_code}{flag_line}"]
    if out and out.strip():
        parts.append(f"stdout:\n{preview}")
    if stderr and stderr.strip():
        parts.append(f"stderr:\n{stderr[:3000]}")
    return "\n".join(parts)


class LLM:
    def __init__(self):
        self.client = None
        if config.has_llm:
            self.client = OpenAI(api_key=config.api_key, base_url=config.api_base,
                                  timeout=httpx.Timeout(120.0, connect=10.0))
        self._hit_tokens = 0
        self._miss_tokens = 0
        self._last_output = ""  # captured output from last inline run_command
        self._agent_text = ""    # captured agent text for display ordering

    def run(self, user_input, error=None, exec_callback=None, ask_user_callback=None, bg_callback=None):
        if not self.client:
            return "No LLM configured — set DEEPSEEK_API_KEY, OPENAI_API_KEY, or add api_key to ~/.srun/user_config.json", None, None

        failure_text = f"{user_input}\nError: {error}" if error else user_input

        tools = TOOL_DEFINITIONS + mcp.all_tools()

        system_prompt = PROMPT.format()
        messages = state.build_conversation_messages(system_prompt)

        if not state._context_injected:
            state._context_injected = True
            messages.append({
                "role": "user",
                "content": f"[session startup]\n{state.startup_context()}"
            })

        user_content = failure_text
        if state.current_language != state._llm_last_known_language:
            user_content = f"[env: {state.current_language}]\n\n{user_content}"
            state._llm_last_known_language = state.current_language
        cwd = os.getcwd()
        if cwd != state._last_known_cwd:
            state._last_known_cwd = cwd
        if state._context_stale:
            state._context_stale = False
            reminder = state.llm_context() + "\n" + state.workspace_context()
            user_content = f"<system_reminder>\n{reminder}\n</system_reminder>\n\n{user_content}"
        if error and state.session_context():
            user_content += f"\n\n{state.session_context()}"
        elif not error:
            session_ctx = state.session_context()
            if session_ctx:
                user_content += f"\n\n{session_ctx}"

        conv_start = len(messages)  # position before user message — delta includes user + LLM response
        messages.append({"role": "user", "content": user_content})
        prev_log_len = conv_start   # track last logged position for delta logging

        all_commands = []
        reasoning = False
        total_tokens = 0
        retries = 0
        nudges = 0  # bounded "empty final response" retries
        max_steps = int(config_get("max_llm_steps") or 12)  # hard cap on agent-loop steps
        steps = 0
        hit_step_cap = False
        MAX_TOKENS = state._max_context_tokens // 4  # response budget = 1/4 of context window
        while total_tokens < MAX_TOKENS and retries < 5:
            if steps >= max_steps:
                hit_step_cap = True
                break
            steps += 1
            start = time.perf_counter()
            try:
                kwargs = {"model": config.model, "messages": messages,
                          "temperature": config.temperature, "max_tokens": config.max_tokens,
                          "stream": config.stream}
                if config.top_p > 0:
                    kwargs["top_p"] = config.top_p
                if tools:
                    kwargs["tools"] = tools
                    if config.tool_choice:
                        kwargs["tool_choice"] = config.tool_choice
                stream = self.client.chat.completions.create(**kwargs)

                content_parts = []
                reasoning_parts = []
                tool_call_data = {}
                usage = None
                first_content = True
                reasoning_open = False
                last_chunk_at = time.monotonic()

                for chunk in stream:
                    now = time.monotonic()
                    if now - last_chunk_at > _STREAM_IDLE_TIMEOUT_SECONDS:
                        raise TimeoutError(f"stream timeout: idle for {now - last_chunk_at:.0f}s")
                    last_chunk_at = now
                    if chunk.usage:
                        usage = chunk.usage
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if not delta:
                        continue
                    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                        reasoning_parts.append(delta.reasoning_content)
                        if not reasoning_open:
                            reasoning_open = True
                            reasoning = True
                            print("\033[2mReasoning: ", end="", flush=True)
                        print(delta.reasoning_content, end="", flush=True)
                    if delta.content:
                        if reasoning_open:
                            print("\033[0m", flush=True)
                            reasoning_open = False
                        if first_content:
                            first_content = False
                            if reasoning:
                                self._agent_text = "\n\033[2mAgent response: "
                            else:
                                self._agent_text = "\033[2mAgent response: "
                        self._agent_text += delta.content
                        content_parts.append(delta.content)
                    if delta.tool_calls:
                        if reasoning_open:
                            print("\033[0m", flush=True)
                            reasoning_open = False
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

                if content_parts or reasoning:
                    if reasoning_open:
                        print("\033[0m", flush=True)
                        reasoning_open = False
                    print(flush=True)

                self._track_usage(usage)
                if usage and hasattr(usage, "prompt_cache_miss_tokens"):
                    total_tokens += getattr(usage, "prompt_cache_miss_tokens", 0) or 0

                text = "".join(content_parts).strip()
                tool_calls = []
                for i in sorted(tool_call_data.keys()):
                    tc = tool_call_data[i]
                    if tc["name"] and tc["arguments"]:
                        tool_calls.append(types.SimpleNamespace(
                            id=tc["id"], function=types.SimpleNamespace(name=tc["name"], arguments=tc["arguments"])
                        ))

                if tools and tool_calls:
                    reasoning_text = "".join(reasoning_parts)
                    msg_dict = {"role": "assistant", "content": text, "tool_calls": []}
                    if reasoning_text:
                        msg_dict["reasoning_content"] = reasoning_text
                    for tc in tool_calls:
                        tc_dict = {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        msg_dict["tool_calls"].append(tc_dict)
                    messages.append(msg_dict)
                    for tc in tool_calls:
                        try:
                            args = json.loads(tc.function.arguments)
                        except json.JSONDecodeError:
                            messages.append({"role": "tool", "tool_call_id": tc.id,
                                             "content": f"Error: failed to parse tool call arguments as JSON: {tc.function.arguments[:200]}"})
                            continue
                        if tc.function.name == "run_command":
                            cmd = args.get("command", "")
                            lang = args.get("language", "shell")
                            if cmd:
                                all_commands.append({"command": cmd, "language": lang,
                                                     "output": "", "background": bool(args.get("background"))})
                            if cmd and args.get("background") and bg_callback:
                                content = bg_callback("start", command=cmd, language=lang)
                                print(f"\033[1;32m> {cmd} [background]\033[0m", flush=True)
                            elif exec_callback and cmd:
                                result = exec_callback(cmd, lang)
                                if isinstance(result, tuple) and len(result) >= 3 and isinstance(result[2], dict):
                                    ok, out, meta = result[0], result[1], result[2]
                                else:
                                    ok, out = result[0], result[1]
                                    meta = {}
                                if all_commands:
                                    all_commands[-1]["output"] = (out or "").strip()
                                # Print code + output inline (natural interleaved order)
                                for line in cmd.split("\n"):
                                    print(f"\033[1;32m> {line}\033[0m", flush=True)
                                stripped_out = (out or "").strip()
                                if stripped_out:
                                    print(stripped_out)
                                content = _build_run_command_feedback(ok, out, meta)
                            else:
                                content = f"queued: {cmd}"
                            messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
                        elif tc.function.name in ("check_background", "stop_background"):
                            task_id = args.get("task_id", "")
                            if bg_callback:
                                action = "check" if tc.function.name == "check_background" else "stop"
                                content = bg_callback(action, task_id=task_id)
                                print(f"\033[2m  → {action} background task {task_id}\033[0m", flush=True)
                            else:
                                content = f"{tc.function.name} is not available in this context"
                            messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
                        else:
                            if tc.function.name == "ask_user" and ask_user_callback:
                                question = args.get("question", "")
                                details = args.get("details", "")
                                response = ask_user_callback(question, details)
                                content = f"User response: {response}"
                                if response.lower() in ("no", "n", "cancel", "deny", "refuse"):
                                    content += "\nUser denied. Do NOT proceed with this action. Try an alternative approach."
                                elif response.lower() in ("yes", "y", "ok", "proceed", "allow", "approve"):
                                    content += "\nUser approved. You may proceed with the action."
                                print(f"\033[2m  → asking user: {question} → {response}\033[0m", flush=True)
                            else:
                                if tc.function.name == "file_edit":
                                    old = args.get("old_string", "")
                                    new = args.get("new_string", "")
                                    diff = difflib.unified_diff(
                                        old.splitlines(keepends=True),
                                        new.splitlines(keepends=True),
                                        fromfile="a", tofile="b", lineterm=""
                                    )
                                    diff_text = "".join(diff)
                                    if diff_text:
                                        for line in diff_text.split("\n"):
                                            if line.startswith("---") or line.startswith("+++"):
                                                print(f"\033[2m  {line}\033[0m")
                                            elif line.startswith("@@"):
                                                print(f"\033[1;36m  {line}\033[0m")
                                            elif line.startswith("-"):
                                                print(f"\033[1;31m  {line}\033[0m")
                                            elif line.startswith("+"):
                                                print(f"\033[1;32m  {line}\033[0m")
                                            else:
                                                print(f"\033[2m  {line}\033[0m")
                                if tc.function.name.startswith("mcp_"):
                                    result = mcp.call_tool(tc.function.name, args)
                                else:
                                    result = execute_tool(tc.function.name, args)
                                content = _truncate_tool_result(tc.function.name, result)
                            messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
                    state.log_conversation(messages[prev_log_len:])
                    prev_log_len = len(messages)
                    self._maybe_compact()
                    # Without an inline exec_callback (e.g. repair mode or direct
                    # llm.run callers), keep looping through context-gathering
                    # tool calls (get_context, read_file, grep_search, ...) and
                    # only hand back once the model produced run_command calls
                    # or a final text answer. Otherwise a model that explores
                    # first would bail with (None, None) and look like a failure.
                    if not exec_callback and all_commands:
                        return text if text else None, all_commands, messages[conv_start:]
                    continue

                summary = text if text else None
                commands = None
                if tool_calls:
                    commands = []
                    for tc in tool_calls:
                        if tc.function.name == "run_command":
                            try:
                                args = json.loads(tc.function.arguments)
                            except json.JSONDecodeError:
                                continue
                            cmd = args.get("command", "")
                            lang = args.get("language", "shell")
                            if cmd:
                                commands.append({"command": cmd, "language": lang})
                if not all_commands and summary:
                    extracted = _extract_command_from_text(summary)
                    if extracted:
                        lang = extracted.get("language") or state.current_language
                        all_commands = [{"command": extracted["command"], "language": lang}]
                # A reasoning-only terminal message (no text, no tool calls) after
                # exploration is not a usable answer — nudge once so a turn never
                # ends silently with (None, None).
                if summary is None and not all_commands and nudges < 2:
                    nudges += 1
                    messages.append({"role": "user",
                                     "content": "[system] Your previous response contained no content. "
                                                "Provide your final answer or the command to run now."})
                    continue
                # Append final assistant text to messages so conversation is complete
                if summary:
                    messages.append({"role": "assistant", "content": summary})
                state.log_conversation(messages[prev_log_len:])
                prev_log_len = len(messages)
                self._maybe_compact()
                return summary, all_commands if all_commands else None, messages[conv_start:]

            except (json.JSONDecodeError, AttributeError, KeyError) as e:
                state.last_dispatch_error = str(e)
                state.log_conversation(messages[prev_log_len:])
                prev_log_len = len(messages)
                return f"LLM response error: {e}", None, None
            except Exception as e:
                err_msg = str(e).lower()
                if any(kw in err_msg for kw in ("timeout", "connection", "rate limit", "server error", "503", "502", "429")):
                    retries += 1
                    if retries >= 5:
                        return f"LLM error: too many transient failures — {e}", None, None
                    time.sleep(1.5)
                    continue
                if any(kw in err_msg for kw in ("401", "403", "authentication", "invalid api key", "invalid_request_error", "access denied")):
                    return ("Authentication failed — check your API key. Set DEEPSEEK_API_KEY, OPENAI_API_KEY,"
                            " or add api_key to ~/.srun/user_config.json. Type /configure for help."), None, None
                state.last_dispatch_error = str(e)
                state.log_conversation(messages[prev_log_len:])
                prev_log_len = len(messages)
                return f"LLM error: {e}", None, None

        state.log_conversation(messages[prev_log_len:])
        prev_log_len = len(messages)
        if hit_step_cap:
            return (f"Reached max steps ({max_steps}) for this turn; task too complex. "
                    "Break it into smaller steps or continue with a follow-up prompt.",
                    all_commands if all_commands else None, messages[conv_start:])
        return f"Token budget ({MAX_TOKENS}) exceeded; task too complex.", None, None

    def _track_usage(self, usage):
        if usage:
            self._hit_tokens += getattr(usage, "prompt_cache_hit_tokens", 0) or 0
            self._miss_tokens += getattr(usage, "prompt_cache_miss_tokens", 0) or 0

    def _maybe_compact(self):
        """Trigger context compaction if conversation exceeds 80% of token budget."""
        tokens = state.context_tokens()
        limit = int(state._max_context_tokens * 0.8)
        remaining = state._max_context_tokens - tokens
        if tokens > limit or remaining < 30000:
            state.compact_context(llm_module=self)
        # Extract memory every 50K tokens of growth since last extraction,
        # or every ~20 turns, whichever comes first.
        since_last = tokens - state._last_memory_extract_tokens
        since_last_turn = state._turn - state._last_memory_extract_turn
        if since_last > 50000 or since_last_turn >= 20:
            state.extract_memory(llm_module=self)

    def reset_cache(self):
        self._hit_tokens = 0
        self._miss_tokens = 0

    @property
    def cache_hit_rate(self):
        total = self._hit_tokens + self._miss_tokens
        return self._hit_tokens / total if total > 0 else 0

    @property
    def cache_stats(self):
        total = self._hit_tokens + self._miss_tokens
        return {
            "hit_tokens": self._hit_tokens,
            "miss_tokens": self._miss_tokens,
            "total_tokens": total,
            "rate": self.cache_hit_rate,
        }


llm = LLM()
