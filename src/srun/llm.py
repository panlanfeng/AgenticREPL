import json
import os
import re
import time
import types
from openai import OpenAI
from .config import config
from .context import state
from .prompts import PROMPT
from .tools import TOOL_DEFINITIONS, execute_tool
from .mcp import mcp


def _extract_command_from_text(text):
    """Extract command and language from JSON embedded in LLM text output using brace balancing."""
    if not text:
        return None
    for match in re.finditer(r'\{', text):
        depth = 0
        start = match.start()
        i = start
        in_string = False
        escape_next = False
        while i < len(text):
            c = text[i]
            if escape_next:
                escape_next = False
            elif c == '\\' and in_string:
                escape_next = True
            elif c == '"' and not escape_next:
                in_string = not in_string
            elif not in_string:
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        json_str = text[start:i+1]
                        try:
                            obj = json.loads(json_str)
                            if "command" in obj or "code" in obj:
                                cmd = obj.get("command") or obj.get("code")
                                lang = obj.get("language")
                                return {"command": cmd, "language": lang}
                        except json.JSONDecodeError:
                            pass
                        break
            i += 1
    return None


class LLM:
    def __init__(self):
        self.client = None
        if config.has_llm:
            self.client = OpenAI(api_key=config.api_key, base_url=config.api_base)
        self._hit_tokens = 0
        self._miss_tokens = 0
        self._last_output = ""  # captured output from last inline run_command

    def run(self, user_input, error=None, exec_callback=None, ask_user_callback=None):
        if not self.client:
            return "No LLM configured — set SRUN_API_KEY or add api_key to ~/.srun/user_config.json", None

        failure_text = (
            f"The user typed: {user_input}\n"
            + (f"The returned error:\n{error}\n\n" if error else "\n")
            + f"Generate the correct command and execute it via run_command with no explaination."
            #+ f"Do NOT explain the error — just fix and execute."
        )

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
            change_msg = f"[Environment changed to: {state.current_language}"
            if state.current_language != "shell":
                change_msg += " — user entered session"
            else:
                change_msg += " — returned to shell"
            change_msg += "]"
            user_content = f"{change_msg}\n\n{user_content}"
            state._llm_last_known_language = state.current_language
        cwd = os.getcwd()
        if cwd != state._last_known_cwd:
            state._last_known_cwd = cwd
            state._context_stale = True
        if state._context_stale:
            state._context_stale = False
            reminder = state.llm_context() + "\n" + state.workspace_context()
            user_content = f"<system_reminder>\n{reminder}\n</system_reminder>\n\n{user_content}"
        if error and state.session_context():
            user_content += f"\n\n{state.session_context()}"
        elif not error:
            session_ctx = state.session_context()
            if session_ctx:
                user_content += f"\n\n[Recent activity]\n{session_ctx}"

        messages.append({"role": "user", "content": user_content})

        all_commands = []
        reasoning = False
        total_tokens = 0
        MAX_TOKENS = 32000
        while total_tokens < MAX_TOKENS:
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
                tool_call_data = {}
                usage = None
                first_content = True

                for chunk in stream:
                    if chunk.usage:
                        usage = chunk.usage
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if not delta:
                        continue
                    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                        if not reasoning:
                            reasoning = True
                            print("\033[2mReasoning: \033[0m", end="", flush=True)
                        print(delta.reasoning_content, end="", flush=True)
                    if delta.content:
                        if first_content:
                            first_content = False
                            if reasoning:
                                print("\n\033[2mAgent response: \033[0m", end="", flush=True)
                            else:
                                print("\033[2mAgent response: \033[0m", end="", flush=True)
                        print(delta.content, end="", flush=True)
                        content_parts.append(delta.content)
                    if delta.tool_calls:
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

                if content_parts:
                    print(flush=True)

                self._track_usage(usage)
                if usage and hasattr(usage, "total_tokens"):
                    total_tokens += usage.total_tokens

                text = "".join(content_parts).strip()
                tool_calls = []
                for i in sorted(tool_call_data.keys()):
                    tc = tool_call_data[i]
                    if tc["name"] and tc["arguments"]:
                        tool_calls.append(types.SimpleNamespace(
                            id=tc["id"], function=types.SimpleNamespace(name=tc["name"], arguments=tc["arguments"])
                        ))

                if tools and tool_calls:
                    msg_dict = {"role": "assistant", "content": text, "tool_calls": []}
                    for tc in tool_calls:
                        tc_dict = {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        msg_dict["tool_calls"].append(tc_dict)
                    messages.append(msg_dict)
                    for tc in tool_calls:
                        try:
                            args = json.loads(tc.function.arguments)
                        except json.JSONDecodeError:
                            continue
                        if tc.function.name == "run_command":
                            cmd = args.get("command", "")
                            lang = args.get("language", "shell")
                            if cmd:
                                all_commands.append({"command": cmd, "language": lang})
                            if exec_callback and cmd:
                                ok, out, *_ = exec_callback(cmd, lang)
                                if self._last_output:
                                    self._last_output += "\n"
                                self._last_output += out.strip() if out else ""
                                out_lines = out.strip().split("\n") if out else []
                                if len(out_lines) > 20:
                                    out = "\n".join(out_lines[-20:]) + f"\n... ({len(out_lines)} lines)"
                                content = f"Exit: {'0' if ok else '1'}\n{out[:3000]}"
                            else:
                                content = f"queued: {cmd}"
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
                                label = tc.function.name.replace("get_command_help", "reading help").replace("check_command", "checking command").replace("search_files", "searching files").replace("read_file", "reading file").replace("get_env_info", "checking environment").replace("check_repo_info", "checking repo").replace("check_command_versions", "checking versions").replace("ask_user", "requesting confirmation")
                                val = list(args.values())[0] if args else ""
                                print(f"\033[2m  → {label}: {val}\033[0m", flush=True)
                                if tc.function.name.startswith("mcp_"):
                                    result = mcp.call_tool(tc.function.name, args)
                                else:
                                    result = execute_tool(tc.function.name, args)
                                content = result
                            messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
                    state.log_conversation(messages)
                    self._maybe_compact()
                    if not exec_callback:
                        return text if text else None, all_commands if all_commands else None
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
                state.log_conversation(messages)
                self._maybe_compact()
                return summary, all_commands if all_commands else None

            except (json.JSONDecodeError, AttributeError, KeyError) as e:
                state.last_dispatch_error = str(e)
                state.log_conversation(messages)
                return f"LLM response error: {e}", None
            except Exception as e:
                err_msg = str(e).lower()
                if any(kw in err_msg for kw in ("timeout", "connection", "rate limit", "server error", "503", "502", "429")):
                    time.sleep(1.5)
                    continue
                state.last_dispatch_error = str(e)
                state.log_conversation(messages)
                return f"LLM error: {e}", None

        state.log_conversation(messages)
        return f"Token budget ({MAX_TOKENS}) exceeded; task too complex.", None

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
        # Extract memory every 50K tokens of growth since last extraction
        since_last = tokens - state._last_memory_extract_tokens
        if since_last > 50000:
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
