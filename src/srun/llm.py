import json
import os
import re
import time
import types
from datetime import datetime
from openai import OpenAI
from .config import config
from .context import state, CONVERSATIONS_DIR
from .prompts import PROMPT
from .tools import TOOL_DEFINITIONS, execute_tool


def _extract_command_from_text(text):
    if not text:
        return None
    for cmd_key in ("command", "code"):
        pattern = r'\{\s*"' + cmd_key + r'"\s*:\s*"((?:[^"\\]|\\.)*)"'
        m = re.search(pattern, text)
        if m:
            raw = m.group(1)
            try:
                cmd = json.loads('"' + raw + '"')
            except (json.JSONDecodeError, UnicodeDecodeError):
                cmd = raw
            lang = None
            lang_m = re.search(r'"language"\s*:\s*"(\w+)"', text)
            if lang_m:
                lang = lang_m.group(1)
            return {"command": cmd, "language": lang}
    return None


class LLM:
    def __init__(self):
        self.client = None
        if config.has_llm:
            self.client = OpenAI(api_key=config.api_key, base_url=config.api_base)
        self._hit_tokens = 0
        self._miss_tokens = 0

    def run(self, user_input, error=None):
        if not self.client:
            return None, None

        if error:
            failure_text = (
                f"The user typed this command: {user_input}\n"
                f"It failed with this error:\n{error}\n\n"
                f"Fix the command and output the correct version as JSON."
            )
        else:
            failure_text = (
                f"The user typed: {user_input}\n\n"
                f"Generate the correct executable command as JSON."
            )

        tools = TOOL_DEFINITIONS

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
            user_content = f"[Environment changed to: {state.current_language}]\n\n{user_content}"
            state._llm_last_known_language = state.current_language
        cwd = os.getcwd()
        if cwd != state._last_known_cwd:
            state._last_known_cwd = cwd
            files = state.workspace_context()
            user_content = f"[CWD: {cwd}]\n{files}\n\n{user_content}"
        if error and state.session_context():
            user_content += f"\n\n{state.session_context()}"

        messages.append({"role": "user", "content": user_content})

        for _ in range(10):
            start = time.perf_counter()
            try:
                kwargs = {"model": config.model, "messages": messages, "temperature": 0.0, "max_tokens": 500, "stream": True}
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"
                stream = self.client.chat.completions.create(**kwargs)

                content_parts = []
                tool_call_data = {}
                usage = None

                for chunk in stream:
                    if chunk.usage:
                        usage = chunk.usage
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if not delta:
                        continue
                    if delta.content:
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

                text = "".join(content_parts).strip()
                tool_calls = []
                for i in sorted(tool_call_data.keys()):
                    tc = tool_call_data[i]
                    if tc["name"] and tc["arguments"]:
                        tool_calls.append(types.SimpleNamespace(
                            id=tc["id"], function=types.SimpleNamespace(name=tc["name"], arguments=tc["arguments"])
                        ))

                if tools and tool_calls:
                    has_run = any(tc.function.name == "run_command" for tc in tool_calls)
                    if has_run:
                        commands = []
                        msg_content = text
                        msg_dict = {"role": "assistant", "content": msg_content, "tool_calls": []}
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
                                    commands.append({"command": cmd, "language": lang})
                                messages.append({"role": "tool", "tool_call_id": tc.id, "content": f"queued: {cmd}"})
                            else:
                                label = tc.function.name.replace("get_command_help", "reading help").replace("check_command", "checking command").replace("search_files", "searching files").replace("read_file", "reading file").replace("get_env_info", "checking environment")
                                val = list(args.values())[0] if args else ""
                                print(f"\033[2m  → {label}: {val}\033[0m", flush=True)
                                result = execute_tool(tc.function.name, args)
                                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                        self._save_conversation(messages)
                        return text if text else None, commands if commands else None
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
                        label = tc.function.name.replace("get_command_help", "reading help").replace("check_command", "checking command").replace("search_files", "searching files").replace("read_file", "reading file").replace("get_env_info", "checking environment")
                        val = list(args.values())[0] if args else ""
                        print(f"\033[2m  → {label}: {val}\033[0m", flush=True)
                        result = execute_tool(tc.function.name, args)
                        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
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
                if not commands and summary:
                    extracted = _extract_command_from_text(summary)
                    if extracted:
                        lang = extracted.get("language") or state.current_language
                        commands = [{"command": extracted["command"], "language": lang}]
                self._save_conversation(messages)
                return summary, commands if commands else None

            except (json.JSONDecodeError, AttributeError, KeyError) as e:
                state.last_dispatch_error = str(e)
                self._save_conversation(messages)
                return None, None
            except Exception as e:
                state.last_dispatch_error = str(e)
                self._save_conversation(messages)
                return None, None

        self._save_conversation(messages)
        return None, None

    def _track_usage(self, usage):
        if usage:
            self._hit_tokens += getattr(usage, "prompt_cache_hit_tokens", 0) or 0
            self._miss_tokens += getattr(usage, "prompt_cache_miss_tokens", 0) or 0

    def _save_conversation(self, messages):
        os.makedirs(CONVERSATIONS_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(CONVERSATIONS_DIR, f"conv_{ts}.json")
        clean = []
        for m in messages:
            if isinstance(m, dict):
                clean.append({k: v for k, v in m.items() if k != "tool_call_id"})
        with open(path, "w") as f:
            json.dump(clean, f, indent=2, default=str)

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
