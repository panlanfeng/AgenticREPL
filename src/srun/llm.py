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
        if error and state.session_context():
            user_content += f"\n\n{state.session_context()}"

        messages.append({"role": "user", "content": user_content})

        for _ in range(8):
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
                            args = json.loads(tc.function.arguments)
                            if tc.function.name == "run_command":
                                cmd = args.get("command", "")
                                if cmd:
                                    commands.append(cmd)
                                messages.append(msg_dict)
                                messages.append({"role": "tool", "tool_call_id": tc.id, "content": f"queued: {cmd}"})
                            else:
                                label = tc.function.name.replace("get_command_help", "reading help").replace("check_command", "checking command").replace("search_files", "searching files").replace("read_file", "reading file").replace("get_env_info", "checking environment")
                                val = list(args.values())[0] if args else ""
                                print(f"\033[2m  → {label}: {val}\033[0m", flush=True)
                                result = execute_tool(tc.function.name, args)
                                messages.append(msg_dict)
                                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                        self._save_conversation(messages)
                        return text if text else None, commands if commands else None
                    msg_dict = {"role": "assistant", "content": text, "tool_calls": []}
                    for tc in tool_calls:
                        tc_dict = {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        msg_dict["tool_calls"].append(tc_dict)
                        args = json.loads(tc.function.arguments)
                        label = tc.function.name.replace("get_command_help", "reading help").replace("check_command", "checking command").replace("search_files", "searching files").replace("read_file", "reading file").replace("get_env_info", "checking environment")
                        val = list(args.values())[0] if args else ""
                        print(f"\033[2m  → {label}: {val}\033[0m", flush=True)
                        result = execute_tool(tc.function.name, args)
                        messages.append(msg_dict)
                        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                    continue

                summary = text if text else None
                commands = None
                if tool_calls:
                    commands = []
                    for tc in tool_calls:
                        if tc.function.name == "run_command":
                            args = json.loads(tc.function.arguments)
                            cmd = args.get("command", "")
                            if cmd:
                                commands.append(cmd)
                self._save_conversation(messages)
                return summary, commands if commands else None

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

    def _parse(self, text, original_input):
        if not text:
            return None, None
        summary = None
        tool_calls = None

        def _extract(d):
            nonlocal summary, tool_calls
            summary = d.get("summary", "")
            tc = d.get("tool_calls")
            if tc and isinstance(tc, list) and len(tc) > 0:
                tool_calls = [t.get("command", "") for t in tc if t.get("command")]
            return tool_calls or summary

        try:
            parsed = json.loads(text)
            if _extract(parsed):
                return summary, tool_calls
        except json.JSONDecodeError:
            pass
        obj = self._extract_json(text)
        if obj:
            if _extract(obj):
                return summary, tool_calls
        return None, None

    @staticmethod
    def _extract_json(text):
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        i = start
        in_string = False
        escape = False
        while i < len(text):
            c = text[i]
            if escape:
                escape = False
                i += 1
                continue
            if c == "\\":
                escape = True
            elif c == '"':
                in_string = not in_string
            elif not in_string:
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i + 1])
                        except json.JSONDecodeError:
                            return None
            i += 1
        return None


llm = LLM()
