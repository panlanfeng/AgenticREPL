import json
import os
import re
import time
from datetime import datetime
from openai import OpenAI
from .config import config
from .context import state
from .prompts import PROMPT
from .tools import TOOL_DEFINITIONS, execute_tool

REPAIR_TOOLS = [t for t in TOOL_DEFINITIONS if t["function"]["name"] in ("get_command_help", "check_command")]

HISTORY_DIR = os.path.join(os.path.expanduser("~"), ".srun", "conversations")


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
            tools = None
        else:
            failure_text = (
                f"The user typed: {user_input}\n\n"
                f"Generate the correct executable command as JSON. "
                f"Use available tools to check file contents, command versions, or environment if needed."
            )
            tools = TOOL_DEFINITIONS

        system_prompt = PROMPT.format(context=state.llm_context())
        messages = state.build_conversation_messages(system_prompt)
        messages.append({"role": "user", "content": failure_text})

        for _ in range(8):
            start = time.perf_counter()
            try:
                kwargs = {"model": config.model, "messages": messages, "temperature": 0.0, "max_tokens": 500}
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"
                resp = self.client.chat.completions.create(**kwargs)

                self._track_usage(resp.usage)

                choice = resp.choices[0]
                msg = choice.message
                if tools and msg.tool_calls:
                    messages.append(msg)
                    for tc in msg.tool_calls:
                        name = tc.function.name
                        args = json.loads(tc.function.arguments)
                        result = execute_tool(name, args)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        })
                    continue

                text = choice.message.content.strip() if choice.message.content else ""
                lang, code = self._parse(text, user_input)
                self._save_conversation(messages)
                return lang, code

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
        os.makedirs(HISTORY_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(HISTORY_DIR, f"conv_{ts}.json")
        clean = []
        for m in messages:
            if isinstance(m, dict):
                clean.append({k: v for k, v in m.items() if k != "tool_call_id"})
        with open(path, "w") as f:
            json.dump(clean, f, indent=2, default=str)

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
        # 1) Try whole text as JSON
        try:
            parsed = json.loads(text)
            code = parsed.get("code")
            lang = parsed.get("language", "shell")
            if code and code != original_input:
                return lang, code
        except json.JSONDecodeError:
            pass
        # 2) Extract JSON object from mixed text
        obj = self._extract_json(text)
        if obj:
            code = obj.get("code")
            lang = obj.get("language", "shell")
            if code and code != original_input:
                return lang, code
        # 3) Extract code from markdown blocks
        block = re.search(r"```(?:json|bash|shell|sh|python|py)?\s*\n?(.+?)```", text, re.DOTALL)
        if block:
            code = block.group(1).strip()
            try:
                inner = json.loads(code)
                code = inner.get("code", code)
                lang = inner.get("language", "shell")
            except json.JSONDecodeError:
                lang = "shell"
            if code and code != original_input:
                return lang, code
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
