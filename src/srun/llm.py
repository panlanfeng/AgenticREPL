import json
import re
from openai import OpenAI
from .config import config
from .context import state
from .prompts import PROMPT


class LLM:
    def __init__(self):
        self.client = None
        if config.has_llm:
            self.client = OpenAI(api_key=config.api_key, base_url=config.api_base)

    def run(self, user_input, error=None):
        if not self.client:
            return None, None
        if error:
            failure = f"Failed command: {user_input}\nError message: {error}\nFix this command."
        else:
            failure = f"User typed: {user_input}\nGenerate the correct command."
        prompt = PROMPT.format(failure=failure, context=state.llm_context())
        try:
            resp = self.client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_input},
                ],
                temperature=0.0,
                max_tokens=500,
            )
            text = resp.choices[0].message.content.strip()
            return self._parse(text, user_input)
        except Exception as e:
            state.last_dispatch_error = str(e)
            return None, None

    def _parse(self, text, original_input):
        try:
            parsed = json.loads(text)
            code = parsed.get("code")
            lang = parsed.get("language", "shell")
            if code and code != original_input:
                return lang, code
            return None, code
        except json.JSONDecodeError:
            pass
        block = re.search(r"```(?:bash|shell|sh|python|py)?\s*\n?(.+?)```", text, re.DOTALL)
        if block:
            code = block.group(1).strip()
            if code != original_input:
                return "shell", code
        return None, None


llm = LLM()
