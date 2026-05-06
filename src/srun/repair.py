import json
import re
from openai import OpenAI
from .config import config
from .context import state
from .prompts import REPAIR_PROMPT_SHORT


QUICK_FIXES = [
    # Shell aliases that don't survive subprocess
    (r"^ll$", "ls -la"),
    (r"^la$", "ls -a"),
    (r"^l$", "ls -CF"),
    # ls all / ls everything → ls -la
    (r"^ls\s+(all|everything)$", "ls -la"),
    (r"^ls\s+all\s+files?$", "ls -la"),
    # cd ... or cd.. → cd ..
    (r"^cd\.\.$", "cd .."),
    # grep -r <no file>: likely missing .
    (r"^grep\s+-r\s+(['\"]?)(\S+)\1$", r'grep -r "\2" .'),
]


def apply_quick_fix(original, error_msg):
    lower_out = original.strip()
    for pattern, replacement in QUICK_FIXES:
        m = re.match(pattern, lower_out, re.IGNORECASE)
        if m:
            return m.expand(replacement) if hasattr(m, "expand") else replacement
    return None


class Repairer:
    def __init__(self):
        self.client = None
        if config.has_llm:
            self.client = OpenAI(api_key=config.api_key, base_url=config.api_base)

    def _extract_code(self, text):
        block = re.search(r"```(?:bash|shell|sh|python)?\s*\n?(.+?)```", text, re.DOTALL)
        if block:
            return block.group(1).strip()

        match = re.search(r"```(\w+)?\s*\n?(.+?)```", text, re.DOTALL)
        if match:
            return match.group(2).strip()

        try:
            parsed = json.loads(text)
            return parsed.get("code")
        except json.JSONDecodeError:
            pass

        return None

    def fix(self, original_input, error_message, language="shell"):
        quick = apply_quick_fix(original_input, error_message)
        if quick:
            return quick

        if not self.client:
            return None

        context_str = state.llm_context()
        prompt = REPAIR_PROMPT_SHORT.format(
            input=original_input, error=error_message, context=context_str
        )
        try:
            resp = self.client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": original_input},
                ],
                temperature=0.0,
                max_tokens=300,
            )
            content = resp.choices[0].message.content.strip()
            code = self._extract_code(content)
            if code and code != original_input:
                return code
            return None
        except Exception:
            return None


repairer = Repairer()
