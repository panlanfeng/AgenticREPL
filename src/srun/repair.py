import re
from .llm import llm

QUICK_FIXES = [
    (r"^ll$", "ls -la"),
    (r"^la$", "ls -a"),
    (r"^l$", "ls -CF"),
    (r"^ls\s+(all|everything)$", "ls -la"),
    (r"^ls\s+all\s+files?$", "ls -la"),
    (r"^cd\.\.$", "cd .."),
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
    def fix(self, original_input, error_message, language="shell"):
        quick = apply_quick_fix(original_input, error_message)
        if quick:
            return quick, None
        summary, tool_calls = llm.run(original_input, error=error_message)
        if tool_calls and len(tool_calls) > 0:
            tc = tool_calls[0]
            if isinstance(tc, dict):
                return tc["command"], summary
            return tc, summary
        return None, summary


repairer = Repairer()
