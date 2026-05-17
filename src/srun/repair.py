import re
from .llm import llm


def _ask_user(question, details=""):
    """Ask user for permission in repair mode."""
    print()
    if details:
        print(f"\033[1;33m?\033[0m \033[1m{question}\033[0m")
        print(f"  \033[2m{details}\033[0m")
    else:
        print(f"\033[1;33m?\033[0m \033[1m{question}\033[0m")
    print("  [y/N] ", end="", flush=True)
    try:
        answer = input().strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "n"
    return "yes" if answer in ("y", "yes") else "no"


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
        summary, tool_calls, _ = llm.run(original_input, error=error_message, ask_user_callback=_ask_user)
        if tool_calls and len(tool_calls) > 0:
            tc = tool_calls[0]
            if isinstance(tc, dict):
                return tc["command"], summary
            return tc, summary
        return None, summary


repairer = Repairer()
