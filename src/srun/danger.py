import re

DANGER_PATTERNS = [
    (r"rm\s+-(?:rf?|fr)\s+(?:\/|~)", "rm -rf / or ~"),
    (r"rm\s+--(?:recursive|force)\s+(?:\/|~)", "recursive rm on / or ~"),
    (r">\s*/dev/sd[a-z]", "overwrite disk device"),
    (r"\bmkfs\.", "format filesystem"),
    (r"\bdd\s+if=.*of=/dev/", "direct write to disk"),
    (r"chmod\s+-(?:R|--recursive)\s+(?:777|a\+rwx)\s+(?:\/|~)", "recursive world-writable chmod"),
    (r":\(\)\s*\{\s*:\|:&\s*\};:", "fork bomb"),
    (r":\(\)\s*\{\s*:\|:&\s*\s*\};:", "fork bomb"),
    (r"\bwhile\s*:\s*;\s*do\s*\S+\s*&\s*;\s*done", "fork bomb loop"),
    (r"wget\s+.*\|\s*(?:sh|bash)", "pipe wget to shell"),
    (r"curl\s+.*\|\s*(?:sh|bash)", "pipe curl to shell"),
]


def check_danger(code):
    for pattern, desc in DANGER_PATTERNS:
        if re.search(pattern, code):
            return True, desc
    return False, ""
