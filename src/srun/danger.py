import re

DANGER_PATTERNS = [
    (r"rm\s+-rf\s+/", "rm -rf /"),
    (r"rm\s+-rf\s+~", "rm -rf ~"),
    (r">\s*/dev/sd[a-z]", "overwrite disk device"),
    (r"\bmkfs\.", "format filesystem"),
    (r"\bdd\s+if=.*of=/dev/", "direct write to disk"),
    (r"chmod\s+-R\s+777\s+/", "recursive chmod 777 on /"),
    (r":\(\)\s*\{\s*:\|:&\s*\};:", "fork bomb"),
    (r"wget\s+.*\|\s*sh", "pipe wget to shell"),
    (r"curl\s+.*\|\s*sh", "pipe curl to shell"),
]


def check_danger(code):
    for pattern, desc in DANGER_PATTERNS:
        if re.search(pattern, code):
            return True, desc
    return False, ""
