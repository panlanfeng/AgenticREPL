"""Pre-execution policy gate for user and model actions.

The policy is intentionally conservative for model-generated actions. Known
catastrophic commands are denied; higher-risk actions require approval. Direct
user input is still checked for catastrophic patterns, but the user's explicit
intent is not downgraded to an approval prompt.
"""

import os
import re
from datetime import datetime, timezone
from fnmatch import fnmatchcase

from .danger import check_danger


HIGH_RISK_PATTERNS = [
    (r"\bsudo\b", "sudo"),
    (r"\brm\s+-(?:[^\s]*r[^\s]*f|[^\s]*f[^\s]*r)\b", "recursive force delete"),
    (r"\b(chmod|chown)\b", "permission change"),
    (r"\b(pip|pip3|npm|pnpm|yarn|brew|apt|apt-get|yum|dnf)\s+(install|uninstall|remove)\b", "package installation"),
    (r"\bgit\s+(push|reset\s+--hard|clean\s+-fd)\b", "git history/worktree change"),
    (r"\b(docker|podman)\s+(rm|rmi|prune|system\s+prune)\b", "container deletion"),
    (r"\bkubectl\s+delete\b", "kubernetes deletion"),
    (r"\b(terraform|tofu)\s+(apply|destroy)\b", "infrastructure change"),
    (r"\b(shutdown|reboot|halt|poweroff)\b", "system shutdown"),
    (r"\b(launchctl|systemctl|crontab)\b", "system service change"),
    (r"\bkill\s+-9\b", "force kill"),
    (r"\b(mv|cp)\s+[^;&|]*\s+/(?:etc|usr|bin|sbin|var|System|Library)\b", "system path modification"),
]
READ_ONLY_TOOLS = {
    "get_context", "inspect_command", "read_file", "grep_search",
    "search_files", "check_background", "stop_background", "todowrite",
    "ask_user", "check_command", "get_env_info", "check_repo_info",
    "check_command_versions", "get_command_help",
}
PYTHON_RISK = re.compile(
    r"\b(os\.system|os\.popen|subprocess\.|shutil\.rmtree|eval\s*\(|exec\s*\()"
)


def _result(decision, reason, action, source):
    return {
        "decision": decision,
        "reason": reason,
        "action": action,
        "source": source,
    }


def grant_matches(grant, action, args=None, now=None):
    """Check whether an event-sourced permission grant covers an action."""
    if not isinstance(grant, dict):
        return False
    expires_at = grant.get("expires_at")
    if expires_at:
        try:
            expires = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            if expires <= (now or datetime.now(timezone.utc)):
                return False
        except (ValueError, TypeError):
            return False
    granted_action = grant.get("action") or "*"
    if not fnmatchcase(action, granted_action):
        return False
    args = args or {}
    needle = grant.get("contains")
    if needle:
        text = args.get("command") or args.get("path") or ""
        if str(needle).lower() not in str(text).lower():
            return False
    prefix = grant.get("path_prefix")
    if prefix:
        path = os.path.abspath(os.path.expanduser(str(args.get("path") or "")))
        root = os.path.abspath(os.path.expanduser(str(prefix)))
        if not (path == root or path.startswith(root + os.sep)):
            return False
    return True


def _path_is_scoped(path, cwd):
    if not isinstance(path, str) or not path.strip():
        return False
    resolved = os.path.abspath(os.path.expanduser(path.strip()))
    roots = [os.path.abspath(cwd or os.getcwd()), os.path.expanduser("~"),
             "/tmp", "/private/tmp"]
    for root in roots:
        root = os.path.abspath(root)
        if resolved == root or resolved.startswith(root + os.sep):
            return True
    return False


def _check_shell(command):
    danger, description = check_danger(command)
    if danger:
        return "deny", f"blocked destructive command: {description}"
    for pattern, description in HIGH_RISK_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return "ask", f"requires approval: {description}"
    return "allow", "safe shell command"


def check_action(action, args=None, cwd=None, source="llm", grants=None):
    """Return allow/ask/deny for one action before it executes."""
    args = args or {}
    for grant in grants or []:
        if grant_matches(grant, action, args):
            return _result("allow", "matching permission grant", action, source)
    if action == "run_command":
        language = args.get("language") or "shell"
        command = args.get("command") or ""
        if language == "shell":
            decision, reason = _check_shell(command)
        elif language == "python" and PYTHON_RISK.search(command):
            decision, reason = "ask", "requires approval: process/file system access"
        else:
            decision, reason = "allow", f"safe {language} code"
        if source == "user" and decision == "ask":
            # The user typed this command explicitly. Keep catastrophic denies,
            # but do not add an approval prompt for ordinary high-risk actions.
            decision, reason = "allow", "explicit user input"
        return _result(decision, reason, action, source)

    if action in ("file_edit", "file_write"):
        path = args.get("path")
        if _path_is_scoped(path, cwd):
            return _result("allow", "path is inside the workspace or /tmp", action, source)
        decision = "allow" if source == "user" else "ask"
        return _result(decision, "path is outside the workspace", action, source)

    if action.startswith("mcp_"):
        decision = "allow" if source == "user" else "ask"
        return _result(decision, "external MCP tool action", action, source)

    if action in READ_ONLY_TOOLS:
        return _result("allow", "read-only or control tool", action, source)

    return _result("allow", "no policy restriction", action, source)
