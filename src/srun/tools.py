"""Agent tools — callable by LLM to gather context, read files, check commands."""

import os
import re
import json
import subprocess
import shutil


def get_command_help(command):
    parts = command.strip().split()
    if not parts:
        return "No help found for ''"
    cmd = parts[0]
    outs = []
    for method in [f"{cmd} --help 2>&1", f"man {cmd} 2>&1 | col -b"]:
        try:
            r = subprocess.run(
                method,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10,
                env=os.environ.copy(),
            )
            out = (r.stdout + r.stderr).strip()
            if out and len(out) > 20:
                lines = out.split("\n")
                if len(lines) > 40:
                    out = "\n".join(lines[:40])
                if len(out) > 3000:
                    out = out[:3000]
                outs.append(out)
                if len(outs) >= 1:
                    break
        except Exception:
            pass
    return outs[0] if outs else f"No help found for '{cmd}'"


def search_files(pattern):
    cwd = os.getcwd()
    matches = []
    for root, dirs, files in os.walk(cwd):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in files + dirs:
            if re.search(re.escape(pattern).replace(r"\*", ".*"), name, re.IGNORECASE):
                rel = os.path.relpath(os.path.join(root, name), cwd)
                size = os.path.getsize(os.path.join(root, name)) if os.path.isfile(os.path.join(root, name)) else None
                entry = rel
                if size is not None:
                    if size < 1024:
                        entry += f" ({size}B)"
                    elif size < 1024 * 1024:
                        entry += f" ({size/1024:.0f}KB)"
                    else:
                        entry += f" ({size/1024/1024:.1f}MB)"
                matches.append(entry)
    if not matches:
        return f"No files matching '{pattern}'"
    return "\n".join(matches[:20])


def read_file(path, lines=None):
    resolved = os.path.expanduser(path)
    if not os.path.isabs(resolved):
        resolved = os.path.join(os.getcwd(), resolved)
    if not os.path.isfile(resolved):
        return f"File not found: {path}"
    try:
        size = os.path.getsize(resolved)
        with open(resolved, encoding="utf-8", errors="replace") as f:
            if lines:
                content = "".join(f.readline() for _ in range(lines))
            elif size > 100 * 1024:
                content = f.read(5000) + f"\n... (truncated, {size}B total)"
            else:
                content = f.read(10000)
                if len(content) >= 10000:
                    content += f"\n... (truncated)"
        content = _redact_secrets(resolved, content)
        return f"--- {resolved} ({size}B) ---\n{content}"
    except Exception as e:
        return f"Error reading {path}: {e}"


def _redact_secrets(path, content):
    """Strip API keys and credentials from files before sending to LLM."""
    import fnmatch, re
    resolved = os.path.realpath(path)
    sensitive = False
    for pattern in ["*.pem", "*id_rsa*", "*.key", ".env*", "*credentials*", "*secret*", "*.token", "*.pem"]:
        if fnmatch.fnmatch(os.path.basename(resolved), pattern):
            sensitive = True
            break
    for deny in ["/.aws/", "/.ssh/", "/.srun/user_config.json"]:
        if deny in resolved:
            sensitive = True
            break
    if sensitive:
        content = re.sub(r'(?i)(api[_-]?key|secret|password|token|credential)s?\s*[:=]\s*["\']?[^"\'}\s,]+["\']?', r'\1=***', content)
        content = re.sub(r'sk-[a-zA-Z0-9]{20,}', 'sk-***', content)
    return content


def _sys_platform():
    import platform
    name = platform.system().lower()
    if name == "darwin":
        try:
            ver = platform.mac_ver()[0]
            return f"macOS {ver}"
        except Exception:
            return "macOS"
    return name


def _sys_python_version():
    import sys
    return sys.version.split()[0]


GNU_ALTERNATIVES = {
    "grep": "ggrep",
    "sed": "gsed",
    "awk": "gawk",
    "find": "gfind",
    "ls": "gls",
    "make": "gmake",
    "tar": "gtar",
}


def check_command(command):
    cmd = command.strip().split()[0]
    path = shutil.which(cmd)
    if not path:
        return f"Command '{cmd}' not found in PATH"
    info = [f"Command: {cmd}", f"Path: {path}"]
    try:
        r = subprocess.run(f"file {path}", shell=True, capture_output=True, text=True, timeout=5)
        line = r.stdout.strip().split("\n")[0]
        info.append(f"Type: {line}")
    except Exception:
        pass
    for flag in ["--version", "-V", "version"]:
        try:
            r = subprocess.run(f"{path} {flag} 2>&1", shell=True, capture_output=True, text=True, timeout=5)
            out = (r.stdout + r.stderr).strip()
            if out and len(out) < 500:
                out = out.split("\n")[0]
                info.append(f"Version: {out}")
                break
        except Exception:
            continue
    is_bsd = False
    combined = " ".join(info).lower()
    if "bsd" in combined and cmd in ("grep", "sed", "awk", "ls", "sort", "find"):
        is_bsd = True
        info.append("Note: macOS BSD version (differs from GNU). Flags like --color=never, --no-color are NOT available — strip the flag instead.")
    elif "gnu" in combined:
        info.append("Note: GNU version detected")
    if is_bsd and cmd in GNU_ALTERNATIVES:
        gnu_cmd = GNU_ALTERNATIVES[cmd]
        gnu_path = shutil.which(gnu_cmd)
        if gnu_path:
            info.append(f"GNU alternative available: {gnu_cmd} at {gnu_path} (use '{gnu_cmd}' instead of '{cmd}' for GNU-compatible flags)")
    return "\n".join(info)


def get_env_info():
    info = [
        f"OS: {_sys_platform()}",
        f"Arch: {os.uname().machine if hasattr(os, 'uname') else 'unknown'}",
        f"Shell: {os.environ.get('SHELL', 'unknown')}",
        f"Python: {_sys_python_version()}",
    ]
    path = os.environ.get("PATH", "")
    if path:
        paths = path.split(":")
        info.append(f"PATH: {':'.join(paths[:5])}" + ("..." if len(paths) > 5 else ""))
    for cmd in ["grep", "sed", "awk", "ls", "find", "python3", "python"]:
        p = shutil.which(cmd)
        if p:
            info.append(f"  {cmd}: {p}")
    return "\n".join(info)


def check_repo_info():
    """Check current git repository information: branch, remote, status, recent commits."""
    cwd = os.getcwd()
    lines = [f"Repository: {cwd}"]
    try:
        r = subprocess.run("git rev-parse --git-dir 2>/dev/null", shell=True, capture_output=True, text=True, timeout=3, cwd=cwd)
        if r.returncode != 0:
            return "Not a git repository"
        branch = subprocess.run("git rev-parse --abbrev-ref HEAD 2>/dev/null", shell=True, capture_output=True, text=True, timeout=3, cwd=cwd).stdout.strip()
        lines.append(f"Branch: {branch}")
        sha = subprocess.run("git rev-parse --short HEAD 2>/dev/null", shell=True, capture_output=True, text=True, timeout=3, cwd=cwd).stdout.strip()
        if sha:
            lines.append(f"HEAD: {sha}")
        remote = subprocess.run("git remote get-url origin 2>/dev/null", shell=True, capture_output=True, text=True, timeout=3, cwd=cwd).stdout.strip()
        if remote:
            lines.append(f"Remote: {remote}")
        status = subprocess.run("git status --porcelain 2>/dev/null", shell=True, capture_output=True, text=True, timeout=3, cwd=cwd).stdout.strip()
        if status:
            changed = len(status.split("\n"))
            lines.append(f"Status: {changed} file(s) modified")
        else:
            lines.append("Status: clean")
        log = subprocess.run("git log --oneline -5 2>/dev/null", shell=True, capture_output=True, text=True, timeout=3, cwd=cwd).stdout.strip()
        if log:
            lines.append(f"Recent commits:\n{log}")
    except Exception as e:
        return f"Error checking repo: {e}"
    return "\n".join(lines)


def check_command_versions(command):
    """Find all installed versions of a command in PATH (e.g., python, R, node)."""
    import glob as _glob
    cmd = command.strip().split()[0]
    versions = []
    seen = set()
    paths = os.environ.get("PATH", "").split(":")
    for p in paths:
        for pattern in [cmd, f"{cmd}3", f"{cmd}3.*"]:
            for match in _glob.glob(os.path.join(p, pattern)):
                name = os.path.basename(match)
                if name.startswith(cmd) and match not in seen:
                    seen.add(match)
                    try:
                        r = subprocess.run([match, "--version"], capture_output=True, text=True, timeout=5)
                        ver = r.stdout.strip().split("\n")[0] if r.returncode == 0 else f"exit {r.returncode}"
                        versions.append(f"{name}: {ver} ({match})")
                    except Exception:
                        versions.append(f"{name}: {match}")
    if not versions:
        default = shutil.which(cmd)
        if default:
            try:
                r = subprocess.run([default, "--version"], capture_output=True, text=True, timeout=5)
                ver = r.stdout.strip().split("\n")[0] if r.returncode == 0 else ""
                versions.append(f"{cmd}: {ver} ({default})")
            except Exception:
                versions.append(f"{cmd}: {default}")
        else:
            return f"No versions of '{cmd}' found in PATH"
    return "\n".join(versions[:10])


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_command_help",
            "description": "Get help/man page for a shell command. Use to see available flags and syntax.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command name, e.g. 'grep', 'sed'"}
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for files matching a pattern in the current project directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "File name or pattern to search for, e.g. '*.csv', 'test*.py'"}
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a file. Use to understand file structure, data format, or code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read"},
                    "lines": {"type": "integer", "description": "Number of lines to read (default: all up to 10000 chars)"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_command",
            "description": "Check which version of a command is installed (GNU vs BSD, version, path). Also checks for GNU alternatives (ggrep, gsed, gawk) on macOS. Use when you need to know if flags are compatible or if a GNU version is available.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command to check, e.g. 'grep', 'sed', 'python'"}
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_env_info",
            "description": "Get system environment info: OS, PATH, installed command versions, Python version.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_repo_info",
            "description": "Check current git repository: branch, remote URL, HEAD SHA, dirty/clean status, and recent commits. Use to understand the project context before generating commands.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_command_versions",
            "description": "Find all installed versions of a command (e.g., 'python', 'R', 'node') in PATH. Use to check which versions are available before generating version-specific code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command to check versions for, e.g. 'python', 'R', 'node'"}
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a command in a specific REPL environment. Set the language field to indicate the target: 'shell', 'python', or 'r'. The command will be immediately executed in that environment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command or code to execute"},
                    "language": {"type": "string", "enum": ["shell", "python", "r"], "description": "Target execution language: 'shell' for shell commands, 'python' for Python code, 'r' for R code. Default to the current environment language if unsure."},
                },
                "required": ["command", "language"],
            },
        },
    },
]

def _run_command(command):
    return f"Command queued: {command}"


TOOL_HANDLERS = {
    "get_command_help": get_command_help,
    "search_files": search_files,
    "read_file": read_file,
    "check_command": check_command,
    "get_env_info": get_env_info,
    "run_command": _run_command,
    "check_repo_info": check_repo_info,
    "check_command_versions": check_command_versions,
}


def execute_tool(name, arguments):
    handler = TOOL_HANDLERS.get(name)
    if not handler:
        return f"Unknown tool: {name}"
    try:
        return handler(**arguments)
    except Exception as e:
        return f"Tool error: {e}"
