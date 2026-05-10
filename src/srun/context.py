import os
import json
import time
import uuid
import platform
import shutil
import subprocess

BASE_DIR = os.path.join(os.path.expanduser("~"), ".srun")


def _gen_session_id():
    return time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]


def _session_dir():
    sid = os.environ.get("SRUN_SESSION_ID", "")
    if sid:
        return os.path.join(BASE_DIR, "sessions", sid)
    return os.path.join(BASE_DIR, "sessions", _gen_session_id())


SESSION_DIR = _session_dir()
SESSION_ID = os.path.basename(SESSION_DIR)
STATE_FILE = os.path.join(SESSION_DIR, "state.json")
CONVERSATIONS_DIR = os.path.join(SESSION_DIR, "conversations")  # deprecated — use full_history.jsonl
FULL_HISTORY_FILE = os.path.join(SESSION_DIR, "full_history.jsonl")
OUTPUTS_DIR = os.path.join(SESSION_DIR, "outputs")


def _detect_python_versions():
    """Find all Python versions available in PATH."""
    versions = []
    seen = set()
    for name in ["python", "python3", "python3.9", "python3.10", "python3.11", "python3.12", "python3.13"]:
        p = shutil.which(name)
        if p and p not in seen:
            seen.add(p)
            try:
                r = subprocess.run([p, "--version"], capture_output=True, text=True, timeout=5)
                if r.returncode == 0:
                    versions.append(f"{name}: {r.stdout.strip()}")
            except Exception:
                versions.append(f"{name}: {p}")
    return versions


def _detect_gnu_alternatives():
    """Check for GNU alternatives on macOS/BSD."""
    alts = {}
    for cmd, gnu in [("grep", "ggrep"), ("sed", "gsed"), ("awk", "gawk"),
                     ("find", "gfind"), ("ls", "gls"), ("make", "gmake"),
                     ("tar", "gtar")]:
        gnu_path = shutil.which(gnu)
        if gnu_path:
            try:
                r = subprocess.run([gnu_path, "--version"], capture_output=True, text=True, timeout=5)
                ver = r.stdout.strip().split("\n")[0] if r.returncode == 0 else gnu_path
                alts[cmd] = f"{gnu}: {ver}"
            except Exception:
                alts[cmd] = f"{gnu}: {gnu_path}"
    return alts


def get_system_info():
    import sys as _sys
    from datetime import datetime as _datetime
    os_name = platform.system().lower()
    info = {
        "os": os_name,
        "arch": platform.machine(),
        "cpu": os.cpu_count(),
        "shell": os.environ.get("SHELL", "").split("/")[-1],
        "cwd": os.getcwd(),
        "platform": os_name,
        "python_version": _sys.version.split()[0],
        "now": _datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z"),
        "r_available": shutil.which("R") is not None,
    }
    if os_name == "darwin":
        try:
            ver = platform.mac_ver()[0]
            info["platform"] = f"macOS {ver}"
        except Exception:
            info["platform"] = "macOS"
    if os_name == "darwin":
        gnu_alts = _detect_gnu_alternatives()
        if gnu_alts:
            info["gnu_alternatives"] = gnu_alts
            info["tools_note"] = "macOS uses BSD grep/sed/awk by default. GNU alternatives available — use the GNU command names listed below for GNU-compatible flags."
        else:
            info["tools_note"] = "macOS uses BSD grep/sed/awk, not GNU. Install coreutils for GNU versions."
    elif os_name == "linux":
        info["platform"] = "Linux"
        info["tools_note"] = "Linux uses GNU grep/sed/awk"
    python_versions = _detect_python_versions()
    if python_versions:
        info["python_versions"] = python_versions
    venv = os.environ.get("VIRTUAL_ENV", "") or os.environ.get("CONDA_DEFAULT_ENV", "")
    if venv:
        info["virtualenv"] = venv
    r_path = shutil.which("R")
    if r_path:
        try:
            r = subprocess.run([r_path, "--version"], capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                info["r_version"] = r.stdout.split("\n")[0].strip()
        except Exception:
            pass
    return info


def get_file_meta():
    cwd = os.getcwd()
    files = []
    try:
        for f in sorted(os.listdir(cwd)):
            path = os.path.join(cwd, f)
            if os.path.isfile(path) and not f.startswith("."):
                size = os.path.getsize(path)
                if size < 1024:
                    size_str = f"{size}B"
                elif size < 1024 * 1024:
                    size_str = f"{size / 1024:.0f}KB"
                else:
                    size_str = f"{size / 1024 / 1024:.1f}MB"
                files.append({"name": f, "size": size_str})
    except Exception:
        pass
    return files


def _get_git_info(cwd):
    try:
        r = subprocess.run(
            "git rev-parse --abbrev-ref HEAD 2>/dev/null",
            shell=True, capture_output=True, text=True, timeout=3, cwd=cwd,
        )
        branch = r.stdout.strip()
        if branch:
            info = f"branch={branch}"
            r2 = subprocess.run(
                "git rev-parse --short HEAD 2>/dev/null",
                shell=True, capture_output=True, text=True, timeout=3, cwd=cwd,
            )
            sha = r2.stdout.strip()
            if sha:
                info += f" sha={sha}"
            return info
    except Exception:
        pass
    return ""


class SessionState:
    def __init__(self):
        self.vars = {}
        self.active_df = None
        self.last_lang = "shell"
        self._current_language = "shell"
        self.last_dispatch_error = None
        self.session_log = []
        self._turn = 0
        self._conversation = []
        self._context_injected = False
        self._llm_last_known_language = "shell"
        self._last_known_cwd = ""
        self._context_stale = True
        os.makedirs(BASE_DIR, exist_ok=True)
        os.makedirs(SESSION_DIR, exist_ok=True)
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        self.full_history_path = FULL_HISTORY_FILE
        self.outputs_dir = OUTPUTS_DIR
        self.state_path = STATE_FILE

    def reset_session(self):
        self._conversation = []
        self.session_log = []
        self._turn = 0
        self.last_dispatch_error = None
        self._context_injected = False
        self._llm_last_known_language = self._current_language

    def build_conversation_messages(self, system_prompt):
        messages = [{"role": "system", "content": system_prompt}]
        for entry in self._conversation:
            messages.append(entry)
        return messages

    def add_conversation_turn(self, user_msg, assistant_code, error_output=None):
        new_msgs = [{"role": "user", "content": user_msg}]
        if error_output:
            new_msgs.append({"role": "user", "content": f"Command failed with error:\n{error_output}"})
        new_msgs.append({"role": "assistant", "content": json.dumps({"code": assistant_code})})
        # Deduplicate: skip if last entries already match
        tail = self._conversation[-len(new_msgs):] if len(self._conversation) >= len(new_msgs) else []
        if tail != new_msgs:
            self._conversation.extend(new_msgs)
        self._prune_conversation()

    def _prune_conversation(self):
        MAX_TURNS = 8
        assistant_count = sum(1 for m in self._conversation if m.get("role") == "assistant")
        if assistant_count <= MAX_TURNS:
            return
        excess = assistant_count - MAX_TURNS
        removed = 0
        i = 0
        while i < len(self._conversation) and removed < excess:
            if self._conversation[i].get("role") == "assistant":
                removed += 1
                del self._conversation[i]
                if i > 0 and self._conversation[i - 1].get("role") == "user":
                    del self._conversation[i - 1]
                    if i - 1 > 0 and self._conversation[i - 2].get("role") == "user":
                        del self._conversation[i - 2]
                    i -= 1
                i -= 1
            i += 1

    def log_entry(self, **kwargs):
        self._turn += 1
        entry = {"turn": self._turn, "cwd": os.getcwd(), **kwargs}
        self.session_log.append(entry)
        if len(self.session_log) > 50:
            self.session_log = self.session_log[-50:]
        self._write_history(entry)

    def _write_history(self, entry):
        """Write a structured history entry to full_history.jsonl.
        Outputs > 20 lines are stored in separate files and linked."""
        import datetime as _datetime
        output = entry.get("output", "") or ""
        error = entry.get("error", "") or ""
        output_lines = output.split("\n") if output else []
        error_lines = error.split("\n") if error else []

        record = {
            "turn": self._turn,
            "timestamp": _datetime.datetime.now().isoformat(),
            "language": entry.get("language", ""),
            "input": entry.get("input", "")[:200],
            "code_executed": entry.get("code", "")[:500],
            "llm_used": bool(entry.get("llm_generated")),
            "success": entry.get("success", False),
            "elapsed_ms": entry.get("elapsed_ms", 0),
            "type": entry.get("type", "?"),
            "output": None,
            "output_file": None,
            "error_file": None,
        }

        if len(output_lines) > 20:
            out_path = os.path.join(OUTPUTS_DIR, f"turn_{self._turn}_out.txt")
            with open(out_path, "w") as f:
                f.write(output)
            record["output"] = "\n".join(output_lines[:3]) + f"\n... ({len(output_lines)} lines total, see output_file)"
            record["output_file"] = os.path.relpath(out_path, SESSION_DIR)
        elif output:
            record["output"] = output[:2000]

        if len(error_lines) > 20:
            err_path = os.path.join(OUTPUTS_DIR, f"turn_{self._turn}_err.txt")
            with open(err_path, "w") as f:
                f.write(error)
            record["error_file"] = os.path.relpath(err_path, SESSION_DIR)

        with open(FULL_HISTORY_FILE, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_conversation(self, messages):
        """Write raw LLM conversation messages to the unified history file."""
        import datetime as _datetime
        clean = []
        for m in messages:
            if isinstance(m, dict):
                clean.append({k: v for k, v in m.items()})
        record = {
            "type": "llm_conversation",
            "turn": self._turn,
            "timestamp": _datetime.datetime.now().isoformat(),
            "messages": clean,
        }
        with open(FULL_HISTORY_FILE, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _tail_lines(self, text, n=10):
        if not text:
            return ""
        lines = text.strip().split("\n")
        return "\n".join(lines[-n:])

    def add_var(self, name, meta):
        self.vars[name] = meta

    def remove_var(self, name):
        self.vars.pop(name, None)
        if self.active_df == name:
            self.active_df = None

    def set_active(self, name):
        if name in self.vars:
            self.active_df = name

    def get_df_schema(self, name=None):
        name = name or self.active_df
        if name and name in self.vars:
            return self.vars[name]
        return None

    def get_available_columns(self):
        schema = self.get_df_schema()
        if schema and "columns" in schema:
            return schema["columns"]
        return []

    @property
    def current_language(self):
        return self._current_language

    @current_language.setter
    def current_language(self, value):
        if value not in ("shell", "python", "r"):
            return
        self._current_language = value

    def save(self):
        if not os.environ.get("SRUN_DEBUG"):
            return
        os.makedirs(SESSION_DIR, exist_ok=True)
        state_data = {
            "session_id": SESSION_ID,
            "system": get_system_info(),
            "workspace": {"files": get_file_meta()},
            "session": {
                "vars": {k: v for k, v in self.vars.items()},
                "active_df": self.active_df,
                "last_lang": self.last_lang,
                "current_language": self.current_language,
                "log": self.session_log[-20:],
            },
        }
        with open(STATE_FILE, "w") as f:
            json.dump(state_data, f, indent=2)

    def llm_context(self):
        info = get_system_info()
        lines = [
            f"System: {info['platform']} ({info['arch']}), {info['cpu']} cores, shell={info['shell']}",
        ]
        if info.get("tools_note"):
            lines.append(f"Note: {info['tools_note']}")
        if info.get("gnu_alternatives"):
            for cmd, alt in sorted(info["gnu_alternatives"].items()):
                lines.append(f"  GNU {cmd}: {alt}")
        if info.get("python_versions"):
            py_list = "; ".join(info["python_versions"][:4])
            lines.append(f"Python versions: {py_list}")
        if info.get("r_version"):
            lines.append(f"R: {info['r_version']}")
        if info.get("virtualenv"):
            lines.append(f"Virtualenv: {info['virtualenv']}")
        if info.get("now"):
            lines.append(f"Date: {info['now']}")
        lines.append(f"Current environment: {self.current_language}")
        sessions = "shell, python"
        if info.get("r_available"):
            sessions += ", r"
        lines.append(f"Available sessions: {sessions} (use 'language' field in run_command to target one)")
        return "\n".join(lines)

    def startup_context(self):
        sys_info = self.llm_context()
        ws_info = self.workspace_context()
        api_note = ""
        from .config import config
        if not config.has_llm:
            api_note = ("\nAPI: No API key configured. To use natural language and repair, set api_key in ~/.srun/user_config.json "
                        "(or export DEEPSEEK_API_KEY). Tell the user to type 'srun configure-api' or update the config file manually.")
        history_note = f"\nHistory file: {self.full_history_path} (JSONL: all turns + LLM conversations + outputs >20 lines → {self.outputs_dir})"
        state_note = f"\nState file: {self.state_path} (session metadata)"
        return f"{sys_info}\n{ws_info}{api_note}{history_note}{state_note}"

    def workspace_context(self):
        info = get_system_info()
        files = get_file_meta()
        lines = [f"Workspace: {info['cwd']}"]
        git_info = _get_git_info(info['cwd'])
        if git_info:
            lines.append(f"Git: {git_info}")
        if files:
            file_list = ", ".join(f"{f['name']}({f['size']})" for f in files[:15])
            lines.append(f"Files: {file_list}")
        return "\n".join(lines)

    def session_context(self):
        if not self.session_log:
            return ""
        entries = []
        for e in self.session_log[-15:]:
            line = self._format_log_line(e)
            if line:
                entries.append(line)
        if not entries:
            return ""
        return "## Session\n" + "\n".join(entries)

    def _format_log_line(self, e):
        t = self._tail_lines(e.get("error", ""), 3)
        code = e.get("code", "")
        gen = e.get("llm_generated", "")
        tp = e.get("type", "?")
        turn = e.get("turn", 0)
        ok = "\u2713" if e.get("success") else "\u2717"
        ms = e.get("elapsed_ms", 0)
        inp = e.get("input", "")

        if tp == "fast":
            return f"#{turn} {code} {ok} {ms}ms"
        if tp == "llm_dispatch":
            return f"#{turn} \"{inp}\" → {gen} {ok} {ms}ms"
        if tp == "llm_repair":
            err_short = t.replace('\n', '; ')[:60] if t else ""
            return f"#{turn} \"{inp}\" err: {err_short} → fix: {gen} {ok} {ms}ms"
        return f"#{turn} {inp[:60]} {ok}"


state = SessionState()
