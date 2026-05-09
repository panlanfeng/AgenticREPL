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
CONVERSATIONS_DIR = os.path.join(SESSION_DIR, "conversations")


def get_system_info():
    os_name = platform.system().lower()
    info = {
        "os": os_name,
        "arch": platform.machine(),
        "cpu": os.cpu_count(),
        "shell": os.environ.get("SHELL", "").split("/")[-1],
        "cwd": os.getcwd(),
        "platform": os_name,
    }
    if os_name == "darwin":
        try:
            ver = platform.mac_ver()[0]
            info["platform"] = f"macOS {ver}"
        except Exception:
            info["platform"] = "macOS"
        info["tools_note"] = "macOS uses BSD grep/sed/awk, not GNU"
    elif os_name == "linux":
        info["platform"] = "Linux"
        info["tools_note"] = "Linux uses GNU grep/sed/awk"
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
        self.current_language = "shell"
        self.history = []
        self.last_dispatch_error = None
        self.code_cache = {}
        self.session_log = []
        self._turn = 0
        self._conversation = []
        self._context_injected = False
        self._llm_last_known_language = "shell"
        os.makedirs(BASE_DIR, exist_ok=True)
        os.makedirs(SESSION_DIR, exist_ok=True)
        os.makedirs(CONVERSATIONS_DIR, exist_ok=True)

    def reset_session(self):
        self._conversation = []
        self.session_log = []
        self._turn = 0
        self.last_dispatch_error = None
        self._context_injected = False
        self._llm_last_known_language = self.current_language

    def startup_context(self):
        sys_info = self.llm_context()
        ws_info = self.workspace_context()
        return f"{sys_info}\n{ws_info}"

    def build_conversation_messages(self, system_prompt):
        messages = [{"role": "system", "content": system_prompt}]
        for entry in self._conversation:
            messages.append(entry)
        return messages

    def add_conversation_turn(self, user_msg, assistant_code, error_output=None):
        self._conversation.append({"role": "user", "content": user_msg})
        if error_output:
            self._conversation.append({"role": "user", "content": f"Command failed with error:\n{error_output}"})
        self._conversation.append({"role": "assistant", "content": json.dumps({"code": assistant_code})})

    def log_entry(self, **kwargs):
        self._turn += 1
        entry = {"turn": self._turn, "cwd": os.getcwd(), **kwargs}
        self.session_log.append(entry)
        if len(self.session_log) > 50:
            self.session_log = self.session_log[-50:]

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

    def save(self):
        os.makedirs(SESSION_DIR, exist_ok=True)
        os.makedirs(CONVERSATIONS_DIR, exist_ok=True)
        state_data = {
            "session_id": SESSION_ID,
            "system": get_system_info(),
            "workspace": {"files": get_file_meta()},
            "session": {
                "vars": {k: v for k, v in self.vars.items()},
                "active_df": self.active_df,
                "last_lang": self.last_lang,
                "current_language": self.current_language,
                "history": self.history[-10:],
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
        lines.append(f"Current environment: {self.current_language}")
        return "\n".join(lines)

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
