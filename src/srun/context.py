import os
import json
import platform
import shutil

STATE_DIR = os.path.join(os.path.expanduser("~"), ".srun")
STATE_FILE = os.path.join(STATE_DIR, "state.json")


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


class SessionState:
    def __init__(self):
        self.vars = {}
        self.active_df = None
        self.last_lang = "shell"
        self.history = []
        self.last_output = ""
        self.last_dispatch_error = None
        self.code_cache = {}
        os.makedirs(STATE_DIR, exist_ok=True)

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

    def add_history(self, entry):
        self.history.append(entry)
        if len(self.history) > 20:
            self.history = self.history[-20:]

    def save(self):
        os.makedirs(STATE_DIR, exist_ok=True)
        state_data = {
            "system": get_system_info(),
            "workspace": {"files": get_file_meta()},
            "session": {
                "vars": {k: v for k, v in self.vars.items()},
                "active_df": self.active_df,
                "last_lang": self.last_lang,
                "history": self.history[-10:],
            },
        }
        with open(STATE_FILE, "w") as f:
            json.dump(state_data, f, indent=2)

    def llm_context(self):
        info = get_system_info()
        files = get_file_meta()
        lines = [
            f"System: {info['platform']} ({info['arch']}), {info['cpu']} cores, shell={info['shell']}",
        ]
        if info.get("tools_note"):
            lines.append(f"Note: {info['tools_note']}")
        lines.append(f"Workspace: {info['cwd']}")
        if files:
            file_list = ", ".join(f"{f['name']}({f['size']})" for f in files[:10])
            lines.append(f"Files: {file_list}")
        if self.vars:
            var_list = []
            for name, meta in self.vars.items():
                if "columns" in meta:
                    cols = ", ".join(meta["columns"][:10])
                    rows = meta.get("rows", "?")
                    var_list.append(f"{name}[{cols} | {rows} rows]")
                else:
                    var_list.append(f"{name}: {meta.get('type', '?')}")
            lines.append(f"Variables: {'; '.join(var_list)}")
        if self.active_df:
            lines.append(f"Active DataFrame: {self.active_df}")
        if self.last_lang:
            lines.append(f"Last language: {self.last_lang}")
        return "\n".join(lines)


state = SessionState()
