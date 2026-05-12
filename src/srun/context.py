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
        self._stable_summary = None  # compaction summary, inserted after startup
        self._max_context_tokens = 128000  # DeepSeek default, overridable
        self._last_memory_extract_tokens = 0  # token count at last MEMORY.md write
        self._memory_file = os.path.join(SESSION_DIR, "MEMORY.md")
        os.makedirs(BASE_DIR, exist_ok=True)
        os.makedirs(SESSION_DIR, exist_ok=True)
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        self.session_dir = SESSION_DIR
        self.full_history_path = FULL_HISTORY_FILE
        self.outputs_dir = OUTPUTS_DIR
        self.state_path = STATE_FILE

    def reset_session(self):
        self._conversation = []
        self._stable_summary = None
        self.session_log = []
        self._turn = 0
        self.last_dispatch_error = None
        self._context_injected = False
        self._llm_last_known_language = self._current_language

    def resume_session(self, session_id):
        """Resume a previous session by ID. Loads its history and redirects
        all future writes (history, outputs) to that session's directory."""
        target_dir = os.path.join(BASE_DIR, "sessions", session_id)
        target_history = os.path.join(target_dir, "full_history.jsonl")
        if not os.path.isfile(target_history):
            return False
        os.environ["SRUN_SESSION_ID"] = session_id
        self.session_dir = target_dir
        self.full_history_path = target_history
        self.outputs_dir = os.path.join(target_dir, "outputs")
        self.state_path = os.path.join(target_dir, "state.json")
        self._memory_file = os.path.join(target_dir, "MEMORY.md")
        os.makedirs(self.outputs_dir, exist_ok=True)
        self._load_conversation_state()
        return True

    def build_conversation_messages(self, system_prompt):
        messages = [{"role": "system", "content": system_prompt}]
        if os.path.isfile(self._memory_file):
            with open(self._memory_file) as f:
                mem = f.read()
            if mem.strip():
                messages.append({"role": "user", "content": f"[Persistent memory — use this to personalize responses]\n{mem}"})
        agents_md = self._load_agents_md()
        if agents_md:
            messages.append({"role": "user", "content": f"[AGENTS.md — user instructions]\n{agents_md}"})
        from .skills import get_skill_prompts
        skill_prompts = get_skill_prompts()
        if skill_prompts:
            messages.append({"role": "user", "content": skill_prompts})
        if self._stable_summary:
            messages.append({"role": "user", "content": f"[Summary of earlier conversation]\n{self._stable_summary}"})
        for entry in self._conversation:
            messages.append(entry)
        return messages

    def _load_agents_md(self):
        """Load AGENTS.md from ~/.srun/ (global) and current directory (project)."""
        blocks = []
        home_path = os.path.join(BASE_DIR, "AGENTS.md")
        if os.path.isfile(home_path):
            try:
                with open(home_path) as f:
                    content = f.read()
                if content.strip():
                    blocks.append(f"[global]\n{content}")
            except Exception:
                pass
        cwd_path = os.path.join(os.getcwd(), "AGENTS.md")
        if os.path.isfile(cwd_path):
            try:
                with open(cwd_path) as f:
                    content = f.read()
                if content.strip():
                    blocks.append(f"[project]\n{content}")
            except Exception:
                pass
        return "\n\n".join(blocks) if blocks else ""

    def add_conversation_turn(self, user_msg, assistant_code, error_output=None):
        new_msgs = [{"role": "user", "content": user_msg}]
        if error_output:
            new_msgs.append({"role": "user", "content": f"Command failed with error:\n{error_output}"})
        new_msgs.append({"role": "assistant", "content": json.dumps({"code": assistant_code})})
        tail = self._conversation[-len(new_msgs):] if len(self._conversation) >= len(new_msgs) else []
        if tail != new_msgs:
            self._conversation.extend(new_msgs)

    def _approx_tokens(self, text):
        """Rough token count: ~3.5 chars per token for English + code."""
        return max(1, len(text) // 3)

    def _context_tokens(self, messages):
        """Estimate total tokens in a message list."""
        total = 0
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, str):
                total += self._approx_tokens(content)
        return total

    def compact_context(self, llm_module=None):
        """If conversation exceeds 80% of max tokens, generate a stable summary
        via a subagent call with identical context (maximizes cache hit rate).
        Removes old turns and inserts the summary after startup context."""
        from .prompts import PROMPT
        msgs = self.build_conversation_messages(PROMPT.format())
        tokens = self._context_tokens(msgs)
        limit = int(self._max_context_tokens * 0.8)
        remaining = self._max_context_tokens - tokens

        if tokens < limit and remaining > 30000:
            return False

        if not llm_module:
            return False

        summary_prompt = (
            "Summarize the conversation so far into a concise context block. "
            "Include: what the user has been working on, key files and variables, "
            "languages used, important commands and their results, and any errors encountered. "
            "Keep it under 500 words."
        )
        summary_msgs = msgs + [{"role": "user", "content": summary_prompt}]
        try:
            kwargs = {"model": llm_module.client.model, "messages": summary_msgs,
                      "temperature": 0.0, "max_tokens": 800, "stream": False}
            resp = llm_module.client.chat.completions.create(**kwargs)
            summary = resp.choices[0].message.content.strip()
            if llm_module._track_usage:
                llm_module._track_usage(resp.usage)
        except Exception:
            return False

        # Keep last 5 turns for recency, replace older ones with summary
        assistant_indices = [i for i, m in enumerate(self._conversation) if m.get("role") == "assistant"]
        if len(assistant_indices) <= 6:
            return False
        cutoff = assistant_indices[-6]
        self._stable_summary = summary
        self._conversation = self._conversation[cutoff:]
        self._write_compaction_snapshot()
        return True

    def _write_compaction_snapshot(self):
        """Write the current compaction state (summary + conversation) to history file.
        Appended so the file is a chronological log; on load we read the last snapshot."""
        import datetime as _datetime
        record = {
            "type": "compaction_snapshot",
            "timestamp": _datetime.datetime.now().isoformat(),
            "summary": self._stable_summary,
            "conversation": self._conversation,
        }
        os.makedirs(self.session_dir, exist_ok=True)
        with open(self.full_history_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _load_conversation_state(self):
        """Restore _stable_summary from the last compaction snapshot in history.
        _conversation starts fresh each session — only the summary persists."""
        if not os.path.isfile(self.full_history_path):
            return
        best_summary = None
        try:
            with open(self.full_history_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if entry.get("type") == "compaction_snapshot":
                        best_summary = entry.get("summary")
            if best_summary is not None:
                self._stable_summary = best_summary
        except Exception:
            pass

    def log_entry(self, **kwargs):
        self._turn += 1
        entry = {"turn": self._turn, "cwd": os.getcwd(), **kwargs}
        self.session_log.append(entry)
        if len(self.session_log) > 200:
            self.session_log = self.session_log[-200:]
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
            out_path = os.path.join(self.outputs_dir, f"turn_{self._turn}_out.txt")
            with open(out_path, "w") as f:
                f.write(output)
            record["output"] = "\n".join(output_lines[:3]) + f"\n... ({len(output_lines)} lines total, see output_file)"
            record["output_file"] = os.path.relpath(out_path, self.session_dir)
        elif output:
            record["output"] = output[:2000]

        if len(error_lines) > 20:
            err_path = os.path.join(self.outputs_dir, f"turn_{self._turn}_err.txt")
            with open(err_path, "w") as f:
                f.write(error)
            record["error_file"] = os.path.relpath(err_path, self.session_dir)

        with open(self.full_history_path, "a") as f:
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
        with open(self.full_history_path, "a") as f:
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

    def context_tokens(self):
        """Estimate total tokens in the current conversation + system prompt."""
        from .prompts import PROMPT
        msgs = self.build_conversation_messages(PROMPT.format())
        return self._context_tokens(msgs)

    def extract_memory(self, llm_module=None):
        """Use a subagent sharing the SAME context (maximizing cache hit rate)
        to extract persistent memory from the conversation into MEMORY.md.
        Only extracts: user profile, feedback/corrections, project context, and
        pointers to important files. Never extracts discoverable info."""
        if not llm_module or not llm_module.client:
            return
        from .prompts import PROMPT

        existing = ""
        if os.path.isfile(self._memory_file):
            with open(self._memory_file) as f:
                existing = f.read()

        msgs = self.build_conversation_messages(PROMPT.format())
        memory_prompt = (
            "Extract information for a persistent memory file. Write in Markdown with these sections:\n\n"
            "## User Profile\n"
            "- Knowledge level, job role, preferred collaboration style\n"
            "- How they like to receive information (concise? detailed? examples first?)\n\n"
            "## Feedback & Corrections\n"
            "- When the user corrected you, what was the mistake and the fix\n"
            "- WHY the fix is correct — the underlying principle, not the syntax\n"
            "- How to apply this correction in similar situations next time\n\n"
            "## Project Context\n"
            "- Current project topic, purpose, and goals\n"
            "- Deadlines or constraints mentioned\n"
            "- Key decisions the user made and WHY they made them\n"
            "- Dependencies or external systems mentioned\n\n"
            "## Important Files\n"
            "- Only describe what each file CONTAINS and its ROLE, never its path or name\n"
            "- E.g., 'The main CSV has 3 columns: region, product, revenue' not 'path/to/sales.csv'\n\n"
            "RULES:\n"
            "- NEVER include file paths, coding style, git history, or anything in AGENTS.md\n"
            "- NEVER repeat information that can be found by searching the repo\n"
            "- Focus on WHY (intent, reasoning) not WHAT (commands, syntax)\n"
            "- If existing memory exists, MERGE new info — don't duplicate, don't remove correct info\n"
            "- Keep under 800 words total\n\n"
        )
        if existing:
            memory_prompt += f"Existing MEMORY.md content to merge with:\n```markdown\n{existing[:2000]}\n```\n\n"

        summary_msgs = msgs + [{"role": "user", "content": memory_prompt}]
        try:
            kwargs = {"model": llm_module.client.model, "messages": summary_msgs,
                      "temperature": 0.0, "max_tokens": 1200, "stream": False}
            resp = llm_module.client.chat.completions.create(**kwargs)
            memory = resp.choices[0].message.content.strip()
            if llm_module._track_usage:
                llm_module._track_usage(resp.usage)
            with open(self._memory_file, "w") as f:
                f.write(memory)
            self._last_memory_extract_tokens = self.context_tokens()
        except Exception:
            pass

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
        history_note = f"\nHistory: {self.full_history_path} (all turns + LLM conversations)"
        state_note = f"\nState: {self.state_path} (session metadata)"
        mem_note = ""
        if os.path.isfile(self._memory_file):
            mem_note = f"\nMemory: {self._memory_file} (persistent user profile, feedback, project context)"
        return f"{sys_info}\n{ws_info}{api_note}{history_note}{state_note}{mem_note}"

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
        for e in self.session_log[-30:]:
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
