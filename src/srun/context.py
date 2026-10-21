import os
import time
import uuid
import platform
import shutil
import subprocess
from datetime import datetime, timedelta, timezone

from .eventlog import EventLog

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
OUTPUTS_DIR = os.path.join(SESSION_DIR, "outputs")
EVENTS_FILE = os.path.join(SESSION_DIR, "events.jsonl")
ATIF_FILE = os.path.join(SESSION_DIR, "trajectory.json")


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
        self._cached_messages = None  # cache for build_conversation_messages
        self._cached_conv_len = 0
        self._cached_summary = None
        self._max_context_tokens = 256000  # per-turn context window
        self._last_memory_extract_tokens = 0  # token count at last MEMORY.md write
        self._last_memory_extract_turn = 0    # turn count at last MEMORY.md write
        self._memory_file = os.path.join(BASE_DIR, "MEMORY.md")
        os.makedirs(BASE_DIR, exist_ok=True)
        os.makedirs(SESSION_DIR, exist_ok=True)
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        self.session_dir = SESSION_DIR
        self.outputs_dir = OUTPUTS_DIR
        self.events_path = EVENTS_FILE
        self.trajectory_path = ATIF_FILE
        self.event_log = EventLog(self.events_path, session_id=os.path.basename(self.session_dir))
        self.last_contract = None
        self.last_verification = None
        self.event_log_degraded = False
        self._event_warning_shown = False
        self._atif_last_turn = -1
        self.permission_grants = []

    def reset_session(self):
        self.event_log = EventLog(self.events_path, session_id=os.path.basename(self.session_dir))
        self._conversation = []
        self._stable_summary = None
        self._cached_messages = None
        self._cached_conv_len = 0
        self._cached_summary = None
        self.session_log = []
        self._turn = 0
        self._last_memory_extract_turn = 0
        self.last_dispatch_error = None
        self._context_injected = False
        self._llm_last_known_language = self._current_language
        self.last_contract = None
        self.last_verification = None
        self.event_log_degraded = False
        self._event_warning_shown = False
        self._atif_last_turn = -1
        self.permission_grants = []

    def resume_session(self, session_id):
        """Resume a previous session by ID. Loads its history and redirects
        all future writes (history, outputs) to that session's directory."""
        target_dir = os.path.join(BASE_DIR, "sessions", session_id)
        target_events = os.path.join(target_dir, "events.jsonl")
        if not os.path.isfile(target_events):
            return False
        os.environ["SRUN_SESSION_ID"] = session_id
        self.session_dir = target_dir
        self.outputs_dir = os.path.join(target_dir, "outputs")
        self.events_path = target_events
        self.trajectory_path = os.path.join(target_dir, "trajectory.json")
        self.event_log = EventLog(self.events_path, session_id=session_id)
        self.last_contract = None
        self.last_verification = None
        self.event_log_degraded = False
        self._event_warning_shown = False
        self._atif_last_turn = -1
        self.permission_grants = []
        self._memory_file = os.path.join(BASE_DIR, "MEMORY.md")
        os.makedirs(self.outputs_dir, exist_ok=True)
        self._load_conversation_state()
        self.write_atif()
        return True

    def build_conversation_messages(self, system_prompt):
        # Return cached messages if nothing changed
        conv_len = len(self._conversation)
        if (self._cached_messages is not None
                and self._cached_conv_len == conv_len
                and self._cached_summary == self._stable_summary):
            return self._cached_messages
        messages = [{"role": "system", "content": system_prompt}]
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
        self._cached_messages = messages
        self._cached_conv_len = conv_len
        self._cached_summary = self._stable_summary
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
        new_msgs.append({"role": "assistant", "content": assistant_code})
        tail = self._conversation[-len(new_msgs):] if len(self._conversation) >= len(new_msgs) else []
        if tail != new_msgs:
            self._conversation.extend(new_msgs)

    def extend_conversation(self, messages):
        """Append LLM conversation messages (assistant, tool) to conversation history.
        Deduplicates against the tail of the existing conversation."""
        if not messages:
            return
        tail = self._conversation[-len(messages):] if len(self._conversation) >= len(messages) else []
        if tail != messages:
            self._conversation.extend(messages)
            self._cached_messages = None  # invalidate — conversation changed

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
            "For every unfinished subtask, state the current conclusion and WHY (what was "
            "decided and the reasoning), so work can continue without re-discovery. "
            "Keep it under 500 words."
        )
        summary_msgs = msgs + [{"role": "user", "content": summary_prompt}]
        try:
            from .config import config
            kwargs = {"model": llm_module.client.model, "messages": summary_msgs,
                      "temperature": config.temperature, "max_tokens": config.max_tokens, "stream": False}
            if config.top_p > 0:
                kwargs["top_p"] = config.top_p
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

        # P1-3: preserve dropped tool outputs as recoverable artifact files so
        # compaction never destroys evidence — the summary keeps absolute paths.
        artifact_note = ""
        dropped = self._conversation[:cutoff]
        dropped_outputs = [m.get("content", "") for m in dropped
                           if m.get("role") == "tool"
                           and isinstance(m.get("content"), str)
                           and m.get("content", "").strip()]
        if dropped_outputs:
            import datetime as _datetime
            os.makedirs(self.outputs_dir, exist_ok=True)
            fname = (f"compaction_{_datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_"
                     f"{uuid.uuid4().hex[:6]}.txt")
            fpath = os.path.join(self.outputs_dir, fname)
            with open(fpath, "w") as f:
                for i, content in enumerate(dropped_outputs):
                    f.write(f"--- tool result {i + 1} ---\n{content}\n\n")
            artifact_note = (
                f"\n\n[Recoverable artifacts from compacted turns: {fpath} — "
                "use read_file if you need a dropped tool result verbatim.]"
            )

        self._stable_summary = summary + artifact_note
        self._conversation = self._conversation[cutoff:]
        self._cached_messages = None  # invalidate — conversation truncated + summary changed
        self._write_compaction_snapshot()
        return True

    def _write_compaction_snapshot(self):
        """Record the current compaction projection as an event."""
        self.append_event(
            "compaction",
            {"summary": self._stable_summary, "conversation": self._conversation},
            turn=self._turn + 1,
        )

    def _load_conversation_state(self):
        """Restore all session state by replaying the append-only event log."""
        if os.path.isfile(self.events_path):
            self._replay_event_log()

    def _replay_event_log(self):
        """Rebuild in-memory session state from events.jsonl after a crash."""
        replayed = self.event_log.replay()
        self._conversation = list(replayed.get("conversation") or [])
        self._stable_summary = replayed.get("summary")
        self.session_log = list(replayed.get("session_log") or [])
        self._turn = int(replayed.get("turn") or 0)
        self._current_language = replayed.get("current_language") or "shell"
        self.last_lang = replayed.get("last_lang")
        self.vars = dict(replayed.get("vars") or {})
        self.active_df = replayed.get("active_df")
        self.last_contract = replayed.get("contract")
        self.last_verification = replayed.get("last_verification")
        self.permission_grants = list(replayed.get("permission_grants") or [])
        self._cached_messages = None
        self._cached_conv_len = 0
        self._cached_summary = None
        if self._conversation:
            self._context_injected = True
            self._last_known_cwd = os.getcwd()

    def append_event(self, event_type, payload=None, turn=None, round=None):
        """Append one authoritative event. Failures are surfaced, not hidden."""
        try:
            event = self.event_log.append(event_type, payload, turn=turn, round=round)
            if event_type == "permission_grant" and payload and payload.get("grant"):
                grant = payload["grant"]
                if grant not in self.permission_grants:
                    self.permission_grants.append(grant)
            elif event_type == "permission_revoke" and payload:
                grant_id = payload.get("grant_id")
                self.permission_grants = [
                    grant for grant in self.permission_grants
                    if grant.get("grant_id") != grant_id
                ]
            return event
        except Exception as exc:
            self.event_log_degraded = True
            self.last_dispatch_error = f"event log write failed: {exc}"
            if not self._event_warning_shown:
                self._event_warning_shown = True
                print(f"\033[1;31mWARNING: event log write failed; audit/replay may be incomplete: {exc}\033[0m")
            return None

    def grant_permission(self, action, contains=None, path_prefix=None,
                         ttl_seconds=None, reason="user approved"):
        """Append a durable permission grant event and update the projection."""
        grant = {
            "grant_id": uuid.uuid4().hex,
            "action": action,
            "reason": reason,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if contains:
            grant["contains"] = contains
        if path_prefix:
            grant["path_prefix"] = path_prefix
        if ttl_seconds:
            grant["expires_at"] = (
                datetime.now(timezone.utc) + timedelta(seconds=int(ttl_seconds))
            ).isoformat()
        self.append_event("permission_grant", {"grant": grant}, turn=self._turn)
        return grant

    def revoke_permission(self, grant_id):
        self.append_event("permission_revoke", {"grant_id": grant_id}, turn=self._turn)

    def write_atif(self):
        """Atomically rebuild the derived ATIF trajectory from events.jsonl."""
        try:
            return self.event_log.write_atif(self.trajectory_path)
        except Exception as exc:
            self.event_log_degraded = True
            self.last_dispatch_error = f"ATIF export failed: {exc}"
            if not self._event_warning_shown:
                self._event_warning_shown = True
                print(f"\033[1;31mWARNING: ATIF export failed; events.jsonl is still authoritative: {exc}\033[0m")
            return None

    def maybe_write_atif(self, force=False):
        """Throttle the O(history) ATIF rebuild while keeping events authoritative."""
        from .user_config import get as config_get
        interval = int(config_get("atif_export_every_turns") or 5)
        if not force and interval > 1 and self._turn - self._atif_last_turn < interval:
            return None
        result = self.write_atif()
        self._atif_last_turn = self._turn
        return result

    def log_entry(self, **kwargs):
        self._turn += 1
        entry = {"turn": self._turn, "cwd": os.getcwd(), **kwargs}
        self.session_log.append(entry)
        if len(self.session_log) > 200:
            self.session_log = self.session_log[-200:]
        self.append_event("turn_result", {"entry": entry}, turn=self._turn)
        self.maybe_write_atif()

    def log_conversation(self, messages):
        """Append raw LLM conversation messages to the event log."""
        clean = []
        for m in messages:
            if isinstance(m, dict):
                clean.append({k: v for k, v in m.items()})
        self.append_event(
            "llm_conversation", {"messages": clean}, turn=self._turn + 1
        )

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
            "Extract information for a persistent memory file shared across sessions. "
            "Write in Markdown with these sections:\n\n"
            "## User Profile\n"
            "- Knowledge level, job role, preferred collaboration style\n"
            "- How they like to receive information (concise? detailed? examples first?)\n\n"
            "## Reflection & Learnings\n"
            "- What did you learn from this session? What would you do differently next time and WHY?\n"
            "- Which approaches worked well and should be repeated?\n"
            "- Which commands or patterns caused errors and what was the root cause?\n\n"
            "## Environment Learnings\n"
            "- Python versions that work (paths, known incompatibilities)\n"
            "- Commands that are NOT in the default PATH and how to run them (full path)\n"
            "- Unfamiliar commands encountered and how they work (flags, syntax, gotchas)\n"
            "- R package paths, virtualenv locations, conda environments discovered\n\n"
            "## Feedback & Corrections\n"
            "- When the user corrected you, what was the mistake and WHY the fix works\n"
            "- The underlying principle, not just syntax\n"
            "- How to apply this correction in similar situations next time\n\n"
            "RULES:\n"
            "- Always record both WHAT happened and WHY — not just the action but the reasoning.\n"
            "- NEVER include file paths or anything discoverable in AGENTS.md\n"
            "- The ongoing session context takes priority — if the user is doing things\n"
            "  differently now, respect their current approach over stored memory.\n"
            "- Drop entries that are obsolete (outdated versions, resolved issues, old projects).\n"
            "- Prefer recent learnings over old ones. The file should reflect current preferences.\n"
            "- If existing memory exists, MERGE new info — don't duplicate, don't remove correct info\n"
            "- Keep under 800 words total\n\n"
        )
        if existing:
            memory_prompt += f"Existing MEMORY.md content to merge with:\n```markdown\n{existing[:2000]}\n```\n\n"

        summary_msgs = msgs + [{"role": "user", "content": memory_prompt}]
        try:
            from .config import config
            kwargs = {"model": llm_module.client.model, "messages": summary_msgs,
                      "temperature": config.temperature, "max_tokens": config.max_tokens, "stream": False}
            if config.top_p > 0:
                kwargs["top_p"] = config.top_p
            resp = llm_module.client.chat.completions.create(**kwargs)
            memory = resp.choices[0].message.content.strip()
            if llm_module._track_usage:
                llm_module._track_usage(resp.usage)
            with open(self._memory_file, "w") as f:
                f.write(memory)
            self._last_memory_extract_tokens = self.context_tokens()
            self._last_memory_extract_turn = self._turn
        except Exception:
            pass

    def save_state(self):
        """Append a state projection event; events.jsonl is authoritative."""
        self.append_event(
            "session_state",
            {
                "current_language": self.current_language,
                "last_lang": self.last_lang,
                "vars": {k: v for k, v in self.vars.items()},
                "active_df": self.active_df,
            },
            turn=self._turn,
        )
        self.maybe_write_atif()

    def save(self):
        if not os.environ.get("SRUN_DEBUG"):
            self.save_state()
            return
        self.append_event(
            "session_debug",
            {
                "system": get_system_info(),
                "workspace": {"files": get_file_meta()},
                "current_language": self.current_language,
                "last_lang": self.last_lang,
                "vars": {k: v for k, v in self.vars.items()},
                "active_df": self.active_df,
                "log": self.session_log[-20:],
            },
            turn=self._turn,
        )
        self.maybe_write_atif()

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
            api_note = ("\nAPI: No API key configured. Set DEEPSEEK_API_KEY, OPENAI_API_KEY, or provider-specific"
                        " env var, or add api_key to ~/.srun/user_config.json. Type /configure for help.")
        history_note = f"\nEvents: {self.events_path} (authoritative runtime log)"
        state_note = f"\nTrajectory: {self.trajectory_path} (derived ATIF projection)"
        mem_note = ""
        if os.path.isfile(self._memory_file):
            mem_note = f"\nMemory: use read_file to load {self._memory_file} — contains user profile, environment learnings, past corrections"
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
        if tp == "llm":
            return f"#{turn} \"{inp}\" → {gen} {ok} {ms}ms"
        return f"#{turn} {inp[:60]} {ok}"


state = SessionState()
