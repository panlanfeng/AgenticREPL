import os
import time
import shutil
import subprocess

SESSION_BASE = os.path.join(os.path.expanduser("~"), ".srun", "sessions")
SESSION_MAX_AGE_DAYS = 90
_init_done = False


def clean_old_sessions():
    if not os.path.isdir(SESSION_BASE):
        return
    cutoff = time.time() - SESSION_MAX_AGE_DAYS * 86400
    for name in os.listdir(SESSION_BASE):
        path = os.path.join(SESSION_BASE, name)
        try:
            if os.path.isdir(path) and os.path.getmtime(path) < cutoff:
                shutil.rmtree(path)
        except Exception:
            pass


def _shell_name():
    shell = os.environ.get("SHELL", "/bin/zsh")
    return shell.split("/")[-1]


def load_shell_env():
    shell = _shell_name()
    rc_file = os.path.expanduser(f"~/.{shell}rc")
    if not os.path.isfile(rc_file):
        return
    try:
        result = subprocess.run(
            f"source {rc_file} >/dev/null 2>&1; env",
            shell=True,
            executable=os.environ.get("SHELL", "/bin/zsh"),
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.strip().split("\n"):
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            if key and key not in os.environ and val:
                os.environ[key] = val
    except Exception:
        pass


def init():
    global _init_done
    if _init_done:
        return
    _init_done = True
    load_shell_env()
    clean_old_sessions()


class Config:
    def __init__(self):
        from .user_config import get_api_config
        self.api_key, self.api_base, self.model = get_api_config()
        self.max_retries = 3

    @property
    def has_llm(self):
        return bool(self.api_key)


config = Config()
