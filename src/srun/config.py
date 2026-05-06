import os
import subprocess


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


load_shell_env()


class Config:
    def __init__(self):
        self.api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        self.api_base = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
        self.model = os.environ.get("SRUN_MODEL", "deepseek-chat")
        self.max_retries = 3

    @property
    def has_llm(self):
        return bool(self.api_key)


config = Config()
