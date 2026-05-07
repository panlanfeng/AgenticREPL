import subprocess
import os
import re
import shutil

SHELL_BIN = os.environ.get("SHELL", "/bin/zsh")


def _build_env():
    env = os.environ.copy()
    env["CLICOLOR"] = "1"
    env["CLICOLOR_FORCE"] = "1"
    if "LSCOLORS" not in env:
        env["LSCOLORS"] = "GxFxCxDxBxegedabagaced"
    return env


_SSH_RE = re.compile(r"^ssh(?:\s+(?:-[^\s]+(?:\s+[^\s]+)?))*\s+([^\s]+)$")


class ShellExecutor:
    def __init__(self):
        self.shell = shutil.which(SHELL_BIN) or SHELL_BIN
        self.env = _build_env()
        self._ssh_prefix = None
        self._remote_label = None

    @property
    def remote(self):
        return self._remote_label

    def connect_ssh(self, cmd):
        stripped = cmd.strip()
        if not stripped.startswith("ssh ") and stripped != "ssh":
            return None
        if re.match(r"^ssh\s*-", stripped):
            return None
        m = _SSH_RE.match(stripped)
        if not m:
            return "Invalid SSH syntax. Use: ssh [options] user@host"
        target = m.group(m.lastindex)
        self._ssh_prefix = cmd.strip()
        self._remote_label = target
        try:
            r = subprocess.run(
                f"{self._ssh_prefix} 'echo connected'",
                shell=True,
                executable=self.shell,
                capture_output=True,
                text=True,
                timeout=10,
                cwd=os.getcwd(),
                env=self.env,
            )
            if r.returncode != 0:
                self._ssh_prefix = None
                self._remote_label = None
                return f"SSH connection failed:\n{r.stderr or r.stdout}"
        except subprocess.TimeoutExpired:
            self._ssh_prefix = None
            self._remote_label = None
            return "SSH connection timed out"
        except Exception as e:
            self._ssh_prefix = None
            self._remote_label = None
            return f"SSH error: {e}"
        return None

    def disconnect(self):
        self._ssh_prefix = None
        self._remote_label = None

    def execute(self, code):
        stripped = code.strip()
        if self._ssh_prefix:
            code = f"{self._ssh_prefix} '{stripped}'"
        try:
            result = subprocess.run(
                code,
                shell=True,
                executable=self.shell,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=os.getcwd(),
                env=self.env,
            )
        except subprocess.TimeoutExpired:
            return False, "Command timed out (30s)", "", -1
        except Exception as e:
            return False, f"Error: {e}", "", -1

        output = result.stdout
        if result.stderr:
            output += result.stderr

        if self._is_pure_cd(stripped) and not self._ssh_prefix:
            new_cwd = self._sync_cd(stripped)
            if new_cwd:
                os.chdir(new_cwd)

        if result.returncode != 0:
            return False, output, result.stderr, result.returncode
        return True, output, result.stderr

    def _is_cd(self, code):
        return code.startswith("cd ") or code == "cd"

    def _is_pure_cd(self, code):
        if not self._is_cd(code):
            return False
        for sep in ("&&", "||", ";", "|"):
            if sep in code:
                return False
        return True

    def _sync_cd(self, code):
        try:
            r = subprocess.run(
                f"{code} && pwd",
                shell=True,
                executable=self.shell,
                capture_output=True,
                text=True,
                timeout=5,
                cwd=os.getcwd(),
                env=self.env,
            )
            if r.returncode == 0:
                return r.stdout.strip().split("\n")[-1]
        except Exception:
            pass
        return None
