import subprocess
import os
import re
import shutil
import time

SHELL_BIN = os.environ.get("SHELL", "/bin/zsh")

_TTY_COMMANDS = {"less", "more", "vim", "vi", "nano", "emacs", "top", "htop",
                  "man", "ssh", "tmux", "screen", "watch", "htop"}


def _needs_tty(command):
    stripped = command.strip()
    if not stripped:
        return False
    base = stripped.split()[0]
    if base in _TTY_COMMANDS:
        return True
    if base == "git":
        parts = stripped.split()
        if len(parts) > 1 and parts[1] in ("log", "diff", "show", "blame", "reflog"):
            return True
    return False


def _build_env():
    env = os.environ.copy()
    env["CLICOLOR"] = "1"
    env["CLICOLOR_FORCE"] = "1"
    if "LSCOLORS" not in env:
        env["LSCOLORS"] = "GxFxCxDxBxegedabagaced"
    return env


_SSH_RE = re.compile(r"^ssh(?:\s+(?:-[^\s]+(?:\s+[^\s]+)?))*\s+([^\s]+)$")
_MARKER = "__SRUN_SSH_DONE__"


class ShellExecutor:
    def __init__(self):
        self.shell = shutil.which(SHELL_BIN) or SHELL_BIN
        self.env = _build_env()
        self._ssh_process = None  # persistent SSH shell session
        self._remote_label = None

    @property
    def remote(self):
        return self._remote_label

    # ── SSH connection ────────────────────────────────────────────

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
        self._remote_label = target

        # Open a persistent SSH shell session (like R executor pattern)
        try:
            ssh_cmd = ["ssh", "-o", "LogLevel=QUIET", "-T", target]
            self._ssh_process = subprocess.Popen(
                ssh_cmd,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
                cwd=os.getcwd(),
            )
            os.set_blocking(self._ssh_process.stdout.fileno(), False)
            # Warm up — send a no-op to confirm connection
            ok, out, _ = self._send_ssh_command("echo ok", timeout=10)
            if not ok or "ok" not in out:
                self._ssh_process.kill()
                self._ssh_process = None
                self._remote_label = None
                return f"SSH connection failed:\n{out}"
            return None
        except FileNotFoundError:
            self._remote_label = None
            return "SSH: ssh command not found"
        except Exception as e:
            self._ssh_process = None
            self._remote_label = None
            return f"SSH error: {e}"

    def disconnect(self):
        if self._ssh_process and self._ssh_process.poll() is None:
            try:
                self._ssh_process.stdin.write("exit\n")
                self._ssh_process.stdin.flush()
                self._ssh_process.wait(timeout=3)
            except Exception:
                self._ssh_process.kill()
        self._ssh_process = None
        self._remote_label = None

    def _send_ssh_command(self, cmd, timeout=30):
        """Send a command to the persistent SSH session and read output until marker.
        Returns (ok, output, exit_code)."""
        if not self._ssh_process or self._ssh_process.poll() is not None:
            return False, "SSH session disconnected", -1
        try:
            # Append marker with exit code capture
            self._ssh_process.stdin.write(f"{cmd}\necho {_MARKER} $?\n")
            self._ssh_process.stdin.flush()
        except Exception:
            return False, "SSH write error", -1

        output_lines = []
        exit_code = -1
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                line = self._ssh_process.stdout.readline()
            except Exception:
                time.sleep(0.05)
                self._ssh_process.poll()
                if self._ssh_process.returncode is not None:
                    break
                continue
            if not line:
                time.sleep(0.05)
                self._ssh_process.poll()
                if self._ssh_process.returncode is not None:
                    break
                continue
            if _MARKER in line:
                # Parse exit code from: __SRUN_SSH_DONE__ 0
                parts = line.strip().split()
                if len(parts) >= 2 and parts[-1].lstrip("-").isdigit():
                    exit_code = int(parts[-1])
                out = "\n".join(output_lines)
                return True, out, exit_code
            stripped = line.rstrip("\n")
            if stripped:
                output_lines.append(stripped)
        return False, "SSH command timed out", -1

    # ── Execute ───────────────────────────────────────────────────

    def execute(self, code):
        stripped = code.strip()

        # Remote: send through persistent SSH session
        if self._ssh_process:
            ok, out, exit_code = self._send_ssh_command(stripped)
            if not ok:
                return False, out, out, exit_code
            if exit_code != 0:
                return False, out, out, exit_code
            return True, out, "", 0

        # Local: subprocess.run
        if _needs_tty(stripped):
            rc = subprocess.call(stripped, shell=True, executable=self.shell,
                                  cwd=os.getcwd(), env=self.env)
            return rc == 0, "", "", 0 if rc == 0 else rc

        try:
            result = subprocess.run(
                stripped,
                shell=True,
                executable=self.shell,
                capture_output=True,
                text=True,
                cwd=os.getcwd(),
                env=self.env,
            )
        except Exception as e:
            return False, f"Error: {e}", "", -1

        output = result.stdout
        if result.stderr:
            output += result.stderr

        if self._is_pure_cd(stripped):
            new_cwd = self._sync_cd(stripped)
            if new_cwd:
                os.chdir(new_cwd)

        if result.returncode != 0:
            return False, output, result.stderr, result.returncode
        return True, output, result.stderr, 0

    # ── Helpers ───────────────────────────────────────────────────

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
