import subprocess
import os
import shutil

SHELL_BIN = os.environ.get("SHELL", "/bin/zsh")


class ShellExecutor:
    def __init__(self):
        self.shell = shutil.which(SHELL_BIN) or SHELL_BIN

    def execute(self, code):
        stripped = code.strip()
        try:
            result = subprocess.run(
                code,
                shell=True,
                executable=self.shell,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=os.getcwd(),
            )
        except subprocess.TimeoutExpired:
            return False, "Command timed out (30s)", "", -1
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
            )
            if r.returncode == 0:
                return r.stdout.strip().split("\n")[-1]
        except Exception:
            pass
        return None
