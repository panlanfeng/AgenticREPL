import subprocess
import shutil
import os


class RExecutor:
    def __init__(self):
        self._available = shutil.which("R") is not None
        self._process = None

    @property
    def available(self):
        return self._available

    def _ensure_process(self):
        if self._process is None or self._process.poll() is not None:
            self._process = subprocess.Popen(
                ["R", "--slave", "--no-save", "--no-restore"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True,
                cwd=os.getcwd(),
            )

    def execute(self, code):
        if not self.available:
            return False, "R not available (install R)", ""
        try:
            self._ensure_process()
            marker = "cat('\\n__SRUN_END__\\n')"
            cmd = f"{code}\n{marker}\n"
            self._process.stdin.write(cmd)
            self._process.stdin.flush()
            output_lines = []
            for line in self._process.stdout:
                if "__SRUN_END__" in line:
                    break
                stripped = line.rstrip("\n")
                if stripped and not stripped.startswith("> ") and not stripped.startswith("+ "):
                    output_lines.append(stripped)
            out = "\n".join(output_lines)
            ok = "rror:" not in out[:200] and "Error:" not in out[:200]
            if not ok:
                self._process = None
            return ok, out, out
        except Exception as e:
            self._process = None
            return False, f"R Error: {e}", ""
