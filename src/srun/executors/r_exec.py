import subprocess
import shutil
import os
import time


class RExecutor:
    def __init__(self):
        self._available = shutil.which("R") is not None
        self._process = None
        self._read_timeout = 30

    @property
    def available(self):
        return self._available

    def _ensure_process(self):
        if self._process is None or self._process.poll() is not None:
            self._process = subprocess.Popen(
                ["R", "--slave", "--no-save", "--no-restore"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
                cwd=os.getcwd(),
            )
            os.set_blocking(self._process.stdout.fileno(), False)

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
            deadline = time.monotonic() + self._read_timeout

            while time.monotonic() < deadline:
                try:
                    line = self._process.stdout.readline()
                except Exception:
                    time.sleep(0.1)
                    self._process.poll()
                    if self._process.returncode is not None:
                        break
                    continue
                if not line:
                    time.sleep(0.05)
                    self._process.poll()
                    if self._process.returncode is not None:
                        break
                    continue
                if "__SRUN_END__" in line:
                    out = "\n".join(output_lines)
                    self._process.poll()
                    rc = self._process.returncode
                    ok = rc is None or rc == 0
                    if not ok:
                        self._process = None
                    return ok, out, out
                stripped = line.rstrip("\n")
                if stripped and not stripped.startswith("> ") and not stripped.startswith("+ "):
                    output_lines.append(stripped)

            if output_lines:
                out = "\n".join(output_lines)
                self._process.poll()
                rc = self._process.returncode
                ok = rc is None or rc == 0
                if not ok:
                    self._process = None
                return ok, out, out

            try:
                self._process.kill()
                self._process.wait(timeout=2)
            except Exception:
                pass
            self._process = None
            return False, "R command timed out", ""
        except Exception as e:
            if self._process:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None
            return False, f"R Error: {e}", ""
