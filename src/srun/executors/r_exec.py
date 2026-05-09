import subprocess
import shutil
import os
import threading
import queue


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
            result_queue = queue.Queue()

            def _reader():
                try:
                    for line in self._process.stdout:
                        if "__SRUN_END__" in line:
                            result_queue.put(("ok", output_lines))
                            return
                        stripped = line.rstrip("\n")
                        if stripped and not stripped.startswith("> ") and not stripped.startswith("+ "):
                            output_lines.append(stripped)
                    result_queue.put(("eof", output_lines))
                except Exception as e:
                    result_queue.put(("error", str(e)))

            thread = threading.Thread(target=_reader, daemon=True)
            thread.start()
            thread.join(timeout=self._read_timeout)

            if thread.is_alive():
                try:
                    self._process.kill()
                except Exception:
                    pass
                self._process = None
                return False, "R command timed out", ""

            if result_queue.empty():
                return False, "R read error", ""

            status, result = result_queue.get()
            if status == "error":
                return False, f"R read error: {result}", ""

            out = "\n".join(result)
            self._process.poll()
            rc = self._process.returncode
            ok = rc is None or rc == 0
            if not ok:
                self._process = None
            return ok, out, out
        except Exception as e:
            if self._process:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None
            return False, f"R Error: {e}", ""
