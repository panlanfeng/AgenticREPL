import subprocess
import shutil
import os
import time
import threading


def _balance_r_code(code):
    """Add closing delimiters to make R code syntactically complete.
    Prevents R from entering continuation mode on unclosed parens/brackets/braces
    which would cause a 30s timeout instead of a proper error message."""
    pairs = {'(': ')', '[': ']', '{': '}'}
    stack = []
    in_str = None  # None, '"', "'", '`'
    i = 0
    while i < len(code):
        c = code[i]
        if in_str:
            if c == '\\':
                i += 1  # skip escaped char
            elif c == in_str:
                in_str = None
        else:
            if c in ('"', "'", '`'):
                in_str = c
            elif c in pairs:
                stack.append(c)
            elif c in pairs.values():
                if stack and pairs[stack[-1]] == c:
                    stack.pop()
        i += 1
    # Close any unclosed string, then close remaining open delimiters
    closing = (in_str if in_str else '') + ''.join(pairs[c] for c in reversed(stack))
    return code + closing


class RExecutor:
    def __init__(self):
        self._available = shutil.which("R") is not None
        self._process = None
        self._read_timeout = 30
        self._stderr_lines = []
        self._stderr_thread = None

    @property
    def available(self):
        return self._available

    def _ensure_process(self):
        if self._process is None or self._process.poll() is not None:
            self._process = subprocess.Popen(
                ["R", "--slave", "--no-save", "--no-restore"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, bufsize=1,
                cwd=os.getcwd(),
            )
            os.set_blocking(self._process.stdout.fileno(), False)
            os.set_blocking(self._process.stderr.fileno(), False)
            self._stderr_lines = []
            self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
            self._stderr_thread.start()

    def _read_stderr(self):
        """Read stderr in a background thread so it stays separate from stdout."""
        try:
            while self._process and self._process.poll() is None:
                try:
                    line = self._process.stderr.readline()
                except Exception:
                    break
                if not line:
                    time.sleep(0.05)
                    continue
                stripped = line.rstrip('\n')
                if stripped:
                    self._stderr_lines.append(stripped)
            # Process exited — drain any remaining stderr from the pipe buffer
            if self._process and self._process.stderr:
                for line in self._process.stderr:
                    stripped = line.rstrip('\n')
                    if stripped:
                        self._stderr_lines.append(stripped)
        except Exception:
            pass

    def _collect_stderr(self):
        """Return collected stderr lines and reset for next command.
        Also drains any remaining stderr from the pipe directly."""
        if self._process and self._process.stderr:
            try:
                for line in self._process.stderr:
                    stripped = line.rstrip('\n')
                    if stripped:
                        self._stderr_lines.append(stripped)
            except Exception:
                pass
        lines = self._stderr_lines[:]
        self._stderr_lines = []
        return "\n".join(lines)

    def execute(self, code):
        if not self.available:
            return False, "R not available (install R)", ""
        try:
            self._ensure_process()
            # Balance delimiters so incomplete code won't cause continuation mode
            code = _balance_r_code(code)
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
                    stderr_out = self._collect_stderr()
                    self._process.poll()
                    rc = self._process.returncode
                    ok = rc is None or rc == 0
                    if not ok:
                        self._process = None
                    return ok, out, stderr_out if stderr_out else out
                stripped = line.rstrip("\n")
                if stripped and not stripped.startswith("> ") and not stripped.startswith("+ "):
                    output_lines.append(stripped)

            # Process died or timed out — collect stderr which may have error info
            stderr_out = self._collect_stderr()
            if output_lines or stderr_out:
                out = "\n".join(output_lines)
                if stderr_out:
                    out = out + ("\n" + stderr_out if out else stderr_out)
                self._process.poll()
                rc = self._process.returncode
                ok = rc is None or rc == 0
                if not ok:
                    self._process = None
                return ok, out, stderr_out if stderr_out else out

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
