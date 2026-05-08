import subprocess
import shutil
import os


class RExecutor:
    def __init__(self):
        self._available = shutil.which("Rscript") is not None

    @property
    def available(self):
        return self._available

    def execute(self, code):
        if not self.available:
            return False, "R not available (install R)", ""
        try:
            result = subprocess.run(
                ["Rscript", "-e", code],
                capture_output=True, text=True, timeout=30,
                cwd=os.getcwd(),
            )
            output = result.stdout
            if result.stderr:
                output += result.stderr
            return result.returncode == 0, output.strip(), output
        except Exception as e:
            return False, f"R Error: {e}", ""
