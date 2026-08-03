import ast
import sys
import io
import signal
import threading
from ..context import state
from ..user_config import get as _config_get


class _ExecTimeout(BaseException):
    """Raised when Python code exceeds the execution timeout.
    Inherits BaseException so user code cannot swallow it with `except Exception`."""
    pass


def _timeout_handler(signum, frame):
    raise _ExecTimeout("Python execution timed out")


class PythonExecutor:
    def __init__(self):
        self.namespace = {"__builtins__": __builtins__}
        self._init_namespace()

    def _init_namespace(self):
        try:
            import pandas as pd
            import numpy as np
            self.namespace["pd"] = pd
            self.namespace["np"] = np
        except ImportError:
            pass

    def _resolve_columns(self, code):
        columns = state.get_available_columns()
        if not columns:
            return code
        try:
            tree = ast.parse(code, mode="exec")
            transformer = ColumnResolver(columns)
            transformed = transformer.visit(tree)
            ast.fix_missing_locations(transformed)
            return ast.unparse(transformed)
        except SyntaxError:
            return code

    def execute(self, code, timeout=None):
        if timeout is None:
            try:
                timeout = int(_config_get("exec_timeout_seconds") or 0)
            except Exception:
                timeout = 0
        code = self._resolve_columns(code)
        stdout = io.StringIO()
        stderr = io.StringIO()
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = stdout
        sys.stderr = stderr
        old_handler = None
        # SIGALRM only works from the main thread; otherwise fall back to no timeout
        if (timeout and timeout > 0 and hasattr(signal, "setitimer")
                and threading.current_thread() is threading.main_thread()):
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.setitimer(signal.ITIMER_REAL, timeout)
        try:
            tree = ast.parse(code, mode="exec")
            is_expr = (
                len(tree.body) == 1
                and isinstance(tree.body[0], ast.Expr)
                and not isinstance(tree.body[0].value, ast.Constant)
            )
            if is_expr:
                expr_code = ast.unparse(tree.body[0].value)
                result = eval(expr_code, self.namespace)
                if result is not None:
                    print(repr(result))
            else:
                exec(code, self.namespace)
        except _ExecTimeout:
            return (False,
                    f"Python execution timed out after {timeout}s — code interrupted. "
                    "Note: child threads/subprocesses may still be running.",
                    stdout.getvalue(), stderr.getvalue(), 124, {"timed_out": True})
        except Exception as e:
            return False, f"{type(e).__name__}: {e}", stdout.getvalue(), stderr.getvalue(), 1, {}
        finally:
            if old_handler is not None:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old_handler)
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        output = stdout.getvalue()
        error_output = stderr.getvalue()
        self._track_vars()
        return True, output, error_output, 0, {}

    def _track_vars(self):
        try:
            import pandas as pd
            for name, val in self.namespace.items():
                if name.startswith("_"):
                    continue
                if isinstance(val, pd.DataFrame):
                    state.add_var(
                        name,
                        {
                            "type": "DataFrame",
                            "columns": list(val.columns),
                            "rows": len(val),
                            "dtypes": {k: str(v) for k, v in val.dtypes.items()},
                        },
                    )
                    if state.active_df is None:
                        state.set_active(name)
                elif isinstance(val, pd.Series):
                    state.add_var(
                        name,
                        {
                            "type": "Series",
                            "length": len(val),
                            "dtype": str(val.dtype),
                        },
                    )
                elif isinstance(val, (list, tuple, set)):
                    state.add_var(name, {"type": type(val).__name__, "length": len(val)})
                elif isinstance(val, (int, float, str, bool)):
                    state.add_var(name, {"type": type(val).__name__})
        except ImportError:
            pass


class ColumnResolver(ast.NodeTransformer):
    def __init__(self, columns):
        self.columns = set(columns)

    def visit_Name(self, node):
        if node.id in self.columns:
            return ast.copy_location(
                ast.Constant(value=node.id), node
            )
        return node
