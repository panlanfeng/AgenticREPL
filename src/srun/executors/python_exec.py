import ast
import sys
import io
from ..context import state


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

    def execute(self, code):
        code = self._resolve_columns(code)
        stdout = io.StringIO()
        stderr = io.StringIO()
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = stdout
        sys.stderr = stderr
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
        except Exception as e:
            return False, f"{type(e).__name__}: {e}", stdout.getvalue(), stderr.getvalue(), 1
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        output = stdout.getvalue()
        error_output = stderr.getvalue()
        self._track_vars()
        return True, output, error_output, 0

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
