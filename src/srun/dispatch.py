import ast
import re
from .context import state

SHELL_COMMANDS = {
    "ls", "cd", "cat", "grep", "awk", "sed", "sort", "find", "head", "tail",
    "less", "more", "pwd", "mkdir", "rm", "cp", "mv", "echo", "export", "unset",
    "which", "whoami", "date", "df", "du", "ps", "kill", "top", "htop", "man",
    "chmod", "chown", "touch", "wc", "cut", "uniq", "diff", "tar", "gzip", "gunzip",
    "zip", "unzip", "curl", "wget", "ssh", "scp", "git", "docker", "kubectl",
    "python", "python3", "node", "npm", "npx", "pip", "pip3", "make", "cmake",
    "gcc", "g++", "brew", "apt", "yum", "cargo", "rustc", "go", "java", "mvn",
    "gradle", "rsync", "ln", "file", "stat", "env", "history", "clear", "open",
    "source", ".", "code", "vim", "nvim", "nano", "ping", "traceroute", "netstat",
    "ifconfig", "lsof", "mount", "umount", "tee", "xargs", "jq", "yq", "watch",
    "crontab", "alias", "type", "fg", "bg", "jobs", "dirname", "basename",
    "ll", "la", "l", "sudo", "su", "exec", "nice", "nohup", "time", "perf",
}

NL_KEYWORDS = [
    "sort by", "filter by", "filter where", "group by", "groupby",
    "find all", "find the", "show me", "show the", "list all", "list the",
    "get the", "get all", "calculate", "compute", "count of", "number of",
    "rename to", "convert to", "change to", "extract", "download",
    "please", "帮我", "帮我找",
]

SHELL_PATTERNS = [
    r"\|", r">>", r">(?![=])", r"<(?!=)", r"&&", r"\|\|", r";",
    r"\$\(", r"`[^`]+`", r"\\\n",
]


def _looks_like_pseudocode(code):
    lower = code.lower()
    for kw in NL_KEYWORDS:
        if kw in lower:
            idx = lower.index(kw)
            if idx > 0 and lower[idx - 1] == ".":
                continue
            # Ensure keyword is at word boundary, not embedded in a path
            before = idx == 0 or not lower[idx - 1].isalnum()
            after = (idx + len(kw)) >= len(lower) or not lower[idx + len(kw)].isalnum()
            if before and after:
                return True
    return False


def _is_numeric_expr(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return True
    if isinstance(node, ast.BinOp):
        return _is_numeric_expr(node.left) and _is_numeric_expr(node.right)
    if isinstance(node, ast.UnaryOp):
        return _is_numeric_expr(node.operand)
    if isinstance(node, ast.Name):
        return node.id in ("True", "False", "None")
    return False


def _contains_numeric_constant(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return True
    if isinstance(node, ast.BinOp):
        return _contains_numeric_constant(node.left) or _contains_numeric_constant(node.right)
    if isinstance(node, ast.UnaryOp):
        return _contains_numeric_constant(node.operand)
    return False


class Dispatcher:
    def __init__(self):
        pass

    def classify(self, user_input):
        stripped = user_input.strip()
        if not stripped:
            return "empty"
        if _looks_like_pseudocode(stripped):
            return "unknown"
        if self._is_python(stripped):
            return "python"
        if self._is_shell(stripped):
            return "shell"
        return "unknown"

    def _is_python(self, code):
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef,
                                     ast.ClassDef, ast.AnnAssign, ast.Call,
                                     ast.Lambda, ast.DictComp, ast.SetComp,
                                     ast.ListComp, ast.GeneratorExp, ast.For,
                                     ast.While, ast.If, ast.With, ast.Try,
                                     ast.Raise, ast.Assert, ast.Delete)):
                    return True
                if isinstance(node, ast.Attribute):
                    return True
                if isinstance(node, ast.Subscript):
                    return True
                if isinstance(node, ast.Assign):
                    return True
                if isinstance(node, ast.Expr) and len(tree.body) == 1:
                    if isinstance(node.value, (ast.BinOp, ast.UnaryOp)):
                        if _is_numeric_expr(node.value) or _contains_numeric_constant(node.value):
                            return True
            return False
        except SyntaxError:
            return False

    def _is_shell(self, code):
        first_word = code.split()[0].split("/")[-1] if code.split() else ""
        if first_word in SHELL_COMMANDS:
            return True
        for pattern in SHELL_PATTERNS:
            if re.search(pattern, code):
                return True
        if re.match(r"^\.\/", code):
            return True
        if re.match(r"^[a-zA-Z0-9_\-\.]+\s+--?\w+", code):
            return True
        return False


dispatcher = Dispatcher()
