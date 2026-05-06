import ast
import json
import re
from openai import OpenAI
from .config import config
from .context import state
from .prompts import DISPATCH_PROMPT_SHORT

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
    "ll", "la", "l",
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
            return True
    return False


class Dispatcher:
    def __init__(self):
        self.client = None
        if config.has_llm:
            self.client = OpenAI(api_key=config.api_key, base_url=config.api_base)

    def classify(self, user_input):
        stripped = user_input.strip()
        if not stripped:
            return "empty"

        if _looks_like_pseudocode(stripped):
            return "unknown"

        if self._is_shell(stripped):
            return "shell"

        if self._is_python(stripped):
            return "python"

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

    def _parse_response(self, text):
        text = text.strip()
        try:
            parsed = json.loads(text)
            return parsed.get("language", "shell"), parsed.get("code")
        except json.JSONDecodeError:
            pass
        block = re.search(r"```(?:bash|shell|sh)\s*\n?(.+?)```", text, re.DOTALL)
        if block:
            return "shell", block.group(1).strip()
        block = re.search(r"```(?:python|py)\s*\n?(.+?)```", text, re.DOTALL)
        if block:
            return "python", block.group(1).strip()
        return "shell", None

    def llm_dispatch(self, user_input):
        if not self.client:
            return "shell", user_input

        context_str = state.llm_context()
        prompt = DISPATCH_PROMPT_SHORT.format(context=context_str)

        try:
            resp = self.client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_input},
                ],
                temperature=0.0,
                max_tokens=500,
            )
            content = resp.choices[0].message.content.strip()
            lang, code = self._parse_response(content)
            if code:
                return lang, code
            return "shell", user_input
        except Exception as e:
            state.last_dispatch_error = str(e)
            return "shell", user_input


dispatcher = Dispatcher()
