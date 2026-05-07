import os
import sys
import time
import atexit

try:
    import readline
    HISTFILE = os.path.join(os.path.expanduser("~"), ".srun", "history")
    try:
        readline.read_history_file(HISTFILE)
    except FileNotFoundError:
        pass
    readline.set_history_length(1000)
    atexit.register(readline.write_history_file, HISTFILE)
except ImportError:
    pass

from .dispatch import dispatcher
from .context import state
from .repair import repairer, apply_quick_fix
from .danger import check_danger
from .executors.python_exec import PythonExecutor
from .executors.shell_exec import ShellExecutor
from .executors.r_exec import RExecutor


def main():
    py_exec = PythonExecutor()
    sh_exec = ShellExecutor()
    r_exec = RExecutor()

    print("srun - Smart Run (type 'exit' to quit)")
    print(f"LLM: {'available' if dispatcher.client else 'unavailable'}")
    state.save()

    while True:
        try:
            remote = sh_exec.remote
            if remote:
                prompt = f"srun:{remote}$ "
            else:
                prompt = f"srun:{os.getcwd()}$ "
            user_input = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            if sh_exec.remote:
                sh_exec.disconnect()
                print("Disconnected from remote.")
                continue
            break

        if not user_input:
            continue

        if sh_exec.remote and user_input.lower() in ("exit", "quit"):
            sh_exec.disconnect()
            print("Disconnected from remote.")
            continue
        if not sh_exec.remote and user_input.lower() in ("exit", "quit"):
            break

        if not sh_exec.remote and user_input.startswith("ssh "):
            result = _handle_ssh(user_input, sh_exec)
            if result is not None:
                print(result)
                state.save()
                continue

        start = time.perf_counter()
        category = dispatcher.classify(user_input)
        result = execute(category, user_input, py_exec, sh_exec, r_exec)
        elapsed_ms = (time.perf_counter() - start) * 1000

        print_result(result, elapsed_ms)
        state.save()


def _handle_ssh(command, sh_exec):
    error = sh_exec.connect_ssh(command)
    if error:
        return error
    return f"Connected to {sh_exec.remote} (type 'exit' to disconnect)"


def _has_stderr_errors(stderr):
    if not stderr:
        return False
    lower = stderr.lower()
    for pattern in ["no such file", "command not found", "error", "permission denied",
                    "not a directory", "cannot access", "not found"]:
        if pattern in lower:
            return True
    return False


def _retry_loop_shell(initial_input, sh_exec, max_rounds=4, initial_llm=False):
    current_input = initial_input
    llm_used = initial_llm
    for attempt in range(max_rounds):
        success, output, *rest = sh_exec.execute(current_input)
        stderr = rest[0] if len(rest) > 0 else ""
        if success and not _has_stderr_errors(stderr):
            state.last_lang = "shell"
            return {
                "success": True, "output": output.strip(),
                "llm_used": llm_used, "language": "shell",
                "fixed_code": current_input if llm_used else None,
            }
        if attempt >= max_rounds - 1:
            return {
                "success": False, "output": output.strip(),
                "llm_used": llm_used, "language": "shell",
            }
        error_msg = output if not success else stderr
        fixed, used_llm = try_repair(current_input, error_msg, "shell")
        if fixed is None or fixed == current_input:
            return {
                "success": False, "output": output.strip(),
                "llm_used": llm_used, "language": "shell",
            }
        llm_used = llm_used or used_llm
        danger, desc = check_danger(fixed)
        if danger:
            return {
                "success": False,
                "output": f"BLOCKED: {desc}\nFixed: {fixed}",
                "llm_used": llm_used, "language": "shell",
            }
        current_input = fixed
    return {"success": False, "output": "max retries", "llm_used": llm_used, "language": "shell"}


def _retry_loop_python(initial_input, py_exec, max_rounds=4, initial_llm=False):
    current_input = initial_input
    llm_used = initial_llm
    for attempt in range(max_rounds):
        success, output, *extra = py_exec.execute(current_input)
        if success:
            state.last_lang = "python"
            return {
                "success": True, "output": output.strip(),
                "llm_used": llm_used, "language": "python",
                "fixed_code": current_input if llm_used else None,
            }
        if attempt >= max_rounds - 1:
            return {
                "success": False, "output": output.strip(),
                "llm_used": llm_used, "language": "python",
            }
        fixed, used_llm = try_repair(current_input, output, "python")
        if fixed is None or fixed == current_input:
            return {
                "success": False, "output": output.strip(),
                "llm_used": llm_used, "language": "python",
            }
        llm_used = llm_used or used_llm
        current_input = fixed
    return {"success": False, "output": "max retries", "llm_used": True, "language": "python"}


def execute(category, user_input, py_exec, sh_exec, r_exec):
    if category == "empty":
        return {"success": True, "output": "", "llm_used": False, "language": None}

    if category == "python":
        return _retry_loop_python(user_input, py_exec)

    if category == "shell":
        return _retry_loop_shell(user_input, sh_exec)

    lang, code = dispatcher.llm_dispatch(user_input)
    if getattr(state, "last_dispatch_error", None):
        return {
            "success": False,
            "output": f"LLM dispatch failed: {state.last_dispatch_error}",
            "llm_used": True,
            "language": lang,
        }

    return execute_dispatched(lang, code, user_input, py_exec, sh_exec, r_exec)


def execute_dispatched(lang, code, original_input, py_exec, sh_exec, r_exec):
    if lang == "python":
        return _retry_loop_python(code, py_exec, initial_llm=True)

    if lang == "r":
        if not r_exec.available:
            return {
                "success": False,
                "output": "R not available (install rpy2)",
                "llm_used": True,
                "language": "r",
            }
        return _retry_loop_r(code, original_input, r_exec)

    return _retry_loop_shell(code, sh_exec, initial_llm=True)


def _retry_loop_r(initial_input, original_input, r_exec, max_rounds=4):
    current_input = initial_input
    for attempt in range(max_rounds):
        success, output, *_ = r_exec.execute(current_input)
        if success:
            state.last_lang = "r"
            return {
                "success": True, "output": output.strip(),
                "llm_used": True, "language": "r",
                "fixed_code": current_input,
            }
        if attempt >= max_rounds - 1:
            return {
                "success": False, "output": output.strip(),
                "llm_used": True, "language": "r",
            }
        fixed, _ = try_repair(original_input, output, "r")
        if fixed is None or fixed == current_input:
            return {
                "success": False, "output": output.strip(),
                "llm_used": True, "language": "r",
            }
        current_input = fixed
    return {"success": False, "output": "max retries", "llm_used": True, "language": "r"}


def try_repair(original, error_msg, language):
    quick = apply_quick_fix(original, error_msg)
    if quick:
        return quick, False

    for _ in range(3):
        fixed = repairer.fix(original, error_msg, language)
        if fixed is None:
            return None, False
        if fixed != original:
            return fixed, True
        original = fixed
    return None, False


def print_result(result, elapsed_ms):
    output = result.get("output", "")
    llm = result.get("llm_used", False)
    lang = result.get("language", "")
    fixed = result.get("fixed_code")
    generated = result.get("generated_code")

    if fixed:
        print(f"\033[2m→ {fixed}\033[0m")
    if generated:
        print(f"\033[2m→ {generated}\033[0m")

    if output:
        print(output.rstrip())

    timing = f"[{lang}]" if lang else ""
    llm_tag = " +LLM" if llm else ""
    print(f"{timing}{llm_tag} {elapsed_ms:.0f}ms")


if __name__ == "__main__":
    main()
