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

    def _tab_completer(text, state):
        import glob
        if state == 0:
            expanded = os.path.expanduser(text)
            if os.path.isdir(expanded) and not expanded.endswith("/"):
                expanded += "/"
            candidates = glob.glob(expanded + "*")
            candidates = sorted(c for c in candidates if not os.path.basename(c).startswith("."))
            candidates += sorted(c for c in glob.glob(expanded + ".*") if os.path.basename(c).startswith("."))
            _tab_completer._matches = candidates
        try:
            match = _tab_completer._matches[state]
            if os.path.isdir(match):
                return match + "/"
            return match
        except IndexError:
            return None

    _tab_completer._matches = []
    readline.set_completer(_tab_completer)
    readline.parse_and_bind("tab: complete")
except ImportError:
    pass

from .dispatch import dispatcher
from .context import state
from .logo import LOGO
from .llm import llm
from .repair import repairer, apply_quick_fix
from .danger import check_danger
from .executors.python_exec import PythonExecutor
from .executors.shell_exec import ShellExecutor
from .executors.r_exec import RExecutor


def main():
    py_exec = PythonExecutor()
    sh_exec = ShellExecutor()
    r_exec = RExecutor()
    state.reset_session()

    print(LOGO)
    print(f"LLM: {'available' if llm.client else 'unavailable'}")

    if len(sys.argv) > 1:
        _run_file(sys.argv[1], py_exec, sh_exec, r_exec)
        return

    _run_repl(py_exec, sh_exec, r_exec)


def _run_file(path, py_exec, sh_exec, r_exec):
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        return
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]
    if not lines:
        print("No commands found in file")
        return
    print(f"Executing {len(lines)} commands from {path}\n")
    failed = 0
    llm_calls = 0
    total_start = time.perf_counter()
    for line in lines:
        start = time.perf_counter()
        category = dispatcher.classify(line)
        result = execute(category, line, py_exec, sh_exec, r_exec)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if result.get("llm_used"):
            llm_calls += 1
        if not result["success"]:
            failed += 1
        print(f"{'✓' if result['success'] else '✗'} {line[:50]:50s} {elapsed_ms:5.0f}ms")
        if result.get("fixed_code"):
            print(f"  → {result['fixed_code']}")
        elif result.get("generated_code"):
            print(f"  → {result['generated_code']}")
    total_ms = int((time.perf_counter() - total_start) * 1000)
    print(f"\n{len(lines)-failed}/{len(lines)} passed, {llm_calls} LLM calls, {total_ms}ms total")


def _run_repl(py_exec, sh_exec, r_exec):
    state.save()

    while True:
        try:
            remote = sh_exec.remote
            if remote:
                prompt = f"{remote}\n\033[1;32msrun>\033[0m "
            else:
                prompt = f"{os.getcwd()}\n\033[1;32msrun>\033[0m "
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

        _log_turn(user_input, result, elapsed_ms)
        print_result(result, elapsed_ms)
        state.save()


def _handle_ssh(command, sh_exec):
    error = sh_exec.connect_ssh(command)
    if error:
        return error
    return f"Connected to {sh_exec.remote} (type 'exit' to disconnect)"


def _log_turn(user_input, result, elapsed_ms):
    fixed = result.get("fixed_code")
    generated = result.get("generated_code")
    llm_used = result.get("llm_used", False)
    success = result.get("success", False)
    lang = result.get("language", "")
    repair_errors = result.get("repair_errors", [])

    if llm_used:
        code = fixed or generated or user_input
        if code:
            error_text = "\n".join(repair_errors) if repair_errors else None
            state.add_conversation_turn(
                user_msg=f"The user typed: {user_input}",
                assistant_code=code,
                error_output=error_text,
            )

    state.log_entry(
        type="llm_repair" if (llm_used and fixed) else
             "llm_dispatch" if (llm_used and generated) else "fast",
        input=user_input,
        code=fixed or generated or user_input,
        llm_generated=fixed or generated,
        language=lang,
        success=success,
        elapsed_ms=int(elapsed_ms),
        error="\n".join(repair_errors) if repair_errors else None,
    )


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
    repair_errors = []
    for attempt in range(max_rounds):
        success, output, *rest = sh_exec.execute(current_input)
        stderr = rest[0] if len(rest) > 0 else ""
        if success and not _has_stderr_errors(stderr):
            state.last_lang = "shell"
            return {
                "success": True, "output": output.strip(),
                "llm_used": llm_used, "language": "shell",
                "fixed_code": current_input if current_input != initial_input else None,
                "repair_errors": repair_errors,
            }
        if attempt >= max_rounds - 1:
            return {
                "success": False, "output": output.strip(),
                "llm_used": llm_used, "language": "shell",
                "repair_errors": repair_errors,
            }
        error_msg = output if not success else stderr
        fixed, used_llm, _ = try_repair(current_input, error_msg, "shell")
        if fixed is None or fixed == current_input:
            return {
                "success": False, "output": output.strip(),
                "llm_used": llm_used, "language": "shell",
                "repair_errors": repair_errors,
            }
        repair_errors.append(error_msg)
        llm_used = llm_used or used_llm
        danger, desc = check_danger(fixed)
        if danger:
            return {
                "success": False,
                "output": f"BLOCKED: {desc}\nFixed: {fixed}",
                "llm_used": llm_used, "language": "shell",
                "repair_errors": repair_errors,
            }
        current_input = fixed
    return {"success": False, "output": "max retries", "llm_used": llm_used, "language": "shell", "repair_errors": repair_errors}


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
                    "fixed_code": current_input if current_input != initial_input else None,
                }
        if attempt >= max_rounds - 1:
            return {
                "success": False, "output": output.strip(),
                "llm_used": llm_used, "language": "python",
            }
        fixed, used_llm, _ = try_repair(current_input, output, "python")
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

    lang, code, summary = dispatcher.llm_dispatch(user_input)
    if getattr(state, "last_dispatch_error", None):
        return {
            "success": False,
            "output": f"LLM dispatch failed: {state.last_dispatch_error}",
            "llm_used": True,
            "language": lang,
        }

    return execute_dispatched(lang, code, summary, user_input, py_exec, sh_exec, r_exec)


def execute_dispatched(lang, code, summary, original_input, py_exec, sh_exec, r_exec):
    if lang == "python":
        result = _retry_loop_python(code, py_exec, initial_llm=True)
    elif lang == "r":
        if not r_exec.available:
            return {
                "success": False,
                "output": "R not available (install rpy2)",
                "llm_used": True,
                "language": "r",
            }
        result = _retry_loop_r(code, original_input, r_exec)
    else:
        result = _retry_loop_shell(code, sh_exec, initial_llm=True)
    if summary:
        result["summary"] = summary
    return result


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
        fixed, _, _ = try_repair(original_input, output, "r")
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
        return quick, False, None

    for _ in range(3):
        fixed, summary = repairer.fix(original, error_msg, language)
        if fixed is None:
            return None, False, None
        if fixed != original:
            return fixed, True, summary
        original = fixed
    return None, False, None


def print_result(result, elapsed_ms):
    output = result.get("output", "")
    llm = result.get("llm_used", False)
    lang = result.get("language", "")
    summary = result.get("summary")
    fixed = result.get("fixed_code")
    generated = result.get("generated_code")

    if fixed:
        print(f"\033[1;33m⟳  {fixed}\033[0m")
    if generated:
        print(f"\033[1;33m⟳  {generated}\033[0m")
    if summary:
        print(f"\033[2m   {summary}\033[0m")

    if output:
        print(output.rstrip())

    timing = f"\033[2m[{lang}]\033[0m" if lang else ""
    llm_tag = " \033[1;33m+LLM\033[0m" if llm else ""
    from .llm import llm
    stats = llm.cache_stats
    cache_part = ""
    if stats["total_tokens"] > 0:
        cache_part = f" cache:{stats['rate']:.0%}"
    print(f"{timing}{llm_tag} \033[2m{elapsed_ms:.0f}ms{cache_part}\033[0m")


if __name__ == "__main__":
    main()
