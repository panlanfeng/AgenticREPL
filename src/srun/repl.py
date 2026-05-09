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
    tab_completer = _tab_completer
    readline.set_completer(_tab_completer)
    readline.parse_and_bind("tab: complete")
except ImportError:
    pass

from .dispatch import dispatcher
from .context import state
from .logo import LOGO
from .llm import llm
from .repair import repairer
from .danger import check_danger
from .config import init as config_init
from .executors.python_exec import PythonExecutor
from .executors.shell_exec import ShellExecutor
from .executors.r_exec import RExecutor
from .user_config import get as config_get


def main():
    config_init()
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
            lang = state.current_language
            if lang == "python":
                prompt = "\033[1;35mpython>\033[0m "
            elif lang == "r":
                prompt = "\033[1;34mR>\033[0m "
            elif sh_exec.remote:
                prompt = f"{sh_exec.remote}\n\033[1;32msrun>\033[0m "
            else:
                prompt = f"{os.getcwd()}\n\033[1;32msrun>\033[0m "
            user_input = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            lang = state.current_language
            if lang == "python":
                state.current_language = "shell"
                print("Exited Python session.")
                continue
            if lang == "r":
                state.current_language = "shell"
                print("Exited R session.")
                continue
            if sh_exec.remote:
                sh_exec.disconnect()
                print("Disconnected from remote.")
                continue
            break

        if not user_input:
            continue

        lang = state.current_language
        if lang == "python":
            if user_input.lower() in ("exit()", "quit()", "exit", "quit"):
                state.current_language = "shell"
                print("Exited Python session.")
                state.save()
                continue
        elif lang == "r":
            if user_input.lower() in ("exit()", "quit()", "exit", "quit", "q()"):
                state.current_language = "shell"
                print("Exited R session.")
                state.save()
                continue
        elif sh_exec.remote and user_input.lower() in ("exit", "quit"):
            sh_exec.disconnect()
            print("Disconnected from remote.")
            continue
        elif lang == "shell" and user_input.lower() in ("exit", "quit"):
            break

        if lang == "shell" and not sh_exec.remote and user_input.lower() == "python":
            state.current_language = "python"
            print("Entered Python session (type 'exit()' to leave).")
            state.save()
            continue

        if lang == "shell" and not sh_exec.remote and user_input.lower() == "r":
            state.current_language = "r"
            if r_exec.available:
                print("Entered R session (type 'exit()' to leave).")
            else:
                print("R not available. Install rpy2 to enable R support.")
                state.current_language = "shell"
            state.save()
            continue

        if lang == "shell" and not sh_exec.remote and user_input.startswith("ssh "):
            result = _handle_ssh(user_input, sh_exec)
            if result is not None:
                print(result)
                state.save()
                continue

        start = time.perf_counter()
        category = dispatcher.classify(user_input)

        lang = state.current_language
        if category in ("shell", "python", "r") and lang in ("python", "r") and category != lang:
            category = "unknown"

        result = execute(category, user_input, py_exec, sh_exec, r_exec)

        if result.get("llm_used") and config_get("confirm_llm_code"):
            code = result.get("fixed_code") or result.get("generated_code") or ""
            if code:
                result = _confirm_execution(code, result, sh_exec, py_exec, r_exec)
        elapsed_ms = (time.perf_counter() - start) * 1000

        _log_turn(user_input, result, elapsed_ms)
        print_result(result, elapsed_ms)
        state.save()


def _handle_ssh(command, sh_exec):
    error = sh_exec.connect_ssh(command)
    if error:
        return error
    return f"Connected to {sh_exec.remote} (type 'exit' to disconnect)"


def _confirm_execution(code, result, sh_exec, py_exec, r_exec):
    print(f"\033[1;33m⟳  {code}\033[0m")
    print("\033[2m[Enter] execute  [Ctrl+C] skip\033[0m")
    try:
        response = input().strip()
    except (EOFError, KeyboardInterrupt):
        return {"success": True, "output": "skipped", "llm_used": True, "language": result.get("language")}
    if response == "":
        lang = result.get("language", "shell")
        if lang == "python":
            ok, out, *_ = py_exec.execute(code)
        elif lang == "r":
            ok, out, *_ = r_exec.execute(code)
        else:
            ok, out, *_ = sh_exec.execute(code)
        return {"success": ok, "output": out.strip() if out else "", "llm_used": True, "language": lang, "fixed_code": code}
    return {"success": True, "output": "skipped", "llm_used": True, "language": result.get("language")}


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
    return False


def _retry_loop(initial_input, executor, language, max_rounds=None, initial_llm=False):
    if max_rounds is None:
        max_rounds = config_get("max_retry_rounds")
    is_shell = language == "shell"
    current_input = initial_input
    llm_used = initial_llm
    repair_errors = []
    attempts = []
    last_summary = None

    for attempt in range(max_rounds):
        success, output, *rest = executor.execute(current_input)
        stderr = rest[0] if len(rest) > 0 else ""
        ok = success
        if is_shell and ok and _has_stderr_errors(stderr):
            ok = False

        if ok:
            state.last_lang = language
            state.current_language = language
            out = output.strip() if output else ""
            return {
                "success": True, "output": out,
                "llm_used": llm_used, "language": language,
                "fixed_code": current_input if current_input != initial_input else None,
                "repair_errors": repair_errors,
                "summary": last_summary,
            }

        if attempt >= max_rounds - 1:
            summary = f"Gave up after {max_rounds} {'rounds' if max_rounds > 1 else 'round'}."
            if attempts:
                summary += " Tried: " + "; ".join(a[:60] for a in attempts) + "."
            result = {
                "success": False, "output": output.strip() if output else "",
                "llm_used": llm_used, "language": language,
                "summary": summary,
            }
            if is_shell:
                result["repair_errors"] = repair_errors
            return result

        error_msg = output if not success else stderr
        fixed, used_llm, summary = try_repair(current_input, error_msg, language)
        if summary:
            last_summary = summary
        if fixed is None or fixed == current_input:
            summary = "Unable to fix this command."
            if attempts:
                summary += " Tried: " + "; ".join(a[:60] for a in attempts) + "."
            result = {
                "success": False, "output": output.strip() if output else "",
                "llm_used": llm_used, "language": language,
                "summary": summary,
            }
            if is_shell:
                result["repair_errors"] = repair_errors
            return result

        if is_shell:
            repair_errors.append(error_msg)
        attempts.append(fixed)
        llm_used = llm_used or used_llm

        if is_shell:
            danger, desc = check_danger(fixed)
            if danger:
                return {
                    "success": False,
                    "output": f"BLOCKED: {desc}\nFixed: {fixed}",
                    "llm_used": llm_used, "language": language,
                    "repair_errors": repair_errors,
                }

        current_input = fixed

    return {"success": False, "output": "max retries", "llm_used": llm_used, "language": language}


def execute(category, user_input, py_exec, sh_exec, r_exec):
    EXEC_MAP = {
        "shell": (sh_exec, "shell"),
        "python": (py_exec, "python"),
        "r": (r_exec, "r"),
    }

    if category == "empty":
        return {"success": True, "output": "", "llm_used": False, "language": None}

    if category == "python":
        return _retry_loop(user_input, py_exec, "python")

    if category == "r":
        return _retry_loop(user_input, r_exec, "r")

    if category == "shell":
        return _retry_loop(user_input, sh_exec, "shell")

    summary, tool_calls = llm.run(user_input)
    if tool_calls is None and summary is None:
        return {
            "success": False,
            "output": "Unable to understand this input. Try rephrasing.",
            "llm_used": True,
            "summary": "The input could not be translated into an executable command.",
        }
    if not tool_calls:
        return {
            "success": True,
            "output": "",
            "llm_used": True,
            "language": "text",
        }

    for tc in tool_calls:
        if isinstance(tc, dict):
            cmd = tc["command"]
            lang = tc.get("language", state.current_language if state.current_language in EXEC_MAP else "shell")
        else:
            cmd = tc
            lang = state.current_language if state.current_language in EXEC_MAP else "shell"
        executor, lang_name = EXEC_MAP.get(lang, (sh_exec, "shell"))
        result = _retry_loop(cmd, executor, lang_name, initial_llm=True)
        if not result["success"]:
            result["summary"] = summary
            return result

    first_cmd = tool_calls[0]
    gen_code = first_cmd["command"] if isinstance(first_cmd, dict) else first_cmd
    result = {"success": True, "output": "", "llm_used": True,
              "language": "shell",
              "generated_code": gen_code if len(tool_calls) == 1 else None,
              "summary": summary}
    return result


def try_repair(original, error_msg, language):
    for _ in range(3):
        fixed, summary = repairer.fix(original, error_msg, language)
        if fixed is None:
            return None, False, None
        if fixed != original:
            return fixed, summary is not None, summary
        original = fixed
    return None, False, None


def print_result(result, elapsed_ms):
    output = result.get("output", "")
    llm = result.get("llm_used", False)
    fixed = result.get("fixed_code")
    generated = result.get("generated_code")
    gave_up = result.get("success") is False and result.get("summary")

    if gave_up:
        print(f"\033[2m   {result['summary']}\033[0m")
    if fixed:
        print(f"\033[1;33m⟳  {fixed}\033[0m")
    if generated:
        print(f"\033[1;33m⟳  {generated}\033[0m")

    if output:
        print(output.rstrip())

    llm_tag = " \033[1;33m+LLM\033[0m" if llm else ""
    from .llm import llm as llm_mod
    stats = llm_mod.cache_stats
    cache_part = ""
    if stats["total_tokens"] > 0:
        cache_part = f" cache:{stats['rate']:.0%}"
    print(f"{llm_tag} \033[2m{elapsed_ms:.0f}ms{cache_part}\033[0m")


if __name__ == "__main__":
    main()
