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
    readline.parse_and_bind('"\\C-z": "\\C-d"')
except ImportError:
    pass

import re as _re


def _rl_prompt(s):
    """Wrap ANSI escapes so readline knows they are zero-width."""
    return _re.sub(r"(\x1b\[[0-9;]*m)", r"\001\1\002", s)


from .dispatch import dispatcher
from .context import state
from .logo import LOGO
from .llm import llm
from .repair import repairer
from .danger import check_danger
from .config import config, init as config_init
from .executors.python_exec import PythonExecutor
from .executors.shell_exec import ShellExecutor
from .executors.r_exec import RExecutor
from .user_config import get as config_get


def _is_incomplete(code, language):
    """Return True if the code is an incomplete statement that needs more input."""
    code = code.strip()
    if not code:
        return False

    if language == "python":
        try:
            compile(code, "<input>", "exec")
            return False
        except SyntaxError as e:
            msg = str(e)
            if "unexpected EOF" in msg or "expected" in msg:
                return True
            return False
        except Exception:
            return False

    if language == "r":
        stripped = code.rstrip()
        if not stripped:
            return False
        open_parens = stripped.count("(") - stripped.count(")")
        open_braces = stripped.count("{") - stripped.count("}")
        open_brackets = stripped.count("[") - stripped.count("]")
        if open_parens > 0 or open_braces > 0 or open_brackets > 0:
            return True
        continuation_tokens = ("+", "-", "*", "/", "&", "|", "<", ">", "=",
                               "%>%", "|>", "%<>%", "%$%", "%%",
                               "in", "else", "then")
        for tok in sorted(continuation_tokens, key=len, reverse=True):
            if stripped.endswith(tok):
                return True
        return False

    if language == "shell":
        stripped = code.rstrip()
        if not stripped:
            return False
        if stripped.endswith("\\"):
            return True
        if stripped.endswith("|") or stripped.endswith("&"):
            last = stripped.split()[-1] if stripped.split() else ""
            if last in ("|", "||", "|&", "&&"):
                return True
        if stripped.rstrip(";").endswith("do") or stripped.rstrip(";").endswith("then"):
            return True
        return False

    return False


def main():
    config_init()
    py_exec = PythonExecutor()
    sh_exec = ShellExecutor()
    r_exec = RExecutor()
    state.reset_session()

    print(LOGO)
    if llm.client:
        print(f"LLM: ready ({config.model})")
    else:
        print("LLM: no API key — set api_key in ~/.srun/user_config.json or export DEEPSEEK_API_KEY")

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
    _exit_pending = False

    while True:
        try:
            lang = state.current_language
            if lang == "python":
                prompt = _rl_prompt("\033[1;35mpython>\033[0m ")
            elif lang == "r":
                prompt = _rl_prompt("\033[1;34mR>\033[0m ")
            elif sh_exec.remote:
                prompt = f"{sh_exec.remote}\n{_rl_prompt('\033[1;32msrun>\033[0m ')}"
            else:
                prompt = f"{os.getcwd()}\n{_rl_prompt('\033[1;32msrun>\033[0m ')}"
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
            if _exit_pending:
                break
            _exit_pending = True
            print("Press Ctrl+D again to exit srun.")
            continue

        if not user_input:
            continue

        # --- built-in: configure API key ---
        if not llm.client and user_input.lower() in ("configure", "configure-api", "setup api", "set api key"):
            _print_config_help()
            continue

        lang = state.current_language

        # --- accumulate multi-line input ---
        if _is_incomplete(user_input, lang):
            lines = [user_input]
            cont_prompt = _rl_prompt("... ")
            while True:
                try:
                    next_line = input(cont_prompt).rstrip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    user_input = "\n".join(lines)
                    break
                if not next_line:
                    user_input = "\n".join(lines)
                    break
                lines.append(next_line)
                combined = "\n".join(lines)
                if not _is_incomplete(combined, lang):
                    user_input = combined
                    break
            else:
                user_input = "\n".join(lines)

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
        try:
            result = execute(category, user_input, py_exec, sh_exec, r_exec)
        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            result = {"success": False, "output": f"Internal error: {e}", "language": state.current_language, "llm_used": False}

        if result.get("llm_used") and config_get("confirm_llm_code"):
            code = result.get("fixed_code") or result.get("generated_code") or ""
            if code:
                result = _confirm_execution(code, result, sh_exec, py_exec, r_exec)
        elapsed_ms = (time.perf_counter() - start) * 1000

        _log_turn(user_input, result, elapsed_ms)
        print_result(result, elapsed_ms)
        state.save()


def _print_config_help():
    print("Configure your API key in ~/.srun/user_config.json:")
    print("  {")
    print('    "api_key": "sk-...",')
    print('    "api_base": "https://api.openai.com/v1",  // optional, defaults to DeepSeek')
    print('    "api_model": "gpt-4o"                       // optional, defaults to deepseek-chat')
    print("  }")
    print("")
    print("Or export DEEPSEEK_API_KEY in your shell for DeepSeek compatibility.")


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

    if llm_used and not fixed:
        code = generated or user_input
        if code:
            state.add_conversation_turn(
                user_msg=f"The user typed: {user_input}",
                assistant_code=code,
                error_output=None,
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
        output=result.get("output", ""),
    )


def _has_stderr_errors(stderr):
    if not stderr:
        return False
    lower = stderr.lower()
    for pattern in ["error", "traceback", "command not found", "no such file", "cannot access",
                    "permission denied", "fatal", "syntax", "unexpected", "not found", "invalid"]:
        if pattern in lower:
            return True
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

        repair_errors.append(error_msg)
        attempts.append(fixed)
        llm_used = llm_used or used_llm

        if used_llm and fixed:
            state.add_conversation_turn(
                user_msg=f"The user typed: {current_input}",
                assistant_code=fixed,
                error_output=error_msg,
            )

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

    cur = state.current_language
    if category in EXEC_MAP and cur in EXEC_MAP and category != cur and cur != "shell":
        category = "unknown"

    if category == "python":
        return _retry_loop(user_input, py_exec, "python")

    if category == "r":
        return _retry_loop(user_input, r_exec, "r")

    if category == "shell":
        return _retry_loop(user_input, sh_exec, "shell")

    summary, tool_calls = llm.run(user_input)
    if tool_calls is None and summary is None:
        if not llm.client and any(kw in user_input.lower() for kw in ("api key", "api_key", "configure", "setup api", "set up llm")):
            _print_config_help()
            return {"success": True, "output": "", "llm_used": False, "language": "text"}
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
        exec_result = _retry_loop(cmd, executor, lang_name, initial_llm=True)
        if not exec_result["success"]:
            exec_result["summary"] = summary
            return exec_result

    first_cmd = tool_calls[0]
    gen_code = first_cmd["command"] if isinstance(first_cmd, dict) else first_cmd
    return {"success": True, "output": exec_result.get("output", ""), "llm_used": True,
            "language": exec_result.get("language", lang_name),
            "generated_code": gen_code if len(tool_calls) == 1 else None,
            "summary": summary}


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
