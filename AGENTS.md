
## SRUN Project Overview

This is ligthweight AI agent focusing on efficient code execution with high tolerance to user input or code errors. It provide REPL for interactive usage `srun` and one time execution `srun test.sh`. The SRUN puts command execution first. Normal command will be executed with no extra latency and AI agent is only invoked when user user input is ambiguous or not directly executable. 

## Code style

- Separate REPL UI code from the AI agent. 
- Add clear comments for each main lines. Document every key designs. 
- When user input is clearly code, execute it immediately. LLM intervenes only when the input is ambiguous or when execution fails. 
- When Users execute files or multi-line code — understand the overall user intention first but recommend to execute commands one by one, repairing errors individually.

## Build & Test

```bash
pip install -e ".[dev]"               # install with dev deps
pytest tests/ -v                      # all tests
pytest tests/ -v -m "not slow and not llm"  # fast tests (<5s, no LLM)
pytest tests/ -v -m "llm"             # LLM tests only
pytest tests/test_synthetic_project.py -v -s  # simulated user session
```

Always run fast tests after code changes. Only run LLM tests when LLM-related code changed.

## Usage examples

Type `srun` in your terminal to enter REPL mode.

```
$ srun
srun> ls -la              # normal shell, 0 delay
srun> cat data.csv        # normal shell, 0 delay
srun> 100/4               # expression → 25.0
srun> ll                  # quick fix → ls -la
srun> cat xx.csv sort by student name filter by scores > 80  # LLM translates → awk + sort
```

Or execute one time:
```
$ srun test.sh
```

## Setup

- `DEEPSEEK_API_KEY` required in `~/.zshrc` or env (auto-loaded on startup)
- State persisted to `~/.srun/state.json`

## Architecture

```
User Input → classify (shell / python / unknown)
  ├─ Shell / Python → execute → success? done
  │                            └─ fail → LLM repair → retry (max 4 rounds)
  └─ Unknown → LLM generates (language, code) → execute → ...
```

## Tests

**Don't assume tests pass because they ran.** Verify each output against expected results. A failure is a failure — diagnose root cause first, then decide whether to fix code, fix test, or document as known limitation.

### Checklist

- [ ] Normal shell commands execute within 20ms
- [ ] Quick fixes (aliases, simple rules) complete within 20ms
- [ ] Pseudocode/natural language correctly translated by LLM
- [ ] Syntax errors/typos fixed by LLM and re-executed successfully
- [ ] Dangerous commands properly blocked
- [ ] `cd` persists across commands
- [ ] `ll`/`la` aliases work
- [ ] srun runs from any directory
- [ ] LLM fix shown as `⟳` preview before execution

