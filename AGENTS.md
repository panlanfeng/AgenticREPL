
## AgenticREPL Project Overview

This is ligthweight AI agent focusing on efficient code execution with high tolerance to user input or code errors. It provide REPL for interactive usage `srun` and one time execution `srun test.sh`. The SRUN puts command execution first. Normal command will be executed with no extra latency and AI agent is only invoked when user user input is ambiguous or not directly executable. 

## Core routing logic

After user input, the classifer identify the input category. If the category is clear and match the current active language in REPL, immediately execute it in the current environment. If the input is ambiguous or category differs from the current active langauge, go to llm. The llm decides the langauge to be used and where it should be executed. The LLM firstly chooses the langauge based on the user intention; say user stating this is XX langauge, it should firstly respect user intention;
secondly, it should then choose the same language as the current environment. During the tool call generation, LLM should generate a code that is directly executable without relying on another classifer. If the language of the generated code matches the current environment, it sends the code to the current active session so the results are persistent. If the langauge of generated code is different from the current environment, identify if there is matching sessions in the backend. if yes, send it to the most recent matcing session. 
If user intention is not to generate any code, LLM replies back in text as well. The tool call in this case can be any langauge appropriate and executable.

Add a system reminder message about the current active execution language, python or shell and inject into the context; Update and send the system reminder again when user entered a different session mode. Simiarly, when user used a different python version in the user input. if the new version python works, update the state about this new python version.

## Code style

- Separate REPL UI code from the AI agent. 
- Add clear comments for each main lines. Document every key designs. 
- The deisgn should be generally applicable for all languages and minimize language specific customizations.
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

**Don't assume tests pass because they ran.** Verify each output against expected results. A failure is a failure — diagnose root cause first, then decide whether to fix code, fix test, or document as known limitation.

Use the deepseek API token to run the llm tests.


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
User input
  └─ Execute in current session (shell/Python/R)
       ├─ Success → done (zero latency)
       └─ Failure → LLM agent loop
            ├─ LLM uses tools: search files, check commands, read data
            ├─ LLM generates code → executed inline, output shown to user
            ├─ LLM sees output, can call more tools or stop
            └─ Loop until LLM stops or max rounds (up to 4 repair retries)
```
