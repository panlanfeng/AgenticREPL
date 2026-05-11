# AgenticREPL — Feature List & Design

## Key Features

| # | Feature | Why essential |
|---|---------|--------------|
| 1 | **Zero-latency execution** | Shell/Python commands execute directly (<10ms). LLM only called on errors or ambiguous input. Without this, every command incurs 1-3s LLM delay — unusable. |
| 2 | **Error repair loop** | Failed commands trigger LLM fix → retry (up to 4 rounds, configurable). Each retry feeds the latest error back to LLM. |
| 3 | **Quick fixes** | Common aliases/typos bypass LLM entirely (<1ms): `ll`→`ls -la`, `cd..`→`cd ..`. |
| 4 | **Natural language → command** | Ambiguous input goes directly to LLM. `"list all csv files"` → `find . -name "*.csv"`. |
| 5 | **Multi-language sessions** | `python` / `r` / default shell. Each session has persistent state (Python in-process, R persistent subprocess). |
| 6 | **Streaming token output** | LLM responses stream token-by-token in real-time. Tool call progress shown as `→ checking command: grep`. |
| 7 | **Server-side prompt caching** | Multi-turn conversation architecture. DeepSeek caches the entire conversation prefix — 90%+ cache hit rate. |
| 8 | **File execution mode** | `srun script.sh` executes shell scripts line-by-line, repairing errors per line. |
| 9 | **SSH remote mode** | `ssh user@host` enters transparent remote execution. Commands auto-wrapped through SSH. |
| 10 | **Agent tools** | `check_command`, `search_files`, `read_file`, `get_env_info`, `get_command_help` — LLM calls them to gather context. |
| 11 | **Danger detection** | Blocks destructive shell patterns (`rm -rf /`, `mkfs`, fork bombs) from LLM-generated code. |
| 12 | **Tab completion + history** | Readline: file path tab completion, arrow key history, `~/.srun/history`. |

## Key Design Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | **Single LLM entry point** | `llm.run(input, error=None)` — one method for both dispatch (new code) and repair (fix errors). No separate dispatch/repair logic. |
| 3 | **Persistent session state** | Python: in-process `exec()`. R: persistent subprocess with stdin/stdout pipe. Shell: stateless subprocess. Each language gets its own executor with state persistence where appropriate. |
| 4 | **`run_command` as the only execution tool** | LLM outputs code through `run_command(command)` function call. No JSON wrapping, no `language` field — the code IS the language. |
| 5 | **Environment-aware context** | `current_language` tracked in state, shown in LLM context. LLM generates code matching the current environment. Environment change notifications injected only on transition. |
| 10 | **Session isolation** | Each srun invocation creates unique `~/.srun/sessions/<id>/`. State, conversations, debug logs per-session. Auto-clean >90 days. |

## Core Routing Logic

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

## To do

- plan and execute mode
- ~~snip the old tool results. ~~
- ~~UX, transparency, different font~~
- ~~install the dependencies and ask for permission~~
- for danger commands, output the command, do not execute it until user approve. 
- ~~add the summary for context compact into conversation history. ~~
- ~~skill and mcp support~~
- ~~no need to specify the language. should always be the same as environment~~
- ~~allow user to add agents.md~~
- resume from conversation
- self evovling
- ~~ctrl + C only stop the current command in R, not the whole srun. ~~