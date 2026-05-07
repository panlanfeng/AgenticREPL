# srun — Smart REPL

This is not another AI terminal but an LLM-powered lightweight REPL. Type shell commands or other code naturally and add natural langauges descriptions when you don't remember the details. Correct commands execute instantly, typos get fixed automatically and natural langauge description will be translate into the correct command.
This REPL is able to execute normal commands, psuedo code, code mixed with natural languages or even natural langauge description itself. 

The SRUN puts command execution first and AI agent as a suplementary tool when user user input is not a executable command. 

I uses coding agent as terminal to execute code in many cases instead of running it directly in terminal because coding agent is smarter with error tolerance. I don't want to hear from the shell telling me `xx` dependency is missing. I want it to not stop and wait for my input but just fix the minor errors and continue to wrok. However, a coding agent could still wait there for my instruction and a coding agent could be too heavy weight do the code execution job. This SRUN repl is particularly built for this cases and aim to execute it fast than a coding agent. 

It is impossible for me to remember the meaning of the long list of arguments of `tar`, `rsync` etc. 

## Usage Examples

```bash
$ srun
> ls -la              # normal shell, 0 delay
> 100/4               # arithmetic expression → 25.0
> ll                  # alias auto-expanded → ls -la
> cat data.csv sort by student name filter by scores > 80  # bad command → LLM fixes → runs
```


## LLLM API Key setting

Bring your own API Key, for example add the following to your .bashrc or .zshrc.

```bash
export DEEPSEEK_API_KEY="sk-xxx"
```


## Features

- **Zero-latency execution**: normal shell and Python commands execute directly (<10ms)
- **Error auto-repair**: typos, wrong flags, broken pipes → LLM fixes and retries (up to 4 rounds)
- **Quick fixes**: common aliases (`ls -all`→`ls -la`) resolve instantly
- **Natural language**: type pseudo-code or plain language — LLM translates to executable commands
- **Cross-language**: Python, shell, and R in one REPL
- **SSH remote mode**: `ssh user@host` enters transparent remote execution, LLM repair works remotely
