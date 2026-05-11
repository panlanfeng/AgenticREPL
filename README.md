# srun — The terminal that thinks with you

**Talk to your terminal in broken grammar, half-formed thoughts, or plain English. It figures out what you meant, writes the correct command, runs it, fixes errors without asking, and keeps your sessions alive.**

**Why not just use a coding agent?** Most coding agents send every keystroke through an LLM. `ls` takes 2 seconds. Variables, imports, and dataframes die between commands because each runs in a fresh process. `srun` keeps sessions alive. Python and R run as persistent processes — your data, variables, and packages stay loaded. The LLM only intervenes when there's an error or ambiguity. Normal commands execute at terminal speed: `<10ms` for shell, `<1ms` for Python expressions.

```bash
$ srun
srun> ls -la                                      # normal shell: zero delay
srun> 100/4                                       # 25.0

srun> compress the folder myproject into a tar.gz file
⟳  tar -czf myproject.tar.gz myproject

srun> find python files modified in last 7 days that contain the word TODO
⟳  find . -name '*.py' -mtime -7 -exec grep -l 'TODO' {} +
  ./tests/test_synthetic_project.py

srun> count lines in all .py files recursively
⟳  find . -name '*.py' -type f | xargs wc -l

srun> r                                           # enter R session
R> filter mtcars where mpg > 20, show mpg and cyl only
⟳  mtcars[mtcars$mpg > 20, c("mpg", "cyl")]
     mpg cyl
Mazda RX4      21.0   6
Mazda RX4 Wag  21.0   6
Datsun 710     22.8   4
Hornet 4 Drive 21.4   6
Merc 240D      24.4   4
Merc 230       22.8   4
Fiat 128       32.4   4
Honda Civic    30.4   4

R> group mtcars by cyl and gear, count rows, show average mpg
⟳  mtcars %>% group_by(cyl, gear) %>% summarise(count = n(), avg_mpg = mean(mpg), .groups = "drop")
     # A tibble: 8 × 4
     cyl  gear count avg_mpg
   <dbl> <dbl> <int>   <dbl>
 1     4     3     1    21.5
 2     4     4     8    26.9
 3     4     5     2    28.2
 4     6     3     2    19.8

R> in mtcars, group by cyl, compute mean of mpg, hp and wt, rounded to 1 decimal
⟳  mtcars %>% group_by(cyl) %>% summarise(across(c(mpg, hp, wt), ~ round(mean(.x), 1)))
     # A tibble: 3 × 4
       cyl   mpg    hp    wt
     <dbl> <dbl> <dbl> <dbl>
   1     4  26.7  82.6   2.3
   2     6  19.7 122.    3.1
   3     8  15.1 209.    4.0

R> plot mtcars with mpg on x, hp on y, color points by cyl as factor, add smooth regression line
⟳  library(ggplot2)
⟳  ggplot(mtcars, aes(x = mpg, y = hp, color = as.factor(cyl))) +
⟳    geom_point() +
⟳    geom_smooth(method = "lm", se = FALSE) +
⟳    labs(color = "cyl")
```

## How it works

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

- **Try first, ask later** — execute directly. Only call LLM when something fails.
- **Locked to current session** — shell commands in shell mode, R code in R mode. No surprise language switches.
- **Agent loop** — LLM can do multi-step: search → read → generate → execute → check → refine.
- **LLM sees output** — every `run_command` tool call returns actual exit code and last 20 lines of output.

## Error auto-repair

```bash
srun> grep --nocolor error /var/log/app.log
✗ grep: unrecognized option '--nocolor'
  ⟳ grep error /var/log/app.log

srun> pritn('hello world')
✗ NameError: name 'pritn' is not defined
  ⟳ print('hello world')
  hello world
```

## SSH remote + file execution

```bash
srun> ssh admin@prod-server
admin@prod-server$ find large files in /var/log    # NL works remotely too
```

```bash
$ srun deploy.sh
✓ docker build -t app .                         3200ms
✗ kubectl apply -f k8s/deploy.yaml                35ms
  → kubectl apply -f k8s/deployment.yaml          # LLM fixes typo, resumes
✓ kubectl apply -f k8s/deployment.yaml            28ms
```

## Quick start

```bash
pip install -e .
export DEEPSEEK_API_KEY="sk-xxx"
srun
```

No API key? srun still works as a smart REPL — tab completion, history, Python/R sessions, quick fixes.

## License

MIT
