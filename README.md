# srun — The terminal that thinks with you

**Normal commands at full speed. Ambiguous thoughts translated to code. Errors fixed before you notice.**

```bash
$ srun
srun> ls -la                      # executes in 5ms — regular terminal speed
srun> 100/4                       # 25.0
srun> ll                          # ⟳ ls -la — alias fixed instantly, no LLM
srun> cat data.csv sort by student name filter by scores > 80
← LLM understands the intent, generates:
⟳  awk -F, 'NR==1 || $3>80' data.csv | sort -t, -k2
srun> python                      # enter persistent Python session
python> import pandas as pd
python> df = pd.read_csv('sales.csv')
python> df.groupby('region').revenue.mean()  # state persists
python> exit()
srun> r                           # enter persistent R session
R> library(dplyr)
R> mtcars %>% group_by(cyl) %>% summarise(mpg = mean(mpg))
R> exit()
```

## Wait, why does 100/4 work in a terminal?

srun classifies your input before executing it. Expressions get routed to Python. Commands to the shell. Ambiguous natural language to the LLM.

```
Your input
  ├─ Valid shell/Python/R command? → execute immediately (<10ms)
  │   └─ Failed? → LLM reads the error, fixes it, retries
  │
  └─ Ambiguous / natural language? → LLM generates executable code
       ├─ Respects your stated language ("in Python, load csv")
       ├─ Matches your current session (Python mode → Python code)
       └─ Routes to the right executor (shell, Python, or R)
```

## Why not just use a coding agent?

Most coding agents send every keystroke through an LLM. `ls` takes 2 seconds. Variables, imports, and dataframes die between commands because each runs in a fresh process.

srun keeps sessions alive. Python and R run as persistent processes — your data, variables, and packages stay loaded. The LLM only intervenes when there's an error or ambiguity. Normal commands execute at terminal speed: `<10ms` for shell, `<1ms` for Python expressions.

## Quick start

```bash
pip install -e .
export DEEPSEEK_API_KEY="sk-xxx"
srun
```

LLM is optional. Without an API key, srun works as a fast REPL with Python, R, tab completion, history, and quick fixes.

## Real examples

**Mixed shell/Python data pipeline:**
```
srun> head -3 sales.csv
  region,product,revenue
  East,Widget A,1200
  West,Widget B,800
srun> wc -l sales.csv
  243 sales.csv
srun> python
python> import pandas as pd
python> df = pd.read_csv('sales.csv')
python> df.groupby('region').revenue.sum()
  East     45200
  North    38100
  South    29400
  West     41200
python> df[df.revenue > 1000].to_csv('top_sales.csv', index=False)
python> exit()
srun> wc -l top_sales.csv
  87 top_sales.csv
```

**Flexible grammar — srun understands intent, not syntax:**
```
srun> find all csv files                             # natural language
  ⟳ find . -name "*.csv"

srun> load sales.csv and show average revenue by region   # NL → pandas
  ⟳ python3 -c "
import pandas as pd
df = pd.read_csv('sales.csv')
print(df.groupby('region').revenue.mean())
"
  East     4520.0
  North    3810.0
  South    2940.0
  West     4120.0

srun> count lines in all .py files recursively
  ⟳ find . -name "*.py" | xargs wc -l | tail -1

srun> df | group_by(char1) | num1=mean(num1), num2=mean(num2)
← Even broken pipe syntax works — LLM infers the dplyr intent
  ⟳ df %>% group_by(char1) %>% summarise(num1 = mean(num1), num2 = mean(num2))
```

**Error auto-repair — typos, wrong flags, broken syntax:**
```
srun> grep --nocolor error /var/log/app.log
✗ grep: unrecognized option '--nocolor'
  ⟳ grep error /var/log/app.log                      # LLM strips bad flag

srun> pritn('hello world')                            # typo
✗ NameError: name 'pritn' is not defined
  ⟳ print('hello world')
  hello world

srun> kubectl apply -f k8s/deploy.yaml                # wrong filename
✗ error: the path "k8s/deploy.yaml" does not exist
  ⟳ kubectl apply -f k8s/deployment.yaml              # LLM guesses correct file
  deployment.apps/app configured
```

**Multi-line input — incomplete statements don't trigger LLM:**
```
python> def transform(df):
...      return df.assign(total = df.x + df.y)
python> transform(df)

R> df %>%
...   group_by(char1) %>%
...   summarise(num1 = mean(num1), num2 = mean(num2))
```

Incomplete lines (`def`, `for`, `df %>%`, shell `\`) trigger a `...` continuation prompt instead of executing prematurely.

**SSH remote with full repair:**
```
srun> ssh admin@prod-server
Connected to admin@prod-server (type 'exit' to disconnect)
admin@prod-server$ df -h /data
admin@prod-server$ grep error /var/log/app.log | sort | uniq -c
```

**File execution with per-line repair:**
```
$ srun deploy.sh
Executing 12 commands from deploy.sh

✓ docker build -t app .                                         3200ms
✓ docker push registry/app:latest                               4500ms
✗ kubectl apply -f k8s/deploy.yaml                                35ms
  → kubectl apply -f k8s/deployment.yaml
✓ kubectl apply -f k8s/deployment.yaml                            28ms
✓ curl -s https://app.example.com/health | grep OK               150ms

11/12 passed, 2 LLM calls, 9800ms total
```

## Sessions

Sessions keep state alive. Cross-language execution doesn't hijack your context — only explicit `python`/`r`/`exit()` switch sessions.

```
srun> python          # enter Python (persistent, in-process)
python> x = [1, 2, 3]
python> sum(x)        # 6
python> exit()

srun> r               # enter R (persistent subprocess)
R> x <- c(1, 2, 3)
R> mean(x)            # 2
R> exit()
```

## Config

Edit `~/.srun/user_config.json`:

```json
{
  "confirm_llm_code": false,
  "max_retry_rounds": 4
}
```

## License

MIT
