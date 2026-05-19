# AgenticREPL — Make your terminal think for you

AgenticREPL unifies your terminal and AI agent into one. Talk to your terminal in plain language, half-formed thoughts, or any syntax you like. It figures out what you mean, translates it into the correct command, and runs it. Your agent knows what commands you've run, and your terminal knows to send your instructions to the agent.

**Why not just use a coding agent?** AgenticREPL puts code execution first and adds no additional latency unless your input needs repair or it is in natural language. Regular coding agents send every keystroke through an LLM — heavyweight and slow. We need AI agents that can do end-to-end work, but also agents that execute efficiently and quietly help only when needed.

For exploratory data analysis in particular, iterative data exploration gives researchers their first intuition about a problem. Some syntax in well-known packages like `dplyr`, `pandas`, or `ggplot2` is too verbose to remember. AgenticREPL lets you explore data in any syntax, or just in plain English. For example, this runs correctly in AgenticREPL's R session: `mtcars.filter(wt>5).mean(X) for X in (gear, carb)`. 

See more examples below.

```bash
$ srun
shell> ls -la                                      # normal shell: zero delay
shell> 100/4                                       # 25.0

shell> compress the folder myproject into a tar.gz file  # No need to remember those tar flags!
  | tar -czf myproject.tar.gz myproject

shell> find python files modified in last 7 days that contain the word TODO
  | find . -name '*.py' -mtime -7 -exec grep -l 'TODO' {} +
  ./tests/test_synthetic_project.py

shell> count lines in all .py files recursively
  | find . -name '*.py' -type f | xargs wc -l

shell> r                                           # enter R session
R> load mtcars
  | data(mtcars)
  | head(mtcars)
mpg cyl disp  hp drat    wt  qsec vs am gear carb
Mazda RX4         21.0   6  160 110 3.90 2.620 16.46  0  1    4    4
Mazda RX4 Wag     21.0   6  160 110 3.90 2.875 17.02  0  1    4    4
Datsun 710        22.8   4  108  93 3.85 2.320 18.61  1  1    4    1
Hornet 4 Drive    21.4   6  258 110 3.08 3.215 19.44  1  0    3    1
Hornet Sportabout 18.7   8  360 175 3.15 3.440 17.02  0  0    3    2
Valiant           18.1   6  225 105 2.76 3.460 20.22  1  0    3    1

R> mcars | group by cyl | mean(carb) | sort by carb  # if you don't want to type %>%
  | library(dplyr); mtcars |>
  | group_by(cyl) |>
  | summarise(mean_carb = mean(carb)) |>
  | arrange(mean_carb)
# A tibble: 3 × 2
    cyl mean_carb
  <dbl>     <dbl>
1     4      1.55
2     6      3.43
3     8      3.5

R> mtcars.filter(wt>5).mean(X) for X in (gear, carb)  # pandas syntax in R
  | library(dplyr)
  | for (X in c("gear", "carb")) {
  |   result <- mtcars |> filter(wt > 5) |> summarise(mean_val = mean(.data[[X]]))
  |   cat(X, ":", result$mean_val, "\n")
  | }
  gear : 3
  carb : 4


R> in mtcars, group by cyl, compute mean of mpg, hp and wt, rounded to 1 decimal
  | mtcars %>% group_by(cyl) %>% summarise(across(c(mpg, hp, wt), ~ round(mean(.x), 1)))
  # A tibble: 3 × 4
    cyl   mpg    hp    wt
  <dbl> <dbl> <dbl> <dbl>
1     4  26.7  82.6   2.3
2     6  19.7 122.    3.1
3     8  15.1 209.    4.0

R> plot mtcars with mpg on x, hp on y, color points by cyl as factor, add smooth regression line
  | library(ggplot2)
  | ggplot(mtcars, aes(x = mpg, y = hp, color = as.factor(cyl))) +
  |   geom_point() +
  |   geom_smooth(method = "lm", se = FALSE) +
  |   labs(color = "cyl")
```

## AgenticREPL is a full fledged AI Agent, understand all natural lanaguage instructions

It is not just translating your command as a one time request but it view all the full interaction history and understand your intention. You can ask it to read your repo, edit your files, fix a bug and run tests for you, just like a regular coding agent. 
```shell
shell> add comments for the file test.py, run it and fix any bugs
#or 
shell> analyze the following repo and explain the *.py files
```

## Error auto-repair and psuedo code being translated

```bash
shell> pritn('hello world')
✗ NameError: name 'pritn' is not defined
  | print('hello world')
  hello world
```

## Normal commands runs with no llm overhead
```bash
shell> git pull  # runs normally, does not go to llm but history being visible to the agent
```


## Quick start
Install via Homebrew or pip, then start with `srun`.

```bash
# Homebrew (macOS/Linux)
brew tap panlanfeng/agenticrepl
brew install agenticrepl

srun #start the REPL 
```
Or pip
```bash
pip install git+https://github.com/panlanfeng/AgenticREPL.git

srun                                # start the REPL
```
If `srun` is not found after install via pip:
```bash
which srun
export PATH="$(python -c 'import sysconfig; print(sysconfig.get_path("scripts"))'):$PATH"
```


## LLM Providers

AgenticREPL auto-detects provider API keys from environment variables. Set one in your `~/.zshrc`:

```bash
export DEEPSEEK_API_KEY="sk-xxx"     # for DeepSeek or
#OPENAI_API_KEY="sk-xxx"     # for OpenAI
#ANTHROPIC_API_KEY="sk-xxx"  # for Anthropic
```

No need to set model names or base URLs — they're filled from presets automatically.

| Provider | Env Var | Default Model | Base URL |
|----------|---------|---------------|----------|
| **DeepSeek** (default) | `DEEPSEEK_API_KEY` | `deepseek-v4-pro` | `https://api.deepseek.com/v1` |
| **OpenAI** | `OPENAI_API_KEY` | `gpt-5.5` | `https://api.openai.com/v1` |
| **Anthropic** | `ANTHROPIC_API_KEY` | `claude-opus-4-7` | `https://api.anthropic.com/v1` |
| **Google Gemini** | `GOOGLE_API_KEY` | `gemini-3.1-pro-preview` | `https://generativelanguage.googleapis.com/v1beta/openai` |
| **xAI** | `XAI_API_KEY` | `grok-4.3` | `https://api.x.ai/v1` |
| **Moonshot Kimi** | `KIMI_API_KEY` | `kimi-k2-thinking` | `https://api.moonshot.ai/v1` |
| **Zhipu GLM** | `GLM_API_KEY` | `glm-5.1` | `https://open.bigmodel.cn/api/paas/v4` |
| **Alibaba Qwen** | `QWEN_API_KEY` | `qwen3.6-plus` | `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` |
| **MiniMax** | `MINIMAX_API_KEY` | `MiniMax-M2.7` | `https://api.minimax.chat/v1` |
| **OpenRouter** | `OPENROUTER_API_KEY` | `openrouter/auto` | `https://openrouter.ai/api/v1` |
| **SiliconFlow** | `SILICONFLOW_API_KEY` | `deepseek-ai/DeepSeek-V4-Flash` | `https://api.siliconflow.cn/v1` |
| **Perplexity** | `PERPLEXITY_API_KEY` | `sonar-pro` | `https://api.perplexity.ai` |
| **Mistral AI** | `MISTRAL_API_KEY` | `mistral-large-2512` | `https://api.mistral.ai/v1` |
| **Amazon Bedrock** | `AWS_ACCESS_KEY_ID` | `anthropic.claude-opus-4-7-v1:0` | `https://bedrock-runtime.us-east-1.amazonaws.com` |

Or configure via `~/.srun/user_config.json`:

```json
{ "provider": "openai", "api_key": "sk-..." }
```

Override the default model:
```json
{ "provider": "deepseek", "api_key": "sk-...", "api_model": "deepseek-v4-flash" }
```

Or use the REPL interactive setup:
```
srun> /configure
```

Priority: `SRUN_API_KEY` env > provider-specific env > `~/.srun/user_config.json` > defaults.



## License

MIT
