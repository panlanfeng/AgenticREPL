# srun - Smart Run

在终端里输入 `srun` 进入智能 REPL。

## 核心理念

正常命令正常执行，有问题时才介入。零额外延迟。

## 使用方式

```
$ srun
> ls -la              # 正常 shell，直接执行，0 延迟
> cat data.csv        # 正常 shell，直接执行，0 延迟
> cat xx.csv sort by student name | grep scores > 80  # 错误命令，LLM 修复后执行
```

## 平台

macOS, Linux

## 关键原则

- 用户直接输入的命令不做安全检查，直接执行
- 正常命令零感知延迟
- 仅错误/不规范时调用 LLM
- LLM prompt 极短，输出仅限可执行命令，不做解释
- LLM 可调工具获取上下文（文件元数据、系统信息等）
- LLM 默认: DeepSeek v4 Flash (OpenAI 兼容 API)
- LLM 输出的 shell 命令过危险检测，python 代码直接执行
- 状态文件: `.srun/state.json`，记录变量 schema、文件列表、系统信息

## API Key

从环境变量 `DEEPSEEK_API_KEY` 读取，不写进代码。测试时在 `~/.zshrc` 中 export 即可：
```bash
export DEEPSEEK_API_KEY="sk-xxx"
```

## 架构

用户输入 → 分类(Python/Shell/未知)
  ├─ Python AST? → 列名自动加引号(region→"region") → exec()
  ├─ Shell?     → subprocess
  └─ 未知       → LLM 调度 → 生成代码 → 执行
                     │
                  失败 → LLM 修复(max 3次) → 重试

## 测试

### 测试原则

**不要想当然认为测试通过。** 必须逐条检查实际输出是否符合预期，不论问题是 srun 代码、LLM 还是环境导致。一个测试用例失败就是失败，需要定位根因后再决定是修代码、改测试还是标记为已知限制。

### 准备测试数据

```
tests/
├── data/
│   └── test.csv                    # 基础测试数据
├── synthetic_project/              # 模拟真实项目
│   ├── data/sales.csv              # 销售数据
│   ├── data/config.json
│   ├── src/utils.py                # Python 工具模块
│   ├── src/app.py                  # 可执行入口
│   ├── logs/
│   └── README.md                   # 含 TODO 标记
├── test_dispatch.py                # 分类测试
├── test_shell_exec.py              # Shell 执行测试
├── test_python_exec.py             # Python 执行测试
├── test_repair.py                  # 快速修复测试
├── test_danger.py                  # 危险检测测试
├── test_integration.py             # 端到端集成测试
└── test_synthetic_project.py       # 模拟用户会话测试
```

### 运行测试

```bash
# 全部测试
pytest tests/ -v

# 仅快速测试（不调 LLM）
pytest tests/ -v -m "not slow and not llm"

# 仅 LLM 测试
pytest tests/ -v -m "llm"

# 模拟用户会话
pytest tests/test_synthetic_project.py -v -s
```

### 测试检查项

每次测试必须逐项确认：
- [ ] 正常 shell 命令延迟 < 20ms（零感知）
- [ ] 快速修复（别名、简单规则）延迟 < 20ms
- [ ] 伪代码/自然语言被 LLM 正确翻译为可执行命令
- [ ] 语法错误/拼写错误被 LLM 修复后成功执行
- [ ] 危险命令被正确拦截
- [ ] `cd` 命令目录切换持久生效
- [ ] `ll`/`la` 等别名正常工作
- [ ] srun 可在任意目录启动运行

### 已知限制

- LLM 修复质量受模型能力限制，可能不感知当前平台差异（如 macOS BSD vs GNU 命令选项）
