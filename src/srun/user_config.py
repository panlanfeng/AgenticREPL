"""User configuration — persisted to ~/.srun/user_config.json"""

import json
import os

CONFIG_FILE = os.path.join(os.path.expanduser("~"), ".srun", "user_config.json")

DEFAULTS = {
    "confirm_llm_code": False,
    "max_retry_rounds": 4,
    "provider": "deepseek",  # provider preset (deepseek, openai, anthropic, ...)
    "api_key": "",
    "api_base": "",          # override provider's default base URL
    "api_model": "",         # override provider's default model
    "temperature": 0.0,
    "top_p": 0.0,            # 0 = not set; alternative to temperature
    "max_tokens": 2000,
    "stream": True,          # set false for models with broken streaming
    "tool_choice": "auto",   # set "" to omit (some models don't support it)
    "mcp_servers": {},
}

TYPES = {
    "confirm_llm_code": bool,
    "max_retry_rounds": int,
    "provider": str,
    "api_key": str,
    "api_base": str,
    "api_model": str,
    "temperature": float,
    "top_p": float,
    "max_tokens": int,
    "stream": bool,
    "tool_choice": str,
    "mcp_servers": dict,
}

# Provider presets — each maps to an OpenAI-compatible /chat/completions endpoint.
# The env_var is auto-detected: if set in the environment, that provider activates.
# Priority: SRUN_API_KEY env > provider-specific env > user_config > defaults.
PROVIDERS = {
    "deepseek":    {"name": "DeepSeek",
                    "api_base": "https://api.deepseek.com/v1",
                    "api_model": "deepseek-chat",
                    "env_var": "DEEPSEEK_API_KEY"},
    "openai":      {"name": "OpenAI",
                    "api_base": "https://api.openai.com/v1",
                    "api_model": "gpt-4o",
                    "env_var": "OPENAI_API_KEY"},
    "anthropic":   {"name": "Anthropic",
                    "api_base": "https://api.anthropic.com/v1",
                    "api_model": "claude-sonnet-4-5",
                    "env_var": "ANTHROPIC_API_KEY"},
    "google":      {"name": "Google Gemini",
                    "api_base": "https://generativelanguage.googleapis.com/v1beta/openai",
                    "api_model": "gemini-2.5-pro",
                    "env_var": "GOOGLE_API_KEY"},
    "glm":         {"name": "Zhipu GLM",
                    "api_base": "https://open.bigmodel.cn/api/paas/v4",
                    "api_model": "glm-4-plus",
                    "env_var": "GLM_API_KEY"},
    "kimi":        {"name": "Moonshot Kimi",
                    "api_base": "https://api.moonshot.cn/v1",
                    "api_model": "moonshot-v1-8k",
                    "env_var": "KIMI_API_KEY"},
    "minimax":     {"name": "MiniMax",
                    "api_base": "https://api.minimax.chat/v1",
                    "api_model": "abab6.5s-chat",
                    "env_var": "MINIMAX_API_KEY"},
    "qwen":        {"name": "Alibaba Qwen",
                    "api_base": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                    "api_model": "qwen-max",
                    "env_var": "QWEN_API_KEY"},
    "xai":         {"name": "xAI",
                    "api_base": "https://api.x.ai/v1",
                    "api_model": "grok-2-1212",
                    "env_var": "XAI_API_KEY"},
    "openrouter":  {"name": "OpenRouter",
                    "api_base": "https://openrouter.ai/api/v1",
                    "api_model": "openai/gpt-4o",
                    "env_var": "OPENROUTER_API_KEY"},
    "siliconflow": {"name": "SiliconFlow",
                    "api_base": "https://api.siliconflow.cn/v1",
                    "api_model": "deepseek-ai/DeepSeek-V3",
                    "env_var": "SILICONFLOW_API_KEY"},
    "perplexity":  {"name": "Perplexity",
                    "api_base": "https://api.perplexity.ai",
                    "api_model": "sonar-pro",
                    "env_var": "PERPLEXITY_API_KEY"},
    "mistral":     {"name": "Mistral AI",
                    "api_base": "https://api.mistral.ai/v1",
                    "api_model": "mistral-large-latest",
                    "env_var": "MISTRAL_API_KEY"},
    "bedrock":     {"name": "Amazon Bedrock",
                    "api_base": "https://bedrock-runtime.us-east-1.amazonaws.com",
                    "api_model": "anthropic.claude-sonnet-4-5-v1:0",
                    "env_var": "AWS_ACCESS_KEY_ID"},
    "custom":      {"name": "Custom",
                    "api_base": "",
                    "api_model": "",
                    "env_var": None},
}

_cache = None
_cache_mtime = 0


def load():
    global _cache, _cache_mtime
    try:
        mtime = os.path.getmtime(CONFIG_FILE)
    except OSError:
        mtime = 0
    if _cache is not None and mtime == _cache_mtime:
        return dict(_cache)
    cfg = dict(DEFAULTS)
    if os.path.isfile(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                loaded = json.load(f)
                for k, v in loaded.items():
                    if k in TYPES and not isinstance(v, TYPES[k]):
                        v = DEFAULTS.get(k, v)
                    cfg[k] = v
        except (json.JSONDecodeError, IOError):
            pass
    _cache = dict(cfg)
    _cache_mtime = mtime
    return cfg


def save(cfg):
    global _cache, _cache_mtime
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)
    _cache = dict(cfg)
    try:
        _cache_mtime = os.path.getmtime(CONFIG_FILE)
    except OSError:
        _cache_mtime = 0


def get(key):
    return load().get(key, DEFAULTS.get(key))


def set(key, value):
    cfg = load()
    cfg[key] = value
    save(cfg)


def _auto_detect_provider():
    """Scan env vars for known provider-specific keys. Return (provider_key, api_key, api_base, api_model)
    for the first match found, or (None, "", "", "")."""
    for key, preset in PROVIDERS.items():
        if key == "custom":
            continue
        env_var = preset["env_var"]
        if env_var:
            env_val = os.environ.get(env_var, "")
            if env_val:
                return key, env_val, preset["api_base"], preset["api_model"]
    return None, "", "", ""


def get_api_config():
    """Return (api_key, api_base, api_model) with priority:
    1. SRUN_API_KEY / SRUN_API_BASE / SRUN_MODEL env vars (highest — explicit override)
    2. Provider-specific env var auto-detection (DEEPSEEK_API_KEY, OPENAI_API_KEY, ...)
    3. Config file: provider key pulls preset defaults for missing fields
    4. Hardcoded defaults (deepseek)"""
    cfg = load()

    # Priority 1: Generic SRUN_ env vars — explicit manual override, highest priority
    env_key = os.environ.get("SRUN_API_KEY", "")
    if env_key:
        env_base = os.environ.get("SRUN_API_BASE", "") or "https://api.deepseek.com/v1"
        env_model = os.environ.get("SRUN_MODEL", "") or "deepseek-chat"
        return env_key, env_base, env_model

    # Priority 2: Auto-detect from provider-specific env vars
    auto_provider, auto_key, auto_base, auto_model = _auto_detect_provider()
    if auto_provider:
        return auto_key, auto_base, auto_model

    # Priority 3: Config file — provider preset fills in missing fields
    provider = cfg.get("provider", "")
    api_key = cfg.get("api_key", "")
    api_base = cfg.get("api_base", "")
    api_model = cfg.get("api_model", "")

    if provider and provider in PROVIDERS:
        preset = PROVIDERS[provider]
        api_base = api_base or preset["api_base"]
        api_model = api_model or preset["api_model"]
        if not api_key and preset["env_var"]:
            api_key = os.environ.get(preset["env_var"], "")

    api_key = api_key or cfg.get("api_key", "")

    # Priority 4: Hardcoded defaults (deepseek)
    api_base = api_base or "https://api.deepseek.com/v1"
    api_model = api_model or "deepseek-chat"

    return api_key, api_base, api_model
