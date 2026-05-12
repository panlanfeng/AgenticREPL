"""User configuration — persisted to ~/.srun/user_config.json"""

import json
import os

CONFIG_FILE = os.path.join(os.path.expanduser("~"), ".srun", "user_config.json")

DEFAULTS = {
    "confirm_llm_code": False,
    "max_retry_rounds": 4,
    "api_key": "",
    "api_base": "",
    "api_model": "",
    "mcp_servers": {},
}

TYPES = {
    "confirm_llm_code": bool,
    "max_retry_rounds": int,
    "api_key": str,
    "api_base": str,
    "api_model": str,
    "mcp_servers": dict,
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


def get_api_config():
    """Return (api_key, api_base, api_model) with priority:
    1. SRUN_API_KEY / SRUN_API_BASE / SRUN_MODEL env vars
    2. api_key / api_base / api_model in user_config.json"""
    cfg = load()
    api_key = os.environ.get("SRUN_API_KEY", "") or cfg.get("api_key", "")
    api_base = os.environ.get("SRUN_API_BASE", "") or cfg.get("api_base", "")
    api_model = os.environ.get("SRUN_MODEL", "") or cfg.get("api_model", "")
    return api_key, api_base, api_model
