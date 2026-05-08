"""User configuration — persisted to ~/.srun/user_config.json"""

import json
import os

CONFIG_FILE = os.path.join(os.path.expanduser("~"), ".srun", "user_config.json")

DEFAULTS = {
    "confirm_llm_code": False,
    "max_retry_rounds": 4,
}


def load():
    cfg = dict(DEFAULTS)
    if os.path.isfile(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                loaded = json.load(f)
                cfg.update(loaded)
        except (json.JSONDecodeError, IOError):
            pass
    return cfg


def save(cfg):
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


def get(key):
    return load().get(key, DEFAULTS.get(key))


def set(key, value):
    cfg = load()
    cfg[key] = value
    save(cfg)
