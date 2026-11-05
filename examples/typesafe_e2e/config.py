import json
import os
import shlex
from pathlib import Path
from urllib.request import Request, urlopen


def read_config(path):
    keys = ("TYPESAFE_API_KEY", "TRACEROOT_API_KEY", "TRACEROOT_HOST_URL")
    config = {key: os.environ[key] for key in keys if os.environ.get(key)}
    if path:
        for line in Path(path).read_text().splitlines():
            key, separator, value = line.strip().removeprefix("export ").partition("=")
            key = key.strip()
            if separator and key in keys and key not in config:
                parts = shlex.split(value, comments=True)
                if len(parts) == 1:
                    config[key] = parts[0]
    for key in keys[:2]:
        if not config.get(key):
            raise RuntimeError(f"Missing {key}")
    return config


def get_json(host, api_key, path):
    request = Request(host + path, headers={"Authorization": f"Bearer {api_key}"})
    with urlopen(request, timeout=30) as response:
        return json.load(response)
