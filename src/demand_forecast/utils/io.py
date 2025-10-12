from __future__ import annotations
import os
from pathlib import Path
from typing import Dict
import yaml


def load_yaml(path: str | os.PathLike) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(*paths: str | os.PathLike) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)