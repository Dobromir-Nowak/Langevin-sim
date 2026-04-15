from pathlib import Path
from typing import Any
import yaml

def load_config(file_name: str) -> dict:
    path = Path(__file__).parent.parent.parent.parent / "configs" / f"{file_name}.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)
