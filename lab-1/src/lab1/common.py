import os
from pathlib import Path

import yaml

CONFIG = Path(__file__).parents[2] / "config.yaml"


def load_config(path=CONFIG):
    return yaml.safe_load(Path(path).read_text())


def out_dir(cfg, *parts):
    path = Path(os.environ.get("LAB1_OUTPUT_DIR", cfg["output_dir"])).joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
