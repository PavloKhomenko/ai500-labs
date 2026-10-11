from pathlib import Path

import yaml

CONFIG = Path(__file__).parents[2] / "config.yaml"


def load_config(path=CONFIG):
    return yaml.safe_load(Path(path).read_text())

# Seeds: use transformers.set_seed(cfg["seed"]).
