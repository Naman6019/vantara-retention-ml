from pathlib import Path
from typing import Any, Dict

import yaml


def get_project_root() -> Path:
    """Returns the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """
    Loads and parses the YAML configuration file.

    Args:
        config_path: Optional custom path to config.yaml. Defaults to config/config.yaml.

    Returns:
        Dict containing project configuration settings.
    """
    root = get_project_root()
    if config_path is None:
        config_file = root / "config" / "config.yaml"
    else:
        config_file = Path(config_path)
        if not config_file.is_absolute():
            config_file = root / config_file

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_file}")

    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config
