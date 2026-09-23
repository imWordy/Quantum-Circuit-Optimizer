"""Config loading utility."""

from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(config_path: Path | str | None = None) -> Dict[str, Any]:
    """Load configuration yaml file.

    Args:
        config_path: Path to configuration YAML. If None, default is used.

    Returns:
        Dictionary containing configuration.
    """
    if config_path is None:
        # Default: project_root/config/config.yaml
        config_path = Path(__file__).resolve().parent.parent.parent.parent / "config" / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config
